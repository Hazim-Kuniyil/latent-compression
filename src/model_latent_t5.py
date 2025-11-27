# src/model_latent_t5.py
"""
Latent Bottleneck Retriever-Reader model.

High-level:
- A shared encoder (e.g., DistilBERT) encodes the long retrieved context.
- A small set of learnable latents cross-attend over the context (Perceiver-style).
- The latents are projected into T5's hidden size and passed to T5's encoder output
  with a Flamingo-style tanh gate.
- T5's decoder generates the answer.

Expected batch fields (see forward() docstring for details):
    input_ids:              [B, L_q]       T5 tokenizer ids (question / prompt).
    attention_mask:         [B, L_q]
    context_input_ids:      [B, L_ctx]     Encoder tokenizer ids (retrieved context).
    context_attention_mask: [B, L_ctx]
    labels:                 [B, L_y]       T5 tokenizer ids for answer (shifted by collator).
    support_token_mask:     [B, L_ctx]     0/1 mask for support tokens (optional).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput


# -----------------------------
#  Latent cross-attention block
# -----------------------------

class LatentCrossAttentionBlock(nn.Module):
    """
    One layer of:
        latents = latents + CrossAttention(latents -> context)
        latents = latents + FFN(latents)

    Uses nn.MultiheadAttention with batch_first=True.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(context_dim, latent_dim)
        self.v_proj = nn.Linear(context_dim, latent_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln1 = nn.LayerNorm(latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)

        self.ff = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,            # [B, M, D_latent]
        context: torch.Tensor,            # [B, L_ctx, D_ctx]
        context_mask: torch.Tensor,       # [B, L_ctx] (1 = real, 0 = pad)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project to attn space
        q = self.q_proj(latents)          # [B, M, D_latent]
        k = self.k_proj(context)          # [B, L_ctx, D_latent]
        v = self.v_proj(context)          # [B, L_ctx, D_latent]

        # key_padding_mask: True for PAD, False for real tokens
        key_padding_mask = context_mask.eq(0)  # [B, L_ctx]

        attn_out, attn_weights = self.attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=None,
        )
        # attn_out:     [B, M, D_latent]
        # attn_weights: [B, M, L_ctx]  (averaged over heads)

        # Residual + layernorm
        latents = self.ln1(latents + attn_out)

        # Feedforward
        ff_out = self.ff(latents)
        latents = self.ln2(latents + ff_out)

        return latents, attn_weights


# -----------------------------
#  Latent selector module
# -----------------------------

class LatentSelector(nn.Module):
    """
    Perceiver-style latent bottleneck over long context.

    - Start from a fixed set of learned latent vectors.
    - Run several LatentCrossAttentionBlock layers where latents attend to
      the full context.
    - Optionally project context hidden size -> latent_dim.

    Returns:
        latents:      [B, M, latent_dim]
        attn_weights: [B, M, L_ctx] (from the last layer)
    """

    def __init__(
        self,
        num_latents: int,
        latent_dim: int,
        context_hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim

        # Fixed learned latents
        self.latent_init = nn.Parameter(
            torch.randn(num_latents, latent_dim) * 0.02
        )

        # Project context hidden states into latent_dim if needed
        if context_hidden_dim != latent_dim:
            self.context_proj = nn.Linear(context_hidden_dim, latent_dim)
            context_dim_for_blocks = latent_dim
        else:
            self.context_proj = nn.Identity()
            context_dim_for_blocks = context_hidden_dim

        self.layers = nn.ModuleList(
            [
                LatentCrossAttentionBlock(
                    latent_dim=latent_dim,
                    context_dim=context_dim_for_blocks,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        context_hidden: torch.Tensor,        # [B, L_ctx, D_ctx]
        context_attention_mask: torch.Tensor # [B, L_ctx]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = context_hidden.size(0)

        # Expand latents across batch
        latents = self.latent_init.unsqueeze(0).expand(B, -1, -1)  # [B, M, D_latent]

        # Project context into latent_dim if needed
        context_proj = self.context_proj(context_hidden)           # [B, L_ctx, latent_dim]

        last_attn = None
        for layer in self.layers:
            latents, attn_weights = layer(latents, context_proj, context_attention_mask)
            last_attn = attn_weights  # [B, M, L_ctx] from last layer

        return latents, last_attn


# -----------------------------
#  Full latent-augmented T5 model
# -----------------------------

@dataclass
class LatentT5Config:
    t5_model_name: str = "google/flan-t5-base"
    # CHANGE: Default to Longformer for long context support
    encoder_model_name: str = "allenai/longformer-base-4096"

    num_latents: int = 64  # Increased from 32
    num_latent_layers: int = 2
    num_latent_heads: int = 8
    latent_dropout: float = 0.1

    aux_loss_weight: float = 0.5  # weight for support-attention loss

    # NEW: training / memory options
    freeze_t5_initially: bool = False
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True  # used by the training script


class LatentT5Model(nn.Module):
    """
    Wrapper model combining:
        - context encoder (e.g., DistilBERT)
        - latent selector
        - T5 seq2seq with Flamingo-style gate

    forward() expects a batch dict with at least:
        input_ids:              [B, L_q]       (T5 input ids)
        attention_mask:         [B, L_q]
        context_input_ids:      [B, L_ctx]     (encoder input ids)
        context_attention_mask: [B, L_ctx]
        labels:                 [B, L_y]       (T5 labels)  (optional at eval)
        support_token_mask:     [B, L_ctx]     (0/1 mask)   (optional)
    """

    def __init__(self, config: LatentT5Config):
        super().__init__()
        self.config = config

        # Context encoder (for long retrieved context)
        self.context_encoder = AutoModel.from_pretrained(config.encoder_model_name)
        context_hidden_dim = self.context_encoder.config.hidden_size

        # T5 model for question/answering
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(config.t5_model_name)
        t5_hidden_dim = self.t5.config.d_model

        # NEW: optionally freeze T5 params initially
        if config.freeze_t5_initially:
            for p in self.t5.parameters():
                p.requires_grad = False

        # NEW: gradient checkpointing for memory
        if getattr(config, "use_gradient_checkpointing", False):
            if hasattr(self.t5, "gradient_checkpointing_enable"):
                # required for HF seq2seq + checkpointing
                self.t5.config.use_cache = False
                self.t5.gradient_checkpointing_enable()
            if hasattr(self.context_encoder, "gradient_checkpointing_enable"):
                self.context_encoder.gradient_checkpointing_enable()

        # Latent selector
        self.latent_selector = LatentSelector(
            num_latents=config.num_latents,
            latent_dim=t5_hidden_dim,                # match T5 hidden size
            context_hidden_dim=context_hidden_dim,
            num_layers=config.num_latent_layers,
            num_heads=config.num_latent_heads,
            dropout=config.latent_dropout,
        )

        # Flamingo-style gate (scalar)
        # Initialize to 0.5 → tanh(0.5) ≈ 0.46
        # This allows ~46% of latent signal through immediately,
        # forcing T5 to use the latent context from the start
        # while avoiding the instability of a fully-open gate.
        self.latent_gate = nn.Parameter(torch.tensor(0.5))

        # Loss weight for support-fact supervision
        self.aux_loss_weight = config.aux_loss_weight

    def _build_encoder_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build encoder_hidden and encoder_attention_mask, shared
        between forward() (teacher forcing) and generate().

        Returns:
            encoder_hidden: [B, L_q + M, D_t5]
            encoder_attention_mask: [B, L_q + M]
            attn_weights: [B, M, L_ctx] - attention weights from latent selector
        """
        # Encode context
        context_outputs = self.context_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
        )
        context_hidden = context_outputs.last_hidden_state  # [B, L_ctx, D_ctx]

        # Latent selector
        latents, attn_weights = self.latent_selector(
            context_hidden=context_hidden,
            context_attention_mask=context_attention_mask,
        )

        gate = torch.tanh(self.latent_gate)
        gated_latents = latents * gate  # [B, M, D_t5]

        # Encode question with T5 encoder
        t5_encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        question_hidden = t5_encoder_outputs.last_hidden_state  # [B, L_q, D_t5]

        # Concatenate question tokens and latent tokens
        encoder_hidden = torch.cat([question_hidden, gated_latents], dim=1)  # [B, L_q + M, D_t5]

        B = input_ids.size(0)
        latent_mask = torch.ones(
            (B, latents.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        encoder_attention_mask = torch.cat(
            [attention_mask, latent_mask], dim=1
        )  # [B, L_q + M]

        return encoder_hidden, encoder_attention_mask, attn_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        support_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids:              [B, L_q]       T5 question tokens.
            attention_mask:         [B, L_q]       1/0.
            context_input_ids:      [B, L_ctx]     encoder tokens (retrieved context).
            context_attention_mask: [B, L_ctx]     1/0.
            labels:                 [B, L_y]       T5 labels; required for training.
            support_token_mask:     [B, L_ctx]     0/1 mask for support tokens (optional).

        Returns:
            A dict with keys:
                loss:         total loss (if labels is not None)
                answer_loss:  language modeling loss
                aux_loss:     support-attention loss (or 0 if not used)
                logits:       [B, L_y, vocab_size]
        """

        # Build encoder states using the helper
        encoder_hidden, encoder_attention_mask, attn_weights = self._build_encoder_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
        )

        # Run T5 decoder with custom encoder outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
            return_dict=True,
        )

        answer_loss = outputs.loss if labels is not None else None

        # 7) Auxiliary support-fact attention loss (optional)
        aux_loss = None
        if support_token_mask is not None and attn_weights is not None:
            # attn_weights: [B, M, L_ctx]; average over latents -> [B, L_ctx]
            attn_over_tokens = attn_weights.mean(dim=1)  # [B, L_ctx]

            # Normalize support_token_mask to float
            support_mask = support_token_mask.float()  # [B, L_ctx]

            # Mass of attention on support tokens
            pos_mass = (attn_over_tokens * support_mask).sum(dim=-1)  # [B]

            # CRITICAL FIX: Add epsilon (1e-6) inside log to prevent explosion
            # This guarantees loss never exceeds ~13.8, and usually stays lower.
            # No clamping needed - the epsilon handles numerical stability gracefully.
            aux_loss = -torch.log(pos_mass + 1e-6).mean()

        # 8) Combine losses
        loss = None
        if labels is not None:
            if aux_loss is not None:
                loss = answer_loss + self.aux_loss_weight * aux_loss
            else:
                loss = answer_loss

        # For HF Trainer compatibility, just return a dict
        return {
            "loss": loss,
            "answer_loss": answer_loss,
            "aux_loss": aux_loss,
            "logits": outputs.logits,  # [B, L_y, vocab_size]
            "gate_value": torch.tanh(self.latent_gate).detach(),  # scalar for logging
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        **gen_kwargs,
    ) -> torch.Tensor:
        """
        Generate answers using T5, conditioned on latent-compressed context.

        Returns:
            generated_ids: [B, L_gen]
        """
        encoder_hidden, encoder_attention_mask, _ = self._build_encoder_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
        )

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        generated_ids = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            **gen_kwargs,
        )
        return generated_ids
