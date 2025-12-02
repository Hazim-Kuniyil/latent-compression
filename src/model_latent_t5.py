from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput


class LatentCrossAttentionBlock(nn.Module):
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
        latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(latents)
        k = self.k_proj(context)
        v = self.v_proj(context)

        key_padding_mask = context_mask.eq(0)

        attn_out, attn_weights = self.attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=None,
        )

        latents = self.ln1(latents + attn_out)

        ff_out = self.ff(latents)
        latents = self.ln2(latents + ff_out)

        return latents, attn_weights


class LatentSelector(nn.Module):
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

        self.latent_init = nn.Parameter(
            torch.randn(num_latents, latent_dim) * 0.02
        )

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
        context_hidden: torch.Tensor,
        context_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = context_hidden.size(0)

        latents = self.latent_init.unsqueeze(0).expand(B, -1, -1)

        context_proj = self.context_proj(context_hidden)

        last_attn = None
        for layer in self.layers:
            latents, attn_weights = layer(latents, context_proj, context_attention_mask)
            last_attn = attn_weights

        return latents, last_attn


@dataclass
class LatentT5Config:
    t5_model_name: str = "google/flan-t5-base"
    encoder_model_name: str = "allenai/longformer-base-4096"
    num_latents: int = 64
    num_latent_layers: int = 2
    num_latent_heads: int = 8
    latent_dropout: float = 0.1
    aux_loss_weight: float = 0.5
    freeze_t5_initially: bool = False
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True


class LatentT5Model(nn.Module):
    def __init__(self, config: LatentT5Config):
        super().__init__()
        self.config = config

        self.context_encoder = AutoModel.from_pretrained(config.encoder_model_name)
        context_hidden_dim = self.context_encoder.config.hidden_size

        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(config.t5_model_name)
        t5_hidden_dim = self.t5.config.d_model

        if config.freeze_t5_initially:
            for p in self.t5.parameters():
                p.requires_grad = False

        if getattr(config, "use_gradient_checkpointing", False):
            if hasattr(self.t5, "gradient_checkpointing_enable"):
                self.t5.config.use_cache = False
                self.t5.gradient_checkpointing_enable()
            if hasattr(self.context_encoder, "gradient_checkpointing_enable"):
                self.context_encoder.gradient_checkpointing_enable()

        self.latent_selector = LatentSelector(
            num_latents=config.num_latents,
            latent_dim=t5_hidden_dim,
            context_hidden_dim=context_hidden_dim,
            num_layers=config.num_latent_layers,
            num_heads=config.num_latent_heads,
            dropout=config.latent_dropout,
        )

        self.latent_gate = nn.Parameter(torch.tensor(0.5))

        self.aux_loss_weight = config.aux_loss_weight

    def _build_encoder_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_outputs = self.context_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
        )
        context_hidden = context_outputs.last_hidden_state

        latents, attn_weights = self.latent_selector(
            context_hidden=context_hidden,
            context_attention_mask=context_attention_mask,
        )

        gate = torch.tanh(self.latent_gate)
        gated_latents = latents * gate

        t5_encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        question_hidden = t5_encoder_outputs.last_hidden_state

        encoder_hidden = torch.cat([question_hidden, gated_latents], dim=1)

        B = input_ids.size(0)
        latent_mask = torch.ones(
            (B, latents.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        encoder_attention_mask = torch.cat(
            [attention_mask, latent_mask], dim=1
        )

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
        encoder_hidden, encoder_attention_mask, attn_weights = self._build_encoder_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
        )

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
            return_dict=True,
        )

        answer_loss = outputs.loss if labels is not None else None

        aux_loss = None
        if support_token_mask is not None and attn_weights is not None:
            attn_over_tokens = attn_weights.mean(dim=1)
            support_mask = support_token_mask.float()
            pos_mass = (attn_over_tokens * support_mask).sum(dim=-1)
            aux_loss = -torch.log(pos_mass + 1e-6).mean()

        loss = None
        if labels is not None:
            if aux_loss is not None:
                loss = answer_loss + self.aux_loss_weight * aux_loss
            else:
                loss = answer_loss

        return {
            "loss": loss,
            "answer_loss": answer_loss,
            "aux_loss": aux_loss,
            "logits": outputs.logits,
            "gate_value": torch.tanh(self.latent_gate).detach(),
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
