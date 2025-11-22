# train_latent_t5.py
"""
Train + eval script for the LatentT5Model on Hotpot-style data.

Expected input files: JSONL with columns:
    - id: str
    - question: str
    - answer: str
    - supporting_facts: dict {'title': [...], 'sent_id': [...]}
    - context: dict {'title': [...], 'sentences': [[...], ...]}

Adjust the pandas loading logic if your files are in a different format.
"""

import argparse
import math
import os
import re
import string
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    T5TokenizerFast,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

from model_latent_t5 import LatentT5Model, LatentT5Config
from data_hotpot import HotpotDataset, collate_latent


# =========================
#   Text metrics utilities
# =========================

def _normalize_answer(s: str) -> str:
    """Lowercase, strip, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score(pred: str, truth: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    truth_tokens = _normalize_answer(truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(truth_tokens)
    return 2 * prec * rec / (prec + rec)


def _exact_match(pred: str, truth: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(truth))


# =========================
#   Training / eval loops
# =========================

def _build_optimizer(model: LatentT5Model, lr_latent=1e-4, lr_t5=1e-5, lr_encoder=1e-5):
    """Separate learning rates for latent modules, T5, and context encoder."""
    def _trainable(params):
        return [p for p in params if p.requires_grad]

    latent_params = list(model.latent_selector.parameters()) + [model.latent_gate]
    t5_params = list(model.t5.parameters())
    encoder_params = list(model.context_encoder.parameters())

    optimizer = AdamW(
        [
            {
                "params": _trainable(latent_params),
                "lr": lr_latent,
            },
            {
                "params": _trainable(t5_params),
                "lr": lr_t5,
            },
            {
                "params": _trainable(encoder_params),
                "lr": lr_encoder,
            },
        ]
    )
    return optimizer


def train_one_epoch(
    model: LatentT5Model,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    grad_accum_steps: int = 1,
    max_norm: float = 1.0,
    log_every: int = 50,
    epoch_idx: int = 0,
    use_mixed_precision: bool = False,
    scaler: Optional[GradScaler] = None,
):
    model.train()
    global_step = 0
    running_loss = 0.0

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with autocast("cuda", enabled=use_mixed_precision):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                context_input_ids=batch["context_input_ids"],
                context_attention_mask=batch["context_attention_mask"],
                labels=batch["labels"],
                support_token_mask=batch["support_token_mask"],
            )
            loss = outputs["loss"]

        # --- NaN/Inf guard ---
        if not torch.isfinite(loss):
            print(
                f"[Epoch {epoch_idx}] Non-finite loss detected: "
                f"loss={loss}, answer_loss={outputs['answer_loss']}, aux_loss={outputs['aux_loss']}"
            )
            optimizer.zero_grad(set_to_none=True)
            continue
        # ---------------------

        loss = loss / grad_accum_steps

        if use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item()

        if (step + 1) % grad_accum_steps == 0:
            if use_mixed_precision:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = running_loss / log_every
                print(
                    f"[Epoch {epoch_idx} | Step {global_step}] "
                    f"loss = {avg_loss:.4f}"
                )
                running_loss = 0.0


@torch.no_grad()
def evaluate(
    model: LatentT5Model,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer: T5TokenizerFast,
    compute_metrics: bool = True,
    max_gen_len: int = 32,
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Returns:
        mean_loss, EM, F1
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_refs = []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            context_input_ids=batch["context_input_ids"],
            context_attention_mask=batch["context_attention_mask"],
            labels=batch["labels"],
            support_token_mask=batch["support_token_mask"],
        )
        loss = outputs["loss"]
        total_loss += loss.item()
        num_batches += 1

        if compute_metrics:
            gen_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                context_input_ids=batch["context_input_ids"],
                context_attention_mask=batch["context_attention_mask"],
                max_new_tokens=max_gen_len,
            )
            preds = tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            all_preds.extend(preds)

            refs = batch.get("target_texts", None)
            if refs is None:
                labels = batch["labels"].clone()
                labels[labels == -100] = tokenizer.pad_token_id
                refs = tokenizer.batch_decode(
                    labels,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            all_refs.extend(refs)

    mean_loss = total_loss / max(1, num_batches)
    print(f"[Eval] mean loss = {mean_loss:.4f}")

    em = f1 = None
    if compute_metrics and all_preds:
        em_scores = [_exact_match(p, r) for p, r in zip(all_preds, all_refs)]
        f1_scores = [_f1_score(p, r) for p, r in zip(all_preds, all_refs)]
        em = sum(em_scores) / len(em_scores)
        f1 = sum(f1_scores) / len(f1_scores)
        print(f"[Eval] EM = {em:.4f}, F1 = {f1:.4f}")

    return mean_loss, em, f1


# =========================
#   Main / CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, required=False,
                        help="(Deprecated) Path to train JSONL file. Now downloads from HuggingFace.")
    parser.add_argument("--eval_file", type=str, required=False,
                        help="(Deprecated) Path to eval/validation JSONL file. Now downloads from HuggingFace.")
    parser.add_argument("--output_dir", type=str, default="outputs_latent",
                        help="Directory to save best model checkpoint.")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--lr_latent", type=float, default=1e-4)
    parser.add_argument("--lr_t5", type=float, default=1e-5)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)

    parser.add_argument("--max_question_tokens", type=int, default=64)
    parser.add_argument("--max_answer_tokens", type=int, default=32)
    parser.add_argument("--max_context_tokens", type=int, default=768)
    parser.add_argument("--max_gen_len", type=int, default=32)

    parser.add_argument("--freeze_t5_initially", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_mixed_precision", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load data from Hugging Face ----
    print("Loading HotpotQA from Hugging Face datasets...")
    hf_ds = load_dataset("hotpot_qa", "distractor")
    df_train = hf_ds["train"].to_pandas()
    df_eval = hf_ds["validation"].to_pandas()
    print(f"Train shape: {df_train.shape}, Eval shape: {df_eval.shape}")

    train_dataset = HotpotDataset(df_train)
    eval_dataset = HotpotDataset(df_eval)

    # ---- Tokenizers ----
    print("Loading tokenizers...")
    t5_tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-base")
    encoder_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # ---- Dataloaders ----
    print("Building dataloaders...")
    collate_fn = lambda batch: collate_latent(
        batch,
        t5_tokenizer=t5_tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        max_question_tokens=args.max_question_tokens,
        max_answer_tokens=args.max_answer_tokens,
        max_context_tokens=args.max_context_tokens,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ---- Model config & init ----
    print("Initializing model...")
    config = LatentT5Config(
        t5_model_name="google/flan-t5-base",
        encoder_model_name="distilbert-base-uncased",
        num_latents=32,
        num_latent_layers=2,
        num_latent_heads=8,
        latent_dropout=0.1,
        aux_loss_weight=1.0,
        freeze_t5_initially=args.freeze_t5_initially,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        use_mixed_precision=not args.no_mixed_precision,
    )

    model = LatentT5Model(config)
    model.to(device)

    # ---- Optimizer & scheduler ----
    optimizer = _build_optimizer(
        model,
        lr_latent=args.lr_latent,
        lr_t5=args.lr_t5,
        lr_encoder=args.lr_encoder,
    )

    num_training_steps = args.num_epochs * math.ceil(
        len(train_dataloader) / args.grad_accum_steps
    )
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ---- Mixed precision ----
    use_mixed_precision = config.use_mixed_precision and device.type == "cuda"
    scaler = GradScaler(enabled=use_mixed_precision)
    if use_mixed_precision:
        print("Using mixed precision (AMP).")

    # ---- Train loop ----
    best_eval_loss = None
    best_ckpt_path = os.path.join(args.output_dir, "best_latent_t5.pt")

    for epoch in range(args.num_epochs):
        print(f"\n===== Epoch {epoch} =====")
        train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            epoch_idx=epoch,
            use_mixed_precision=use_mixed_precision,
            scaler=scaler,
        )

        eval_loss, em, f1 = evaluate(
            model,
            eval_dataloader,
            device=device,
            tokenizer=t5_tokenizer,
            compute_metrics=True,
            max_gen_len=args.max_gen_len,
        )

        if best_eval_loss is None or eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            print(f"New best eval loss: {best_eval_loss:.4f}. Saving checkpoint to {best_ckpt_path}")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                },
                best_ckpt_path,
            )


if __name__ == "__main__":
    main()
