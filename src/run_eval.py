#!/usr/bin/env python3
# src/run_eval.py
"""
Evaluate a trained LatentT5 checkpoint with optional latent ablations.

Ablation modes:
- none:   Use learned latents normally (default)
- zeros:  Replace latents with zeros before gating
- random: Replace latents with random noise before gating
- bypass: Skip latents entirely, use only T5 encoder (vanilla T5)
"""

import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import string
from collections import Counter

from model_latent_t5 import LatentT5Model, LatentT5Config
from data_hotpot import HotpotDataset, collate_latent


# ---------------------------
# Evaluation metrics (from train_latent_t5.py)
# ---------------------------

def normalize_answer(s: str) -> str:
    """Normalize answer string for EM/F1 computation."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------
# Ablation patches
# ---------------------------

def patch_model_for_ablation(model: LatentT5Model, mode: str):
    """
    Monkey-patch the model's _build_encoder_states method to ablate latents.

    Args:
        model: LatentT5Model instance
        mode: One of 'none', 'zeros', 'random', 'bypass'
    """
    if mode == "none":
        # No ablation, use model as-is
        return

    # Store original method
    original_build = model._build_encoder_states

    if mode == "zeros":
        def ablated_build(input_ids, attention_mask, context_input_ids, context_attention_mask):
            # Build normally but replace latents with zeros
            encoder_hidden, encoder_attention_mask, attn_weights = original_build(
                input_ids, attention_mask, context_input_ids, context_attention_mask
            )
            # Extract question and latent parts
            B, L_total, D = encoder_hidden.shape
            L_q = input_ids.size(1)
            M = L_total - L_q

            # Replace latents with zeros
            encoder_hidden = torch.cat([
                encoder_hidden[:, :L_q, :],  # keep question
                torch.zeros(B, M, D, dtype=encoder_hidden.dtype, device=encoder_hidden.device)
            ], dim=1)

            return encoder_hidden, encoder_attention_mask, attn_weights

        model._build_encoder_states = ablated_build

    elif mode == "random":
        def ablated_build(input_ids, attention_mask, context_input_ids, context_attention_mask):
            # Build normally but replace latents with random noise
            encoder_hidden, encoder_attention_mask, attn_weights = original_build(
                input_ids, attention_mask, context_input_ids, context_attention_mask
            )
            # Extract question and latent parts
            B, L_total, D = encoder_hidden.shape
            L_q = input_ids.size(1)
            M = L_total - L_q

            # Replace latents with random noise (same scale as typical embeddings)
            random_latents = torch.randn(B, M, D, dtype=encoder_hidden.dtype, device=encoder_hidden.device) * 0.02
            encoder_hidden = torch.cat([
                encoder_hidden[:, :L_q, :],  # keep question
                random_latents
            ], dim=1)

            return encoder_hidden, encoder_attention_mask, attn_weights

        model._build_encoder_states = ablated_build

    elif mode == "bypass":
        def ablated_build(input_ids, attention_mask, context_input_ids, context_attention_mask):
            # Skip latents entirely - vanilla T5 behavior
            # Just encode the question with T5 encoder
            t5_encoder_outputs = model.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            question_hidden = t5_encoder_outputs.last_hidden_state  # [B, L_q, D_t5]

            # No latents, just return question encoding
            return question_hidden, attention_mask, None

        model._build_encoder_states = ablated_build

    else:
        raise ValueError(f"Unknown ablation mode: {mode}")


# ---------------------------
# Main evaluation
# ---------------------------

def evaluate(
    model: LatentT5Model,
    dataloader: DataLoader,
    t5_tokenizer,
    device: torch.device,
    max_new_tokens: int = 32,
):
    """Run evaluation and return metrics."""
    model.eval()

    all_em = []
    all_f1 = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            context_input_ids = batch["context_input_ids"].to(device)
            context_attention_mask = batch["context_attention_mask"].to(device)

            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                max_new_tokens=max_new_tokens,
            )

            # Decode predictions
            preds = t5_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Get gold answers (stored as "target_texts" by collate_latent)
            golds = batch["target_texts"]

            # Compute metrics
            for pred, gold in zip(preds, golds):
                all_em.append(compute_exact_match(pred, gold))
                all_f1.append(compute_f1(pred, gold))

    return {
        "exact_match": sum(all_em) / len(all_em) * 100,
        "f1": sum(all_f1) / len(all_f1) * 100,
        "num_examples": len(all_em),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LatentT5 with optional ablations")

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt file)")

    # Ablation mode
    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none", "zeros", "random", "bypass"],
                        help="Ablation mode: none (normal), zeros (replace with zeros), "
                             "random (replace with noise), bypass (skip latents)")

    # Data
    parser.add_argument("--split", type=str, default="validation",
                        choices=["train", "validation"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit evaluation to N samples (for quick testing)")

    # Model config (should match training, but can override)
    parser.add_argument("--t5_model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--encoder_model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--num_latents", type=int, default=32)
    parser.add_argument("--num_latent_layers", type=int, default=2)
    parser.add_argument("--num_latent_heads", type=int, default=8)

    # Inference
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_context_tokens", type=int, default=768)
    parser.add_argument("--max_question_tokens", type=int, default=64)
    parser.add_argument("--max_answer_tokens", type=int, default=32)

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    # Build config (prefer from checkpoint, fallback to args)
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        config = LatentT5Config(**config_dict)
        print("Loaded config from checkpoint")
    else:
        config = LatentT5Config(
            t5_model_name=args.t5_model_name,
            encoder_model_name=args.encoder_model_name,
            num_latents=args.num_latents,
            num_latent_layers=args.num_latent_layers,
            num_latent_heads=args.num_latent_heads,
            use_gradient_checkpointing=False,  # not needed for eval
        )
        print("Using config from arguments (checkpoint has no config)")

    # Create model
    print("\nInitializing model...")
    model = LatentT5Model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Apply ablation if requested
    if args.ablation != "none":
        print(f"\n=' Applying ablation mode: {args.ablation}")
        patch_model_for_ablation(model, args.ablation)
    else:
        print("\nRunning with learned latents (no ablation)")

    # Load tokenizers
    print("\nLoading tokenizers...")
    t5_tokenizer = AutoTokenizer.from_pretrained(config.t5_model_name)
    encoder_tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name)

    # Load dataset
    print(f"\nLoading HotpotQA ({args.split} split)...")
    dataset = load_dataset("hotpot_qa", "distractor", split=args.split)

    # Limit samples if requested
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")

    # Convert to pandas DataFrame for HotpotDataset
    df = dataset.to_pandas()

    # Create dataset and dataloader
    eval_dataset = HotpotDataset(df)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_latent(
            batch,
            t5_tokenizer=t5_tokenizer,
            encoder_tokenizer=encoder_tokenizer,
            max_question_tokens=args.max_question_tokens,
            max_context_tokens=args.max_context_tokens,
            max_answer_tokens=args.max_answer_tokens,
        ),
    )

    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Starting evaluation on {len(eval_dataset)} examples")
    print(f"{'='*60}\n")

    metrics = evaluate(
        model=model,
        dataloader=eval_dataloader,
        t5_tokenizer=t5_tokenizer,
        device=device,
        max_new_tokens=args.max_answer_tokens,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results (ablation={args.ablation})")
    print(f"{'='*60}")
    print(f"Exact Match: {metrics['exact_match']:.2f}%")
    print(f"F1 Score:    {metrics['f1']:.2f}%")
    print(f"Num Examples: {metrics['num_examples']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
