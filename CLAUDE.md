# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing **latent bottleneck compression for retrieval-augmented question answering** on HotpotQA. The project compares two approaches:

1. **Baseline T5**: Vanilla T5 model that concatenates retrieved context + question
2. **Latent T5**: Novel architecture using Perceiver-style latent bottleneck to compress long contexts before feeding to T5

The latent approach enables processing much longer contexts (768+ tokens) while keeping T5's input fixed to a small set of learned latent vectors (default: 32 latents).

## Architecture Overview

### Latent T5 Model ([src/model_latent_t5.py](src/model_latent_t5.py))

The `LatentT5Model` has three main components:

1. **Context Encoder** (DistilBERT): Encodes long retrieved context (up to 768 tokens)
2. **Latent Selector** (Perceiver-style):
   - Starts with learned latent vectors (default: 32 latents)
   - Cross-attends over the full context using `LatentCrossAttentionBlock` layers
   - Compresses context into fixed-size latent representation
   - Returns attention weights over context tokens
3. **T5 Seq2Seq Model**:
   - Encodes the question separately
   - Receives gated latent vectors via Flamingo-style tanh gate
   - Concatenates question embeddings + gated latents as encoder output
   - Decoder generates the answer

**Key design details**:
- The `latent_gate` parameter (initialized to 0) allows the model to start near vanilla T5 behavior
- Supports **auxiliary supervision**: uses `support_token_mask` to encourage latents to attend to supporting facts
- The auxiliary loss is: `-log(attention_mass_on_support_tokens)`
- Gradient checkpointing enabled by default for memory efficiency

### Baseline T5 Model ([src/model_baseline_t5.py](src/model_baseline_t5.py))

Simple wrapper around T5 that takes concatenated `context + question` as input. Used for comparison.

### Data Pipeline ([src/data_hotpot.py](src/data_hotpot.py))

- `HotpotDataset`: Lightweight wrapper around HotpotQA DataFrame
- `collate_latent`: For LatentT5Model
  - Tokenizes questions with T5 tokenizer
  - Tokenizes context with encoder tokenizer (DistilBERT)
  - Builds `support_token_mask`: token-level 0/1 mask aligned with context tokens
  - Each sentence is formatted as `"{title}: {sentence}"` then tokenized without special tokens
  - If `(title, sent_id)` is in `supporting_facts`, all its tokens get mask=1
- `collate_baseline`: For BaselineT5Model
  - Flattens context into single text string
  - Concatenates as `"question: {q} context: {ctx}"`

**Important**: The context encoder tokenization uses `add_special_tokens=False` to keep support mask aligned exactly with tokens.

## Training Commands

Both training scripts now **automatically download HotpotQA from Hugging Face** (`hotpot_qa`, `distractor` split). The `--train_file` and `--eval_file` arguments are deprecated.

### Train Latent T5 Model

```bash
python src/train_latent_t5.py \
  --output_dir outputs_latent \
  --batch_size 8 \
  --num_epochs 3 \
  --lr_latent 1e-4 \
  --lr_t5 1e-5 \
  --lr_encoder 1e-5 \
  --max_context_tokens 768 \
  --max_question_tokens 64 \
  --max_answer_tokens 32
```

**Key arguments**:
- `--lr_latent`, `--lr_t5`, `--lr_encoder`: Separate learning rates for different components
- `--freeze_t5_initially`: Start with frozen T5 (train only latent modules first)
- `--no_gradient_checkpointing`: Disable gradient checkpointing (uses more memory)
- `--no_mixed_precision`: Disable AMP (automatic mixed precision)

The latent LR defaults to `1e-4` (changed from `2e-4` in recent code update).

### Train Baseline T5 Model

```bash
python src/train_baseline_t5.py \
  --output_dir outputs_baseline \
  --batch_size 8 \
  --num_epochs 3 \
  --lr 1e-4 \
  --max_input_tokens 512 \
  --max_answer_tokens 32
```

### Evaluation

Both training scripts:
- Evaluate after each epoch on the validation split
- Compute loss, Exact Match (EM), and F1 scores
- Save best checkpoint based on validation loss to `{output_dir}/best_{model}.pt`

Checkpoints contain:
- `model_state_dict`: Model weights
- `config`: Model configuration dict (LatentT5 only)

## Model Configuration

The `LatentT5Config` dataclass ([src/model_latent_t5.py:192-208](src/model_latent_t5.py#L192-L208)) controls:

- `t5_model_name`: Default `"google/flan-t5-base"`
- `encoder_model_name`: Default `"distilbert-base-uncased"`
- `num_latents`: Number of latent vectors (default: 32)
- `num_latent_layers`: Cross-attention layers (default: 2)
- `num_latent_heads`: Attention heads (default: 8)
- `latent_dropout`: Dropout rate (default: 0.1)
- `aux_loss_weight`: Weight for support-attention loss (default: 0.1 in config, but **1.0 in training script**)
- `freeze_t5_initially`: Whether to freeze T5 params initially
- `use_gradient_checkpointing`: Enable gradient checkpointing (default: True)

## Key Implementation Details

### NaN/Inf Protection

The latent training loop includes guards against non-finite losses ([src/train_latent_t5.py:141-149](src/train_latent_t5.py#L141-L149)):
```python
if not torch.isfinite(loss):
    print(f"Non-finite loss detected: {loss}, ...")
    optimizer.zero_grad(set_to_none=True)
    continue
```

This prevents training crashes from numerical instability in the auxiliary loss.

### Gradient Accumulation

Both scripts support `--grad_accum_steps` to simulate larger batch sizes. Loss is divided by accumulation steps before backward pass.

### Mixed Precision

Both scripts use PyTorch AMP (automatic mixed precision) by default when CUDA is available. The latent script uses `torch.amp.autocast`, while baseline uses `torch.cuda.amp.autocast`.

### Separate Learning Rates

The latent model uses different LRs for three parameter groups ([src/train_latent_t5.py:81-106](src/train_latent_t5.py#L81-L106)):
1. Latent modules (selector + gate): Higher LR (default: 1e-4)
2. T5 parameters: Lower LR (default: 1e-5)
3. Context encoder: Lower LR (default: 1e-5)

### Context Length Limits

The `collate_latent` function clamps `max_context_tokens` to the encoder's `model_max_length` to avoid position embedding errors ([src/data_hotpot.py:240-245](src/data_hotpot.py#L240-L245)).

## Generation

Both models expose a `generate()` method that wraps T5's generation:

```python
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    context_input_ids=context_input_ids,  # LatentT5 only
    context_attention_mask=context_attention_mask,  # LatentT5 only
    max_new_tokens=32,
)
```

The latent model builds encoder states using the same `_build_encoder_states()` helper used in forward pass.

## Metrics

Both training scripts use the same evaluation metrics ([src/train_latent_t5.py:42-74](src/train_latent_t5.py#L42-L74)):

- **Exact Match (EM)**: Normalized string match after removing articles, punctuation, extra whitespace
- **F1 Score**: Token-level F1 between prediction and ground truth

Normalization is identical to SQuAD-style evaluation.

## File Organization

```
src/
├── model_latent_t5.py      # LatentT5Model, LatentSelector, config
├── model_baseline_t5.py    # BaselineT5Model
├── data_hotpot.py          # HotpotDataset, collate functions
├── train_latent_t5.py      # Training script for latent model
├── train_baseline_t5.py    # Training script for baseline model
└── run_eval.py             # (placeholder/empty)

configs/
├── latent_t5_hotpotqa.yaml    # (placeholder/empty)
└── baseline_t5_hotpotqa.yaml  # (placeholder/empty)
```

Config YAML files currently exist but are empty - all configuration is via command-line arguments.

## Development Notes

- All code uses PyTorch and Hugging Face Transformers
- No tests are currently implemented
- The project automatically downloads HotpotQA via `datasets.load_dataset("hotpot_qa", "distractor")`
- Both scripts print training progress every 50 steps by default
- The `run_eval.py` file is a placeholder and not yet implemented
