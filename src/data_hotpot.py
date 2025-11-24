# src/data_hotpot.py

"""
Hotpot-style dataset + collators for:
  - LatentT5Model (latent bottleneck + support-token mask)
  - BaselineT5Model (plain concatenated context + question)

Assumptions about columns (matching your DataFrame sample):
  - id:               str
  - question:         str
  - answer:           str
  - supporting_facts: dict with keys:
        'title':   List[str]
        'sent_id': List[int]
  - context:          dict with keys:
        'title':     List[str]
        'sentences': List[List[str]]  # sentences[i] is list of sentences for title[i]

We assume 'supporting_facts' and 'context' are already parsed into Python dicts.
If they are strings, we try ast.literal_eval() as a fallback.
"""

from typing import List, Dict, Any, Optional, Tuple
import ast

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


# =========================
#   Helper: Normalize HotpotQA fields
# =========================

def _normalize_hotpot_fields(
    supporting_facts, context
):
    """
    Normalize raw HotpotQA fields (HF format or preprocessed) into:
      supporting_facts: {'title': [...], 'sent_id': [...]}
      context: {'title': [...], 'sentences': [[...], ...]}
    """
    # --- supporting_facts ---
    if isinstance(supporting_facts, dict):
        # Assume already in {'title': [...], 'sent_id': [...]} format
        sf_dict = supporting_facts
    elif isinstance(supporting_facts, list):
        # HF format: list of [title, sent_id]
        titles = []
        sent_ids = []
        for pair in supporting_facts:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            t, sid = pair
            titles.append(t)
            sent_ids.append(int(sid))
        sf_dict = {"title": titles, "sent_id": sent_ids}
    else:
        # Fallback: empty
        sf_dict = {"title": [], "sent_id": []}

    # --- context ---
    if isinstance(context, dict):
        # Assume already {'title': [...], 'sentences': [[...], ...]}
        ctx_dict = context
    elif isinstance(context, list):
        # HF format: list of [title, [sentences]]
        titles = []
        sentences_list = []
        for pair in context:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            t, sents = pair
            titles.append(t)
            sentences_list.append(list(sents))
        ctx_dict = {"title": titles, "sentences": sentences_list}
    else:
        ctx_dict = {"title": [], "sentences": []}

    return sf_dict, ctx_dict


# =========================
#   Dataset wrapper
# =========================

class HotpotDataset(Dataset):
    """
    Lightweight Dataset that just returns raw fields from the DataFrame.
    All tokenization & tensorization is handled in collate_fns.

    Expected DataFrame columns:
        'id', 'question', 'answer', 'supporting_facts', 'context'
    """

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        sample_id = row["id"]
        question = row["question"]
        answer = row["answer"]

        supporting_facts = row["supporting_facts"]
        context = row["context"]

        # If these are stored as strings, try to parse them.
        if isinstance(supporting_facts, str):
            supporting_facts = ast.literal_eval(supporting_facts)
        if isinstance(context, str):
            context = ast.literal_eval(context)

        # Normalize HF/raw formats into canonical dicts
        supporting_facts, context = _normalize_hotpot_fields(
            supporting_facts, context
        )

        return {
            "id": sample_id,
            "question": question,
            "answer": answer,
            "supporting_facts": supporting_facts,
            "context": context,
        }


# =========================
#   Helper functions
# =========================

def _build_support_set(supporting_facts: Dict[str, Any]) -> set:
    """
    Build a set of (title, sent_id) pairs that are supporting sentences.

    Expected normalized format:
        supporting_facts: {'title': [...], 'sent_id': [...]}
    """
    titles = supporting_facts.get("title", [])
    sent_ids = supporting_facts.get("sent_id", [])
    return set(zip(titles, sent_ids))


def _encode_context_for_latent(
    context: Dict[str, Any],
    supporting_facts: Dict[str, Any],
    encoder_tokenizer: PreTrainedTokenizerBase,
    max_context_tokens: int,
) -> Tuple[List[int], List[int]]:
    """
    Build:
        - context_input_ids: list[int]
        - support_token_mask: list[int] (0/1, aligned with context_input_ids)

    Strategy:
        - Iterate over paragraphs and sentences.
        - For each sentence, create text = "{title}: {sentence}".
        - Tokenize with add_special_tokens=False.
        - If this (title, sent_idx) is in supporting_facts, mark all its tokens as support (1).
        - Concatenate tokens for all sentences.
        - Truncate to max_context_tokens.

    NOTE:
      We do NOT add [CLS]/[SEP] here; the context encoder will still work
      without them, and this keeps the support mask aligned exactly.
    """
    titles: List[str] = context["title"]
    sentences_per_title: List[List[str]] = context["sentences"]

    support_pairs = _build_support_set(supporting_facts)

    all_token_ids: List[int] = []
    support_mask: List[int] = []

    for para_idx, title in enumerate(titles):
        sentences = sentences_per_title[para_idx]
        for sent_idx, sent_text in enumerate(sentences):
            text = f"{title}: {sent_text}"

            token_ids = encoder_tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,  # we'll truncate at the end globally
            )

            if not token_ids:
                continue

            is_support = (title, sent_idx) in support_pairs
            label_val = 1 if is_support else 0

            all_token_ids.extend(token_ids)
            support_mask.extend([label_val] * len(token_ids))

            # Early stop if we exceed max_context_tokens
            if len(all_token_ids) >= max_context_tokens:
                break
        if len(all_token_ids) >= max_context_tokens:
            break

    # Truncate
    if len(all_token_ids) > max_context_tokens:
        all_token_ids = all_token_ids[:max_context_tokens]
        support_mask = support_mask[:max_context_tokens]

    # Fallback if context is empty
    if len(all_token_ids) == 0:
        pad_id = encoder_tokenizer.pad_token_id
        if pad_id is None:
            # Many BERT-style models use 0 as pad, but use tokenizer's id if available
            pad_id = 0
        all_token_ids = [pad_id]
        support_mask = [0]

    return all_token_ids, support_mask


def _flatten_context_to_text(context: Dict[str, Any]) -> str:
    """
    For the baseline model:
        context_text = " ".join([" ".join(sent_list) for each title])

    i.e., we ignore supporting facts here and just build a big flat context string.
    """
    titles = context["title"]
    sentences_per_title = context["sentences"]

    paragraphs = []
    for title, sentences in zip(titles, sentences_per_title):
        para = " ".join(sentences)
        # Optionally prepend title for extra signal
        para_text = f"{title}. {para}"
        paragraphs.append(para_text)

    return " ".join(paragraphs)


# =========================
#   Collate functions
# =========================

def collate_latent(
    batch: List[Dict[str, Any]],
    t5_tokenizer: PreTrainedTokenizerBase,
    encoder_tokenizer: PreTrainedTokenizerBase,
    max_question_tokens: int = 64,
    max_answer_tokens: int = 32,
    max_context_tokens: int = 512,
) -> Dict[str, torch.Tensor]:
    """
    Collator for LatentT5Model.

    Returns dict with keys:
        input_ids:              [B, L_q]
        attention_mask:         [B, L_q]
        context_input_ids:      [B, L_ctx]
        context_attention_mask: [B, L_ctx]
        labels:                 [B, L_y]
        support_token_mask:     [B, L_ctx]
        target_texts:           list[str] (for EM/F1)
        ids:                    list[str]
    """
    # 1) Questions & answers as lists of raw strings
    questions = [ex["question"] for ex in batch]
    answers = [ex["answer"] for ex in batch]
    ids = [ex["id"] for ex in batch]

    # 2) T5 encoding for questions
    q_enc = t5_tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=max_question_tokens,
        return_tensors="pt",
    )
    input_ids = q_enc["input_ids"]
    attention_mask = q_enc["attention_mask"]

    # 3) T5 encoding for answers (labels)
    with t5_tokenizer.as_target_tokenizer():
        ans_enc = t5_tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=max_answer_tokens,
            return_tensors="pt",
        )
    labels = ans_enc["input_ids"]
    # Replace pad token id with -100 for loss masking
    labels[labels == t5_tokenizer.pad_token_id] = -100

    # 4) Context encoder: per-example tokenization + support mask

    # IMPORTANT: never exceed the encoder's max position embeddings
    encoder_max_len = getattr(encoder_tokenizer, "model_max_length", max_context_tokens)
    # DistilBERT has 512; some tokenizers set this to a very large sentinel, so clamp anyway
    if encoder_max_len is None or encoder_max_len > 4096:
        encoder_max_len = max_context_tokens

    effective_max_ctx = min(max_context_tokens, encoder_max_len)

    context_id_lists: List[List[int]] = []
    support_mask_lists: List[List[int]] = []

    for ex in batch:
        context = ex["context"]
        supporting_facts = ex["supporting_facts"]

        ctx_ids, sup_mask = _encode_context_for_latent(
            context=context,
            supporting_facts=supporting_facts,
            encoder_tokenizer=encoder_tokenizer,
            max_context_tokens=effective_max_ctx,  # <-- use clamped value
        )
        context_id_lists.append(ctx_ids)
        support_mask_lists.append(sup_mask)

    # 5) Pad context sequences to same length
    max_ctx_len = max(len(ids_) for ids_ in context_id_lists)
    pad_id = encoder_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    context_input_ids = []
    context_attention_mask = []
    support_token_mask = []

    for ids_, sup_mask in zip(context_id_lists, support_mask_lists):
        pad_len = max_ctx_len - len(ids_)
        context_input_ids.append(ids_ + [pad_id] * pad_len)
        context_attention_mask.append([1] * len(ids_) + [0] * pad_len)
        support_token_mask.append(sup_mask + [0] * pad_len)

    context_input_ids = torch.tensor(context_input_ids, dtype=torch.long)
    context_attention_mask = torch.tensor(context_attention_mask, dtype=torch.long)
    support_token_mask = torch.tensor(support_token_mask, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "context_input_ids": context_input_ids,
        "context_attention_mask": context_attention_mask,
        "labels": labels,
        "support_token_mask": support_token_mask,
        "target_texts": answers,
        "ids": ids,
    }


def collate_baseline(
    batch: List[Dict[str, Any]],
    t5_tokenizer: PreTrainedTokenizerBase,
    max_input_tokens: int = 512,
    max_answer_tokens: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Collator for BaselineT5Model.

    We feed concatenated "question + context" to T5.

    Returns:
        input_ids:      [B, L]
        attention_mask: [B, L]
        labels:         [B, L_y]
        target_texts:   list[str]
        ids:            list[str]
    """
    questions = [ex["question"] for ex in batch]
    answers = [ex["answer"] for ex in batch]
    ids = [ex["id"] for ex in batch]

    # Build combined text: "question: {q} context: {flat_context}"
    inputs = []
    for ex, q in zip(batch, questions):
        context = ex["context"]
        ctx_text = _flatten_context_to_text(context)
        full_input = f"question: {q} context: {ctx_text}"
        inputs.append(full_input)

    # T5 encoding for input
    inp_enc = t5_tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt",
    )
    input_ids = inp_enc["input_ids"]
    attention_mask = inp_enc["attention_mask"]

    # T5 encoding for labels
    with t5_tokenizer.as_target_tokenizer():
        ans_enc = t5_tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=max_answer_tokens,
            return_tensors="pt",
        )
    labels = ans_enc["input_ids"]
    labels[labels == t5_tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_texts": answers,
        "ids": ids,
    }
