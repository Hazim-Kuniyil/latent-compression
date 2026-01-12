from typing import List, Dict, Any, Optional, Tuple
import ast

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


def _normalize_hotpot_fields(
    supporting_facts, context
):
    if isinstance(supporting_facts, dict):
        sf_dict = supporting_facts
    elif isinstance(supporting_facts, list):
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
        sf_dict = {"title": [], "sent_id": []}

    if isinstance(context, dict):
        ctx_dict = context
    elif isinstance(context, list):
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


class HotpotDataset(Dataset):
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

        if isinstance(supporting_facts, str):
            supporting_facts = ast.literal_eval(supporting_facts)
        if isinstance(context, str):
            context = ast.literal_eval(context)

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


def _build_support_set(supporting_facts: Dict[str, Any]) -> set:
    titles = supporting_facts.get("title", [])
    sent_ids = supporting_facts.get("sent_id", [])
    return set(zip(titles, sent_ids))


def _encode_context_for_latent(
    context: Dict[str, Any],
    supporting_facts: Dict[str, Any],
    question: str,
    encoder_tokenizer: PreTrainedTokenizerBase,
    max_context_tokens: int,
) -> Tuple[List[int], List[int]]:
    titles: List[str] = context["title"]
    sentences_per_title: List[List[str]] = context["sentences"]
    support_pairs = _build_support_set(supporting_facts)

    q_token_ids = encoder_tokenizer.encode(
        question,
        add_special_tokens=False,
        truncation=True,
        max_length=64
    )

    all_token_ids: List[int] = list(q_token_ids)
    support_mask: List[int] = [0] * len(q_token_ids)

    sep_token_id = encoder_tokenizer.sep_token_id
    if sep_token_id is not None:
        all_token_ids.append(sep_token_id)
        support_mask.append(0)

    for para_idx, title in enumerate(titles):
        sentences = sentences_per_title[para_idx]
        for sent_idx, sent_text in enumerate(sentences):
            text = f"{title}: {sent_text}"

            token_ids = encoder_tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False
            )

            if not token_ids:
                continue

            if len(all_token_ids) + len(token_ids) > max_context_tokens:
                break

            is_support = (title, sent_idx) in support_pairs
            label_val = 1 if is_support else 0

            all_token_ids.extend(token_ids)
            support_mask.extend([label_val] * len(token_ids))

        if len(all_token_ids) >= max_context_tokens:
            break

    if len(all_token_ids) > max_context_tokens:
        all_token_ids = all_token_ids[:max_context_tokens]
        support_mask = support_mask[:max_context_tokens]

    if len(all_token_ids) == 0:
        pad_id = encoder_tokenizer.pad_token_id or 0
        all_token_ids = [pad_id]
        support_mask = [0]

    return all_token_ids, support_mask


def _flatten_context_to_text(context: Dict[str, Any]) -> str:
    titles = context["title"]
    sentences_per_title = context["sentences"]

    paragraphs = []
    for title, sentences in zip(titles, sentences_per_title):
        para = " ".join(sentences)
        para_text = f"{title}. {para}"
        paragraphs.append(para_text)

    return " ".join(paragraphs)


def collate_latent(
    batch: List[Dict[str, Any]],
    t5_tokenizer: PreTrainedTokenizerBase,
    encoder_tokenizer: PreTrainedTokenizerBase,
    max_question_tokens: int = 64,
    max_answer_tokens: int = 32,
    max_context_tokens: int = 512,
) -> Dict[str, torch.Tensor]:
    questions = [ex["question"] for ex in batch]
    answers = [ex["answer"] for ex in batch]
    ids = [ex["id"] for ex in batch]

    q_enc = t5_tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=max_question_tokens,
        return_tensors="pt",
    )
    input_ids = q_enc["input_ids"]
    attention_mask = q_enc["attention_mask"]

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

    encoder_max_len = getattr(encoder_tokenizer, "model_max_length", 512)
    if encoder_max_len is None or encoder_max_len > 10000:
        encoder_max_len = 512

    effective_max_ctx = min(max_context_tokens, encoder_max_len)

    context_id_lists: List[List[int]] = []
    support_mask_lists: List[List[int]] = []

    for ex in batch:
        context = ex["context"]
        supporting_facts = ex["supporting_facts"]
        question_text = ex["question"]

        ctx_ids, sup_mask = _encode_context_for_latent(
            context=context,
            supporting_facts=supporting_facts,
            question=question_text,
            encoder_tokenizer=encoder_tokenizer,
            max_context_tokens=effective_max_ctx,
        )
        context_id_lists.append(ctx_ids)
        support_mask_lists.append(sup_mask)

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
    questions = [ex["question"] for ex in batch]
    answers = [ex["answer"] for ex in batch]
    ids = [ex["id"] for ex in batch]

    inputs = []
    for ex, q in zip(batch, questions):
        context = ex["context"]
        ctx_text = _flatten_context_to_text(context)
        full_input = f"question: {q} context: {ctx_text}"
        inputs.append(full_input)

    inp_enc = t5_tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt",
    )
    input_ids = inp_enc["input_ids"]
    attention_mask = inp_enc["attention_mask"]

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
