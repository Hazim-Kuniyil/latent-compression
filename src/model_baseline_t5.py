from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM


class BaselineT5Model(nn.Module):
    def __init__(self, t5_model_name: str = "google/flan-t5-base"):
        super().__init__()
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **gen_kwargs,
    ) -> torch.Tensor:
        return self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
