from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
# from .CubeConfig import CubeConfig
from transformers import (
    GPT2Model,
    GPT2LMHeadModel,
    GenerationMixin,
    GPT2PreTrainedModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import ModelOutput

IGNORE_INDEX = -100


@dataclass
class CubeLMOutput(CausalLMOutputWithCrossAttentions):
    # loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    cube_loss: Optional[torch.FloatTensor] = None
    # logits: Optional[torch.FloatTensor] = None
    cube_logits: Optional[torch.FloatTensor] = None


class CubeLM(GPT2LMHeadModel):

    def __init__(self, config, task="sft", num_heads=24, num_classes=6):
        super().__init__(config)
        assert task in ["sft", "pretrain", "joint"]

        self.task = task
        self.alpha = None
        if hasattr(config, "alpha"):
            self.alpha = config.alpha
        self.vocab_size = config.vocab_size
        self.cube_heads = None
        if task in ["pretrain", "joint"]:
            self.cube_heads = nn.Linear(
                config.n_embd, num_heads * num_classes, bias=False
            )
            self.num_heads = num_heads
            self.num_classes = num_classes
        self.config = config

    def forward(
        self, input_ids, attention_mask=None, labels=None, cube_states=None, **kwargs
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )
        # [batch, seq, d_model]
        hidden_states = outputs.last_hidden_state

        lm_logits = None
        lm_loss = None
        # [batch, seq, n_heads * n_classes]
        if self.task in ["sft", "joint"]:
            lm_logits = self.lm_head(hidden_states)

            if labels is not None:
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                lm_loss = loss_fn(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                )

        cube_logits = None
        cube_loss = None
        if self.cube_heads:
            cube_logits = self.cube_heads(hidden_states)

            if cube_states is not None:
                cube_logits = cube_logits.view(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_heads,
                    self.num_classes,
                )
                # [batch * seq, n_heads, n_classes]
                _logits = cube_logits.view(-1, self.num_heads, self.num_classes)
                # [batch * seq, n_heads]
                _labels = cube_states.view(-1, self.num_heads)
                loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                losses = []
                for head_idx in range(self.num_heads):
                    losses.append(
                        loss_fn(_logits[:, head_idx, :], _labels[:, head_idx])
                    )
                cube_loss = sum(losses) / self.num_heads

        total_loss = None
        if lm_loss is not None and cube_loss is not None:
            assert self.alpha is not None
            total_loss = lm_loss + self.alpha * cube_loss
        elif lm_loss is not None:
            total_loss = lm_loss
        elif cube_loss is not None:
            total_loss = cube_loss

        return CubeLMOutput(
            loss=total_loss,
            lm_loss=lm_loss,
            cube_loss=cube_loss,
            logits=lm_logits,
            cube_logits=cube_logits,
        )
