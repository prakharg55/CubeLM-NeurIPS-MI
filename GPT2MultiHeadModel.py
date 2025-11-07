import torch
import torch.nn as nn
from transformers import GPT2Model
from train_scripts.utils import IGNORE_INDEX


class GPT2MultiHeadModel(nn.Module):
    def __init__(self, config, num_heads=24, num_classes=6):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.config = config
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.lm_heads = nn.Linear(config.n_embd, num_heads * num_classes, bias=False)

    def forward(self, input_ids, attention_mask=None, cube_states=None):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )
        # hidden_states = outputs[0]
        # [batch, seq, d_model]
        hidden_states = outputs.last_hidden_state

        # [batch, seq, n_heads * n_classes]
        logits = self.lm_heads(hidden_states)
        logits = logits.view(
            hidden_states.size(0),
            hidden_states.size(1),
            self.num_heads,
            self.num_classes,
        )

        loss = None
        if cube_states is not None:
            # [batch * seq, n_heads, n_classes]
            _logits = logits.view(-1, self.num_heads, self.num_classes)
            # [batch * seq, n_heads]
            _labels = cube_states.view(-1, self.num_heads)
            loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            losses = []
            for head_idx in range(self.num_heads):
                losses.append(loss_fn(_logits[:, head_idx, :], _labels[:, head_idx]))
            loss = sum(losses) / self.num_heads

        return {"logits": logits, "loss": loss}
