from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch
from typing import List


vocab = {
    "r": 0,
    "u": 1,
    "f": 2,
    "l": 3,
    "d": 4,
    "b": 5,
    "R": 6,
    "RR": 7,
    "RRR": 8,
    "U": 9,
    "UU": 10,
    "UUU": 11,
    "F": 12,
    "FF": 13,
    "FFF": 14,
    "<|endoftext|>": 15,
}

tokenizer_model = WordLevel(vocab, unk_token="<|endoftext|>")
tokenizer_instance = Tokenizer(tokenizer_model)
tokenizer_instance.pre_tokenizer = Whitespace()
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_instance)
hf_tokenizer.add_special_tokens(
    {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "bos_token": "<|endoftext|>",
    }
)


# Below is deprecated.
class CubeTokenizer:
    def __init__(self, max_length=50):
        self.vocab = VOCAB
        self.vocab_size = len(self.vocab)
        self.max_length = max_length
        self.bos_token = "<|endoftext|>"
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token_id = self.vocab[self.pad_token]
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]

    def encode(self, text, max_length=None):
        for j in range(len(text)):
            old_text = text[j]
            new_text = []
            store = ""
            for i in range(len(old_text)):
                if old_text[i].islower():
                    new_text.append(old_text[i])
                else:
                    if i != len(old_text) - 1 and old_text[i] == old_text[i + 1]:
                        store += old_text[i]
                    else:
                        store += old_text[i]
                        new_text.append(store)
                        store = ""
            text[j] = new_text

        inputs = dict()
        inputs["input_ids"] = []
        inputs["attention_mask"] = []
        for i in text:
            inputs["input_ids"].append(
                [self.vocab[word] for word in i]
                + [self.vocab["<|endoftext|>"] for word in range(max_length - len(i))]
            )
            inputs["attention_mask"].append(
                [1 for word in i] + [0 for word in range(max_length - len(i))]
            )
        inputs["input_ids"] = torch.tensor(inputs["input_ids"])
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"])
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    def decode(self, generated_ids):
        vocab_inv = {v: k for k, v in self.vocab.items()}
        vocab_inv.pop(self.vocab[self.pad_token])
        generated_ids = list(map(int, generated_ids))
        return "".join([vocab_inv[word] for word in generated_ids if word in vocab_inv])

    def decode_into_list(self, generated_ids) -> List[str]:
        vocab_inv = {v: k for k, v in self.vocab.items()}
        vocab_inv.pop(self.vocab[self.pad_token])
        generated_ids = list(map(int, generated_ids))
        return [vocab_inv[word] for word in generated_ids if word in vocab_inv]

    def batch_decode(self, generated_ids):
        return [self.decode(ids) for ids in generated_ids]

    def _convert_id_to_token(self, action):
        pass

    def __call__(self, text):
        return self.encode(text)

    def get_cube_state_prompts(self, text):
        encodings = self.encode(text)
        encodings["input_ids"] = encodings["input_ids"][:, :24]
        encodings["attention_mask"] = encodings["attention_mask"][:, :24]
        encodings["labels"] = encodings["labels"][:, :24]
        return encodings
