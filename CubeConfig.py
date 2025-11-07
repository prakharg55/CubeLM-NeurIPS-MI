from transformers import GPT2Config


class CubeConfig(GPT2Config):
    model_type = "CubeLM"

    def __init__(
        self,
        vocab_size=16,
        bos_token_id=15,
        eos_token_id=15,
        pad_token_id=15,
        n_positions=40,
        n_embd=512,
        n_layer=8,
        n_head=8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
