from typing import List
import random
from itertools import groupby
import torch
from pytwisty_moves import *
# from train_scripts.utils import IGNORE_INDEX
IGNORE_INDEX = -100


MOVES = [
    "F",
    "FF",
    "FFF",
    "U",
    "UU",
    "UUU",
    "R",
    "RR",
    "RRR",
]


def get_states(state: str, moves: List[str]):
    ts = [
        [1, ["F", "L", "U"]],
        [2, ["F", "R", "U"]],
        [3, ["F", "R", "D"]],
        [4, ["F", "L", "D"]],
    ]
    bs = [
        [5, ["B", "L", "U"]],
        [6, ["B", "R", "U"]],
        [7, ["B", "R", "D"]],
        [8, ["B", "L", "D"]],
    ]
    get_tsbs(ts, bs, state)
    all_states = [state]
    for move in moves:
        for submove in move:
            assert submove in ["R", "U", "F"]
            if submove == "R":
                R(ts, bs)
            elif submove == "U":
                U(ts, bs)
            elif submove == "F":
                F(ts, bs)
        curr_state = get_state(ts, bs)
        all_states.append(curr_state)
    return all_states


def build_state_tensor(state: str) -> torch.Tensor:
    """
    Convert a string representation of the cube state into a tensor.
    The string should be 24 characters long, each character representing a facelet.
    """
    assert len(state) == 24
    mapping = {
        "r": 0,
        "u": 1,
        "f": 2,
        "l": 3,
        "d": 4,
        "b": 5,
    }
    return torch.tensor([mapping[sq] for sq in state])


class CubeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        initial_states: List[str],
        moves: List[str],
        tokenizer,
        max_length=50,
        num_squares=24,
        num_labels=6,
    ):
        """
        initial_states: list of strings (24 tokens representing the starting board state)
        moves: list of strings (K tokens representing moves)
        board_labels: list of lists, each of shape [K, 24] corresponding to the updated board state for each move
        """
        self.initial_states = initial_states
        self.moves = moves
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_squares = num_squares
        self.num_labels = num_labels

    def __len__(self):
        return len(self.initial_states)

    def __getitem__(self, idx):
        # Concatenate the starting board state and the move sequence
        initial_state = self.initial_states[idx].lower()
        solution = self.moves[idx]
        if solution is None:
            solution = ""

        input_text = initial_state + " " + solution
        # "input_ids": [batch (1), max_length (50)]
        # "attention_mask": [batch (1), max_length (50)]
        encoded = self.tokenizer(
            [input_text],
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)  # shape: [max_length]
        attention_mask = encoded["attention_mask"].squeeze(0)

        _initial_state = "".join(initial_state.split())
        _solution = "".join(solution.split())
        moves = [
            "".join(g) for g in (list(g) for _, g in groupby(_solution)) if "".join(g)
        ]
        _states = get_states(_initial_state, moves)
        assert len(_states) == len(moves) + 1
        states = torch.stack([build_state_tensor(state) for state in _states], dim=0)
        cube_states = torch.full(
            (self.max_length, self.num_squares),
            IGNORE_INDEX,
            dtype=torch.long,
        )
        cube_states[23 : 23 + states.shape[0], :] = states
        return input_ids, attention_mask, cube_states, initial_state


class CubePretrainDataset(CubeDataset):
    def __init__(
        self,
        initial_states: List[str],
        moves: List[str],
        tokenizer,
        max_length=50,
        num_squares=24,
        num_labels=6,
    ):
        super().__init__(
            initial_states, moves, tokenizer, max_length, num_squares, num_labels
        )

    def __getitem__(self, idx):
        # Concatenate the starting board state and the move sequence
        initial_state = self.initial_states[idx].lower()
        orig_solution = self.moves[idx]
        moves = []
        if orig_solution is None:
            moves_txt = ""

        else:
            for num_moves in range(len(orig_solution.split())):
                # Append random move:
                moves.append(random.choice(MOVES))
            moves_txt = " ".join(moves)

        input_text = initial_state + " " + moves_txt
        # "input_ids": [batch (1), max_length (50)]
        # "attention_mask": [batch (1), max_length (50)]
        encoded = self.tokenizer(
            [input_text],
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)  # shape: [max_length]
        attention_mask = encoded["attention_mask"].squeeze(0)

        _initial_state = "".join(initial_state.split())
        
        _states = get_states(_initial_state, moves)
        assert len(_states) == len(moves) + 1
        states = torch.stack([build_state_tensor(state) for state in _states], dim=0)
        cube_states = torch.full(
            (self.max_length, self.num_squares),
            IGNORE_INDEX,
            dtype=torch.long,
        )
        cube_states[23 : 23 + states.shape[0], :] = states
        return input_ids, attention_mask, cube_states, initial_state
