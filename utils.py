from datasets import load_dataset
from torch.utils.data import random_split
import random
from torch.utils.data import IterableDataset
from pytwisty_moves import apply_move, apply_moves
from itertools import groupby
import json
import os

IGNORE_INDEX = -100


def get_data(data_path, tokenizer, max_length, valid_size=None):
    """
    Load data.
    if valid_size is not None, split data into train and valid sets.
    if valid_size is None, the whole data is returned as the first element and return None for the second element.
    """
    dataset = load_dataset(
        "csv",
        data_files=data_path,
    )
    dataset = CubeDataset(
        dataset["train"]["scramble"],
        dataset["train"]["solution"],
        tokenizer,
        max_length=max_length,
    )

    train = dataset
    valid = None
    if valid_size:
        train_size = len(dataset) - valid_size
        train, valid = random_split(dataset, [train_size, valid_size])
    return train, valid


class RandomDataStream(IterableDataset):
    def __init__(self, tokenizer, max_length, seq_type, K, reserved_test_paths=[]):
        self.possible_moves = ["R", "RR", "RRR", "U", "UU", "UUU", "F", "FF", "FFF"]
        self.solved_state = "uuuurrrrffffddddllllbbbb"
        self.opposite_move = {"R": "RRR", "RR": "RR", "RRR": "R", "U": "UUU", "UU": "UU", "UUU": "U", "F": "FFF", "FF": "FF", "FFF": "F"}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.K = K
        self.reserved_test_paths = [path.split() for path in reserved_test_paths]
        self.seq_type = seq_type

        self.complete_data = json.load(open("UPDATE ME"))

    def get_opt_sol_len(self, state):
        return len(["".join(g) for _, g in groupby(self.complete_data[state])])
    
    def get_next_moves(self, state):
        cur_opt_sol_len = self.get_opt_sol_len(state)
        next_moves = {"good": [], "same": [], "bad": []}
        for move in self.possible_moves:
            next_opt_sol_len = self.get_opt_sol_len(apply_move(state, move))
            if next_opt_sol_len < cur_opt_sol_len:
                next_moves['good'].append(move)
            elif next_opt_sol_len == cur_opt_sol_len:
                next_moves['same'].append(move)
            else:
                next_moves['bad'].append(move)
        return next_moves
    
    def get_move_sequence(self, k, initial_state):
        assert self.seq_type in ["random", "incremental"]
        if self.seq_type == "random":
            return random.choices(self.possible_moves, k=k)
        else:
            moves = []
            state = initial_state
            for _ in range(k):
                # print(state,self.get_opt_sol_len(state),self.get_next_moves(state))
                # while True:
                #     random_int = random.randint(1, 10)
                #     if random_int == 1:
                #         moves.append(random.choice(self.get_next_moves(state)['bad']))
                #     elif random_int <= 4:
                #         moves.append(random.choice(self.get_next_moves(state)['same']))
                #     else:
                #         moves.append(random.choice(self.get_next_moves(state)['good']))
                #     if apply_moves(initial_state, moves) == self.solved_state:
                #         moves.pop(-1)
                #         continue
                #     break

                BUCKET_WEIGHTS = {"bad": 70, "same": 30, "good": 0} # distribution is inverted bc we're starting from solved state
                moves_by_kind = self.get_next_moves(state)
                # keep only buckets that still have moves
                available = {kind: lst for kind, lst in moves_by_kind.items() if lst}
                if len(available) == 1: 
                    moves.append(random.choice(available[list(available.keys())[0]]))
                else:
                    # build parallel lists of kinds and weights
                    kinds, weights = zip(*[(kind, BUCKET_WEIGHTS[kind]) for kind in available])
                    # pick one bucket, then pick a move inside it
                    chosen_kind = random.choices(kinds, weights=weights, k=1)[0]
                    moves.append(random.choice(available[chosen_kind]))
                state=apply_move(state,moves[-1])
            return moves

    def generate_train_sequence(self): # What should K be?
        while True:
            k = random.randint(1, self.K) # Do we want k to be uniformly chosen?
            moves = self.get_move_sequence(k, self.solved_state) # Do we want an FF/FFF to follow an F or vice versa?
            
            reserved_path = False
            for path in self.reserved_test_paths:
                if moves[:len(path)] == path:
                    reserved_path = True
                    break
            if reserved_path: continue

            state = self.solved_state
            for move in moves:
                state = apply_move(state, move)
                if state == self.solved_state: continue

            solution = [self.opposite_move[move] for move in moves[::-1]]
            assert apply_moves(state, solution) == self.solved_state
            # print(k, self.get_opt_sol_len(''.join(list(state))))
            return ' '.join(list(state)), ' '.join(solution)
        
    def generate_test_sequence(self):
        while True:
            k = random.randint(1, self.K)
            moves = random.choice(self.reserved_test_paths).copy()
            moves.extend(self.get_move_sequence(max(0, k - len(moves)), apply_moves(self.solved_state, moves)))

            state = self.solved_state
            for move in moves:
                state = apply_move(state, move)
                if state == self.solved_state: continue

            solution = [self.opposite_move[move] for move in moves[::-1]]
            assert apply_moves(state, solution) == self.solved_state
            return ' '.join(list(state)), ' '.join(solution)

    def __iter__(self):
        while True:
            scramble, solution = self.generate_train_sequence()
            yield CubeDataset(
                [scramble],
                [solution],
                self.tokenizer,
                max_length=self.max_length,
            )[0]

    def generate_validation_dataset(self, n):
        scrambles, solutions = [], []
        for i in range(n):
            scramble, solution = self.generate_train_sequence()
            scrambles.append(scramble)
            solutions.append(solution)
        return CubeDataset(
            scrambles,
            solutions,
            self.tokenizer,
            max_length=self.max_length
        )

    def generate_test_dataset(self, n):
        scrambles, solutions = [], []
        for i in range(n):
            scramble, solution = self.generate_test_sequence()
            scrambles.append(scramble)
            solutions.append(solution)
        return CubeDataset(
            scrambles,
            solutions,
            self.tokenizer,
            max_length=self.max_length
        )

def get_pretrain_data(data_path, tokenizer, max_length, valid_size=None):
    """
    Load data.
    if valid_size is not None, split data into train and valid sets.
    if valid_size is None, the whole data is returned as the first element and return None for the second element.
    """
    dataset = load_dataset(
        "csv",
        data_files=data_path,
    )
    dataset = CubePretrainDataset(
        dataset["train"]["scramble"],
        dataset["train"]["solution"],
        tokenizer,
        max_length=max_length,
    )

    train = dataset
    valid = None
    if valid_size:
        train_size = len(dataset) - valid_size
        train, valid = random_split(dataset, [train_size, valid_size])
    return train, valid
