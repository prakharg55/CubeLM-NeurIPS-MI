import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from datasets import load_dataset, DatasetDict
import einops
from fancy_einsum import einsum
from intervene_utils import add_intervene_hooks, remove_hooks
from record_utils import record_activations
from pytwisty_moves import apply_moves, apply_move
from itertools import groupby
from random import randrange, randint
import argparse


complete_data = json.load(open("/home/path/complete_data.json"))


def test_move(prompt, move):
    state = apply_moves(prompt[:24], ["".join(g) for _, g in groupby(prompt[24:])])
    opt_sol_len = len(["".join(g) for _, g in groupby(complete_data[state])])
    new_sol_len = len(
        ["".join(g) for _, g in groupby(complete_data[apply_move(state, move)])]
    )
    if new_sol_len < opt_sol_len:
        return "good"
    elif new_sol_len == opt_sol_len:
        return "neutral"
    else:
        return "bad"


def get_all_good_moves(prompt):
    state = apply_moves(prompt[:24], ["".join(g) for _, g in groupby(prompt[24:])])
    opt_sol_len = len(["".join(g) for _, g in groupby(complete_data[state])])
    good_moves = []
    for move in ["R", "RR", "RRR", "U", "UU", "UUU", "F", "FF", "FFF"]:
        new_sol_len = len(
            ["".join(g) for _, g in groupby(complete_data[apply_move(state, move)])]
        )
        if new_sol_len < opt_sol_len:
            good_moves.append(move)
    return good_moves


def tokenizer_encode(text, vocab, max_length):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
            [vocab[word] for word in i]
            + [vocab["<|endoftext|>"] for word in range(max_length - len(i))]
        )
        inputs["attention_mask"].append(
            [1 for word in i] + [0 for word in range(max_length - len(i))]
        )
    inputs["input_ids"] = torch.tensor(inputs["input_ids"], device=device)
    inputs["attention_mask"] = torch.tensor(inputs["attention_mask"], device=device)
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def tokenizer_decode(generated_ids, vocab):
    vocab_inv = {v: k for k, v in vocab.items()}
    del vocab_inv[vocab["<|endoftext|>"]]
    generated_ids = list(map(int, generated_ids))
    return "".join([vocab_inv[word] for word in generated_ids if word in vocab_inv])


def get_model(model_path, device):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    return model


def get_data(data_path, vocab, init_moves_away):
    """Load data."""

    def encode(examples):
        combined_text = []
        for i in range(max(len(examples["scramble"]), len(examples["solution"]))):
            examples["scramble"][i] = examples["scramble"][i].lower()
            if not (examples["solution"][i]):
                examples["solution"][i] = ""
            combined_text.append(examples["scramble"][i] + examples["solution"][i])
        return tokenizer_encode(combined_text, vocab, max_length=max_length)

    dataset = load_dataset(
        "csv",
        data_files=data_path,
    )
    dataset = dataset.map(encode, batched=True)
    # dataset = dataset.shuffle()
    dataset = dataset.filter(
        lambda example: len(["".join(g) for _, g in groupby(example["solution"])])
        == init_moves_away
    )
    return dataset


@torch.no_grad()
def main(config):
    """Driver"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gpt2 = get_model(config["model_path"], device)
    print(f"Loaded model: {config['model_path']}")
    dataset = get_data(config["data_path"], vocab, config["init_moves_away"])
    print(f"Loaded data: {config['data_path']}")

    gpt2_config = gpt2.config
    d_model = gpt2_config.n_embd

    n_faces = 24
    n_colors = 6

    idx2vocab = {v: k for k, v in vocab.items()}

    record_module_names = config["record_module_names"]
    n_layers = len(record_module_names)
    timestep = config["timestep"]

    # [d_model, layers, 24, 6]
    probe = torch.load(
        f'{config["probe_dir"]}/{config["model_path"].split('/')[-1]}_timestep_{timestep}.pth'
    )

    orig_good = 0
    good = 0
    total = 0
    state_matches_orig_state = 0
    state_matches_orig_state_percent = 0
    pred_state_matches_target_state = 0
    pred_state_matches_target_state_percent = 0
    pred_move_matches_orig_move = 0
    pred_move_good_for_orig_state = 0
    pred_move_matches_target_pred = 0
    sum_prob_mass_diff = 0
    sum_best_good_move_diff = 0
    sum_prob_mass_ratio = 0.0
    sum_best_good_move_ratio = 0.0
    for example in dataset["train"].iter(batch_size=2):
        if len(example["scramble"]) < 2:
            break

        for src_idx, dst_idx in ((0, 1), (1, 0)):
            if total >= config["max_interventions"]:
                break

            input_ids = torch.tensor(
                example["input_ids"], device=device, dtype=torch.long
            )[:, : n_faces + timestep]
            src_input_id = input_ids[src_idx].unsqueeze(0)
            target_input_id = input_ids[dst_idx].unsqueeze(0)

            cube_state = [
                [
                    vocab[sq]
                    for sq in apply_moves(
                        example["scramble"][i_scramble],
                        [
                            "".join(g)
                            for g in (
                                list(g)
                                for _, g in groupby(example["solution"][i_scramble])
                            )
                            if "".join(g)
                        ][:timestep],
                    )
                ]
                for i_scramble in range(len(example["scramble"]))
            ]

            orig_scramble = apply_moves(
                example["scramble"][src_idx],
                [
                    "".join(g)
                    for g in (list(g) for _, g in groupby(example["solution"][src_idx]))
                    if "".join(g)
                ][:timestep],
            )
            target_scramble = apply_moves(
                example["scramble"][dst_idx],
                [
                    "".join(g)
                    for g in (list(g) for _, g in groupby(example["solution"][dst_idx]))
                    if "".join(g)
                ][:timestep],
            )

            if (
                len(
                    set(get_all_good_moves(orig_scramble)).intersection(
                        get_all_good_moves(target_scramble)
                    )
                )
                > 0
            ):
                continue

            # [batch, seq, vocab]
            logits = gpt2(input_ids=src_input_id).logits.clone()
            src_probs_before = logits[0, -1].softmax(-1)
            pred = src_probs_before.argmax(-1)

            logits = gpt2(input_ids=target_input_id).logits.clone()
            pred_target = logits[0, -1].softmax(-1).argmax(-1)

            with record_activations(
                gpt2,
                ["transformer.h.7"],
            ) as recording:
                logits = gpt2(input_ids=src_input_id).logits.clone()

            assert pred == logits[0, -1].softmax(-1).argmax(-1)

            residual_stream = recording["transformer.h.7"][0].squeeze()
            probe_out = einsum(
                "d_model, d_model n_faces n_colors -> n_faces n_colors",
                residual_stream[-1, :],
                probe[:, -1, ...],
            )
            orig_cube_state = probe_out.log_softmax(dim=-1).argmax(-1).tolist()

            hooks = []
            for layer in record_module_names:
                layer_num = int(layer[-1])
                hooks.extend(
                    add_intervene_hooks(
                        gpt2,
                        [layer],
                        probe[:, layer_num, ...],
                        cube_state[src_idx],
                        cube_state[dst_idx],
                        config["margin"],
                    )
                )
            with record_activations(
                gpt2,
                ["transformer.h.7"],
            ) as recording:
                hacked_logits = gpt2(input_ids=src_input_id).logits.clone()

            remove_hooks(hooks)

            # [seq, d_model]
            residual_stream = recording["transformer.h.7"][0].squeeze()
            probe_out = einsum(
                "d_model, d_model n_faces n_colors -> n_faces n_colors",
                residual_stream[-1, :],
                probe[:, -1, ...],
            )
            hacked_cube_state = probe_out.log_softmax(dim=-1).argmax(-1).tolist()

            # Prediction after interventions
            hacked_probs = hacked_logits[0, -1].softmax(-1)
            hacked_pred = hacked_probs.argmax(-1)

            try:
                good_hack = test_move(target_scramble, idx2vocab[hacked_pred.item()])
            except:
                good_hack = "invalid_move"
            if good_hack == "good":
                good += 1
            total += 1

            if cube_state[dst_idx] == hacked_cube_state:
                pred_state_matches_target_state += 1
            pred_state_matches_target_state_percent += (
                sum(
                    [
                        cube_state[dst_idx][square] == hacked_cube_state[square]
                        for square in range(24)
                    ]
                )
                / 24
            )

            if orig_cube_state == cube_state[src_idx]:
                state_matches_orig_state += 1
            state_matches_orig_state_percent += (
                sum(
                    [
                        orig_cube_state[square] == cube_state[src_idx][square]
                        for square in range(24)
                    ]
                )
                / 24
            )

            try:
                orig_good_move = test_move(orig_scramble, idx2vocab[pred.item()])
            except:
                orig_good_move = "invalid_move"
            if orig_good_move == "good":
                orig_good += 1

            if idx2vocab[hacked_pred.item()] == idx2vocab[pred.item()]:
                pred_move_matches_orig_move += 1

            try:
                if test_move(orig_scramble, idx2vocab[hacked_pred.item()]) == "good":
                    pred_move_good_for_orig_state += 1
            except:
                pass

            if idx2vocab[hacked_pred.item()] == idx2vocab[pred_target.item()]:
                pred_move_matches_target_pred += 1

            try:
                target_good_moves = get_all_good_moves(target_scramble)
                if len(target_good_moves) > 0:
                    good_ids = [vocab[m] for m in target_good_moves]
                    good_ids_t = torch.tensor(good_ids, device=device, dtype=torch.long)

                    # diff in total prob mass on all target-good moves
                    before_mass = src_probs_before[good_ids_t].sum().item()
                    after_mass = hacked_probs[good_ids_t].sum().item()
                    sum_prob_mass_diff += (after_mass - before_mass)

                    # diff in prob for the best (highest-prob-after) target-good move
                    best_rel_idx = torch.argmax(hacked_probs[good_ids_t]).item()
                    best_token_id = good_ids_t[best_rel_idx].item()
                    sum_best_good_move_diff += (hacked_probs[best_token_id] - src_probs_before[best_token_id]).item()

                    orig_good_moves = get_all_good_moves(orig_scramble)
                    if len(orig_good_moves) > 0:
                        orig_good_ids = [vocab[m] for m in orig_good_moves]
                        orig_good_ids_t = torch.tensor(orig_good_ids, device=device, dtype=torch.long)

                        # ratio of prob mass on target-good / original-good
                        target_mass_after = hacked_probs[good_ids_t].sum().item()
                        orig_mass_after = hacked_probs[orig_good_ids_t].sum().item()
                        if orig_mass_after > 0.0:
                            sum_prob_mass_ratio += (target_mass_after / orig_mass_after)

                        # ratio of best target-good prob / best original-good prob
                        best_target_after = hacked_probs[good_ids_t].max().item()
                        best_orig_after = hacked_probs[orig_good_ids_t].max().item()
                        if best_orig_after > 0.0:
                            sum_best_good_move_ratio += (best_target_after / best_orig_after)
            except:
                pass

            print("Orig:")
            print("  Scramble:")
            print("  " + orig_scramble)
            print("  Pred:")
            print("  " + idx2vocab[pred.item()])
            print("  " + orig_good_move)
            print("Hacked:")
            print("  Scramble:")
            print("  " + target_scramble)
            print("  Pred Cube State:")
            print(f'  {"".join([idx2vocab[i] for i in hacked_cube_state])}')
            print(f"  Cube state eq: {cube_state[dst_idx] == hacked_cube_state}")
            print("  Pred:")
            print("  " + idx2vocab[hacked_pred.item()])
            print("  " + good_hack)
            print("-----------")
            print("-----------")

            if total >= config["max_interventions"]:
                break

        if total >= config["max_interventions"]:
            break

    if total > 0:

        return {
            "Original move Good %": orig_good/total,  # Pred move after hack is Good for target state
            "Pred move Good for target state %": good/total,  # Pred move after hack matches unhacked model's pred for target state
            "Pred move matches original move %": pred_move_matches_orig_move/total,  # hacked state fully matches target state
            "Pred move is good for original state %": pred_move_good_for_orig_state/total,  # Original probe state fully matches original state
            "Original probe state matches original state %": state_matches_orig_state/total,  # Original move before hack is Good for original state
            "Avg percentage Original probe state matches original state %": state_matches_orig_state_percent/total,  # Pred move after hack matches original move
            "Pred state matches target state %": pred_state_matches_target_state/total,  # % of original probe state that matches original state
            "Avg percentage Pred state matches target state %": pred_state_matches_target_state_percent/total,  # % of hacked state that matches target state
            "Pred move matches target state's model pred %": pred_move_matches_target_pred/total,  # Pred move after hack is good for original state
            "Avg diff in prob mass on target-good moves": sum_prob_mass_diff/total, # diff in total prob mass on all target-good moves
            "Avg diff in prob of best target-good move": sum_best_good_move_diff/total, # diff in prob for the best (highest-prob-after) target-good move
            "Avg ratio prob mass (target-good/original-good) after": sum_prob_mass_ratio/total, # ratio of prob mass on target-good / original-good
            "Avg ratio best good-move prob (target/original) after": sum_best_good_move_ratio/total, # ratio of best target-good prob / best original-good prob
            "total": total,
        }
    else:
        return {}


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    return parser.parse_args()


# IMP READ THIS:
#   UPDATE THE MODEL PATH, TRAIN DATA PATH, PROBE DIRECTORY PATH, AND OUTPUT FILE NAMES BELOW

#   THE FORMAT OF THE OUTPUT FILE IS OF THE FORM:
#       {
#           "MOVES-AWAY COMPLEXITY OF INITIAL SCRAMBLE": {
#               "TIMESTEP/MOVES AFTER INITIAL SCRAMBLE": {
#                   "KEY": 
#               }
#           }
#       }

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
vocab_size = len(vocab)
max_length = 40
possible_moves = ["R", "RR", "RRR", "U", "UU", "UUU", "F", "FF", "FFF"]

if __name__ == "__main__":
    args = get_cli_args()
    data = dict()
    for margin in [1]:
        for init_moves_away in range(3, 12):
            data[str(init_moves_away)] = dict()
            for timestep in range(init_moves_away):
                config = {
                    "model_path": args.model_path, # UPDATE THIS
                    "data_path": "/home/path/data222_complete_faces_threshold_2_test_2starts_new.csv", # UPDATE THIS, DON'T PASS A "SPLIT" FILE
                    "probe_dir": "/home/path/final_probes/intermediate_states_probes/", # UPDATE THIS
                    "record_module_names": [
                        # "transformer.h.0",
                        # "transformer.h.1",
                        # "transformer.h.2",
                        # "transformer.h.3",
                        "transformer.h.4",
                        "transformer.h.5",
                        "transformer.h.6",
                        # "transformer.h.7",
                    ],
                    "max_interventions": 1000,
                    "init_moves_away": init_moves_away,
                    "timestep": timestep,
                    "margin": margin,
                }
                data[str(init_moves_away)][str(timestep)] = main(config)
        json_object = json.dumps(data, indent=4)
        with open(
            f'/home/path/intervene_results/{config["model_path"].split('/')[-1]}_margin_{margin}.json', # UPDATE THIS
            "w",
        ) as outfile:
            outfile.write(json_object)
