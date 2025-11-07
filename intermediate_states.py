import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from datasets import load_dataset, DatasetDict
import einops
from fancy_einsum import einsum
from record_utils import record_activations
from pytwisty_moves import apply_moves
from itertools import groupby
import json
import argparse


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


def get_data(data_path, vocab, valid_size, timestep):
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
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.filter(
        lambda example: len(["".join(g) for _, g in groupby(example["solution"])])
        >= timestep
    )

    if valid_size != 0:
        train_test_split = dataset["train"].train_test_split(
            test_size=valid_size / len(dataset["train"])
        )
        dataset = DatasetDict(
            {"train": train_test_split["train"], "test": train_test_split["test"]}
        )
        print("Test split:", len(dataset["test"]))
    print("Train split:", len(dataset["train"]))
    return dataset


def evaluate(
    test_data_path,
    probe_model,
    batch_size,
    device,
    n_faces,
    n_colors,
    gpt2,
    record_module_names,
    valid_size,
    n_layers,
    timestep,
    test_size,
    random_seq_on_eval,
):
    """
    Evaluate probe model.
    """
    dataset = get_data(
        test_data_path,
        vocab,
        timestep=(0 if random_seq_on_eval else timestep),
        valid_size=0
    )
    test_size = min(test_size, len(dataset["train"]))
    dataset["train"] = dataset["train"].select(range(test_size))

    val_accuracies = []
    layer_accuracies = [[] for _ in range(n_layers)]
    layer_accuracies_full_seq = [[] for _ in range(n_layers)]

    for example in dataset["train"].iter(batch_size=batch_size):
        if not random_seq_on_eval:
            input_ids = torch.tensor(
                [x[: n_faces + timestep] for x in example["input_ids"]], device=device
            )

            curr_batch_size = input_ids.shape[0]
            cube_state = [
                [
                    vocab[sq]
                    for sq in apply_moves(
                        example["scramble"][i_scramble],
                        [
                            "".join(g)
                            for g in (
                                list(g) for _, g in groupby(example["solution"][i_scramble])
                            )
                            if ("".join(g)).isupper()
                        ][:timestep],
                    )
                ]
                for i_scramble in range(len(example["scramble"]))
            ]
        else:
            faces_only = [
                (x[:n_faces].tolist() if isinstance(x, torch.Tensor) else list(x[:n_faces]))
                for x in example["input_ids"]
            ]
            curr_batch_size = len(faces_only)
            rand_moves = [list(np.random.choice(possible_moves, size=max_moves, replace=True))
                        for _ in range(curr_batch_size)]
            input_ids = torch.tensor(
                [faces_only[i] + [vocab[m] for m in rand_moves[i][:timestep]]
                for i in range(curr_batch_size)],
                device=device,
            )
            cube_state = [
                [
                    vocab[sq]
                    for sq in apply_moves(
                        example["scramble"][i_scramble],
                        rand_moves[i_scramble][:timestep],
                    )
                ]
                for i_scramble in range(curr_batch_size)
            ]

        cube_state = F.one_hot(
            torch.tensor(cube_state),
            num_classes=n_colors,
        )
        assert cube_state.shape == (curr_batch_size, n_faces, n_colors)

        with record_activations(
            gpt2,
            record_module_names,
        ) as recording:
            gpt2(input_ids)

        # [batch, n_layers, seq, d_model]
        residual_stream = torch.stack([x[0] for x in recording.values()], dim=1)

        cube_state = cube_state.unsqueeze(dim=1).repeat((1, n_layers, 1, 1))
        _val_probe_out = einsum(
            "batch n_layers d_model, d_model n_layers n_faces n_colors -> batch n_layers n_faces n_colors",
            residual_stream[:, :, -1, :],
            probe_model,
        )

        val_preds = _val_probe_out.argmax(-1).to(device)
        val_gold = cube_state.argmax(-1).to(device)

        val_results = val_preds == val_gold
        val_accuracy = (val_results.sum() / val_results.numel()).item()
        val_accuracies.append(val_accuracy * input_ids.shape[0])

        for layer in range(n_layers):
            layer_val_results = val_results[:, layer, :]
            layer_results_full_seq = layer_val_results.all(dim=-1)
            layer_accuracy = (
                layer_val_results.sum() / layer_val_results.numel()
            ).item()
            layer_accuracy_full_seq = (
                layer_results_full_seq.sum() / layer_results_full_seq.numel()
            ).item()
            layer_accuracies[layer].append(layer_accuracy * input_ids.shape[0])
            layer_accuracies_full_seq[layer].append(
                layer_accuracy_full_seq * input_ids.shape[0]
            )

    validation_accuracy = sum(val_accuracies) / test_size
    print(f"  Inference Accuracy: {validation_accuracy}")

    output = dict()
    print("    Inference Accuracies per layer:")
    print("% sequence correct on average")
    i = 0
    for acc_list in layer_accuracies:
        acc = sum(acc_list) / test_size
        print("\t", acc)
        output[str(i)] = {"% sequence correct on average": acc}
        i += 1

    print("% of sequences entirely correct")
    i = 0
    for acc_list in layer_accuracies_full_seq:
        acc = sum(acc_list) / test_size
        print("\t", acc)
        output[str(i)]["% of sequences entirely correct"] = acc
        i += 1

    return output


def main(config):
    """Driver"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gpt2 = get_model(config["model_path"], device)
    print(f"Loaded model: {config['model_path']}")
    timestep = config["timestep"]
    dataset = get_data(
        config["train_data_path"],
        vocab,
        config["valid_size"],
        (0 if config["random_seq_on_train"] else timestep),
    )
    print(f"Loaded data: {config['train_data_path']}")

    gpt2_config = gpt2.config
    d_model = gpt2_config.n_embd

    n_faces = 24
    n_colors = 6

    lr = config["lr"]
    wd = config["wd"]
    batch_size = config["batch_size"]
    valid_size = len(dataset["test"])
    test_size = config["test_size"]
    valid_patience = config["valid_patience"]
    output_dir = config["output_dir"]
    valid_every = config["valid_every"] // batch_size
    num_epochs = config["num_epochs"]
    record_module_names = config["record_module_names"]
    n_layers = len(record_module_names)
    test_data_path = config["test_data_path"]
    random_seq_on_eval = config["random_seq_on_eval"]
    random_seq_on_train = config["random_seq_on_train"]

    probe_name = config["model_path"].split('/')[-1]
    lowest_val_loss = 999999

    # [d_model, n_layers, n_faces, n_colors]
    probe_model = torch.randn(
        d_model,
        n_layers,
        n_faces,
        n_colors,
        requires_grad=False,
        device=device,
    ) / np.sqrt(d_model)

    probe_model.requires_grad = True
    optimiser = torch.optim.AdamW(
        [probe_model], lr=lr, betas=(0.9, 0.99), weight_decay=wd
    )
    torch.manual_seed(42)
    np.random.seed(42)

    train_seen = 0
    done_training = False
    for epoch in range(num_epochs):
        if done_training:
            print(f"Training seen: {train_seen}")
            break

        idx = 0
        for example in dataset["train"].iter(batch_size=batch_size):
            idx += 1

            if done_training:
                print(f"Training seen: {train_seen}")
                break
            train_seen += batch_size

            if not random_seq_on_train:
                input_ids = torch.tensor(
                    [x[: n_faces + timestep] for x in example["input_ids"]], device=device
                )

                curr_batch_size = input_ids.shape[0]
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
                                if ("".join(g)).isupper()
                            ][:timestep],
                        )
                    ]
                    for i_scramble in range(len(example["scramble"]))
                ]
            else:
                # Build faces + random moves for training (same as eval random path)
                faces_only = [
                    (x[:n_faces].tolist() if isinstance(x, torch.Tensor) else list(x[:n_faces]))
                    for x in example["input_ids"]
                ]
                curr_batch_size = len(faces_only)
                rand_moves = [
                    list(np.random.choice(possible_moves, size=max_moves, replace=True))
                    for _ in range(curr_batch_size)
                ]
                input_ids = torch.tensor(
                    [
                        faces_only[i] + [vocab[m] for m in rand_moves[i][:timestep]]
                        for i in range(curr_batch_size)
                    ],
                    device=device,
                )
                cube_state = [
                    [
                        vocab[sq]
                        for sq in apply_moves(
                            example["scramble"][i_scramble],
                            rand_moves[i_scramble][:timestep],
                        )
                    ]
                    for i_scramble in range(curr_batch_size)
                ]

            cube_state = F.one_hot(
                torch.tensor(cube_state),
                num_classes=n_colors,
            )
            assert cube_state.shape == (curr_batch_size, n_faces, n_colors)

            with record_activations(
                gpt2,
                record_module_names,
            ) as recording:
                gpt2(input_ids)

            # [batch, n_layers, seq, d_model]
            residual_stream = torch.stack([x[0] for x in recording.values()], dim=1)

            cube_state = cube_state.unsqueeze(dim=1).repeat((1, n_layers, 1, 1))
            # --> [batch, n_layers, n_faces, n_colors]
            probe_out = einsum(
                "batch n_layers d_model, d_model n_layers n_faces n_colors -> batch n_layers n_faces n_colors",
                residual_stream[:, :, -1, :],  # [batch n_layers d_model]
                probe_model,  # [d_model n_layers n_faces n_colors]
            )
            # [batch, n_faces, n_colors]
            # Negative log-likelihood
            probe_log_probs = probe_out.log_softmax(dim=-1)
            probe_correct_log_probs = (
                einops.reduce(
                    probe_log_probs * cube_state.to(probe_log_probs.device),
                    "batch n_layers n_faces n_colors -> n_layers n_faces",
                    "mean",
                )
                * n_colors
            )

            train_loss = -1 * probe_correct_log_probs.mean()
            train_loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            print(train_loss)

            if idx % valid_every == 0:
                val_losses = []
                val_accuracies = []
                layer_accuracies = [[] for _ in range(n_layers)]
                for example in dataset["test"].iter(batch_size=batch_size):

                    input_ids = torch.tensor(
                        [x[: n_faces + timestep] for x in example["input_ids"]],
                        device=device,
                    )

                    curr_batch_size = input_ids.shape[0]
                    cube_state = [
                        [
                            vocab[sq]
                            for sq in apply_moves(
                                example["scramble"][i_scramble],
                                [
                                    "".join(g)
                                    for g in (
                                        list(g)
                                        for _, g in groupby(
                                            example["solution"][i_scramble]
                                        )
                                    )
                                    if ("".join(g)).isupper()
                                ][:timestep],
                            )
                        ]
                        for i_scramble in range(len(example["scramble"]))
                    ]

                    cube_state = F.one_hot(
                        torch.tensor(cube_state),
                        num_classes=n_colors,
                    )
                    assert cube_state.shape == (curr_batch_size, n_faces, n_colors)

                    with record_activations(
                        gpt2,
                        record_module_names,
                    ) as recording:
                        gpt2(input_ids)

                    # [batch, n_layers, seq, d_model]
                    residual_stream = torch.stack(
                        [x[0] for x in recording.values()], dim=1
                    )

                    cube_state = cube_state.unsqueeze(dim=1).repeat((1, n_layers, 1, 1))
                    _val_probe_out = einsum(
                        "batch n_layers d_model, d_model n_layers n_faces n_colors -> batch n_layers n_faces n_colors",
                        residual_stream[:, :, -1, :],
                        probe_model,
                    )
                    _val_probe_log_probs = _val_probe_out.log_softmax(dim=-1)
                    val_probe_correct_log_probs = (
                        einops.reduce(
                            _val_probe_log_probs
                            * cube_state.to(_val_probe_log_probs.device),
                            "batch n_layers n_faces n_colors -> n_layers n_faces",
                            "mean",
                        )
                        * n_colors
                    )

                    val_loss = -val_probe_correct_log_probs.mean()
                    val_losses.append(val_loss * input_ids.shape[0])

                    val_preds = _val_probe_out.argmax(-1).to(device)
                    val_gold = cube_state.argmax(-1).to(device)

                    val_results = val_preds == val_gold
                    val_accuracy = (val_results.sum() / val_results.numel()).item()
                    val_accuracies.append(val_accuracy * input_ids.shape[0])

                    for layer in range(n_layers):
                        layer_val_results = val_results[:, layer, :]
                        layer_accuracy = (
                            layer_val_results.sum() / layer_val_results.numel()
                        ).item()
                        layer_accuracies[layer].append(
                            layer_accuracy * input_ids.shape[0]
                        )

                validation_loss = sum(val_losses) / valid_size
                validation_accuracy = sum(val_accuracies) / valid_size
                print(f"  Validation Accuracy: {validation_accuracy}")

                print("    Validation Accuracies per layer:")
                for acc_list in layer_accuracies:
                    print("\t", sum(acc_list) / valid_size)

                print(f"  Validation Loss: {validation_loss}")
                if validation_loss < lowest_val_loss:
                    print(f"  New lowest valid loss! {validation_loss}")
                    curr_patience = 0
                    torch.save(
                        probe_model,
                        f"{output_dir}/{probe_name}_timestep_{timestep}.pth",
                    )

                    lowest_val_loss = validation_loss

                else:
                    curr_patience += 1
                    print(f"  Did not beat previous best ({lowest_val_loss})")
                    print(f"  Current patience: {curr_patience}")
                    if curr_patience >= valid_patience:
                        print("  Ran out of patience! Stopping training.")
                        done_training = True
    return evaluate(
        test_data_path,
        probe_model,
        batch_size,
        device,
        n_faces,
        n_colors,
        gpt2,
        record_module_names,
        valid_size,
        n_layers,
        timestep,
        test_size,
        random_seq_on_eval,
    )


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    return parser.parse_args()


# IMP READ THIS:
#   UPDATE THE MODEL PATH, TRAIN DATA PATH, TEST DATA PATH, AND OUTPUT FILE NAMES BELOW

#   THE FORMAT OF THE OUTPUT FILE IS OF THE FORM:
#       {
#           "TIMESTEP/MOVES AFTER SCRAMBLE": {
#               "LAYER": {
#                   "% sequence correct on average": ,
#                   "% of sequences entirely correct":
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
max_length = 50
possible_moves = ["R","RR","RRR","U","UU","UUU","F","FF","FFF"]
max_moves = 11

if __name__ == "__main__":
    args = get_cli_args()
    data = dict()
    for timestep in range(max_moves + 1):
        config = {
            "model_path": args.model_path,  # UPDATE THIS
            "train_data_path": "/home/path/data222_complete_faces_threshold_2_train_2starts_new.csv",  # UPDATE THIS, DON'T PASS A "SPLIT" FILE
            "test_data_path": "/home/path/data222_complete_faces_threshold_2_test_2starts_new.csv",  # UPDATE THIS, DON'T PASS A "SPLIT" FILE
            "lr": 1e-3,
            "wd": 0.01,
            "valid_every": 200,
            "batch_size": 32,
            "valid_size": 512,
            "test_size": 1024,
            "valid_patience": 10,
            "output_dir": "/home/path/intermediate_states_probes/",
            "record_module_names": [
                "transformer.h.0",
                "transformer.h.1",
                "transformer.h.2",
                "transformer.h.3",
                "transformer.h.4",
                "transformer.h.5",
                "transformer.h.6",
                "transformer.h.7",
            ],
            "num_epochs": 1,
            "timestep": timestep,
            "random_seq_on_eval": False,
            "random_seq_on_train": True,
        }
        data[timestep] = main(config)
    json_object = json.dumps(data, indent=4)
    with open(
        f"/home/path/intermediate_states_results/{config["model_path"].split('/')[-1]}.json", # UPDATE THIS
        "w",
    ) as outfile:
        outfile.write(json_object)
