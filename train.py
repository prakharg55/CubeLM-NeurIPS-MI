import os
import random
from random import randrange, randint
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import einops
from fancy_einsum import einsum
from cubeLM import CubeLM, hf_tokenizer
from cubeLM.utils import IGNORE_INDEX, get_data, RandomDataStream
from cubeLM.tokenizer import vocab
from pytwisty_moves import apply_moves
from transformers import GPT2Config
import wandb
import json


def main(config):
    """Driver"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    wandb.init(project="CubeWorldModel", config=config, mode="offline")
    wandb.run.name = config["exp_name"]
    wandb.run.save("UPDATE ME")

    task = config["task"]
    pretrain_path = config["pretrain_path"]
    lr = config["lr"]
    wd = config["wd"]
    max_length = config["max_length"]
    batch_size = config["batch_size"]
    valid_size = config["valid_size"]
    valid_patience = config["valid_patience"]
    valid_every = config["valid_every"]
    num_epochs = config["num_epochs"]
    save_path = config["model_path"]
    alpha = config["alpha"]
    d_model = config["d_model"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    print(config, flush=True)
    vocab_size = len(vocab)
    gpt2_config = GPT2Config(
        vocab_size=vocab_size,
        bos_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 1,
        pad_token_id=vocab_size - 1,
        n_positions=max_length,
        n_embd=d_model,
        n_layer=n_layers,
        n_head=n_heads,
    )
    if alpha:
        gpt2_config.alpha = alpha

    n_faces = 24
    n_colors = 6
    n_layers = gpt2_config.n_layer

    lowest_val_loss = 999999

    gpt2 = CubeLM(gpt2_config, task=task, num_heads=n_faces, num_classes=n_colors)

    if pretrain_path is not None:
        print("Loading pretrained model from ", pretrain_path, flush=True)
        state_dict = torch.load(pretrain_path, map_location="cpu")
        gpt2.load_state_dict(state_dict, strict=False)

    gpt2 = gpt2.to(device)
    tokenizer = hf_tokenizer
    if config["data_type"] == "generated":
        dataset_stream = RandomDataStream(tokenizer=tokenizer, max_length=max_length, seq_type="incremental", K=config["K"], reserved_test_paths=config["reserved_test_paths"])
        test_dataset = dataset_stream.generate_validation_dataset(valid_size)
        train_dataloader = DataLoader(dataset_stream, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        # check that data_path exists.
        if not os.path.exists(config["data_path"]):
            raise ValueError(f"Data path {config["data_path"]} does not exist.")
        
        train_dataset, test_dataset = get_data(
            config["data_path"], tokenizer, max_length, valid_size=valid_size
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    gpt2_config = gpt2.config
    d_model = gpt2_config.n_embd

    optimizer = torch.optim.AdamW(
        gpt2.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=wd
    )

    done_training = False
    gpt2.train()
    for epoch in range(num_epochs):
        for idx, (input_ids, attention_mask, cube_states, _) in enumerate(
            train_dataloader
        ):

            if done_training:
                break

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            cube_states = cube_states.to(device)
            labels = None
            _cube_states = None
            if task in ["sft", "joint"]:
                labels = input_ids
            if task in ["pretrain", "joint"]:
                _cube_states = cube_states
            output = gpt2(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                cube_states=_cube_states,
            )
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if idx % config["log_every"] == 0:
                print(f"  Training Loss: {loss.item()}", flush=True)
                wandb.log({"train_loss": loss.item()})

            if idx % valid_every == 0:
                gpt2.eval()
                val_losses = []
                val_accuracies = []
                for input_ids, attention_mask, cube_states, _ in test_dataloader:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    cube_states = cube_states.to(device)
                    labels = None
                    _cube_states = None
                    if task in ["sft", "joint"]:
                        labels = input_ids
                    if task in ["pretrain", "joint"]:
                        _cube_states = cube_states
                    with torch.no_grad():
                        output = gpt2(
                            input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            cube_states=_cube_states,
                        )
                    loss = output.loss
                    if task in ["pretrain", "joint"]:
                        logits = output["cube_logits"]
                        preds = logits.argmax(-1)
                        mask = cube_states.view(-1) != IGNORE_INDEX
                        val_results = preds.view(-1)[mask] == cube_states.view(-1)[mask]
                        val_accuracies.append(val_results.float().mean().item())
                    else:
                        logits = output["logits"]
                        preds = logits.argmax(-1)
                        accuracy = []
                        for i in range(len(preds)):
                            # print("input_ids:",tokenizer.decode(input_ids[i], skip_special_tokens=True).split())
                            scramble = tokenizer.decode(input_ids[i][:24], skip_special_tokens=True).split()
                            solution = tokenizer.decode(preds[i][23:], skip_special_tokens=True).split()
                            if i < 3: 
                                try:
                                    print(''.join(scramble), dataset_stream.get_opt_sol_len(''.join(scramble)), solution, dataset_stream.get_opt_sol_len(apply_moves(''.join(scramble),solution)))
                                except:
                                    print('invalid move predicted')
                            try:
                                accuracy.append(int(apply_moves(''.join(scramble), solution) == "uuuurrrrffffddddllllbbbb"))
                            except:
                                accuracy.append(0)
                        val_accuracies.append(sum(accuracy)/len(accuracy))
                    val_losses.append(loss.item())

                gpt2.train()
                validation_loss = sum(val_losses) / len(val_losses)
                validation_accuracy = sum(val_accuracies) / len(val_losses)
                wandb.log(
                    {
                        "valid_loss": validation_loss,
                        "valid_accuracy": validation_accuracy,
                    }
                )
                print(f"  Validation Loss: {validation_loss}", flush=True)
                print(f"  Validation Accuracy: {validation_accuracy}", flush=True)
                if validation_loss < lowest_val_loss:
                    print(f"  New lowest valid loss! {validation_loss}", flush=True)
                    curr_patience = 0
                    torch.save(gpt2.state_dict(), save_path)
                    lowest_val_loss = validation_loss

                else:
                    curr_patience += 1
                    print(f"  Did not beat previous best ({lowest_val_loss})", flush=True)
                    print(f"  Current patience: {curr_patience}", flush=True)
                    if curr_patience >= valid_patience:
                        print("  Ran out of patience! Stopping training.", flush=True)
                        done_training = True


if __name__ == "__main__":
    # Get seed from command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1423)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain_path", type=str, default=None)
    parser.add_argument("--task", type=str)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--reserved_test_paths", type=str, default='[]') # e.g. '["F", "R U"]'

    args = parser.parse_args()
    seed = args.seed
    wd = args.wd
    lr = args.lr
    batch_size = args.batch_size
    pretrain_path = args.pretrain_path
    task = args.task
    alpha = args.alpha
    d_model = args.d_model
    n_layers = args.n_layers
    n_heads = args.n_heads
    data_path = args.data_path
    data_type = args.data_type
    K = args.K
    reserved_test_paths = json.loads(args.reserved_test_paths)

    assert data_type in ["first", "full", "generated"]

    if task == "joint" and alpha is None:
        raise ValueError("Alpha must be set for joint training.")
    if pretrain_path is not None:
        if task in ["pretrain", "joint"]:
            raise ValueError(
                "Pretrain path should not be set for pretraining or joint training."
            )

    assert task in ["pretrain", "sft", "joint"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    base_dir = "UPDATE ME"
    ckpts_dir = os.path.join(base_dir, "ckpts")
    data_dir = os.path.join(base_dir, "data")
    exp_name = f"{task}_seed_{seed}_{d_model}d_{n_layers}L_{n_heads}H_datatype_{data_type}"
    if pretrain_path is not None:
        exp_name += "_pretrain"
    if alpha is not None:
        exp_name += f"_alpha_{alpha}"
    save_path = os.path.join(ckpts_dir, exp_name + ".pt")

    config = {
        "model_path": save_path,
        "data_path": data_path,
        "lr": lr,
        "wd": wd,
        "max_length": 60,
        "valid_every": 200,
        "batch_size": batch_size,
        "valid_size": 512,
        "valid_patience": 99999,
        "num_epochs": 100,
        "log_every": 100,
        "task": task,
        "pretrain_path": pretrain_path,
        "alpha": alpha,
        "exp_name": exp_name,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "data_type": data_type,
        "K": K,
        "reserved_test_paths": reserved_test_paths,
    }
    main(config)
