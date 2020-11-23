import os
import pickle as pkl
from typing import Optional


import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_model_cache(model_save_folder):
    if not os.path.isdir(f"{model_save_folder}"):
        return {}

    model_types = os.listdir(f"{model_save_folder}")
    # model_types = {x: [] for x in model_types}
    models_in_cache = {}
    for mt in model_types:
        model_names = os.listdir(f"{model_save_folder}/{mt}")
        models_in_cache[mt] = model_names
    return models_in_cache


def load_model(model_folder):
    mdl = torch.load(f"{model_folder}/model.pkl", map_location=DEVICE)
    with open(f"{model_folder}/int_to_vocab.pkl", "rb") as itv:
        int_to_vocab = pkl.load(itv)
    with open(f"{model_folder}/vocab_to_int.pkl", "rb") as vti:
        vocab_to_int = pkl.load(vti)
    return mdl, int_to_vocab, vocab_to_int


def predict(
    model: nn.Module,
    device: str,
    starting_string: str,
    vocab_to_int: dict,
    int_to_vocab: dict,
    output_len: int = 100,
    stopping_word: Optional[str] = None,
    top_k: int = 3,
):
    """Predict some stuff"""
    model.eval()

    words = starting_string.split(" ")

    state_h, state_c = model.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = model(ix, (state_h, state_c))

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])

    if stopping_word:
        # Put this in to avoid the songs going on forever (looking at you Sun Kil Moon...)
        output_len = 100000

    for _ in range(output_len):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = model(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])
        if stopping_word:
            if int_to_vocab[choice].lower() == "<songend>":
                break

    return " ".join(words)
