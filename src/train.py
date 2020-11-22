from collections import Counter
from dataclasses import dataclass
import os
import pickle as pkl

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from prep_data import CleanTextData

# Class for model parameters
@dataclass
class ModelConfig:
    seq_size: int
    batch_size: int
    embedding_size: int
    hidden_layer_size: int
    n_layers: int
    dropout: float
    gradients_clipping: int
    epochs: int


class TrainModel:
    def __init__(
        self,
        text: str,
        model_name: str,
        model: nn.Module,
        model_config: ModelConfig,
        model_save_folder: str = "./src/model_files",
        run_in_streamlit: bool = False,
        criterion=None,
        optimiser=None,
    ):

        self.text = text
        (
            self.n_vocab,
            self.int_to_vocab,
            self.vocab_to_int,
            self.int_text,
        ) = self.encode_text(self.text)

        self.model_name = model_name
        self.model_config = model_config
        self.model = model(
            self.n_vocab,
            self.model_config.seq_size,
            self.model_config.embedding_size,
            self.model_config.hidden_layer_size,
            self.model_config.n_layers,
            self.model_config.dropout,
        )

        self.model_save_folder = model_save_folder
        if not os.path.exists(self.model_save_folder):
            os.mkdir(self.model_save_folder)
        self.model_save_path = f"{self.model_save_folder}/{self.model.model_type}"

        self.run_in_streamlit = run_in_streamlit

        if self.run_in_streamlit:
            self.print_fn = st.write
        else:
            self.print_fn = print

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.print_fn(f"Device: {self.device}")

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        if optimiser is None:
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.01)
        else:
            self.optimiser = optimiser

    def encode_text(self, text: str) -> [int, dict, dict, list]:
        word_counts = Counter(text)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
        vocab_to_int = {w: k for k, w in int_to_vocab.items()}
        n_vocab = len(int_to_vocab)
        int_text = [vocab_to_int[w] for w in text]

        print("Vocabulary size", n_vocab)

        return n_vocab, int_to_vocab, vocab_to_int, int_text

    def create_training_data(
        self, int_text: list, vocab_to_int: dict
    ) -> [np.array, np.array]:

        num_batches = int(
            len(int_text) / (self.model_config.seq_size * self.model_config.batch_size)
        )
        in_text = int_text[
            : num_batches * self.model_config.batch_size * self.model_config.seq_size
        ]
        out_text = np.zeros_like(in_text)
        out_text[:-1] = in_text[1:]
        out_text[-1] = in_text[0]
        in_text = np.reshape(in_text, (self.model_config.batch_size, -1))
        out_text = np.reshape(out_text, (self.model_config.batch_size, -1))

        return in_text, out_text

    def get_batches(self, in_text: np.array, out_text: np.array):
        num_batches = np.prod(in_text.shape) // (
            self.model_config.seq_size * self.model_config.batch_size
        )
        for i in range(
            0, num_batches * self.model_config.seq_size, self.model_config.seq_size
        ):
            yield in_text[:, i : i + self.model_config.seq_size], out_text[
                :, i : i + self.model_config.seq_size
            ]

    def save_model_and_maps(self, num_epochs: int) -> None:
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        model_run = self.model_save_path + f"/model-{self.model_name}-{num_epochs}"
        if not os.path.exists(model_run):
            os.mkdir(model_run)
        torch.save(
            self.model,
            f"{model_run}/model.pkl",
        )
        with open(f"{model_run}/int_to_vocab.pkl", "wb") as itv:
            pkl.dump(self.int_to_vocab, itv)
        with open(f"{model_run}/vocab_to_int.pkl", "wb") as vti:
            pkl.dump(self.vocab_to_int, vti)

    def train(self):
        iteration = 0

        losses = []

        for e in range(self.model_config.epochs + 1):

            in_text, out_text = self.create_training_data(
                self.int_text, self.vocab_to_int
            )
            batches = self.get_batches(
                in_text,
                out_text,
            )
            state_h, state_c = self.model.zero_state(self.model_config.batch_size)

            # Transfer data to GPU
            state_h = state_h.to(self.device)
            state_c = state_c.to(self.device)
            for x, y in batches:
                iteration += 1

                # Tell it we are in training mode
                self.model.train()

                # Reset all gradients
                self.optimiser.zero_grad()

                # Transfer data to GPU (if present)
                x = torch.tensor(x).to(self.device)
                y = torch.tensor(y).to(self.device)

                logits, (state_h, state_c) = self.model(x, (state_h, state_c))
                loss = self.criterion(logits.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss_value = loss.item()

                # Perform back-propagation
                loss.backward()

                _ = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.model_config.gradients_clipping
                )

                # Update the network's parameters
                self.optimiser.step()

            losses.append(loss_value)

            self.print_fn(
                "Epoch: {}/{}".format(e, self.model_config.epochs),
                "Loss: {}".format(loss_value),
            )
        self.save_model_and_maps(num_epochs=e)

        return losses

    def run_training(self):

        losses = self.train()
        return losses, self.model, self.vocab_to_int, self.int_to_vocab