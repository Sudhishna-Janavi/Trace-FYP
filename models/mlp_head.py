"""Model 6 of 7: a feedforward neural network on MiniLM embeddings.

This replaces Model 5's logistic-regression head with a multilayer perceptron
over the identical frozen embeddings, so the comparison isolates exactly one
question: **does added depth help?**

Architecture
    384 -> 256 -> 128 -> 17, ReLU between layers, dropout before each hidden
    layer, softmax at the output. A logistic regression is the same network with
    no hidden layer at all, which is why the two are directly comparable.

Components, as taught
    softmax + cross-entropy  the standard multi-class output and loss; in
                             PyTorch, CrossEntropyLoss applies the softmax
                             internally, so the network emits raw logits.
    ReLU                     non-linearity; without it, stacking linear layers
                             collapses back to a single linear layer and the
                             depth buys nothing.
    dropout                  randomly zeroes activations during training so the
                             network cannot lean on any single unit, which is
                             the main defence against overfitting here.
    Adam                     adaptive per-parameter learning rates.
    class weights            the categories are uneven after deduplication
                             (Restaurants 3,110 against Rent 874), and macro-F1
                             penalises a model that favours the larger ones.
    early stopping           training halts when a held-out slice stops
                             improving, and the best weights are restored.

Why early stopping uses a slice of TRAIN, not the validation set
    Every other model in this comparison is scored on validation without having
    seen it. Stopping on validation would let this one model tune against the
    set used to compare them all, so a 10% slice of the training data is held
    back for that purpose instead and validation stays untouched.
"""
from __future__ import annotations

import torch
from torch import nn

SEED = 1131
HIDDEN = (256, 128)
DROPOUT = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 12
EARLY_STOP_FRACTION = 0.10


class MLPClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int,
                 hidden: tuple[int, ...] = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        layers: list[nn.Module] = []
        size = n_features
        for width in hidden:
            layers += [nn.Dropout(dropout), nn.Linear(size, width), nn.ReLU()]
            size = width
        # no softmax here: CrossEntropyLoss applies it internally and is more
        # numerically stable than computing it separately
        layers.append(nn.Linear(size, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
