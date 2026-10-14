#!/usr/bin/env python3
"""Write the Model 6 findings document from the stored results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reporting import FIGURES, write_doc  # noqa: E402


def main() -> int:
    d = json.loads((ROOT / "results" / "model6_mlp.json").read_text())
    prev = json.loads((ROOT / "results" / "model5_minilm.json").read_text())
    svm = json.loads((ROOT / "results" / "model4_linear_svm.json").read_text())

    keys = ["stratified (validation)", "merchant held out (validation)",
            "  of which strictly novel"]
    pretty = ["Stratified (validation)", "Merchant held out (validation)",
              "of which strictly novel"]
    r, cal, arch = d["results"], d["calibration"], d["architecture"]
    fig = {n: FIGURES / n for n in d["figures"]}

    deltas = {k: r[k]["macro_f1"] - prev["results"][k]["macro_f1"] for k in keys}
    helps = any(v > 0.01 for v in deltas.values())
    hurts = all(v < 0.005 for v in deltas.values())

    write_doc(
        "model6_neural_network.docx",
        "Model 6 of 7 - Feedforward Neural Network on MiniLM Embeddings",
        "Unit 2, model comparison. Tests whether added depth improves on a "
        "linear head over the same frozen embeddings.",
        [
        ("h1", "1. What this model is"),
        ("p", "A feedforward neural network placed on top of the same frozen "
              "MiniLM embeddings used by Model 5. Model 5 put a logistic "
              "regression on those 384 numbers; this one puts a multilayer "
              "perceptron on them instead. Because the representation is "
              "identical, the comparison isolates exactly one question: **does "
              "added depth help?**"),
        ("h2", "1.1 Architecture and training"),
        ("table", [["Component", "Choice", "Why"],
                   ["Layers",
                    f"{arch['input']} -> {' -> '.join(str(h) for h in arch['hidden'])} -> {arch['output']}",
                    "a logistic regression is this network with no hidden layer, "
                    "which is what makes the two directly comparable"],
                   ["Non-linearity", "ReLU",
                    "without it, stacked linear layers collapse to a single "
                    "linear layer and the depth buys nothing"],
                   ["Output and loss", "softmax with cross-entropy",
                    "the standard multi-class pairing; PyTorch applies the "
                    "softmax inside the loss for numerical stability, so the "
                    "network emits raw logits"],
                   ["Regularisation", f"dropout {arch['dropout']}",
                    "randomly zeroes activations during training so the network "
                    "cannot lean on any single unit"],
                   ["Optimiser", f"Adam, learning rate {arch['learning_rate']}",
                    "adaptive per-parameter learning rates"],
                   ["Class imbalance", "weighted cross-entropy",
                    "deduplication left Restaurants with 3,110 rows against "
                    "Rent's 874, and macro-F1 penalises favouring the larger "
                    "categories"],
                   ["Stopping", f"early stopping, patience {arch['early_stopping_patience']}",
                    "training halts when a held-out slice stops improving and "
                    "the best weights are restored"]]),
        ("h2", "1.2 Why early stopping does not use the validation set"),
        ("p", "Every other model in this comparison is scored on validation "
              "without having seen it. Stopping on validation would let this one "
              "model tune against the very set used to compare them all, which "
              "would quietly advantage it. A "
              f"{arch['early_stop_slice']:.0%} slice of the **training** data is "
              "therefore held back for that purpose, and validation stays "
              "untouched."),
        ("p", f"Training ran {d['epochs_run']} epochs in {d['train_seconds']:.1f} "
              f"seconds, with the best weights found at epoch {d['best_epoch']}."),
        ("figure", (fig["model6_training_curve.png"],
                    "Figure 1. Training and held-out loss by epoch. The dashed "
                    "line marks the epoch whose weights were restored.")),
        ("h1", "2. Results"),
        ("table", [["Protocol", "n", "Accuracy", "Macro-F1", "ECE"]] +
                  [[label, f"{r[k]['n']:,}", f"{r[k]['accuracy']:.4f}",
                    f"{r[k]['macro_f1']:.4f}", f"{c['ece']:.4f}"]
                   for label, k, c in zip(pretty, keys, cal.values())]),
        ("figure", (fig["model6_protocols.png"],
                    "Figure 2. Accuracy and macro-F1 across the three protocols.")),
        ("figure", (fig["model6_per_class_f1.png"],
                    "Figure 3. F1 per category, merchant held out.")),
        ("figure", (fig["model6_confusion.png"],
                    "Figure 4. Confusion matrix, merchant held out.")),
        ("figure", (fig["model6_reliability.png"],
                    "Figure 5. Reliability diagram, merchant held out.")),
        ("h1", "3. Does depth help?"),
        ("table", [["Protocol", "Logistic head (Model 5)",
                    "Neural network head (Model 6)", "Difference"]] +
                  [[label, f"{prev['results'][k]['macro_f1']:.4f}",
                    f"{r[k]['macro_f1']:.4f}", f"{deltas[k]:+.4f}"]
                   for label, k in zip(pretty, keys)]),
        ("figure", (fig["model6_depth_effect.png"],
                    "Figure 6. The same frozen embeddings with a linear head "
                    "against a neural network head.")),
        ("p", ("**Yes, modestly.** Added capacity improves on the linear head, "
               "which indicates the embedding space is not perfectly linearly "
               "separable and that a non-linear decision boundary recovers some "
               "of what a linear one misses."
               if helps else
               "**No, or not materially.** Added capacity does not improve on "
               "the linear head. This is informative rather than disappointing: "
               "it indicates the limit is not the classifier's expressive power "
               "but the information present in the representation. Three linear "
               "classifiers over TF-IDF already agreed to within a few points, "
               "and a non-linear classifier over embeddings reaches much the "
               "same place. What changed performance in this project was "
               "changing the representation, not deepening the model."),),
        ("p", "The practical reading is that effort is better spent on what the "
              "model is shown than on how many layers process it. A richer "
              "description - a merchant category code, the amount, the account "
              "type - would supply information no architecture can invent from "
              "the text alone."),
        ("h1", "4. Limitations"),
        ("bullet", "The encoder remains frozen. Fine-tuning it would adapt the "
                   "representation itself rather than only the classifier on "
                   "top, which is the more promising direction and the one this "
                   "model does not test."),
        ("bullet", "A neural network over embeddings cannot produce token-level "
                   "explanations. Models 3 and 4 can point at the words behind a "
                   "decision; this one cannot without a separate post-hoc method, "
                   "which matters for the application's explainability "
                   "requirement."),
        ("bullet", "It introduces hyperparameters - widths, dropout rate, "
                   "learning rate, patience - that the simpler models do not "
                   "have. These were set to standard values rather than searched, "
                   "so a tuned network might do somewhat better than reported "
                   "here."),
        ("h1", "5. For the report"),
        ("p", f"A feedforward network of {' and '.join(str(h) for h in arch['hidden'])} "
              "hidden units over frozen MiniLM embeddings, trained with dropout, "
              "weighted cross-entropy, Adam and early stopping, reaches a "
              f"macro-F1 of {r[keys[0]]['macro_f1']:.4f} on familiar merchants, "
              f"{r[keys[1]]['macro_f1']:.4f} with merchants held out and "
              f"{r[keys[2]]['macro_f1']:.4f} on strictly novel merchants. "
              + ("Depth therefore contributes a measurable improvement over a "
                 "linear head on the same representation."
                 if helps else
                 "Compared with a linear head on the same representation the "
                 "difference is marginal, which locates the ceiling of this task "
                 "in the information carried by a transaction description rather "
                 "than in the capacity of the classifier applied to it.")),
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
