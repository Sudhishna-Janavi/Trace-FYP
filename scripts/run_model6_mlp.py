#!/usr/bin/env python3
"""Unit 2, Model 6 of 7: a feedforward neural network on MiniLM embeddings.

Reuses the embeddings cached by Model 5, so nothing is re-encoded.

Run:
    .venv/bin/python scripts/run_model6_mlp.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_model import (calibration, embedding_cache, load,  # noqa: E402
                            print_table, score, strictly_novel)
from reporting import (ACCENT, BAD, GOOD, MUTED, confusion_heatmap,  # noqa: E402
                       grouped_bars, per_class_bars, reliability_diagram, save,
                       write_doc)
from models.mlp_head import (BATCH_SIZE, DROPOUT, EARLY_STOP_FRACTION,  # noqa: E402
                             HIDDEN, LEARNING_RATE, MAX_EPOCHS, PATIENCE,
                             MLPClassifier, set_seed)

MODEL = "MiniLM embeddings + feedforward neural network"
EMB = embedding_cache()


def train(features, labels, classes, log=None):
    """Fit the network, early-stopping on a slice held back from training."""
    set_seed()
    n_classes = len(classes)
    index = {name: i for i, name in enumerate(classes)}
    y = np.array([index[v] for v in labels])

    # hold back a slice of TRAIN for early stopping; validation stays untouched
    rng = np.random.default_rng(1131)
    order = rng.permutation(len(y))
    cut = int(len(y) * (1 - EARLY_STOP_FRACTION))
    fit_rows, stop_rows = order[:cut], order[cut:]

    x_fit = torch.tensor(features[fit_rows], dtype=torch.float32)
    y_fit = torch.tensor(y[fit_rows], dtype=torch.long)
    x_stop = torch.tensor(features[stop_rows], dtype=torch.float32)
    y_stop = torch.tensor(y[stop_rows], dtype=torch.long)

    # class weights, because deduplication left the categories uneven
    counts = np.bincount(y[fit_rows], minlength=n_classes).astype(float)
    weights = torch.tensor(len(y_fit) / (n_classes * np.maximum(counts, 1)),
                           dtype=torch.float32)

    model = MLPClassifier(features.shape[1], n_classes)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(TensorDataset(x_fit, y_fit), batch_size=BATCH_SIZE,
                        shuffle=True)

    best_score, best_state, best_epoch, waited = -1.0, None, 0, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            optimiser.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch_x)
        train_loss = total / len(x_fit)

        model.eval()
        with torch.no_grad():
            logits = model(x_stop)
            stop_loss = float(criterion(logits, y_stop))
            predicted = logits.argmax(1).numpy()
        macro = float(f1_score(y[stop_rows], predicted, average="macro",
                               zero_division=0))
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "holdout_loss": stop_loss, "holdout_macro_f1": macro})

        if macro > best_score + 1e-5:
            best_score, best_epoch, waited = macro, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
        if log and epoch % 10 == 0:
            print(f"      epoch {epoch:>3}  train loss {train_loss:.4f}  "
                  f"holdout macro-F1 {macro:.4f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def predict(model, features):
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()


def main() -> int:
    results, calibrations = {}, {}

    s_train = load("stratified_train.csv")
    s_validation = load("stratified_validation.csv")
    classes = sorted({r["category"] for r in s_train})

    print(f"architecture: {384} -> {' -> '.join(str(h) for h in HIDDEN)} -> "
          f"{len(classes)}   dropout {DROPOUT}   Adam lr {LEARNING_RATE}")
    print(f"early stopping on a {EARLY_STOP_FRACTION:.0%} slice of train "
          f"(validation untouched), patience {PATIENCE}\n")

    x_train = np.load(EMB / "norm_strat_train.npy")
    x_validation = np.load(EMB / "norm_strat_validation.npy")
    started = time.perf_counter()
    model, history, best_epoch = train(
        x_train, [r["category"] for r in s_train], classes, log=True)
    train_seconds = time.perf_counter() - started
    print(f"   stopped at epoch {best_epoch} after {train_seconds:.1f}s "
          f"({len(history)} epochs run)")

    prob_s = predict(model, x_validation)
    predicted_s = [classes[i] for i in prob_s.argmax(1)]
    gold_s = [r["category"] for r in s_validation]
    results["stratified (validation)"] = score(gold_s, predicted_s)
    calibrations["stratified"] = calibration(gold_s, prob_s, classes)

    m_train = load("merchant_train.csv")
    m_validation = load("merchant_validation.csv")
    xm_train = np.load(EMB / "norm_merch_train.npy")
    xm_validation = np.load(EMB / "norm_merch_validation.npy")
    model_m, history_m, _best = train(
        xm_train, [r["category"] for r in m_train], classes)
    prob_m = predict(model_m, xm_validation)
    predicted_m = [classes[i] for i in prob_m.argmax(1)]
    gold_m = [r["category"] for r in m_validation]
    results["merchant held out (validation)"] = score(gold_m, predicted_m)
    calibrations["merchant"] = calibration(gold_m, prob_m, classes)

    novel = strictly_novel(m_train, m_validation)
    position = {r["transaction_id"]: i for i, r in enumerate(m_validation)}
    rows = [position[r["transaction_id"]] for r in novel]
    prob_n = prob_m[rows]
    predicted_n = [classes[i] for i in prob_n.argmax(1)]
    gold_n = [r["category"] for r in novel]
    results["  of which strictly novel"] = score(gold_n, predicted_n)
    calibrations["strictly_novel"] = calibration(gold_n, prob_n, classes)

    print_table(f"MODEL 6 of 7 - {MODEL.upper()}", results)

    print("\ncalibration")
    print(f"  {'protocol':<18}{'ECE':>9}{'mean conf':>11}{'accuracy':>10}{'gap':>9}")
    for name, c in calibrations.items():
        print(f"  {name:<18}{c['ece']:>9.4f}{c['mean_confidence']:>11.4f}"
              f"{c['accuracy']:>10.4f}{c['overconfidence']:>+9.4f}")

    # ---- does depth help? -------------------------------------------------
    previous = json.loads((ROOT / "results" / "model5_minilm.json").read_text())
    keys = list(results)
    print("\ndoes depth help? identical embeddings, different head")
    print(f"  {'protocol':<34}{'logistic':>10}{'MLP':>9}{'diff':>9}")
    for key in keys:
        a = previous["results"][key]["macro_f1"]
        b = results[key]["macro_f1"]
        print(f"  {key.strip():<34}{a:>10.4f}{b:>9.4f}{b - a:>+9.4f}")

    labels = classes
    f1_merchant = f1_score(gold_m, predicted_m, average=None, labels=labels,
                           zero_division=0)
    counts = [gold_s.count(label) for label in labels]

    # ---- figures ----------------------------------------------------------
    print("\nfigures")
    short = ["Stratified", "Merchant held out", "Strictly novel"]
    fig1 = grouped_bars(
        "Neural network on MiniLM embeddings: accuracy and macro-F1", "", short,
        [("Accuracy", [results[n]["accuracy"] for n in keys], ACCENT),
         ("Macro-F1", [results[n]["macro_f1"] for n in keys], GOOD)],
        "model6_protocols.png", percent=True)

    fig2 = per_class_bars(
        "Neural network: F1 per category (merchant held out)",
        labels, list(f1_merchant), "model6_per_class_f1.png", counts=counts)

    matrix = confusion_matrix(gold_m, predicted_m, labels=labels)
    fig3 = confusion_heatmap(
        "Neural network: what gets confused with what (merchant held out)",
        labels, matrix, "model6_confusion.png")

    fig4 = reliability_diagram(
        "Neural network: is its confidence trustworthy?",
        calibrations["merchant"]["bins"], "model6_reliability.png",
        ece=calibrations["merchant"]["ece"], subtitle="merchant held out.")

    fig5 = grouped_bars(
        "Does depth help? Same frozen embeddings, logistic head against a network",
        "macro-F1", short,
        [("Logistic head (Model 5)",
          [previous["results"][n]["macro_f1"] for n in keys], GOOD),
         ("Neural network head (Model 6)",
          [results[n]["macro_f1"] for n in keys], BAD)],
        "model6_depth_effect.png", percent=False)

    # training curve
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train_loss"] for h in history], color=ACCENT,
            linewidth=1.6, label="training loss")
    ax.plot(epochs, [h["holdout_loss"] for h in history], color=BAD,
            linewidth=1.6, label="held-out loss")
    ax.axvline(best_epoch, color=MUTED, linestyle="--", linewidth=1)
    ax.text(best_epoch, ax.get_ylim()[1] * 0.95, f"  best epoch {best_epoch}",
            fontsize=8.5, color=MUTED, va="top")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training curve: early stopping restores the best weights",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e1e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    fig6 = save(fig, "model6_training_curve.png")

    out = ROOT / "results" / "model6_mlp.json"
    out.write_text(json.dumps({
        "model": MODEL, "machine_learning": True,
        "architecture": {"input": 384, "hidden": list(HIDDEN),
                         "output": len(classes), "dropout": DROPOUT,
                         "optimiser": "Adam", "learning_rate": LEARNING_RATE,
                         "batch_size": BATCH_SIZE, "loss": "cross-entropy",
                         "early_stopping_patience": PATIENCE,
                         "early_stop_slice": EARLY_STOP_FRACTION},
        "train_seconds": round(train_seconds, 2),
        "best_epoch": best_epoch, "epochs_run": len(history),
        "results": results,
        "calibration": {k: {m: v for m, v in c.items() if m != "bins"}
                        for k, c in calibrations.items()},
        "training_history": history,
        "per_class_f1_merchant": {"labels": labels,
                                  "values": [float(v) for v in f1_merchant]},
        "figures": [f.name for f in (fig1, fig2, fig3, fig4, fig5, fig6)],
    }, indent=2), encoding="utf-8")
    print(f"\n[PASS] wrote {out}")
    print("[INFO] document is written by scripts/write_model6_doc.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
