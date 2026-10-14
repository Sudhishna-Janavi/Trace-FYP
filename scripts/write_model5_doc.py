#!/usr/bin/env python3
"""Write the Model 5 findings document from the stored results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reporting import FIGURES, write_doc  # noqa: E402


def main() -> int:
    d = json.loads((ROOT / "results" / "model5_minilm.json").read_text())
    others = {k: json.loads((ROOT / "results" / f"{f}.json").read_text())
              for k, f in (("nb", "model2_naive_bayes"),
                           ("lr", "model3_logistic_regression"),
                           ("svm", "model4_linear_svm"))}

    keys = ["stratified (validation)", "merchant held out (validation)",
            "  of which strictly novel"]
    pretty = ["Stratified (validation)", "Merchant held out (validation)",
              "of which strictly novel"]
    r, light, cal = d["results"], d["results_light_text"], d["calibration"]
    lr, svm, nb = others["lr"], others["svm"], others["nb"]

    def f1(source, key):
        return source["results"][key]["macro_f1"]

    best_tfidf = {k: max(f1(lr, k), f1(svm, k), f1(nb, k)) for k in keys}
    wins = {k: f1(d, k) > best_tfidf[k] for k in keys}
    beats_tfidf = all(wins.values())
    loses_everywhere = not any(wins.values())
    # the interesting case: loses where merchants recur, wins where they do not
    crossover = (not wins[keys[0]]) and wins[keys[1]] and wins[keys[2]]
    light_helps = any(light[k]["macro_f1"] - r[k]["macro_f1"] > 0.01 for k in keys)
    novel_gap = f1(d, keys[2]) - f1(lr, keys[2])

    fig = {name: FIGURES / name for name in d["figures"]}

    write_doc(
        "model5_minilm.docx",
        "Model 5 of 5 - MiniLM Sentence Embeddings with a Logistic Regression Head",
        "Unit 2, model comparison. The only model that represents meaning rather "
        "than counting surface patterns.",
        [
        ("h1", "1. What this model is"),
        ("p", "all-MiniLM-L6-v2 is a small transformer, pre-trained by others on "
              "roughly a billion sentence pairs, which maps any text to "
              f"{d['embedding_dim']} numbers arranged so that texts with similar "
              "meaning land near one another. Unlike TF-IDF it does not count "
              "words; it places the whole description in a learned semantic space."),
        ("h2", "1.1 What is actually trained here"),
        ("p", "**The encoder is downloaded and frozen.** No neural network is "
              "trained by this project. Only a logistic regression head is "
              f"fitted on top of the {d['embedding_dim']} numbers the encoder "
              f"produces, taking {d['head_train_seconds']:.2f} seconds. That head "
              "is identical to Model 3, so the comparison between Models 3 and 5 "
              "isolates the representation: counted surface patterns against "
              "learned semantics."),
        ("h2", "1.2 The hypothesis under test"),
        ("p", "Bank descriptions are codes rather than prose: TST*GOLDEN DRAGON, "
              "PPD ID, PYPL*365 MARKET. A semantic encoder should in principle "
              "recognise that an unfamiliar BISTRO relates to a familiar "
              "RESTAURANT, which is exactly the weakness every previous model "
              "displayed once merchants were held out. Whether that advantage "
              "survives on text this far from natural language is the question "
              "this model answers."),
        ("h1", "2. Results"),
        ("table", [["Protocol", "n", "Accuracy", "Macro-F1", "ECE"]] +
                  [[label, f"{r[k]['n']:,}", f"{r[k]['accuracy']:.4f}",
                    f"{r[k]['macro_f1']:.4f}", f"{c['ece']:.4f}"]
                   for label, k, c in zip(pretty, keys, cal.values())]),
        ("figure", (fig["model5_protocols.png"],
                    "Figure 1. Accuracy and macro-F1 across the three protocols.")),
        ("figure", (fig["model5_per_class_f1.png"],
                    "Figure 2. F1 per category, merchant held out.")),
        ("figure", (fig["model5_confusion.png"],
                    "Figure 3. Confusion matrix, merchant held out.")),
        ("figure", (fig["model5_reliability.png"],
                    "Figure 4. Reliability diagram, merchant held out.")),
        ("h1", "3. Did semantics beat surface patterns?"),
        ("table", [["Protocol", "TF-IDF + Logistic regression", "TF-IDF + Linear SVM",
                    "MiniLM + Logistic regression"]] +
                  [[label, f"{f1(lr, k):.4f}", f"{f1(svm, k):.4f}", f"{f1(d, k):.4f}"]
                   for label, k in zip(pretty, keys)]),
        ("figure", (fig["model5_vs_tfidf.png"],
                    "Figure 5. MiniLM against the two strongest TF-IDF models on "
                    "all three protocols.")),
        ("p", ("**It depends entirely on whether the merchant is familiar, and "
               "that is the finding.** MiniLM is the weakest model of the five "
               f"where merchants recur ({f1(d, keys[0]):.4f} against "
               f"{best_tfidf[keys[0]]:.4f} for the best TF-IDF model) and the "
               f"strongest where they do not ({f1(d, keys[2]):.4f} against "
               f"{best_tfidf[keys[2]]:.4f} on strictly novel merchants, a margin "
               f"of {f1(d, keys[2]) - best_tfidf[keys[2]]:+.4f})."
               if crossover else
               "**" + ("Yes." if beats_tfidf else "No.") + "** MiniLM "
               + ("outperforms" if beats_tfidf else "does not outperform")
               + " the TF-IDF models across the protocols tested. On the strictly "
                 "novel subset it differs from TF-IDF with logistic regression by "
               + f"{novel_gap:+.4f} macro-F1."),),
        ("p", "The explanation lies in what each representation can exploit. "
              "TF-IDF wins on familiar merchants because a brand name is a "
              "perfectly discriminative token: having seen SAFEWAY labelled "
              "Groceries, recognising it again is trivial, and no semantic "
              "understanding is required or useful. That advantage disappears "
              "entirely the moment the merchant is one the model has never seen, "
              "because the token simply is not in its vocabulary."),
        ("p", "A sentence encoder has no such cliff. It places an unfamiliar "
              "merchant somewhere in a learned semantic space on the basis of "
              "whatever the description does contain - word fragments, format "
              "cues, the shape of the name - and that placement is still "
              "informative when the exact string is new. The two representations "
              "are therefore strongest in opposite conditions, which is a more "
              "useful result for this project than either one winning outright."),
        ("p", "**This bears directly on deployment.** A finance application meets "
              "unfamiliar merchants constantly: every new user arrives with a set "
              "of shops the system has never encountered. The protocol that "
              "matters most for that situation is the one MiniLM leads."),
        ("h1", "4. Would MiniLM do better on more natural text?"),
        ("p", "Every model so far has received the shared normalised text, which "
              "is uppercase with long digit runs replaced by <NUM>. That suits "
              "TF-IDF but is unlike the text MiniLM was trained on, so the "
              "encoder was run a second time on lightly cleaned text that keeps "
              "natural case and punctuation. This checks that the comparison is "
              "not quietly handicapping the one model it could disadvantage."),
        ("table", [["Protocol", "Shared normalised text", "Lightly cleaned text",
                    "Difference"]] +
                  [[label, f"{r[k]['macro_f1']:.4f}", f"{light[k]['macro_f1']:.4f}",
                    f"{light[k]['macro_f1'] - r[k]['macro_f1']:+.4f}"]
                   for label, k in zip(pretty, keys)]),
        ("figure", (fig["model5_text_variant.png"],
                    "Figure 6. The same encoder on the shared normalised text "
                    "against lightly cleaned text.")),
        ("p", ("Lighter cleaning **does** materially help, which means the shared "
               "normalisation was costing this model real performance and the "
               "result should be reported with that caveat attached."
               if light_helps else
               "Lighter cleaning does **not** materially help. The shared "
               "normalisation is therefore not the reason for MiniLM's standing "
               "in this comparison, and the single shared representation used "
               "across all five models is defensible."),),
        ("h1", "5. Limitations"),
        ("bullet", "The encoder is frozen. Fine-tuning it on this data would "
                   "likely improve matters, at a cost in training time, "
                   "hardware and explainability that this project did not take on."),
        ("bullet", "An embedding is 384 anonymous numbers, so there is no "
                   "token-level explanation to show a user. Models 3 and 4 can "
                   "point at the words that drove a decision; this one cannot "
                   "without a separate post-hoc method."),
        ("bullet", "Encoding is the dominant cost at prediction time, far above "
                   "a TF-IDF transform, which matters for an interactive "
                   "application."),
        ("h1", "6. For the report"),
        ("p", "MiniLM sentence embeddings with a logistic regression head reach a "
              f"macro-F1 of {f1(d, keys[0]):.4f} on familiar merchants, "
              f"{f1(d, keys[1]):.4f} with merchants held out and "
              f"{f1(d, keys[2]):.4f} on strictly novel merchants. "
              + ("The comparison with TF-IDF reverses depending on the protocol: "
                 "counted surface patterns win while merchants recur between "
                 "training and evaluation, and learned semantics win once they do "
                 "not. Reporting a single winner would therefore misrepresent the "
                 "result. Which model is preferable depends on whether the "
                 "deployed system expects to meet merchants it has already seen, "
                 "and a consumer finance application plainly does not."
                 if crossover else
                 "This exceeds the TF-IDF models and supports the view that "
                 "semantic representation helps on this task."
                 if beats_tfidf else
                 "A frozen general-purpose sentence encoder therefore does not "
                 "outperform TF-IDF on bank transaction descriptions, because "
                 "these descriptions are machine-generated codes rather than "
                 "natural language.")),
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
