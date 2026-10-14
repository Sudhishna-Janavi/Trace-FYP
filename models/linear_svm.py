"""Model 4 of 5: a calibrated linear support vector machine over TF-IDF features.

A linear SVM finds the separating boundary with the widest margin between
classes. On sparse, high-dimensional text it is usually the strongest of the
classical linear models, which makes it the natural test of how much accuracy
is available from this representation at all.

It has one property that matters here: **it produces distances, not
probabilities.** LinearSVC returns how far a point sits from each boundary, and
those numbers have no probabilistic meaning - a distance of 2.0 does not mean
twice as likely. This project routes low-confidence predictions to human review,
so a raw distance is unusable as a confidence score.

CalibratedClassifierCV wraps the SVM and learns the mapping from distance to
probability, fitting a sigmoid on held-out folds of the training data (Platt
scaling). This is an extra model fitted on top of the first, and whether it
produces trustworthy confidence is an empirical question that this model's
evaluation answers rather than assumes.

The cost is time: with cv=3 the SVM is fitted three times over, plus the
calibrator.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


def build_classifier():
    return CalibratedClassifierCV(
        LinearSVC(C=0.5, class_weight="balanced"),
        method="sigmoid",   # Platt scaling: standard for SVM outputs
        cv=3,
    )
