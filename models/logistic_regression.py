"""Model 3 of 5: multinomial logistic regression over the same TF-IDF features.

This model matters to the project beyond its accuracy, for two reasons.

**It is linear, so it explains itself exactly.** The contribution of a feature
to a class score is simply its weight multiplied by its value. No post-hoc
approximation such as LIME or SHAP is required to say why a prediction was
made; the explanation is the arithmetic the model already performed.

**Its probabilities are meaningful.** Logistic regression is fitted by
maximising likelihood, which optimises the probabilities themselves rather than
only the decision boundary. The architecture routes low-confidence predictions
to human review, so a model whose confidence cannot be trusted would push wrong
answers through that gate unflagged. Calibration is therefore measured here, not
just accuracy.

class_weight="balanced" is used because deduplication left the categories uneven
(Restaurants 3,110 rows against Rent 874). Without it the model would favour the
larger categories, which macro-F1 penalises. Naive Bayes handles the same
imbalance through its class priors instead.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_classifier():
    return LogisticRegression(
        max_iter=2000,
        C=4.0,                      # light regularisation: the features are sparse
        class_weight="balanced",
    )
