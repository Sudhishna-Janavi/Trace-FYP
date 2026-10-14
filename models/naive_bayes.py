"""Model 2 of 5: multinomial Naive Bayes over TF-IDF features.

The first model here that learns. It counts how strongly each feature is
associated with each category and applies Bayes' rule, assuming features are
independent given the category. That assumption is plainly false for text -
MORTGAGE and PMT occur together constantly - which is why the method is called
naive. It nevertheless remains a standard strong baseline for text
classification, because the assumption affects the calibration of the
probabilities far more than the ranking of the classes.

It is included here as the cheap learned baseline: if a model that trains in a
fraction of a second does almost as well as logistic regression or an SVM, the
extra complexity of those models has to be justified by something other than
accuracy.
"""
from __future__ import annotations

from sklearn.naive_bayes import MultinomialNB


def build_classifier():
    # alpha is Laplace smoothing: it stops a feature never seen with a category
    # during training from forcing that category's probability to exactly zero.
    return MultinomialNB(alpha=0.1)
