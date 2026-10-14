"""Shared TF-IDF feature extraction for Models 2, 3 and 4.

Those three models differ only in the classifier placed on top. Holding the
features fixed means the comparison measures the classifier rather than
accidental differences in text representation.

Two vectorisers are combined:

  word 1-2 grams   captures whole terms and short phrases: MORTGAGE,
                   OVERDRAFT FEE, DIR DEP.
  char_wb 3-4 grams
                   captures fragments inside words, so a merchant never seen
                   before can still be recognised from its shape. WHOLEFDS
                   shares the fragment "FDS" with FOODS, and TST appears as a
                   prefix on every Toast point-of-sale description.

Both are fitted on the TRAINING split only. Fitting on all the data would let
the vocabulary and the inverse document frequencies carry information from the
validation and test rows into training, which inflates every score that follows.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union


def build_vectoriser():
    return make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                        min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                        min_df=3, sublinear_tf=True, max_features=200_000),
    )
