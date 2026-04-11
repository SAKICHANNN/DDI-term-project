from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.tree import DecisionTreeClassifier


@dataclass
class GuidedTree:
    estimator: DecisionTreeClassifier
    feature_indices: np.ndarray


class GuidedRandomForest:
    """
    Lightweight guided-forest implementation.

    Each tree samples a weighted feature pool (without replacement),
    then trains a CART tree with standard split randomness inside that pool.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        tree_feature_pool_size: int | None = None,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.tree_feature_pool_size = tree_feature_pool_size
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_: List[GuidedTree] = []
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, feature_probabilities: np.ndarray) -> "GuidedRandomForest":
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        rng = np.random.default_rng(self.random_state)

        pool_size = self.tree_feature_pool_size or int(np.sqrt(n_features))
        pool_size = max(2, min(pool_size, n_features))

        probs = feature_probabilities.astype(float)
        probs = probs / probs.sum()

        self.trees_ = []
        for tree_idx in range(self.n_estimators):
            if self.bootstrap:
                row_idx = rng.integers(0, n_samples, size=n_samples)
            else:
                row_idx = np.arange(n_samples)

            feature_idx = rng.choice(n_features, size=pool_size, replace=False, p=probs)
            X_tree = X[row_idx][:, feature_idx]
            y_tree = y[row_idx]

            tree = DecisionTreeClassifier(
                criterion="gini",
                splitter="best",
                max_features="sqrt",
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + tree_idx,
            )
            tree.fit(X_tree, y_tree)
            self.trees_.append(GuidedTree(estimator=tree, feature_indices=feature_idx))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        all_probs = []
        for tree in self.trees_:
            p = tree.estimator.predict_proba(X[:, tree.feature_indices])
            # Ensure two-column output [p0, p1]
            if p.shape[1] == 1:
                only = tree.estimator.classes_[0]
                if int(only) == 1:
                    p = np.column_stack([1.0 - p[:, 0], p[:, 0]])
                else:
                    p = np.column_stack([p[:, 0], 1.0 - p[:, 0]])
            all_probs.append(p)
        return np.mean(np.stack(all_probs, axis=0), axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
