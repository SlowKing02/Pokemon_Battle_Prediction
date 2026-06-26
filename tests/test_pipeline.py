# Smoke tests for feature width and holdout split.

from __future__ import annotations

import unittest

from src.features import (
    build_combat_frame,
    build_matchup_features,
    feature_column_names,
    labeled_combat_frame,
    train_test_split_holdout,
)


class TestFeatures(unittest.TestCase):
    def test_labeled_row_count(self) -> None:
        self.assertEqual(len(labeled_combat_frame()), 50_000)

    def test_holdout_split(self) -> None:
        frame = labeled_combat_frame()
        x_train, y_train, x_test, y_test = train_test_split_holdout(frame, test_size=0.15)
        self.assertEqual(len(x_train), 42_500)
        self.assertEqual(len(x_test), 7_500)
        self.assertEqual(x_train.shape[1], x_test.shape[1])

    def test_matchup_width(self) -> None:
        n = len(feature_column_names())
        self.assertEqual(build_matchup_features(163, 7).shape[1], n)
        self.assertEqual(labeled_combat_frame().shape[1] - 1, n)

    def test_feature_count(self) -> None:
        self.assertEqual(len(feature_column_names()), 70)

    def test_kaggle_holdout_unlabeled(self) -> None:
        self.assertEqual(build_combat_frame().query("Test_Set == 1").shape[0], 2_080)


if __name__ == "__main__":
    unittest.main()
