"""Smoke tests for feature build and train/holdout split."""

from __future__ import annotations

import unittest

from src.features import (
    build_combat_frame,
    build_matchup_features,
    feature_column_names,
    labeled_combat_frame,
    train_test_split_holdout,
)


class PipelineSmokeTest(unittest.TestCase):
    def test_labeled_row_count(self) -> None:
        frame = labeled_combat_frame()
        self.assertEqual(len(frame), 50_000)

    def test_holdout_split(self) -> None:
        frame = labeled_combat_frame()
        x_train, y_train, x_test, y_test = train_test_split_holdout(frame, test_size=0.15)
        self.assertEqual(len(x_train) + len(x_test), 50_000)
        self.assertEqual(len(x_train), 42_500)
        self.assertEqual(len(x_test), 7_500)
        self.assertEqual(x_train.shape[1], x_test.shape[1])
        self.assertGreater(y_train.nunique(), 1)

    def test_matchup_feature_width(self) -> None:
        row = build_matchup_features(163, 7)
        expected = len(feature_column_names())
        self.assertEqual(row.shape[1], expected)
        frame = labeled_combat_frame()
        self.assertEqual(frame.shape[1] - 1, expected)

    def test_feature_count(self) -> None:
        self.assertEqual(len(feature_column_names()), 70)

    def test_unlabeled_kaggle_rows_flagged(self) -> None:
        full = build_combat_frame()
        self.assertEqual(full.query("Test_Set == 1").shape[0], 2_080)


if __name__ == "__main__":
    unittest.main()
