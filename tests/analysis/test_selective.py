"""Selective metrics against hand-computed values."""

import numpy as np

from metanoia_analysis.selective import auroc, coverage_at_precision, tertiles


def test_auroc_perfect_inverted_and_tied():
    conf = np.array([0.9, 0.8, 0.4, 0.3])
    assert auroc(conf, np.array([1.0, 1.0, 0.0, 0.0])) == 1.0
    assert auroc(conf, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    assert auroc(np.full(4, 0.5), np.array([1.0, 0.0, 1.0, 0.0])) == 0.5
    assert auroc(conf, np.ones(4)) is None


def test_auroc_hand_computed():
    # correct: 0.9, 0.5; wrong: 0.7, 0.3 -> pairs (0.9>0.7, 0.9>0.3, 0.5<0.7, 0.5>0.3) = 3/4
    conf = np.array([0.9, 0.7, 0.5, 0.3])
    assert auroc(conf, np.array([1.0, 0.0, 1.0, 0.0])) == 0.75


def test_coverage_at_precision():
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 1, 0, 1, 0], dtype=float)
    # top 3 are all right (precision 1.0); top 5 have 4/5 = 0.8; top 6 have 4/6
    full = coverage_at_precision(conf, correct, 1.0)
    assert full["coverage"] == 0.5 and full["precision"] == 1.0
    eighty = coverage_at_precision(conf, correct, 0.8)
    assert eighty["coverage"] == 5 / 6 and eighty["threshold"] == 0.7
    assert coverage_at_precision(conf, np.zeros(6), 0.9)["coverage"] == 0.0


def test_transfer_threshold_comes_from_the_reference_not_the_evaluated_set():
    ref = (np.array([0.9, 0.8, 0.7]), np.array([1.0, 1.0, 0.0]))
    conf, correct = np.array([0.95, 0.85, 0.75]), np.array([1.0, 0.0, 0.0])
    got = coverage_at_precision(conf, correct, 1.0, ref)
    assert got["threshold"] == 0.8  # reference: top two are all right
    assert got["coverage"] == 2 / 3 and got["precision"] == 0.5


def test_ties_are_not_split_by_a_threshold():
    conf = np.array([0.9, 0.9, 0.5])
    correct = np.array([1.0, 0.0, 1.0])
    assert coverage_at_precision(conf, correct, 1.0)["coverage"] == 0.0


def test_tertiles_use_given_edges():
    conf = np.array([0.1, 0.2, 0.5, 0.6, 0.9, 0.95])
    correct = np.array([0, 0, 1, 0, 1, 1], dtype=float)
    parts = tertiles(conf, correct, edges=(0.3, 0.7))
    assert [p["share"] for p in parts] == [2 / 6, 2 / 6, 2 / 6]
    assert [p["accuracy"] for p in parts] == [0.0, 0.5, 1.0]
