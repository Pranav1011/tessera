"""Eval-harness metrics, slicing, multi-seed CIs, and the regression gate."""

from __future__ import annotations

import numpy as np

from tessera.eval_harness import (
    per_class_iou,
    regression_gate,
    run_multiseed,
    run_once,
)


def test_iou_perfect_is_one():
    y = np.array([0, 1, 2, 0, 1])
    assert per_class_iou(y, y, 3)["mIoU"] == 1.0


def test_iou_known_value():
    yt = np.array([0, 0, 1, 1])
    yp = np.array([0, 1, 1, 1])  # class0: inter1/union2=.5 ; class1: inter2/union3=.667
    r = per_class_iou(yt, yp, 2)
    assert abs(r["plane"] - 0.5) < 1e-6
    assert abs(r["cylinder"] - 2 / 3) < 1e-6


def test_evaluate_has_all_slices():
    r = run_once(seed=0, n_per=6)
    assert set(r["slices"]) == {"by_face_count", "by_genus", "by_primitive"}
    assert 0.0 <= r["overall"]["mIoU"] <= 1.0
    assert r["slices"]["by_primitive"]  # per-primitive slices present


def test_multiseed_reports_ci():
    r = run_multiseed(seeds=(0, 1), n_per=6)
    assert "mIoU_mean" in r and "mIoU_std" in r and "mIoU_ci95" in r
    assert len(r["per_seed"]) == 2


def test_regression_gate():
    base = {"slices": {"by_primitive": {"plane": 0.90}, "by_face_count": {}, "by_genus": {}}}
    worse = {"slices": {"by_primitive": {"plane": 0.70}, "by_face_count": {}, "by_genus": {}}}
    assert regression_gate(worse, base, tol=0.02)          # a real regression is caught
    assert regression_gate(base, base, tol=0.02) == []      # identical passes
    tiny = {"slices": {"by_primitive": {"plane": 0.895}, "by_face_count": {}, "by_genus": {}}}
    assert regression_gate(tiny, base, tol=0.02) == []      # within tolerance = noise, not blocked
