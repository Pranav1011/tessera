"""Slice-based evaluation harness (the highest-value phase).

Aggregate mIoU is what everyone reports; *slices* are what an eval engineer reports.
This computes overall per-class IoU + macro-F1, then the same metrics sliced by
part complexity (face count), topology (genus), and primitive mix — so a regression
that hides in the average shows up in a slice. Multi-seed runs give confidence
intervals (real regression vs noise), and `regression_gate` is the CI merge-blocker.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from tessera.dataset import PRIMITIVES, build_dataset
from tessera.model import predict, train


def per_class_iou(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    ious = {}
    for c in range(n_classes):
        t, p = y_true == c, y_pred == c
        inter = int((t & p).sum())
        union = int((t | p).sum())
        ious[PRIMITIVES[c]] = (inter / union) if union else float("nan")
    valid = [v for v in ious.values() if not np.isnan(v)]
    ious["mIoU"] = float(np.mean(valid)) if valid else 0.0
    return ious


def _fc_bucket(fc: int) -> str:
    return "small(<200)" if fc < 200 else ("medium(200-1000)" if fc < 1000 else "large(>=1000)")


def evaluate(y_true, y_pred, meta, n_classes: int) -> dict:
    overall = {
        "mIoU": per_class_iou(y_true, y_pred, n_classes)["mIoU"],
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "n": len(y_true),
    }
    slices: dict[str, dict] = {"by_face_count": {}, "by_genus": {}, "by_primitive": {}}
    fc = np.array([_fc_bucket(x) for x in meta["face_count"]])
    for b in sorted(set(fc)):
        m = fc == b
        slices["by_face_count"][b] = per_class_iou(y_true[m], y_pred[m], n_classes)["mIoU"]
    for g in sorted(set(meta["genus"].tolist())):
        m = meta["genus"] == g
        slices["by_genus"][f"genus_{g}"] = per_class_iou(y_true[m], y_pred[m], n_classes)["mIoU"]
    for c in range(n_classes):
        m = meta["primitive"] == c
        if m.any():
            iou = per_class_iou(y_true[m], y_pred[m], n_classes)["mIoU"]
            slices["by_primitive"][PRIMITIVES[c]] = iou
    return {"overall": overall, "slices": slices}


def run_once(seed: int, n_per: int = 12) -> dict:
    X, y, groups, meta = build_dataset(n_per=n_per, seed=seed)
    n = len(PRIMITIVES)
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, groups))
    clf = train(X[tr], y[tr], n_classes=n, seed=seed)
    pred = predict(clf, X[te])
    te_meta = {k: v[te] for k, v in meta.items()}
    return evaluate(y[te], pred, te_meta, n)


def run_multiseed(seeds=(0, 1, 2), n_per: int = 12) -> dict:
    runs = [run_once(s, n_per) for s in seeds]
    mious = [r["overall"]["mIoU"] for r in runs]
    return {
        "seeds": list(seeds),
        "mIoU_mean": round(float(np.mean(mious)), 4),
        "mIoU_std": round(float(np.std(mious)), 4),
        "mIoU_ci95": round(float(1.96 * np.std(mious) / np.sqrt(len(mious))), 4),
        "per_seed": runs,
    }


def regression_gate(current: dict, baseline: dict, tol: float = 0.02) -> list[str]:
    """Return the list of slices that regressed beyond `tol` vs baseline (empty = OK).
    This is what a CI job runs to block a merge."""
    fails = []
    for group in ("by_face_count", "by_genus", "by_primitive"):
        cur, base = current["slices"][group], baseline["slices"][group]
        for k, bv in base.items():
            cv = cur.get(k)
            if cv is not None and not np.isnan(cv) and not np.isnan(bv) and cv < bv - tol:
                fails.append(f"{group}/{k}: {bv:.3f} -> {cv:.3f}")
    return fails
