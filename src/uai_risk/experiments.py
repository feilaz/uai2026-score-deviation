"""Three numerical-data panels, using frozen splits and the original hyperparameters."""

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

from .core import score, risk_curves, calibrate, evaluate
from .data import load_data, load_splits

ALPHAS = (0.02, 0.03, 0.04, 0.05)


def _splits(tuples, methods, modes, n_cal, n_splits, seed_base, dataset, orders, center=None,
            n_thresholds=200):
    # A scoring rule is evaluated once, then independently calibrated on each
    # split. Repeated random splits overlap; they are not disjoint k-fold CV.
    for method, rule, gamma in methods:
        curves = risk_curves(tuples, [score(t, rule, gamma, center) for t in tuples],
                             np.linspace(0, 1, n_thresholds)[::-1])
        for split in range(n_splits):
            seed = seed_base + split
            order = np.array(orders[f"{len(tuples)}:{seed}"])
            cal, test = order[:n_cal], order[n_cal:]
            # The same partition is shared by all methods/modes for comparison.
            for alpha in ALPHAS:
                for mode in modes:
                    fit = calibrate(curves, cal, alpha, mode=mode)
                    yield {
                        "dataset": dataset, "method": method, "mode": mode, "alpha": alpha,
                        "split": split, "seed": seed, "gamma": gamma,
                        "n_thresholds": n_thresholds,
                        "calibration_tuples": len(cal), "test_tuples": len(test),
                        "threshold": fit.threshold, "calibration_risk": fit.risk,
                        "calibration_p": fit.p, "grid_passed": fit.grid_passed,
                        "reason": fit.reason,
                        "score_scope": "whole_corpus_center" if rule == "ushape" else "within_tuple",
                        **evaluate(curves, test, fit),
                    }


def run_panel(panel: str, data_dir: Path, include_legacy: bool = False,
              n_thresholds: int = 200) -> list[dict]:
    # The final original experiment scripts use 200 candidates. Expose 100 too,
    # as specified by the article, and record the choice in every result row.
    if isinstance(n_thresholds, bool) or not isinstance(n_thresholds, int) or n_thresholds < 2:
        raise ValueError("n_thresholds must be an integer of at least 2")
    modes = ("corrected", "legacy_reported") if include_legacy else ("corrected",)
    main = load_data(data_dir, "homogeneous")
    orders = load_splits(data_dir)
    if panel == "main":
        # Historical baseline only: this centre also sees held-out features.
        center = float(np.median(np.concatenate([t.entropy for t in main])))
        methods = [("Naive", "naive", 0.0), ("Entropy", "entropy", 1.0),
                   ("U-Shape", "ushape", 1.0), ("Score Deviation", "sd", 0.5)]
        return list(_splits(main, methods, modes, 200, 20, 42, "homogeneous_main", orders, center,
                            n_thresholds))
    if panel == "gamma":
        # A sensitivity sweep, not automatic tuning: each gamma gets its own fit.
        methods = [(f"gamma={g}", "sd", g) for g in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]
        return list(_splits(main, methods, modes, 200, 20, 1000, "homogeneous_gamma", orders,
                            n_thresholds=n_thresholds))
    if panel == "heterogeneous":
        # Match by query ID before using the same 70/30 split for both ensembles.
        # Matching questions does not imply matching extracted claims.
        het = load_data(data_dir, "heterogeneous")
        ids = {t.query_id for t in het}
        matched = [t for t in main if t.query_id in ids]
        if len(het) != 100 or [t.query_id for t in matched] != [t.query_id for t in het]:
            raise ValueError("The paired comparison requires 100 matching tuple IDs")
        methods = [("Score Deviation", "sd", 0.5)]
        return [row for name, dataset in (("homogeneous_matched", matched), ("heterogeneous", het))
                for row in _splits(dataset, methods, modes, 70, 50, 5000, name, orders,
                                    n_thresholds=n_thresholds)]
    raise ValueError(f"Unknown panel: {panel}")


def summarize(rows: list[dict]) -> list[dict]:
    # Average split metrics with equal split weights; do not pool all folds into
    # a new dataset or interpret repeated-split SD as a confidence interval.
    groups = defaultdict(list)
    keys = ("dataset", "method", "mode", "alpha", "n_thresholds")
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summaries = []
    for key, values in sorted(groups.items()):
        thresholds = [v["threshold"] for v in values if v["threshold"] is not None]
        result = dict(zip(keys, key))
        result.update(
            n_splits=len(values), gamma=values[0]["gamma"],
            calibration_tuples=values[0]["calibration_tuples"], test_tuples=values[0]["test_tuples"],
            score_scope=values[0]["score_scope"],
            abstain_splits=sum(v["reason"] == "abstain" for v in values),
            grid_failure_splits=sum(not v["grid_passed"] for v in values),
            threshold_one_splits=sum(v["threshold"] == 1.0 for v in values),
            # Empty-set fallbacks have no numerical threshold or p-value.
            threshold_mean=float(np.mean(thresholds)) if thresholds else None,
        )
        for metric in ("recall", "tuple_fdr", "fact_fdr", "selected"):
            result[f"{metric}_mean"] = float(np.mean([v[metric] for v in values]))
        result["recall_sd"] = float(np.std([v["recall"] for v in values]))  # ddof=0; not a confidence interval
        summaries.append(result)
    return summaries


def write_csv(path: Path, rows: list[dict]):
    # Callers create the directory. None becomes an empty CSV cell (abstention).
    if not rows:
        raise ValueError("Cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
