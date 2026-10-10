"""Score Deviation and tuple-level Learn-Then-Test calibration.

Risk bounds assume i.i.d. tuples and scoring rules fixed before calibration.
Losses are relative to the supplied binary judge labels.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class ClaimTuple:
    # One statistical unit: all claims extracted from five answers to ONE query.
    # Entries at the same position in weak/entropy/labels refer to the same claim.
    query_id: str
    weak: np.ndarray
    entropy: np.ndarray
    labels: np.ndarray

    def __post_init__(self):
        weak, entropy, labels = (np.asarray(x) for x in (self.weak, self.entropy, self.labels))
        if not self.query_id or weak.ndim != 1 or not len(weak):
            raise ValueError("Each tuple needs an ID and at least one claim")
        if entropy.shape != weak.shape or labels.shape != weak.shape:
            raise ValueError("Scores, entropies and labels must have equal lengths")
        if not np.all(np.isfinite(weak)) or np.any((weak < 0) | (weak > 1)):
            raise ValueError("Weak scores must be finite and in [0, 1]")
        if not np.all(np.isfinite(entropy)) or np.any(entropy < 0):
            raise ValueError("Entropies must be finite and nonnegative")
        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError("Labels must be binary")
        for name, value, dtype in (("weak", weak, float), ("entropy", entropy, float),
                                   ("labels", labels, np.int8)):
            # Copy and freeze arrays so calibration cannot mutate a shared snapshot.
            array = np.array(value, dtype=dtype, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def score(t: ClaimTuple, method: str, gamma: float = 0.5,
          entropy_center: float | None = None) -> np.ndarray:
    """Higher scores are retained. Standard deviation uses ddof=0."""
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if method == "naive":
        return t.weak.copy()
    if method == "sd":
        # This is dispersion across different claims, not five judges voting on
        # one claim. The scalar denominator rescales every claim in this tuple.
        penalty = np.std(t.weak)
    elif method == "entropy":
        # Low response diversity gives a larger penalty; beta is fixed at 1.
        penalty = np.exp(-t.entropy)
    elif method == "ushape":
        if entropy_center is None or not math.isfinite(entropy_center):
            raise ValueError("U-Shape requires an explicit, finite entropy centre")
        # Matches appendix Table 8's absolute deviation, not main-text Eq. (5)'s
        # squared deviation. The experiment runner supplies the historical centre.
        penalty = np.abs(t.entropy - entropy_center)
    else:
        raise ValueError(f"Unknown method: {method}")
    return t.weak / (1 + gamma * penalty)


@dataclass(frozen=True)
class RiskCurves:
    # Count matrices avoid re-filtering claims for every seed, alpha and mode.
    grid: np.ndarray
    selected: np.ndarray  # [threshold, tuple]
    false: np.ndarray
    supported: np.ndarray  # [tuple]


def risk_curves(tuples: list[ClaimTuple], scores: list[np.ndarray],
                grid: np.ndarray | None = None) -> RiskCurves:
    """Precompute counts, without changing the fixed descending test order."""
    # 200 candidates are the repository default; the paper specifies 100.
    grid = np.linspace(0, 1, 200)[::-1] if grid is None else np.asarray(grid, dtype=float)
    if (grid.ndim != 1 or not len(grid) or not np.all(np.isfinite(grid))
            or np.any((grid < 0) | (grid > 1)) or np.any(np.diff(grid) >= 0)
            or grid[0] != 1):
        raise ValueError("Grid must start at 1 and strictly decrease within [0, 1]")
    if not tuples or len(tuples) != len(scores):
        raise ValueError("Provide one score array per tuple")
    selected, false = [], []
    for t, s in zip(tuples, scores):
        s = np.asarray(s)
        if s.shape != t.weak.shape or not np.all(np.isfinite(s)) or np.any((s < 0) | (s > 1)):
            raise ValueError("Scores must match the claims and lie in [0, 1]")
        # Broadcasting gives [threshold, claim]. Equality is retained, so a
        # threshold of 1 does NOT mean abstention when some scores equal 1.
        keep = s[None, :] >= grid[:, None]
        selected.append(keep.sum(axis=1))
        false.append((keep & (t.labels == 0)).sum(axis=1))
    return RiskCurves(grid.copy(), np.array(selected).T, np.array(false).T,
                      np.array([int(t.labels.sum()) for t in tuples]))


def _indices(indices, size):
    # Indices address whole tuples, never individual claims. Calibration and test
    # disjointness is established by experiments._splits, not this helper.
    result = np.asarray(indices)
    if (result.ndim != 1 or not len(result) or result.dtype.kind not in "iu"
            or np.any(result < 0) or np.any(result >= size)
            or len(np.unique(result)) != len(result)):
        raise ValueError("Indices must be nonempty, unique, valid integer positions")
    return result


@lru_cache(maxsize=4096)
def _bentkus(n: int, alpha: float, cutoff: int) -> float:
    # Lower binomial tail multiplied by Euler's e; reused across many thresholds.
    return float(math.e * binom.cdf(cutoff, n, alpha))


def p_value(false: np.ndarray, selected: np.ndarray, alpha: float,
            mode: str = "corrected") -> tuple[float, float]:
    """Return (p, mean tuple FDP). Corrected rounding uses exact rational sums."""
    false, selected = np.asarray(false), np.asarray(selected)
    if (false.ndim != 1 or not len(false) or false.shape != selected.shape
            or false.dtype.kind not in "iu" or selected.dtype.kind not in "iu"
            or np.any(false < 0) or np.any(selected < false)):
        raise ValueError("Provide nonempty integer counts with 0 <= false <= selected")
    if not 0 < alpha < 1 or mode not in ("corrected", "legacy_reported"):
        raise ValueError("Require 0 < alpha < 1 and a known mode")
    n = len(false)
    if mode == "corrected":
        # The loss is the mean of per-tuple fractions, not total false / total
        # retained. Empty selections contribute zero. Exact fractions prevent a
        # floating-point rounding error from moving the Bentkus integer cutoff.
        total = sum((Fraction(int(f), int(d)) for f, d in zip(false, selected) if f), Fraction())
        risk = float(total / n)
        cutoff = -(-total.numerator // total.denominator)
    else:
        # Historical code path: floor rounding and quadratic Hoeffding below.
        # This is not a literal implementation of the paper's Eqs. (7)-(9).
        risk = float(np.mean(false / np.maximum(1, selected)))
        cutoff = math.floor(n * risk)
    if risk >= alpha:
        # Calibration loss at/above the target cannot certify this candidate.
        return 1.0, risk
    if mode == "corrected":
        # Bernoulli KL divergence D(risk || alpha), with 0*log(0) treated as 0.
        kl = ((risk * math.log(risk / alpha) if risk else 0.0)
              + (1 - risk) * math.log((1 - risk) / (1 - alpha)))
        hoeffding = math.exp(-n * kl)
    else:
        hoeffding = math.exp(-2 * n * (alpha - risk) ** 2)
    # Corrected mode follows the HB concentration bound, not a generic minimum
    # of arbitrary p-values. Neither mode uses the factor 2 printed in Eq. (9).
    return min(1.0, hoeffding, _bentkus(n, alpha, cutoff)), risk


@dataclass(frozen=True)
class Calibration:
    # index=None means an explicit empty output set; threshold=1 is a real filter.
    index: int | None
    threshold: float | None
    risk: float | None
    p: float | None
    grid_passed: bool
    reason: str


def calibrate(curves: RiskCurves, indices, alpha: float, delta: float = 0.1,
              mode: str = "corrected") -> Calibration:
    """Fixed-sequence LTT: stop at the FIRST failure, even if risk is nonmonotone."""
    indices = _indices(indices, len(curves.supported))
    if not 0 < delta < 1:
        raise ValueError("Require 0 < delta < 1")
    best = None
    first = None
    for i, threshold in enumerate(curves.grid):
        p, risk = p_value(curves.false[i, indices], curves.selected[i, indices], alpha, mode)
        if i == 0:
            first = Calibration(0, 1.0, risk, p, False, "legacy_uncertified_fallback")
        if p > delta:
            # Do not skip a failed test to search for later successes. Tuple FDP
            # can rise or fall as claims are removed; it need not be monotone.
            break
        best = Calibration(i, float(threshold), risk, p, True, "grid_passed")
    if best is not None:
        return best
    if mode == "legacy_reported":
        # Kept only to expose historical behavior: this may retain score-1 claims
        # even though the first candidate failed its risk test.
        return first
    return Calibration(None, None, None, None, False, "abstain")


def evaluate(curves: RiskCurves, indices, calibration: Calibration) -> dict:
    # Held-out labels measure performance; they do not choose the threshold.
    indices = _indices(indices, len(curves.supported))
    if calibration.index is None:
        selected = false = np.zeros(len(indices), dtype=int)
    else:
        selected = curves.selected[calibration.index, indices]
        false = curves.false[calibration.index, indices]
    discoveries, errors = int(selected.sum()), int(false.sum())
    supported = int(curves.supported[indices].sum())
    return {
        # Recall pools supported claims. tuple_fdr gives every query equal weight;
        # fact_fdr pools retained claims and is a different diagnostic quantity.
        "recall": (discoveries - errors) / max(1, supported),
        "tuple_fdr": float(np.mean(false / np.maximum(1, selected))),
        "fact_fdr": errors / max(1, discoveries),
        "selected": discoveries, "false_selected": errors,
        "supported_selected": discoveries - errors, "supported_total": supported,
    }
