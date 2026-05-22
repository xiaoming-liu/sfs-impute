from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln


def _logbinom(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Log of binomial coefficient C(n, k), elementwise. Returns -inf where invalid."""
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    invalid = (k < 0) | (k > n) | (n < 0)
    out = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    out = np.where(invalid, -np.inf, out)
    return out


def _truncate_log_row(log_a: np.ndarray, rel_threshold: float) -> np.ndarray:
    """Return a boolean mask of entries to keep in this kernel row.

    Drops -inf entries always. If rel_threshold > 0, also drops entries below
    log(rel_threshold) relative to the row maximum. Used to keep nnz manageable
    at high N where hypergeometric tails extend over thousands of bins.
    """
    finite = np.isfinite(log_a)
    if rel_threshold <= 0 or not finite.any():
        return finite
    log_max = log_a[finite].max()
    cutoff = log_max + np.log(rel_threshold)
    return finite & (log_a >= cutoff)


def build_kernel(
    configs: Sequence[Tuple[int, int, int, int]],
    *,
    N: int,
    folded: bool,
    truncation: float = 1e-15,
) -> Tuple[sp.csr_matrix, np.ndarray, int]:
    """Build sparse kernel A and count vector c.

    Parameters
    ----------
    configs : sequence of (m, count, n1, n2) tuples
        m     = number of missing haplotypes
        count = observed count for this configuration
        n1    = allele-1 count
        n2    = allele-2 count
    N : int
        Total number of haplotypes (observed + missing).
    folded : bool
        If True, use the folded (minor-allele) kernel (Task 6).
        If False, use the unfolded kernel.
    truncation : float
        Drop kernel entries with relative weight < truncation per row.
        Default 1e-15 (machine precision floor for float64). Set to 0 to
        disable (recovers Phase-1 behavior exactly). Cuts nnz dramatically
        at high N with high missing rates.

    Returns
    -------
    A : csr_matrix of shape (S, K+1)
        Sparse hypergeometric kernel; A[s, j] = P(observe config s | full count = j).
    c : ndarray of shape (S,)
        Observed counts for each configuration.
    K : int
        Maximum full allele count (N if unfolded, floor(N/2) if folded).
    """
    if folded:
        return _build_folded(configs, N=N, truncation=truncation)
    return _build_unfolded(configs, N=N, truncation=truncation)


def _build_unfolded(
    configs: Sequence[Tuple[int, int, int, int]],
    *, N: int, truncation: float = 1e-15,
) -> Tuple[sp.csr_matrix, np.ndarray, int]:
    """Unfolded sparse hypergeometric kernel.

    A_{s,j} = C(j, n1) * C(N-j, n2) / C(N, n1+n2)
    for j in [n1, n1+m], where m is the number of missing haplotypes.
    """
    K = N
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    counts = np.empty(len(configs), dtype=float)
    for s, (m, count, n1, n2) in enumerate(configs):
        counts[s] = count
        n = n1 + n2
        log_denom = _logbinom(np.array([N]), np.array([n]))[0]
        j_lo, j_hi = n1, n1 + m
        j_arr = np.arange(j_lo, j_hi + 1)
        log_a = (
            _logbinom(j_arr, n1)
            + _logbinom(N - j_arr, n2)
            - log_denom
        )
        keep = _truncate_log_row(log_a, truncation)
        if not keep.any():
            continue
        # log_a values are bounded above by 0 (each row is a hypergeometric PMF
        # in observed-count space), so direct exp() cannot overflow. We still
        # subtract the row max for symmetry with the folded path where logaddexp
        # of two terms benefits from the same numerically-stable scaling pattern.
        m_max = log_a[keep].max()
        a_kept = np.exp(log_a[keep] - m_max) * np.exp(m_max)
        kept_j = j_arr[keep]
        positive = a_kept > 0
        if not positive.any():
            continue
        rows.extend([s] * int(positive.sum()))
        cols.extend(kept_j[positive].tolist())
        data.extend(a_kept[positive].tolist())
    A = sp.csr_matrix((data, (rows, cols)), shape=(len(configs), K + 1))
    return A, counts, K


def _build_folded(
    configs: Sequence[Tuple[int, int, int, int]],
    *, N: int, truncation: float = 1e-15,
) -> Tuple[sp.csr_matrix, np.ndarray, int]:
    K = N // 2
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    counts = np.empty(len(configs), dtype=float)
    for s, (m, count, n1, n2) in enumerate(configs):
        counts[s] = count
        n = n1 + n2
        log_denom = _logbinom(np.array([N]), np.array([n]))[0]
        # Bounding box: k in [n2, n1+m] intersected with [0, K]
        k_lo = max(0, n2)
        k_hi = min(K, n1 + m)
        if k_hi < k_lo:
            continue
        k_arr = np.arange(k_lo, k_hi + 1)
        # Term 1: C(k, n2) * C(N-k, n1)
        log_t1 = (
            _logbinom(k_arr, n2)
            + _logbinom(N - k_arr, n1)
            - log_denom
        )
        if n1 == n2:
            log_a = log_t1
        else:
            # Term 2: C(k, n1) * C(N-k, n2)
            log_t2 = (
                _logbinom(k_arr, n1)
                + _logbinom(N - k_arr, n2)
                - log_denom
            )
            # log-sum-exp via numpy
            log_a = np.logaddexp(log_t1, log_t2)
        keep = _truncate_log_row(log_a, truncation)
        if not keep.any():
            continue
        # Same row-max rescale rationale as _build_unfolded: log_a values are
        # bounded above by 0 (folded PMF), so direct exp() cannot overflow,
        # but we keep the rescale for symmetry and future numerical robustness.
        m_max = log_a[keep].max()
        a_kept = np.exp(log_a[keep] - m_max) * np.exp(m_max)
        kept_k = k_arr[keep]
        positive = a_kept > 0
        if not positive.any():
            continue
        rows.extend([s] * int(positive.sum()))
        cols.extend(kept_k[positive].tolist())
        data.extend(a_kept[positive].tolist())
    A = sp.csr_matrix((data, (rows, cols)), shape=(len(configs), K + 1))
    return A, counts, K
