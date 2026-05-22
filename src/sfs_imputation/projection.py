"""Hypergeometric down-projection: convert an SFS at sample size N to an SFS
at a smaller sample size n_star. Used as both a standalone postprocessor
(produce an SFS suitable for dadi/moments) and as a baseline comparator
in the Phase-2 benchmark."""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def _logbinom_scalar(n: int, k: int) -> float:
    """Log of C(n, k), or -inf if invalid."""
    if k < 0 or k > n:
        return -np.inf
    return float(gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0))


def project(
    p_full: np.ndarray, *, N: int, n_star: int, folded: bool = False,
) -> np.ndarray:
    """Down-project an SFS at sample size N to an SFS at sample size n_star.

    For each k' = 0..n_star, computes
        p_proj[k'] = sum over j of C(j,k') * C(N-j, n_star-k') / C(N, n_star) * p[j]

    Parameters
    ----------
    p_full : ndarray
        SFS at sample size N. Length N+1 if unfolded, ⌊N/2⌋+1 if folded.
    N : int
        Original full sample size.
    n_star : int
        Target sample size; must satisfy 0 < n_star <= N.
    folded : bool
        If True, both input and output are folded SFSs.

    Returns
    -------
    p_proj : ndarray of length n_star+1 (unfolded) or ⌊n_star/2⌋+1 (folded).
    """
    if not (0 < n_star <= N):
        raise ValueError(f"n_star ({n_star}) must satisfy 0 < n_star <= N ({N})")
    if folded:
        return _project_folded(p_full, N=N, n_star=n_star)
    return _project_unfolded(p_full, N=N, n_star=n_star)


def _project_unfolded(p: np.ndarray, *, N: int, n_star: int) -> np.ndarray:
    """Vectorized down-projection of unfolded SFS."""
    log_denom = _logbinom_scalar(N, n_star)
    out = np.zeros(n_star + 1)
    for kp in range(n_star + 1):
        # j ranges over [kp, N - (n_star - kp)] for nonzero C(j,kp)*C(N-j, n_star-kp)
        j_lo = kp
        j_hi = N - (n_star - kp)
        if j_hi < j_lo:
            continue
        j_arr = np.arange(j_lo, j_hi + 1, dtype=np.int64)
        # log C(j, kp) + log C(N-j, n_star-kp) - log C(N, n_star)
        # vectorized via gammaln
        log_w = (
            gammaln(j_arr + 1.0) - gammaln(kp + 1.0) - gammaln(j_arr - kp + 1.0)
            + gammaln(N - j_arr + 1.0) - gammaln(n_star - kp + 1.0)
            - gammaln(N - j_arr - (n_star - kp) + 1.0)
            - log_denom
        )
        # gammaln of negatives may produce nan/inf; mask those out before exp
        finite = np.isfinite(log_w)
        if not finite.any():
            continue
        out[kp] = float(np.sum(np.exp(log_w[finite]) * p[j_lo:j_hi + 1][finite]))
    return out


def _project_folded(p_folded: np.ndarray, *, N: int, n_star: int) -> np.ndarray:
    """Project a folded SFS at N to a folded SFS at n_star.

    Strategy: unfold p_folded into an unfolded SFS at N by symmetry, project
    that to n_star unfolded, then re-fold. Half the mass at each folded bin is
    assigned to k and half to N-k (or all of it if k == N-k).
    """
    K = N // 2
    p_unfold = np.zeros(N + 1)
    for k in range(K + 1):
        if k == N - k:
            p_unfold[k] = p_folded[k]
        else:
            p_unfold[k] = p_folded[k] / 2.0
            p_unfold[N - k] = p_folded[k] / 2.0
    p_proj_unfold = _project_unfolded(p_unfold, N=N, n_star=n_star)
    Kp = n_star // 2
    p_proj_folded = np.zeros(Kp + 1)
    for kp in range(Kp + 1):
        if kp == n_star - kp:
            p_proj_folded[kp] = p_proj_unfold[kp]
        else:
            p_proj_folded[kp] = p_proj_unfold[kp] + p_proj_unfold[n_star - kp]
    return p_proj_folded
