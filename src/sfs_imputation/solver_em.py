"""SQUAREM-accelerated EM solver for the convex SFS imputation NPMLE.

Note on solver interface: ``solve`` (and ``solve_plain_em``) returns an
``EMResult`` dataclass exposing ``p`` plus convergence metadata. This
differs from ``solver_cvxpy.solve`` which returns a bare ``np.ndarray``.
The CLI bridges the two; downstream callers should pick the variant
matching their needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp


_EPS = 1e-300


def em_step(
    A: sp.csr_matrix, c: np.ndarray, p: np.ndarray,
    *, return_n_eps: bool = False,
) -> "np.ndarray | tuple[np.ndarray, int]":
    """One Vardi-Lee-Kaufman / EM step. Returns updated p on the simplex.

    If return_n_eps=True, also returns the number of denom entries that were
    clipped by the _EPS guard this step (an EM-trap diagnostic).
    """
    L = float(c.sum())
    denom = A @ p  # shape (S,)
    n_eps = int(np.sum(denom < _EPS)) if return_n_eps else 0
    ratio = c / np.maximum(denom, _EPS)
    factor = A.T @ ratio  # shape (K+1,)
    p_new = (p * factor) / L
    if return_n_eps:
        return p_new, n_eps
    return p_new


def loglik(A: sp.csr_matrix, c: np.ndarray, p: np.ndarray) -> float:
    denom = A @ p
    return float(c @ np.log(np.maximum(denom, _EPS)))


@dataclass
class EMResult:
    p: np.ndarray
    iters: int
    converged: bool
    final_delta: float
    final_loglik: float


def solve_plain_em(
    A: sp.csr_matrix,
    c: np.ndarray,
    *,
    tol: float = 1e-10,
    tol_loglik: float = 1e-9,
    max_iter: int = 5000,
    p0: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> EMResult:
    K_plus_1 = A.shape[1]
    p = np.ones(K_plus_1) / K_plus_1 if p0 is None else np.array(p0, dtype=float)
    L = float(c.sum())
    ll_prev = loglik(A, c, p)
    converged = False
    delta = np.inf
    total_eps_fires = 0
    iters_done = 0
    for k in range(1, max_iter + 1):
        iters_done = k
        if verbose:
            p_new, n_eps = em_step(A, c, p, return_n_eps=True)
            total_eps_fires += n_eps
        else:
            p_new = em_step(A, c, p)
        delta = float(np.max(np.abs(p_new - p)))
        p = p_new
        ll_curr = loglik(A, c, p)
        d_ll = abs(ll_curr - ll_prev)
        ll_prev = ll_curr
        if delta < tol or d_ll < tol_loglik * L:
            converged = True
            break
    if verbose and total_eps_fires > 0:
        import sys
        print(f"[sfs-impute] WARNING: _EPS guard fired {total_eps_fires} times "
              f"across {iters_done} EM steps", file=sys.stderr)
    final_loglik = loglik(A, c, p) if not converged else ll_curr
    return EMResult(
        p=p, iters=iters_done, converged=converged, final_delta=delta,
        final_loglik=final_loglik,
    )


def solve(
    A: sp.csr_matrix,
    c: np.ndarray,
    *,
    tol: float = 1e-10,
    tol_loglik: float = 1e-9,
    max_iter: int = 5000,
    p0: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> EMResult:
    """SQUAREM-accelerated EM (monotone variant S3).

    Uses a monotone alpha step-back (halving up to 10 times) instead of
    hard simplex-projection: the accelerated point is only accepted when
    p_acc >= 0 componentwise and its log-likelihood exceeds that of two
    plain EM steps.  This prevents early iterations from permanently
    zeroing components and trapping the solver at a constrained suboptimum.
    """
    K_plus_1 = A.shape[1]
    p = np.ones(K_plus_1) / K_plus_1 if p0 is None else np.array(p0, dtype=float)
    ll_curr = loglik(A, c, p)
    converged = False
    delta = np.inf
    iters = 0
    L_total = float(c.sum())
    total_eps_fires = 0
    for k in range(1, max_iter + 1):
        iters = k
        ll_curr_prev = ll_curr  # snapshot before this iteration's update
        if verbose:
            p1, n_eps1 = em_step(A, c, p, return_n_eps=True)
            p2, n_eps2 = em_step(A, c, p1, return_n_eps=True)
            total_eps_fires += n_eps1 + n_eps2
        else:
            p1 = em_step(A, c, p)
            p2 = em_step(A, c, p1)
        r = p1 - p
        v = (p2 - p1) - r
        v_norm = float(np.linalg.norm(v))
        if v_norm == 0.0:
            p_next = p2
            ll_next = loglik(A, c, p2)
        else:
            alpha = -float(np.linalg.norm(r)) / v_norm
            ll_p2 = loglik(A, c, p2)
            # Monotone step-back: halve |alpha| until p_acc is non-negative and
            # its log-likelihood exceeds that of two plain EM steps.  This
            # prevents the simplex projection from permanently zeroing out
            # components that the unconstrained EM would keep positive.
            for _ in range(10):
                p_acc = p - 2.0 * alpha * r + alpha * alpha * v
                if p_acc.min() >= 0.0:
                    p_acc /= p_acc.sum()
                    ll_acc = loglik(A, c, p_acc)
                    if ll_acc >= ll_p2:
                        break
                alpha /= 2.0
            else:
                # All step-back attempts failed; fall through to p2.
                p_acc = p2
                ll_acc = ll_p2
            if ll_acc >= ll_p2:
                p_next = p_acc
                ll_next = ll_acc
            else:
                p_next = p2
                ll_next = ll_p2
        delta = float(np.max(np.abs(p_next - p)))
        p = p_next
        ll_curr = ll_next
        d_ll = abs(ll_next - ll_curr_prev)
        if delta < tol or d_ll < tol_loglik * L_total:
            converged = True
            break
    if verbose and total_eps_fires > 0:
        import sys
        print(f"[sfs-impute] WARNING: _EPS guard fired {total_eps_fires} times "
              f"across {iters} EM steps", file=sys.stderr)
    return EMResult(
        p=p, iters=iters, converged=converged, final_delta=delta,
        final_loglik=ll_curr,
    )
