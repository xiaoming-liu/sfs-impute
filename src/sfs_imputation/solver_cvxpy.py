from __future__ import annotations

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def solve(A: sp.csr_matrix, c: np.ndarray, *, solver: str = "ECOS") -> np.ndarray:
    """Solve max_p sum_s c_s log((A p)_s) subject to p >= 0, sum(p) = 1.

    Returns p as a 1-D numpy array of length K+1.
    """
    A_dense = A.toarray()
    K_plus_1 = A_dense.shape[1]
    p = cp.Variable(K_plus_1, nonneg=True)
    objective = cp.Maximize(c @ cp.log(A_dense @ p))
    constraints = [cp.sum(p) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)
    if p.value is None:
        raise RuntimeError(f"cvxpy failed to solve (status={prob.status})")
    p_val = np.maximum(np.asarray(p.value).ravel(), 0.0)
    p_val /= p_val.sum()
    return p_val
