from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from sfs_imputation import io as sfs_io
from sfs_imputation import kernel as sfs_kernel
from sfs_imputation import monomorphic as sfs_mono
from sfs_imputation import solver_em


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sfs-impute",
        description="Impute the full-sample SFS from observed AC/AN configs.",
    )
    parser.add_argument("inputs", nargs="+", help="Input SFS files")
    parser.add_argument("--total-length", type=float, default=None,
                        help="Total sequence length L (required when --hasmono=false)")
    parser.add_argument("--hasmono", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--folded", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--solver", choices=["em", "cvxpy"], default="em")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--tol-loglik", type=float, default=1e-9,
                        help="Convergence tolerance on per-iter delta_loglik / L (default 1e-9)")
    parser.add_argument("--truncation", type=float, default=1e-15,
                        help="Drop kernel entries with relative weight < REL "
                             "(default 1e-15; set 0 for Phase-1 reproducibility)")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--out-suffix", default=".imputed")
    parser.add_argument("--project-to", type=int, default=None, metavar="N_STAR",
                        help="Down-project the imputed full-N SFS to sample size N_STAR "
                             "and write to <input>.imputed.proj_<N_STAR>")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _resolve_bool(opt: str, auto_value: bool) -> bool:
    if opt == "auto":
        return auto_value
    return opt == "true"


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    for in_path in args.inputs:
        cfg = sfs_io.read_sfs_file(in_path)
        folded = _resolve_bool(args.folded, cfg.folded)
        hasmono = _resolve_bool(args.hasmono, cfg.hasmono)
        configs = list(cfg.configs)
        # Resolve the total sequence length L. The L in the SFS header reflects
        # only what was put in the input file; for VCF-derived data, that is
        # typically the polymorphic sites + the small fraction of monomorphic
        # sites that the VCF reports, which is much less than the true callable
        # length. --total-length on the CLI overrides the header value.
        L_total: Optional[float] = None
        if args.total_length is not None:
            L_total = args.total_length
        elif cfg.L_total is not None:
            L_total = cfg.L_total
        # v1.2 unified augmentation: synthesize monomorphic-reference configs
        # to bring total up to L_total, distributed by the missing-pattern
        # distribution of ALL input configs (poly + reported mono).  Triggered
        # whenever L_total is supplied and exceeds the input's sum(c). Skipped
        # if L_total is unset (just impute on input as-is) or if L_total
        # equals sum(c) (input already represents L_total). Errors if
        # L_total < sum(c).
        sum_c = sum(count for (_, count, _, _) in configs)
        if L_total is None and not hasmono:
            raise SystemExit(
                "--total-length required when input has no monomorphic configs "
                "(hasmono=false) and header has no 'Total sequence length' value"
            )
        if L_total is not None and L_total > sum_c:
            configs = sfs_mono.augment(configs, N=cfg.N, L_total=L_total)
        A, c, K = sfs_kernel.build_kernel(configs, N=cfg.N, folded=folded,
                                          truncation=args.truncation)
        if args.solver == "cvxpy":
            from sfs_imputation import solver_cvxpy  # lazy import — optional dep
            p = solver_cvxpy.solve(A, c)
            iters = -1
            converged = True
            final_delta = 0.0
            final_ll = float("nan")
        else:
            res = solver_em.solve(
                A, c, tol=args.tol, tol_loglik=args.tol_loglik,
                max_iter=args.max_iter, verbose=args.verbose,
            )
            p = res.p
            iters = res.iters
            converged = res.converged
            final_delta = res.final_delta
            final_ll = res.final_loglik
        # Output count scaling: use sum(c) (the count over sites actually
        # represented in the input). The imputer estimates p[k] as a
        # probability over those sites, so count[k] = p[k] * sum(c) is
        # the imputed number of polymorphic sites of frequency k present
        # in the input. We do NOT scale by L_total in hasmono=True mode
        # because the unreported monomorphic sites in the rest of the
        # genome have unknown missing patterns and were never in the kernel
        # -- scaling output by L_total would inflate every polymorphic-bin
        # count by the ratio L_total/sum(c), inventing sites that have no
        # support in the data.
        L = float(c.sum())
        out_path = Path(in_path + args.out_suffix)
        sfs_io.write_imputed(p, L=L, folded=folded, N=cfg.N, out_path=out_path)
        if args.project_to is not None:
            from sfs_imputation.projection import project
            n_star = args.project_to
            p_proj = project(p, N=cfg.N, n_star=n_star, folded=folded)
            proj_path = Path(in_path + args.out_suffix + f".proj_{n_star}")
            sfs_io.write_imputed(p_proj, L=L, folded=folded, N=n_star,
                                 out_path=proj_path)
        if args.verbose or not converged:
            status = "converged" if converged else "MAX_ITER"
            print(
                f"{in_path}: N={cfg.N} folded={folded} configs={len(configs)} "
                f"iters={iters} delta={final_delta:.3e} loglik={final_ll:.6g} [{status}]",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
