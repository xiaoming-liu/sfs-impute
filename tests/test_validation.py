"""Release-confidence validation suite for the v1.2-minimal distribution.

Run with the stdlib test runner — no extra dependencies required:

    python -m unittest tests.test_validation -v

Or, if pytest is installed:

    pytest tests/

Covers: package import + version, entry-point registration, kernel
correctness on a deterministic fully-observed case, EM solver simplex
invariants, monomorphic augmentation rule, projection roundtrip + tail
invariants, and an end-to-end CLI smoke test on the bundled
example_input.SFS.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from importlib import metadata
from pathlib import Path

import numpy as np

import sfs_imputation
from sfs_imputation import kernel as sfs_kernel
from sfs_imputation import monomorphic as sfs_mono
from sfs_imputation import projection as sfs_proj
from sfs_imputation import solver_em

EXAMPLE = Path(__file__).resolve().parent.parent / "example_input.SFS"


class TestInstall(unittest.TestCase):
    """The package imports, reports the expected version, and registers
    its console-script entry points."""

    def test_version_is_1_2_3(self) -> None:
        self.assertEqual(sfs_imputation.__version__, "1.2.3")

    def test_console_scripts_registered(self) -> None:
        try:
            installed_version = metadata.version("sfs_imputation")
        except metadata.PackageNotFoundError:
            installed_version = None

        if installed_version == sfs_imputation.__version__:
            eps = {ep.name for ep in metadata.entry_points(group="console_scripts")}
        else:
            # Source-tree runs via PYTHONPATH do not install distribution
            # metadata, so validate the script declarations in pyproject.toml.
            pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
            project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
            eps = set(project.get("scripts", {}))

        for cmd in ("sfs-impute", "vcf-to-sfs"):
            with self.subTest(cmd=cmd):
                self.assertIn(cmd, eps, f"entry point '{cmd}' missing")


class TestKernel(unittest.TestCase):
    """Kernel correctness on cases with closed-form expected values."""

    def test_no_missing_kernel_is_identity(self) -> None:
        # With m=0 and a fully-observed (n1=k, n2=N-k) config, the only true
        # frequency consistent with the observation is j = k. So the kernel
        # row should be a one-hot vector at column k.
        N = 20
        configs = [(0, 1, k, N - k) for k in (1, 7, 13, 19)]
        A, _, _ = sfs_kernel.build_kernel(configs, N=N, folded=False, truncation=0.0)
        for r, (_, _, k, _) in enumerate(configs):
            row = np.asarray(A[r].todense()).ravel()
            np.testing.assert_allclose(row[k], 1.0, atol=1e-12,
                                       err_msg=f"row {r} (k={k}) should be 1.0")
            mask = np.ones_like(row, dtype=bool); mask[k] = False
            np.testing.assert_allclose(row[mask], 0.0, atol=1e-12)

    def test_entries_are_probabilities(self) -> None:
        N = 50
        configs = [(0, 1, 30, 20), (5, 1, 25, 20), (10, 1, 20, 20)]
        A, _, _ = sfs_kernel.build_kernel(configs, N=N, folded=False, truncation=0.0)
        dense = np.asarray(A.todense())
        self.assertGreaterEqual(dense.min(), 0.0 - 1e-12)
        self.assertLessEqual(dense.max(), 1.0 + 1e-12)
        self.assertEqual(dense.shape, (len(configs), N + 1))


class TestSolver(unittest.TestCase):
    """SQUAREM-EM converges to a valid simplex probability vector."""

    def test_p_is_nonneg_and_sums_to_one(self) -> None:
        N = 50
        configs = [(0, 100, 30, 20), (5, 50, 25, 20), (10, 25, 20, 20)]
        A, c, _ = sfs_kernel.build_kernel(configs, N=N, folded=False, truncation=0.0)
        res = solver_em.solve(A, c, tol=1e-10, max_iter=1000)
        self.assertTrue(res.converged, f"solver failed to converge ({res.iters} iters)")
        self.assertGreaterEqual(res.p.min(), -1e-12)
        self.assertAlmostEqual(float(res.p.sum()), 1.0, places=10)

    def test_loglik_is_finite_after_convergence(self) -> None:
        N = 30
        configs = [(0, 200, 20, 10), (3, 100, 15, 12)]
        A, c, _ = sfs_kernel.build_kernel(configs, N=N, folded=False, truncation=0.0)
        res = solver_em.solve(A, c, tol=1e-10, max_iter=2000)
        self.assertTrue(res.converged)
        self.assertTrue(np.isfinite(res.final_loglik))


class TestMonomorphicAugment(unittest.TestCase):
    """v1.2 unified augmentation rule: synthesizes mono configs from the
    missing-pattern distribution of the input."""

    def test_total_sums_to_L_total(self) -> None:
        configs = [(0, 200, 6, 4), (4, 100, 4, 2)]
        out = sfs_mono.augment(configs, N=10, L_total=1000)
        self.assertEqual(sum(c[1] for c in out), 1000)

    def test_no_op_when_L_equals_sum(self) -> None:
        configs = [(0, 200, 6, 4), (4, 100, 4, 2)]
        out = sfs_mono.augment(configs, N=10, L_total=300)
        self.assertEqual(out, list(configs))

    def test_rejects_L_below_sum(self) -> None:
        with self.assertRaises(ValueError):
            sfs_mono.augment([(0, 200, 6, 4)], N=10, L_total=50)


class TestProjection(unittest.TestCase):
    """project() is a row-stochastic linear operator on the SFS simplex."""

    def test_project_to_same_N_is_identity(self) -> None:
        N = 20
        p = np.zeros(N + 1)
        p[1:N] = 1.0 / np.arange(1, N)
        p /= p.sum()
        out = sfs_proj.project(p, N=N, n_star=N, folded=False)
        np.testing.assert_allclose(out, p, atol=1e-12)

    def test_project_preserves_simplex(self) -> None:
        N, n_star = 50, 25
        p = np.full(N + 1, 1.0 / (N + 1))
        out = sfs_proj.project(p, N=N, n_star=n_star, folded=False)
        self.assertEqual(out.shape, (n_star + 1,))
        self.assertAlmostEqual(float(out.sum()), 1.0, places=10)
        self.assertGreaterEqual(out.min(), -1e-12)


class TestEndToEndCLI(unittest.TestCase):
    """End-to-end smoke test on the bundled example_input.SFS."""

    def test_sfs_impute_runs_on_example(self) -> None:
        self.assertTrue(EXAMPLE.exists(), f"missing bundled example: {EXAMPLE}")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "example_input.SFS"
            shutil.copy(EXAMPLE, target)
            r = subprocess.run(
                [sys.executable, "-m", "sfs_imputation.cli", str(target)],
                capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(r.returncode, 0, f"sfs-impute failed:\n{r.stderr}")
            out = target.with_suffix(".SFS.imputed")
            self.assertTrue(out.exists(), f"no output produced at {out}")
            # The imputed file is headerless; rows are tab-sep "0 count j N-j".
            counts = []
            for line in out.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 4:
                    continue
                counts.append(float(parts[1]))
            self.assertGreater(len(counts), 0, "imputed file is empty")
            arr = np.array(counts)
            self.assertGreater(arr.sum(), 0.0)
            self.assertGreaterEqual(arr.min(), -1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
