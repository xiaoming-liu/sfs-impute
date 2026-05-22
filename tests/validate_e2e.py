"""End-to-end validation of the sfs_imputation pipeline.

Runs the full pipeline from coalescent simulation to imputed SFS,
exactly the way a user would run it on real data:

    msprime coalescent simulation
        -> ground-truth full-sample SFS (saved for comparison)
        -> diploid VCF written via tskit's write_vcf
    introduce per-genotype missing data (set GT to "./.")
        -> recompute INFO/AC_pop, INFO/AN_pop per site from masked GTs
        -> rewrite VCF
    vcf-to-sfs --pops sim --total-an N --total-length L
        -> .SFS file with header, configs (m, count, n1, n2)
    sfs-impute
        -> .SFS.imputed file with one row per derived-allele class
    compare imputed counts to ground truth

Reports L1, KL, summary statistics (Watterson's theta, pi), and a
PASS/FAIL verdict against thresholds. Exit code 0 = pass, 1 = fail.

Usage (after `pip install '.[validate]'`):

    python tests/validate_e2e.py                  # default config
    python tests/validate_e2e.py --n-diploid 100 --L 1e7  # bigger run
    python tests/validate_e2e.py --miss-rate 0.20 --l1-threshold 0.45

Wall time: ~30-60 seconds on the default (N=100, L=5Mb).
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import msprime
except ImportError:
    print("ERROR: msprime not installed. Run:", file=sys.stderr)
    print("    pip install '.[validate]'", file=sys.stderr)
    print("(or: pip install msprime>=1.3)", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Step 1: simulate coalescent + mutations, capture ground-truth SFS
# ---------------------------------------------------------------------------

def simulate(
    n_diploid: int, sequence_length: float, mutation_rate: float,
    pop_size: int, recomb_rate: float, seed: int,
):
    """Returns (ts, true_count) where true_count[k] = #variants with derived
    count k at the FULL sample (no missing data).

    Counts only strictly biallelic sites so the comparison is consistent with
    vcf-to-sfs's default --snp-only filter, which drops multi-allelic sites.
    """
    ts = msprime.sim_ancestry(
        samples=n_diploid,
        sequence_length=sequence_length,
        recombination_rate=recomb_rate,
        population_size=pop_size,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed)
    N = 2 * n_diploid
    true_count = np.zeros(N + 1, dtype=np.int64)
    for var in ts.variants():
        if len(var.alleles) != 2:
            continue
        j = int((var.genotypes == 1).sum())
        true_count[j] += 1
    return ts, true_count


# ---------------------------------------------------------------------------
# Step 2: write VCF, then post-process to introduce missing genotypes and
# annotate INFO/AC_pop, INFO/AN_pop
# ---------------------------------------------------------------------------

def write_vcf_with_missing(
    ts, *, n_diploid: int, miss_rate: float, pop_label: str,
    out_path: Path, rng: np.random.Generator,
) -> tuple[int, int]:
    """Write VCF from ts, mask each GT with prob miss_rate, recompute AC_pop,
    AN_pop. Returns (n_variants_written, n_polymorphic_after_masking)."""
    contig = "1"
    sample_names = [f"S{i}" for i in range(n_diploid)]
    raw_path = out_path.with_suffix(".raw.vcf")
    with raw_path.open("w", encoding="utf-8") as f:
        ts.write_vcf(f, contig_id=contig, individual_names=sample_names)

    n_variants = 0
    n_poly = 0
    with raw_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("##"):
                fout.write(line)
                continue
            if line.startswith("#CHROM"):
                # Inject INFO header lines just before the column header.
                fout.write(
                    f'##INFO=<ID=AC_{pop_label},Number=A,Type=Integer,'
                    f'Description="Allele count in {pop_label} after masking">\n'
                )
                fout.write(
                    f'##INFO=<ID=AN_{pop_label},Number=1,Type=Integer,'
                    f'Description="Total allele number in {pop_label} after masking">\n'
                )
                fout.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            # Skip multi-allelic — vcf-to-sfs's --snp-only default drops them
            if "," in fields[4] or len(fields[3]) != 1 or len(fields[4]) != 1:
                continue
            n_variants += 1
            # Columns: 0 CHROM 1 POS 2 ID 3 REF 4 ALT 5 QUAL 6 FILTER 7 INFO
            # 8 FORMAT 9.. samples (one per diploid)
            # FORMAT is "GT" from tskit; values are e.g. "0|1", "1|1".
            gts = fields[9:]
            assert len(gts) == n_diploid, (
                f"expected {n_diploid} sample columns, got {len(gts)}"
            )
            n_alt = 0
            n_called = 0
            new_gts = []
            for g in gts:
                a, b = g.split("|") if "|" in g else g.split("/")
                # Mask each haplotype independently with prob miss_rate
                miss_a = rng.random() < miss_rate
                miss_b = rng.random() < miss_rate
                if miss_a and miss_b:
                    new_gts.append("./.")
                    continue
                if miss_a:
                    a = "."
                if miss_b:
                    b = "."
                new_gts.append(f"{a}|{b}")
                if a != ".":
                    n_called += 1
                    if a == "1":
                        n_alt += 1
                if b != ".":
                    n_called += 1
                    if b == "1":
                        n_alt += 1
            if n_called == 0:
                # All-missing site — skip (no information)
                continue
            if n_alt > 0 and n_alt < n_called:
                n_poly += 1
            # FILTER must be "PASS" for the default vcf-to-sfs filter
            fields[6] = "PASS"
            # Replace INFO with AC_pop, AN_pop
            fields[7] = f"AC_{pop_label}={n_alt};AN_{pop_label}={n_called}"
            fields[8] = "GT"
            fields[9:] = new_gts
            fout.write("\t".join(fields) + "\n")
    raw_path.unlink()
    return n_variants, n_poly


# ---------------------------------------------------------------------------
# Step 3: invoke vcf-to-sfs and sfs-impute as subprocesses (the user-facing
# pathway, so we exercise the real CLI)
# ---------------------------------------------------------------------------

def run_pipeline(
    vcf_path: Path, *, pop_label: str, total_an: int, total_length: float,
    out_dir: Path,
) -> Path:
    """Run vcf-to-sfs then sfs-impute. Returns path to the .imputed file."""
    stem = out_dir / "sim"
    cmd_vcf = [
        sys.executable, "-m", "sfs_imputation.vcf_to_sfs",
        str(vcf_path),
        "--pops", pop_label,
        "--total-an", str(total_an),
        "--total-length", str(total_length),
        "--out-stem", str(stem),
    ]
    r = subprocess.run(cmd_vcf, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"vcf-to-sfs failed:\n{r.stderr}")
    sfs_path = Path(f"{stem}.{pop_label}.SFS")
    if not sfs_path.exists():
        raise RuntimeError(f"vcf-to-sfs produced no output at {sfs_path}")

    cmd_imp = [
        sys.executable, "-m", "sfs_imputation.cli",
        "--verbose", str(sfs_path),
    ]
    r = subprocess.run(cmd_imp, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"sfs-impute failed:\n{r.stderr}")
    imputed_path = Path(f"{sfs_path}.imputed")
    if not imputed_path.exists():
        raise RuntimeError(f"sfs-impute produced no output at {imputed_path}")
    return imputed_path


def parse_imputed(path: Path, N: int) -> np.ndarray:
    """Parse the .imputed file (headerless, 4-col rows: m count n1 n2) into
    counts indexed by derived-allele count.

    For the unfolded output written by sfs-impute, n1 is the REF (ancestral)
    count and n2 is the DERIVED count, so we index `counts[n2]`.
    """
    counts = np.zeros(N + 1, dtype=float)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        _, count, _n1, n2 = parts
        counts[int(n2)] = float(count)
    return counts


# ---------------------------------------------------------------------------
# Step 4: scoring
# ---------------------------------------------------------------------------

def watterson_theta(counts: np.ndarray, N: int) -> float:
    s = float(counts[1:N].sum())
    a_n = sum(1.0 / k for k in range(1, N))
    return s / a_n if a_n > 0 else 0.0


def pi_diversity(counts: np.ndarray, N: int) -> float:
    k = np.arange(N + 1)
    weight = k * (N - k) / (N * (N - 1) / 2)
    return float(np.sum(weight * counts))


def score(true_count: np.ndarray, imputed_count: np.ndarray, N: int) -> dict:
    poly_lo, poly_hi = 1, N
    s_true = float(true_count[poly_lo:poly_hi].sum())
    s_imp = float(imputed_count[poly_lo:poly_hi].sum())
    p_true = true_count[poly_lo:poly_hi] / s_true if s_true > 0 else np.zeros(N - 1)
    p_imp = imputed_count[poly_lo:poly_hi] / s_imp if s_imp > 0 else np.zeros(N - 1)
    l1 = float(np.abs(p_true - p_imp).sum())
    eps = 1e-12
    kl = float(np.sum(p_true * np.log((p_true + eps) / (p_imp + eps))))
    return {
        "S_true": s_true,
        "S_imputed": s_imp,
        "L1_normalized": l1,
        "KL_true_to_imp": kl,
        "theta_W_true": watterson_theta(true_count, N),
        "theta_W_imputed": watterson_theta(imputed_count, N),
        "pi_true": pi_diversity(true_count, N),
        "pi_imputed": pi_diversity(imputed_count, N),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-diploid", type=int, default=50,
                   help="Diploid samples (default 50 = N=100 chromosomes)")
    p.add_argument("--L", type=float, default=5e6,
                   help="Sequence length in bp (default 5e6)")
    p.add_argument("--mu", type=float, default=1e-8,
                   help="Mutation rate per bp per gen (default 1e-8)")
    p.add_argument("--pop-size", type=float, default=1e4,
                   help="Effective population size (default 1e4)")
    p.add_argument("--recomb", type=float, default=1e-8,
                   help="Recombination rate per bp per gen (default 1e-8)")
    p.add_argument("--miss-rate", type=float, default=0.05,
                   help="Per-genotype masking probability (default 0.05)")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed for both msprime and masking (default 1)")
    p.add_argument("--l1-threshold", type=float, default=0.30,
                   help="L1(p_true, p_imp) above this fails (default 0.30; "
                        "small-N noise dominates this metric — empirically ~0.15-0.22 "
                        "across seeds at N=100, L=5Mb)")
    p.add_argument("--theta-rel-threshold", type=float, default=0.02,
                   help="Watterson theta relative error above this fails "
                        "(default 0.02; empirically ~0.001 across seeds)")
    p.add_argument("--pi-rel-threshold", type=float, default=0.02,
                   help="Pi relative error above this fails (default 0.02)")
    p.add_argument("--keep-tmp", action="store_true",
                   help="Don't delete the temporary VCF/SFS files (for inspection)")
    args = p.parse_args(argv)

    N = 2 * args.n_diploid
    pop = "sim"
    print(f"sfs_imputation end-to-end validation")
    print(f"  N (chromosomes)   : {N}")
    print(f"  sequence length   : {args.L:,.0f} bp")
    print(f"  mutation rate     : {args.mu}")
    print(f"  Ne                : {args.pop_size:,.0f}")
    print(f"  per-GT miss rate  : {args.miss_rate:.1%}")
    print(f"  seed              : {args.seed}")
    print()

    t0 = time.time()
    print("[1/4] simulating coalescent + mutations ...", flush=True)
    ts, true_count = simulate(
        n_diploid=args.n_diploid, sequence_length=args.L,
        mutation_rate=args.mu, pop_size=int(args.pop_size),
        recomb_rate=args.recomb, seed=args.seed,
    )
    s_true = int(true_count[1:N].sum())
    print(f"      {ts.num_sites:,} variant sites, {s_true:,} polymorphic at full N.")

    rng = np.random.default_rng(args.seed + 1)
    tmp_root = Path(tempfile.mkdtemp(prefix="sfs_e2e_"))
    try:
        vcf_path = tmp_root / "sim.vcf"
        print("[2/4] writing VCF and applying per-GT missing mask ...", flush=True)
        n_var, n_poly_after = write_vcf_with_missing(
            ts, n_diploid=args.n_diploid, miss_rate=args.miss_rate,
            pop_label=pop, out_path=vcf_path, rng=rng,
        )
        print(f"      wrote {n_var:,} variant records "
              f"({n_poly_after:,} still polymorphic after masking).")

        print("[3/4] vcf-to-sfs + sfs-impute ...", flush=True)
        imputed_path = run_pipeline(
            vcf_path, pop_label=pop, total_an=N, total_length=args.L,
            out_dir=tmp_root,
        )
        imputed_count = parse_imputed(imputed_path, N)

        print("[4/4] scoring against ground truth ...", flush=True)
        scores = score(true_count, imputed_count, N)
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            print(f"      tmp dir kept: {tmp_root}")

    rel_theta = abs(scores["theta_W_imputed"] - scores["theta_W_true"]) \
                / max(scores["theta_W_true"], 1e-30)
    rel_pi = abs(scores["pi_imputed"] - scores["pi_true"]) \
             / max(scores["pi_true"], 1e-30)

    print()
    print("Results")
    print("-------")
    print(f"  S (polymorphic sites)   true={scores['S_true']:>10,.0f}   "
          f"imputed={scores['S_imputed']:>10,.1f}")
    print(f"  Watterson's theta       true={scores['theta_W_true']:>10.4f}   "
          f"imputed={scores['theta_W_imputed']:>10.4f}   relerr={rel_theta:.4f}")
    print(f"  pi (nuc. diversity)     true={scores['pi_true']:>10.4f}   "
          f"imputed={scores['pi_imputed']:>10.4f}   relerr={rel_pi:.4f}")
    print(f"  L1(p_true, p_imputed)            = {scores['L1_normalized']:.4f}")
    print(f"  KL(p_true || p_imputed)          = {scores['KL_true_to_imp']:.4f}")
    print(f"  Wall time                        = {time.time() - t0:.1f}s")
    print()

    fails = []
    if not math.isfinite(scores["L1_normalized"]):
        fails.append("L1 is non-finite")
    elif scores["L1_normalized"] > args.l1_threshold:
        fails.append(
            f"L1={scores['L1_normalized']:.4f} > threshold {args.l1_threshold}"
        )
    if not math.isfinite(rel_theta):
        fails.append("theta relative error is non-finite")
    elif rel_theta > args.theta_rel_threshold:
        fails.append(
            f"theta_W relerr={rel_theta:.4f} > threshold {args.theta_rel_threshold}"
        )
    if not math.isfinite(rel_pi):
        fails.append("pi relative error is non-finite")
    elif rel_pi > args.pi_rel_threshold:
        fails.append(
            f"pi relerr={rel_pi:.4f} > threshold {args.pi_rel_threshold}"
        )

    if fails:
        print("FAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS: all thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
