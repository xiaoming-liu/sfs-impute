# sfs_imputation

Recover the complete-sample site-frequency spectrum (SFS) from VCF allele-count
data with missing genotypes. Formulates the imputation as a convex
non-parametric maximum-likelihood problem (Kiefer-Wolfowitz) on the SFS simplex
and solves it with a sparse SQUAREM-accelerated EM. Replaces the legacy Java
`Impute_SFS` for `N` up to ~130k chromosomes; handles both unfolded and folded
input.

## Install

A Python 3.12+ environment is required. The two recommended installs are
identical on both platforms; only the activation command differs.

**Linux / macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install dist/sfs_imputation-1.2.3-py3-none-any.whl
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install dist/sfs_imputation-1.2.3-py3-none-any.whl
```

(The wheel is in the `dist/` subfolder of this distribution; adjust the path if running from elsewhere.)

After install, the `sfs-impute` and `vcf-to-sfs` commands are on your `PATH`.

When you're done, leave the venv with `deactivate` (works on both Linux/macOS
and Windows). Re-activate later with the same `source .venv/bin/activate` /
`.\.venv\Scripts\Activate.ps1` command — no need to re-install.

## Run

```bash
sfs-impute --verbose example_input.SFS
```

Produces `example_input.SFS.imputed` next to the input. The command takes
~30-60 seconds on the bundled `example_input.SFS` (N=3472 chromosomes, ~25k
unique configs). `--verbose` prints convergence info (iterations,
delta, final log-likelihood) to stderr.

For ecological-scale samples (N < 200), the plain EM solver is recommended:

```bash
sfs-impute --solver plain-em --verbose example_input.SFS
```

Multiple files in one invocation:

```bash
sfs-impute --verbose pop1.SFS pop2.SFS pop3.SFS
```

Common flags:

```
--folded {auto,true,false}    auto-detect from input header (default)
--hasmono {auto,true,false}   auto-detect from input rows (default)
--total-length L              total sequenced/callable length L. When L exceeds
                              the sum of input counts, the imputer synthesizes
                              monomorphic-reference configs to fill the gap,
                              distributed by the missing-pattern distribution
                              of all input configs (poly + reported mono). This
                              gives whole-genome-scale output counts. See
                              "About --total-length" below.
--project-to N_STAR           also write <input>.imputed.proj_<N_STAR>
                              (down-projected SFS for tools like dadi/moments)
--solver {em,plain-em,cvxpy}   em = SQUAREM-accelerated (default, best for N>=5000);
                              plain-em = unaccelerated Vardi EM (recommended for
                              ecological samples N<200, avoids MLE overfitting);
                              cvxpy = convex optimisation (validation only)
--verbose                     print convergence info to stderr
```

### About --total-length

`--total-length L` (or the `Total sequence length (L)` line in the SFS header)
is the **true total sequenced or callable length** of the genome region being
studied — the number that comes from your sequencing-depth or accessibility-mask
analysis (e.g., "770 Mb of the genome had ≥10x coverage in ≥80% of samples").

**As of v1.2.0:** the imputer uses L the same way regardless of `hasmono`. When
`L > sum-of-input-counts`, it synthesizes `L − sum(c)` additional
monomorphic-reference configs and adds them to the input. Each synthetic
config gets the missing pattern of one of the input sites (sampled per the
distribution of missing patterns across **all** input configs — polymorphic
and reported-monomorphic together).

**Underlying assumption:** the unreported monomorphic sites (the bulk of the
callable genome that doesn't appear in the VCF) have the same per-site
missing-pattern distribution as the reported sites. Reasonable for standard
variant-calling pipelines where per-sample call success is governed by read
depth and is symmetric for polymorphic and monomorphic sites; can fail if
monomorphic sites are filtered with different criteria.

**Practical effect on output:** with the assumption holding, output counts
represent **whole-genome polymorphic counts**. Without supplying L (or with
L set to sum(c)), output counts represent **just the polymorphic sites
present in the input file** — useful for relative SFS shape but not for
absolute downstream statistics like Watterson's θ that need genome-wide
segregating-site counts.

If `L < sum-of-input-counts`, the imputer raises an error: that combination
implies a smaller genome than your input file already represents, which is
nonsensical.

## Input format

The imputer reads SFS files produced by the bundled `vcf-to-sfs` script
(installed alongside `sfs-impute`; source at `src/sfs_imputation/vcf_to_sfs.py`,
which is an example template — modify it for your VCF format). Each file has
a small header followed by 4-column rows `m  count  n1  n2`:

```
Data
Input_data: <pop name>
Data_format: SFS
Alleles polarized: true       # true = unfolded (polarized), false = folded
Total sequence length (L): 7.35864277E8
Total number of sequences (full sample size): 3472
[
4   1   757   2711
0   1   612   2860
...
]
```

- `m` = missing chromosomes at this site
- `count` = number of sites with this exact `(m, n1, n2)`
- `n1`, `n2` = observed allele counts; `n1 + n2 = N - m`

### Producing input from a VCF (template — adapt to your data)

The bundled `vcf-to-sfs` command converts a VCF (or `.vcf.gz`) into the
format above. **It is an example/template, not a universal converter.** It
assumes per-population `AC_<pop>`/`AN_<pop>` INFO fields, single-line records,
and `FILTER == PASS` for inclusion — conventions common in gnomAD-style files.

VCFs from other sources may use different INFO names, multi-allelic splits,
or per-sample genotype counts instead of summary tags. If your VCF differs,
edit the parsing in `src/sfs_imputation/vcf_to_sfs.py` to match. The
`process_vcfs` function is small and well-commented for that purpose.

```bash
vcf-to-sfs my_calls.vcf.gz \
    --pops ASJ,FIN \
    --total-an 6944,21264 \
    --total-length 735864277 \
    --out-stem mydata
# produces: mydata.ASJ.SFS  mydata.FIN.SFS  (one pass through the VCF)
```

`--total-an` is the **full-sample chromosome count per population**
(2 × number of diploid samples in that population). It must match the
maximum AN you'd see if every sample were called.

Default filters: PASS-only, SNPs only (single-base REF/ALT), include
monomorphic sites with their missing patterns. Each can be disabled
(`--no-pass-only`, `--no-snp-only`, `--no-monomorphic`). Add `--folded`
for folded output.

## Output format

`<input>.imputed` is a bare 4-column file, one row per polymorphic class,
descending order:

```
0    <imputed count at k=K>      <K>      <N-K>
0    <imputed count at k=K-1>    <K-1>    <N-K+1>
...
```

`K = N - 1` for unfolded; `K = ⌊N/2⌋` for folded. Counts are floats (the
expected number of sites with that derived allele count under the imputed
SFS).

## Recommendation: use unfolded SFS as input even when downstream uses folded

When ancestral alleles are known (e.g., from outgroup sequences), supply the
unfolded SFS to this imputer rather than pre-folding. The unfolded
representation has **twice as many decision variables** (`N` bins vs `N/2`),
which gives the EM solver more degrees of freedom to fit the observed
configs and yields **measurably more accurate imputation** — especially in
the rare-variant regime that drives most demographic and selection analyses.

If your downstream analysis (dadi/moments/fastsimcoal2/etc.) expects a
folded SFS, **fold the imputed output afterwards**, not the input. You can
either fold the `.imputed` file by hand (a few lines of Python) or use
`--project-to N_STAR` to produce a sample-size-projected output and then
fold that. Folding before imputation discards the ancestral-state
information you paid to determine.

## Reproducing the install

```bash
python -c "import sfs_imputation; print(sfs_imputation.__version__)"  # prints 1.2.3
sfs-impute --verbose example_input.SFS  # ~30-60s; produces example_input.SFS.imputed
head example_input.SFS.imputed  # first rows are highest-frequency derived classes
```

## Validation suite

Two complementary checks are bundled.

### 1. Quick unit/smoke tests (no extra deps)

Exercises core invariants — kernel correctness, EM solver simplex,
monomorphic augmentation, projection, plus an end-to-end CLI smoke
test on the bundled `example_input.SFS`. Runs with the stdlib test
runner:

```bash
python -m unittest discover -s tests -v
```

12 tests, ~10-15 seconds. Useful right after install or after
upgrading numpy/scipy/Python.

### 2. Real simulation-based end-to-end validation

Drives the full user-facing pipeline and scores the recovered SFS
against the ground truth from the simulation:

```
msprime coalescent + mutations
    -> ground-truth SFS (kept for comparison)
    -> diploid VCF (real GTs via tskit's write_vcf)
mask each genotype with prob miss_rate, recompute INFO/AC,AN
    -> VCF with missing data
vcf-to-sfs --pops sim --total-an N --total-length L
    -> .SFS file
sfs-impute --verbose
    -> .SFS.imputed file
score: L1, KL, Watterson's theta, pi, with PASS/FAIL thresholds
```

Requires `msprime` (declared as the `validate` extra):

```bash
pip install '.[validate]'                  # source-tree install
# OR install msprime separately if you installed from the wheel:
pip install msprime>=1.3
```

Then:

```bash
python tests/validate_e2e.py               # default: N=100, L=5 Mb
python tests/validate_e2e.py --n-diploid 100 --L 1e7   # bigger run
python tests/validate_e2e.py --miss-rate 0.20 --l1-threshold 0.45  # high-missingness stress test
```

Wall time: ~30-90 seconds on the default. Reports L1 between true and
imputed normalized SFS, KL divergence, and relative error in
Watterson's theta and pi. Exits 0 on PASS, 1 on FAIL.

## Programmatic API

```python
from sfs_imputation.io import read_sfs_file, write_imputed
from sfs_imputation.kernel import build_kernel
from sfs_imputation.solver_em import solve, solve_plain_em

cfg = read_sfs_file("input.SFS")
A, c, K = build_kernel(cfg.configs, N=cfg.N, folded=cfg.folded)

# For large samples (N >= 5000): SQUAREM-accelerated EM (default)
res = solve(A, c)

# For ecological samples (N < 200): plain EM (recommended)
res = solve_plain_em(A, c)

write_imputed(res.p, L=float(c.sum()), folded=cfg.folded, N=cfg.N,
              out_path="input.SFS.imputed")
```
