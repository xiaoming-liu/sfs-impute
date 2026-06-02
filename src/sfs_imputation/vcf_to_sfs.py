"""Convert a VCF (or gzipped VCF) into the 4-column SFS file format that the
sfs-impute CLI reads.

*** TEMPLATE / EXAMPLE — VCFs vary widely; adapt this script to your input. ***

This script makes assumptions about the VCF that match common gnomAD-style
files:
- Per-population AC_<pop> and AN_<pop> are present in the INFO column.
- The site is a single-line record (multi-allelic sites are rejected by the
  SNP-only filter; if you need them, split first or modify this script).
- FILTER == "PASS" identifies sites to keep.

Real VCFs from different sources may have different INFO field names
(AC/AN_pop_v3, ALLELE_FREQ_<pop>, AC1/AC2 for multiallelics, ...), different
filter conventions, phased vs unphased GTs that you'd rather count from
genotype calls than from INFO summaries, etc. **Treat this file as a starting
point and edit the parsing in `process_vcfs` to match your VCF's reality.**

Default filters (override with flags):
- PASS-only: keep only sites with FILTER == PASS
- SNP-only: keep only single-base REF and ALT
- Include monomorphic sites with their missing patterns (informative for
  imputing low-frequency variants)
"""

from __future__ import annotations

import argparse
import gzip
import sys
from collections import Counter
from pathlib import Path
from typing import IO, List, Sequence


def _open_vcf(path: str) -> IO[str]:
    """Open a VCF that may or may not be gzipped, returning a text stream."""
    p = Path(path)
    if p.suffix.lower() == ".gz":
        return gzip.open(p, mode="rt", encoding="utf-8")
    return open(p, mode="rt", encoding="utf-8")


def _parse_info(info_field: str) -> dict[str, str]:
    """Parse the VCF INFO field into a dict of key -> value."""
    out: dict[str, str] = {}
    for entry in info_field.split(";"):
        if "=" in entry:
            key, val = entry.split("=", 1)
            out[key] = val
        else:
            out[entry] = ""
    return out


def process_vcfs(
    vcf_paths: Sequence[str],
    pops: Sequence[str],
    total_an: Sequence[int],
    *,
    pass_only: bool = True,
    snp_only: bool = True,
    count_monomorphic: bool = True,
    folded: bool = False,
) -> dict[str, Counter]:
    """Process one or more VCF files and return per-population config counters.

    Returns
    -------
    dict mapping pop_id -> Counter of (m, n1, n2) -> count.
    """
    if len(pops) != len(total_an):
        raise ValueError(
            f"--pops has {len(pops)} entries but --total-an has {len(total_an)}; "
            "they must match"
        )

    counters: dict[str, Counter] = {p: Counter() for p in pops}

    for vcf_path in vcf_paths:
        n_lines = 0
        n_kept = 0
        with _open_vcf(vcf_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                n_lines += 1
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 8:
                    continue
                ref = fields[3]
                alt = fields[4]
                filt = fields[6]
                info = fields[7]

                if snp_only and (len(ref) != 1 or len(alt) != 1):
                    continue
                if pass_only and filt.upper() != "PASS":
                    continue

                info_dict = _parse_info(info)

                for pop, totan in zip(pops, total_an):
                    ac_key = f"AC_{pop}"
                    an_key = f"AN_{pop}"
                    if ac_key not in info_dict or an_key not in info_dict:
                        continue
                    try:
                        ac = int(info_dict[ac_key])
                        an = int(info_dict[an_key])
                    except ValueError:
                        continue

                    n_alt = ac
                    n_ref = an - ac
                    n_missing = totan - an

                    if n_missing < 0:
                        # AN exceeds the user-supplied total -- skip this site
                        continue
                    if n_ref < 0 or n_alt < 0:
                        continue

                    is_monomorphic = (n_ref == 0 or n_alt == 0)
                    if is_monomorphic and not count_monomorphic:
                        continue

                    if folded:
                        # Order so n1 >= n2 (major then minor)
                        if n_ref >= n_alt:
                            n1, n2 = n_ref, n_alt
                        else:
                            n1, n2 = n_alt, n_ref
                    else:
                        # Unfolded: n1 = ref count, n2 = alt count
                        n1, n2 = n_ref, n_alt

                    counters[pop][(n_missing, n1, n2)] += 1
                n_kept += 1
        print(f"  {vcf_path}: read {n_lines:,} variant lines, "
              f"kept {n_kept:,} after filters", file=sys.stderr)

    return counters


def write_sfs_file(
    counter: Counter,
    pop: str,
    total_an: int,
    total_length: float,
    folded: bool,
    out_path: str,
) -> None:
    """Write a per-population SFS file in the format read by sfs-impute."""
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("Data\n")
        out.write(f"Input_data: {pop}\n")
        out.write("Data_format: SFS\n")
        out.write(f"Alleles polarized: {'false' if folded else 'true'}\n")
        out.write(f"Total sequence length (L): {total_length}\n")
        out.write(f"Total number of sequences (full sample size): {total_an}\n")
        out.write("[\n")
        # Sort keys for stable output: by m then n1 then n2
        for (m, n1, n2) in sorted(counter.keys()):
            count = counter[(m, n1, n2)]
            out.write(f"{m}\t{count}\t{n1}\t{n2}\n")
        out.write("]\n")


def _comma_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def _comma_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",")]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="vcf-to-sfs",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("vcfs", nargs="+", help="One or more VCF files (.vcf or .vcf.gz)")
    p.add_argument("--pops", required=True, type=_comma_str_list,
                   help="Comma-separated population IDs matching INFO field "
                        "names AC_<pop> and AN_<pop> (e.g. ASJ,FIN)")
    p.add_argument("--total-an", required=True, type=_comma_int_list,
                   help="Comma-separated full-sample chromosome counts per "
                        "population (e.g. 6944,21264 = 2 * #diploid_samples)")
    p.add_argument("--total-length", required=True, type=float,
                   help="Total sequence length L (callable bp; recorded in header)")
    p.add_argument("--out-stem", required=True,
                   help="Output file stem; one file written per population as "
                        "<stem>.<pop>.SFS")
    p.add_argument("--folded", action="store_true",
                   help="Emit folded SFS (no ancestral-allele info needed)")
    p.add_argument("--no-pass-only", action="store_true",
                   help="Disable PASS-only filter (keep all FILTER values)")
    p.add_argument("--no-snp-only", action="store_true",
                   help="Disable SNP-only filter (keep multi-base REF/ALT)")
    p.add_argument("--no-monomorphic", action="store_true",
                   help="Drop monomorphic sites (default: include them with "
                        "their missing patterns, useful for imputing rare variants)")
    args = p.parse_args(argv)

    counters = process_vcfs(
        args.vcfs,
        pops=args.pops,
        total_an=args.total_an,
        pass_only=not args.no_pass_only,
        snp_only=not args.no_snp_only,
        count_monomorphic=not args.no_monomorphic,
        folded=args.folded,
    )

    for pop, totan in zip(args.pops, args.total_an):
        out_path = f"{args.out_stem}.{pop}.SFS"
        write_sfs_file(
            counters[pop], pop=pop, total_an=totan,
            total_length=args.total_length, folded=args.folded,
            out_path=out_path,
        )
        n_configs = len(counters[pop])
        n_sites = sum(counters[pop].values())
        print(f"wrote {out_path}: {n_configs:,} unique configs, "
              f"{n_sites:,} total sites", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
