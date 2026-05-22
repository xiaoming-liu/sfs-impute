from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
Config = Tuple[int, int, int, int]  # (m, count, n1, n2)


@dataclass
class ObservedConfigs:
    N: int
    folded: bool
    hasmono: bool
    L_total: Optional[float]
    configs: List[Config]


def read_sfs_file(path: PathLike) -> ObservedConfigs:
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    N, folded, L_total, data_start = _parse_header(lines)
    configs = _parse_data_rows(lines, data_start, N)
    if folded:
        configs, swaps = _maybe_swap_folded(configs)
        if swaps > 0:
            warnings.warn(
                f"Folded input had {swaps} rows with n1<n2; swapped defensively",
                stacklevel=2,
            )
    hasmono = any(n1 == 0 or n2 == 0 for (_, _, n1, n2) in configs)
    if N is None and configs:
        N = configs[0][0] + configs[0][2] + configs[0][3]
    return ObservedConfigs(
        N=N, folded=folded, hasmono=hasmono, L_total=L_total, configs=configs
    )


def _parse_header(lines: List[str]):
    N: Optional[int] = None
    folded: Optional[bool] = None
    L_total: Optional[float] = None
    data_start = 0
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line == "[":
            data_start = i + 1
            break
        # Current header: "Known ancestral alleles: true/false"
        # Legacy header: "Has_an_ancestral/outgroup_sequence: true/false"
        # Both encode the same semantic: true = ancestral known => unfolded;
        # false = ancestral unknown => folded.
        if "Known ancestral alleles" in line or "Has_an_ancestral/outgroup_sequence" in line:
            folded = "false" in line.lower()
        elif "Total sequence leng" in line:  # matches "length" and legacy "lengh" typo
            try:
                L_total = float(line.split(":", 1)[1].strip())
            except (IndexError, ValueError):
                pass
        elif "Total number of sequences" in line:
            try:
                N = int(line.split(":", 1)[1].strip())
            except (IndexError, ValueError):
                pass
    if folded is None:
        raise ValueError(
            "Header missing 'Known ancestral alleles' line "
            "(or legacy 'Has_an_ancestral/outgroup_sequence')"
        )
    return N, folded, L_total, data_start


def _parse_data_rows(lines: List[str], start_idx: int, N: Optional[int]) -> List[Config]:
    configs: List[Config] = []
    for offset, raw in enumerate(lines[start_idx:], start=start_idx):
        line = raw.strip()
        if line in ("", "]"):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(
                f"Line {offset + 1}: expected 4 columns, got {len(parts)}: {line!r}"
            )
        try:
            m, count, n1, n2 = (int(x) for x in parts)
        except ValueError as e:
            raise ValueError(f"Line {offset + 1}: non-integer column: {line!r}") from e
        row_N = m + n1 + n2
        if N is None:
            N = row_N
        elif row_N != N:
            raise ValueError(
                f"Line {offset + 1}: m+n1+n2={row_N} != N={N}: {line!r}"
            )
        if not (0 <= n1 <= N and 0 <= n2 <= N):
            raise ValueError(
                f"Line {offset + 1}: n1={n1} or n2={n2} out of [0, {N}]: {line!r}"
            )
        configs.append((m, count, n1, n2))
    return configs


def _maybe_swap_folded(configs: List[Config]) -> Tuple[List[Config], int]:
    swapped: List[Config] = []
    swaps = 0
    for (m, count, n1, n2) in configs:
        if n1 < n2:
            n1, n2 = n2, n1
            swaps += 1
        swapped.append((m, count, n1, n2))
    return swapped, swaps


def write_imputed(
    p: "np.ndarray",
    *,
    L: float,
    folded: bool,
    N: int,
    out_path: PathLike,
) -> None:
    counts = np.asarray(p, dtype=float) * float(L)
    out_path = Path(out_path)
    upper = N // 2 if folded else N - 1
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for j in range(upper, 0, -1):
            f.write(f"0\t{counts[j]:.10g}\t{j}\t{N - j}\n")
