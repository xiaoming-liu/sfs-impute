from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence, Tuple

Config = Tuple[int, int, int, int]


def augment(
    configs: Sequence[Config], *, N: int, L_total: float
) -> List[Config]:
    """Add synthetic monomorphic-reference configs to bring the input total
    up to L_total.

    Uses the missing-pattern distribution from ALL input configs (polymorphic
    and reported-monomorphic together) to distribute the synthetic
    L_total - sum(input_counts) sites across missing-count buckets.
    Synthetic configs are appended as monomorphic-reference: (m, count, N-m, 0).
    The last bucket absorbs the integer rounding remainder so totals balance.

    Underlying assumption: unreported monomorphic sites (the bulk of the
    callable genome that never appears in the VCF) have the same per-site
    missing-pattern distribution as the reported sites. Reasonable for
    standard variant-calling pipelines where per-sample call success is
    governed by read depth / base quality and is symmetric for poly and
    mono sites; can fail if monomorphic sites are filtered differently.

    Behavior:
    - If L_total == sum(input_counts), returns the input unchanged.
    - If L_total < sum(input_counts), raises ValueError.
    - Otherwise, appends synthetic monomorphic-reference configs.
    """
    by_m: "defaultdict[int, int]" = defaultdict(int)
    total_reported = 0
    for (m, count, _n1, _n2) in configs:
        by_m[m] += count
        total_reported += count
    if total_reported == 0:
        raise ValueError("Cannot augment when no input configs exist")
    L_to_add = int(L_total - total_reported)
    if L_to_add < 0:
        raise ValueError(
            f"L_total ({L_total}) is less than total reported sites "
            f"({total_reported}); cannot have negative monomorphic count"
        )
    if L_to_add == 0:
        return list(configs)
    sorted_ms = sorted(by_m.keys())
    out: List[Config] = list(configs)
    L_remaining = L_to_add
    for i, m in enumerate(sorted_ms):
        if i == len(sorted_ms) - 1:
            count_m = L_remaining
        else:
            count_m = int(L_to_add * by_m[m] / total_reported)
            L_remaining -= count_m
        if count_m > 0:
            out.append((m, count_m, N - m, 0))
    return out
