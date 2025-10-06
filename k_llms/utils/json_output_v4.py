"""
json_output_v4 - Key selection with optional fuzzy fallback (numeric bucketing)

Goal:
- Keep the existing v3 cascade unchanged for normal selection
- If no key passes Stage 0 OR if fuzzy improves stability, use a fuzzy pass
- Fuzzy pass buckets close scalar values (notably numerics) before metrics

Notes:
- Dates remain plain strings (no parsing)
- Lists/dicts are NOT considered as keys (same as v3). Only scalar paths
- This module depends on json_output_v3_recursion_1 for core types/metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import math

# Reuse v3 components
from .json_output_v3_recursion_1 import ( 
    CascadeConfig,
    KeyMetrics,
    KeySelectionResult,
    discover_scalar_paths,
    values_for_path,
    _evaluate_per_vals,  # internal, but OK for controlled reuse
    select_best_keys as v3_select_best_keys,
)


def _normalize_string(value: str) -> str:
    s = value.strip().lower()
    # collapse whitespace
    s = " ".join(s.split())
    return s


def _canonicalize_scalar(value: Any, numeric_round_decimals: int) -> Any:
    """Canonicalize a scalar for fuzzy grouping.

    - Numeric: round to configured decimals
    - String: normalized lower/trim/collapse
    - Bool/None: returned as-is
    - Others: returned as-is (kept scalar-only upstream)
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return round(float(value), numeric_round_decimals)
        except Exception:
            return value
    if isinstance(value, str):
        return _normalize_string(value)
    return value


def _evaluate_single_key_fuzzy(
    extractions: List[Dict[str, Any]],
    path: str,
    list_key: Optional[str],
    numeric_round_decimals: int,
) -> KeyMetrics:
    """Evaluate a single scalar key with fuzzy canonicalization before metrics."""
    # Per-extraction scalar values for this path
    per_vals_raw: List[List[Any]] = [values_for_path(e, path, list_key=list_key) for e in extractions]

    # Apply canonicalization to each scalar value
    per_vals_canon: List[List[Any]] = []
    for vals in per_vals_raw:
        per_vals_canon.append([_canonicalize_scalar(v, numeric_round_decimals) for v in vals])

    # Reuse v3 metric evaluator on canonicalized values
    m = _evaluate_per_vals(
        extractions,
        per_vals_canon,
        depth_hint=path.count("."),
        n_paths=1,
        list_key=list_key,
    )

    # Build a KeyMetrics with this path
    return KeyMetrics(
        path=(path,),
        coverage_min=m.coverage_min,
        coverage_mean=m.coverage_mean,
        uniqueness_min=m.uniqueness_min,
        uniqueness_mean=m.uniqueness_mean,
        jaccard_min=m.jaccard_min,
        jaccard_mean=m.jaccard_mean,
        I_E=m.I_E,
        I_E_minus_1=m.I_E_minus_1,
        I_ge_2=m.I_ge_2,
        union_size=m.union_size,
        score_tuple=m.score_tuple,
    )


def _stability_tuple(m: KeyMetrics) -> Tuple:
    return (round(m.jaccard_min, 6), m.I_E, m.I_E_minus_1, round(m.jaccard_mean, 6))


def _cascade_select_keys_fuzzy(
    extractions: List[Dict[str, Any]],
    candidates: List[str],
    config: CascadeConfig,
    list_key: Optional[str],
    numeric_round_decimals: int,
) -> KeyMetrics:
    """Run a fuzzy single-key cascade selection (singles only).

    Returns the best single KeyMetrics after cascade. Raises ValueError if none pass Stage 0.
    """
    # Evaluate singles (fuzzy)
    singles: List[KeyMetrics] = [
        _evaluate_single_key_fuzzy(extractions, p, list_key, numeric_round_decimals)
        for p in candidates
    ]

    # Stage 0: gating (same as v3)
    pool0 = [
        m for m in singles
        if (
            m.I_ge_2 > 0 and
            m.jaccard_min > 0.0 and
            m.coverage_min >= config.min_coverage and
            m.uniqueness_min >= config.min_uniqueness
        )
    ]
    if not pool0:
        raise ValueError("No keys pass Stage 0 (fuzzy)")

    # Stage 1: stability-first
    pool1 = sorted(
        pool0,
        key=lambda m: (m.I_E, m.I_E_minus_1, round(m.jaccard_min, 6), round(m.jaccard_mean, 6)),
        reverse=True,
    )[: config.topk_stage1]

    # Stage 2: intra-JSON quality
    pool2 = sorted(
        pool1,
        key=lambda m: (round(m.uniqueness_min, 6), round(m.coverage_min, 6)),
        reverse=True,
    )[: config.topk_stage2]

    # Stage 3: parsimony/globality
    pool3 = sorted(
        pool2,
        key=lambda m: (m.union_size,),  # smaller union is better
        reverse=False,
    )[: config.topk_stage3]

    # Stage 4: tie-breakers (prefer deeper, fewer paths)
    final_sorted = sorted(
        pool3,
        key=lambda m: (sum(p.count(".") for p in m.path), -len(m.path)),
        reverse=True,
    )
    return final_sorted[0]


@dataclass(frozen=True)
class SelectionComparison:
    normal_best: Optional[KeyMetrics]
    fuzzy_best: Optional[KeyMetrics]
    chosen: str  # "normal" | "fuzzy"


def select_best_keys_v4(
    extractions: List[Dict[str, Any]],
    cascade_cfg: CascadeConfig = CascadeConfig(),
    list_key: Optional[str] = None,
    fuzzy_numeric_round_decimals: int = 2,
    enable_fuzzy_fallback: bool = True,
    prefer_fuzzy_if_better: bool = True,
) -> SelectionComparison:
    """Run normal v3 selection first. If it fails or fuzzy is better, use fuzzy singles.

    Returns a SelectionComparison summarizing both and the chosen one.
    """
    normal_best: Optional[KeyMetrics] = None
    try:
        normal_result = v3_select_best_keys(
            extractions,
            cascade_cfg=cascade_cfg,
            list_key=list_key,
        )
        normal_best = normal_result.best_single
    except ValueError:
        normal_result = None

    fuzzy_best: Optional[KeyMetrics] = None
    if enable_fuzzy_fallback:
        candidates = discover_scalar_paths(extractions, list_key=list_key)
        if candidates:
            try:
                fuzzy_best = _cascade_select_keys_fuzzy(
                    extractions, candidates, cascade_cfg, list_key, fuzzy_numeric_round_decimals
                )
            except ValueError:
                fuzzy_best = None

    # Decision logic
    if normal_best is None and fuzzy_best is None:
        raise ValueError("No keys pass Stage 0 (normal or fuzzy)")

    if normal_best is not None and (not enable_fuzzy_fallback or fuzzy_best is None):
        return SelectionComparison(normal_best=normal_best, fuzzy_best=None, chosen="normal")

    if normal_best is None and fuzzy_best is not None:
        return SelectionComparison(normal_best=None, fuzzy_best=fuzzy_best, chosen="fuzzy")

    # Both exist: choose by stability if prefer_fuzzy_if_better
    if prefer_fuzzy_if_better and fuzzy_best is not None and normal_best is not None:
        if _stability_tuple(fuzzy_best) > _stability_tuple(normal_best):
            return SelectionComparison(normal_best=normal_best, fuzzy_best=fuzzy_best, chosen="fuzzy")
    return SelectionComparison(normal_best=normal_best, fuzzy_best=fuzzy_best, chosen="normal")


def main() -> None:
    """CLI: print selection comparison for quick manual checks."""
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="json_output_v4 - key selection with fuzzy fallback")
    ap.add_argument("inputs", nargs="+", help="JSON files to analyze")
    ap.add_argument("--list-key", help="Top-level list key to iterate records (optional)")
    ap.add_argument("--min-coverage", type=float, default=0.0)
    ap.add_argument("--min-uniqueness", type=float, default=0.0)
    ap.add_argument("--fuzzy-decimals", type=int, default=2)
    ap.add_argument("--no-fallback", action="store_true")
    args = ap.parse_args()

    files = [Path(p) for p in args.inputs]
    extractions: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            extractions.append(json.load(f))

    cfg = CascadeConfig(min_coverage=args.min_coverage, min_uniqueness=args.min_uniqueness)
    comp = select_best_keys_v4(
        extractions,
        cascade_cfg=cfg,
        list_key=args.list_key,
        fuzzy_numeric_round_decimals=args.fuzzy_decimals,
        enable_fuzzy_fallback=not args.no_fallback,
        prefer_fuzzy_if_better=True,
    )

    def fmt(m: Optional[KeyMetrics]) -> Optional[Dict[str, Any]]:
        if m is None:
            return None
        return {
            "path": list(m.path),
            "coverage_min": m.coverage_min,
            "uniqueness_min": m.uniqueness_min,
            "jaccard_min": m.jaccard_min,
            "I_E": m.I_E,
            "I_E_minus_1": m.I_E_minus_1,
            "I_ge_2": m.I_ge_2,
            "union_size": m.union_size,
        }

    out = {
        "chosen": comp.chosen,
        "normal_best": fmt(comp.normal_best),
        "fuzzy_best": fmt(comp.fuzzy_best),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


