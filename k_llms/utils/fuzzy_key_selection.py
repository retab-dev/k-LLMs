"""
Fuzzy Key Selection

Extends the standard key selection with fuzzy matching fallback for improved
stability when dealing with near-identical values (e.g., 1.29 vs 1.30).

Features:
- Buckets similar numeric values via rounding
- Normalizes strings (lowercase, whitespace collapse)
- Falls back to fuzzy matching if standard selection fails or if it improves stability

Note: Only applies to scalar paths. Lists/dicts are not considered as keys.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

from .key_selection import (
    CascadeConfig,
    KeyMetrics,
    discover_scalar_paths,
    values_for_path,
    _evaluate_per_vals,
    select_best_keys as select_best_keys_standard,
)


def _normalize_string(value: str) -> str:
    """Normalize a string for fuzzy matching (lowercase, trim, collapse whitespace)."""
    normalized = value.strip().lower()
    # Collapse whitespace
    normalized = " ".join(normalized.split())
    return normalized


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


class SelectionComparison(BaseModel):
    """Comparison result between standard and fuzzy key selection runs.

    - normal_best: best key metrics from standard selection (None if failed)
    - fuzzy_best: best key metrics from fuzzy selection (None if disabled/failed)
    - chosen: which strategy is chosen ("normal" | "fuzzy")
    """
    normal_best: Optional[KeyMetrics]
    fuzzy_best: Optional[KeyMetrics]
    chosen: str  # "normal" | "fuzzy"

    class Config:
        frozen = True


def select_best_keys_with_fuzzy_fallback(
    extractions: List[Dict[str, Any]],
    cascade_cfg: CascadeConfig = CascadeConfig(),
    list_key: Optional[str] = None,
    fuzzy_numeric_round_decimals: int = 2,
    enable_fuzzy_fallback: bool = True,
    prefer_fuzzy_if_better: bool = True,
) -> SelectionComparison:
    """Run standard selection, then optionally a fuzzy fallback, and compare.

    Args:
        extractions: list of extraction dicts
        cascade_cfg: cascade configuration for selection
        list_key: top-level list key to iterate records (optional)
        fuzzy_numeric_round_decimals: rounding precision for numeric canonicalization
        enable_fuzzy_fallback: if True, evaluate fuzzy singles when standard fails
        prefer_fuzzy_if_better: if True, choose fuzzy if its stability improves

    Returns:
        SelectionComparison: summary of both strategies and which one is chosen
    """
    normal_best: Optional[KeyMetrics] = None
    try:
        standard_result = select_best_keys_standard(
            extractions,
            cascade_cfg=cascade_cfg,
            list_key=list_key,
        )
        normal_best = standard_result.best_single
    except ValueError:
        normal_best = None

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



