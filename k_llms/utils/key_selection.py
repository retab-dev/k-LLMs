"""
Key Selection for JSON Alignment

Implements a cascade-based selection strategy to find the best single or composite
key for aligning lists of JSON records. Prioritizes stability (Jaccard similarity)
and coverage across multiple extractions.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel

# Type aliases for clarity
JSONScalar = Any
JSONPath = str


# --------------------- Normalization ---------------------

def normalize_scalar(value: Any) -> Any:
    """Normalize scalars for robust matching (lowercase, collapse whitespace)."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized
    return value


# --------------------- Value access ----------------------

# Configurable record list keys; can be overridden via CLI
RECORD_LIST_KEYS: List[str] = ["products"]

def iter_records(extraction: Dict[str, Any], list_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return list of record dicts from a specific list key.
    
    Args:
        extraction: JSON extraction dict
        list_key: Optional name of the key containing the list (e.g., "products", "production_table")
                  If None, uses fallback behavior (checks RECORD_LIST_KEYS, then auto-detect)
    
    Returns:
        List of record dicts
    """
    records: List[Dict[str, Any]] = []
    
    # If list_key is specified, use it directly
    if list_key is not None:
        seq = extraction.get(list_key)
        if isinstance(seq, list):
            for item in seq:
                if isinstance(item, dict):
                    records.append(item)
        return records
    
    # Fallback behavior: check RECORD_LIST_KEYS first
    for candidate_key in RECORD_LIST_KEYS:
        seq = extraction.get(candidate_key)
        if isinstance(seq, list):
            for item in seq:
                if isinstance(item, dict):
                    records.append(item)
    if records:
        return records
    
    # Auto-detect: find first list of dicts
    for value in extraction.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    records.append(item)
    return records

def values_for_path(extraction: Dict[str, Any], path: JSONPath, list_key: Optional[str] = None) -> List[JSONScalar]:
    """Return scalar values for a dot path across all products of one extraction."""
    parts = path.split(".")
    out: List[JSONScalar] = []
    products = iter_records(extraction, list_key=list_key)
    for record in products:
        if not isinstance(record, dict):
            continue
        cur: Any = record
        ok = True
        for token in parts:
            if isinstance(cur, dict) and token in cur:
                cur = cur[token]
            else:
                ok = False
                break
        if ok and cur is not None and not isinstance(cur, (dict, list)):
            out.append(normalize_scalar(cur))
    return out


def discover_scalar_paths(extractions: List[Dict[str, Any]], list_key: Optional[str] = None) -> List[JSONPath]:
    """Enumerate candidate dot paths that resolve to scalar values (ignores lists/dicts)."""
    candidates: Set[str] = set()
    for extraction in extractions:
        products = iter_records(extraction, list_key=list_key)
        for record in products:
            if not isinstance(record, dict):
                continue
            stack: List[Tuple[str, Any]] = [("", record)]
            while stack:
                base, node = stack.pop()
                if not isinstance(node, dict):
                    continue
                for key, value in node.items():
                    path = f"{base}.{key}" if base else key
                    if isinstance(value, dict):
                        stack.append((path, value))
                    elif isinstance(value, list):
                        continue  # do not consider list-valued paths as keys
                    else:
                        candidates.add(path)
    return sorted(candidates)


# --------------------- Metrics & scoring -----------------

def jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 1.0


class KeyMetrics(BaseModel):
    path: Tuple[str, ...]            # 1 path for single, >1 for composite
    coverage_min: float
    coverage_mean: float
    uniqueness_min: float
    uniqueness_mean: float
    jaccard_min: float
    jaccard_mean: float
    I_E: int                         # values present in all extractions
    I_E_minus_1: int                 # present in E-1 extractions
    I_ge_2: int                      # present in at least 2 extractions
    union_size: int
    score_tuple: Tuple               # lexicographic score used for ranking

    class Config:
        frozen = True  # Make the model immutable


def _evaluate_per_vals(extractions: List[Dict[str, Any]], per_vals: List[List[Any]], depth_hint: int, n_paths: int, list_key: Optional[str] = None) -> KeyMetrics:
    E = len(extractions)
    per_sets = [set(vs) for vs in per_vals]

    coverage: List[float] = []
    uniqueness: List[float] = []
    for vs, e in zip(per_vals, extractions):
        total = len(iter_records(e, list_key=list_key))
        non_null = len(vs)
        cov = non_null / max(1, total)
        coverage.append(cov)
        cnt = Counter(vs)
        uniq = sum(1 for _v, c in cnt.items() if c == 1)
        uniqueness.append(uniq / max(1, non_null) if non_null else 0.0)

    # inter-JSON stability
    j_scores: List[float] = []
    for i in range(E):
        for j in range(i + 1, E):
            j_scores.append(jaccard(per_sets[i], per_sets[j]))
    j_mean = sum(j_scores) / len(j_scores) if j_scores else 1.0
    j_min = min(j_scores) if j_scores else 1.0

    # support histogram
    support = Counter()
    for s in per_sets:
        for v in s:
            support[v] += 1
    counts_by_sup = Counter(support.values())
    I_E = counts_by_sup.get(E, 0)
    I_Em1 = counts_by_sup.get(E - 1, 0) if E >= 2 else 0
    I_2p = sum(c for sup, c in counts_by_sup.items() if sup >= 2)
    U = len(set().union(*per_sets)) if per_sets else 0

    # Stability-first lexicographic score (higher is better)
    score_tuple = (
        round(j_min, 6),              # 1) maximize worst-pair Jaccard
        I_E,                          # 2) maximize values present in all files
        I_Em1,                        # 3) then present in E-1 files
        round(j_mean, 6),             # 4) maximize mean Jaccard
        round(min(uniqueness), 6),    # 5) intra-JSON uniqueness (min)
        round(min(coverage), 6),      # 6) intra-JSON coverage (min)
        -U,                           # 7) discourage locality (large union)
        depth_hint,                   # 8) prefer deeper (tie-break)
        -n_paths,                     # 9) prefer fewer keys if still tie
    )

    return KeyMetrics(
        path=tuple(),  # filled by callers
        coverage_min=min(coverage) if coverage else 0.0,
        coverage_mean=sum(coverage) / len(coverage) if coverage else 0.0,
        uniqueness_min=min(uniqueness) if uniqueness else 0.0,
        uniqueness_mean=sum(uniqueness) / len(uniqueness) if uniqueness else 0.0,
        jaccard_min=j_min,
        jaccard_mean=j_mean,
        I_E=I_E,
        I_E_minus_1=I_Em1,
        I_ge_2=I_2p,
        union_size=U,
        score_tuple=score_tuple,
    )


def evaluate_single_key(extractions: List[Dict[str, Any]], path: JSONPath, list_key: Optional[str] = None) -> KeyMetrics:
    per_vals = [values_for_path(e, path, list_key=list_key) for e in extractions]
    m = _evaluate_per_vals(extractions, per_vals, depth_hint=path.count("."), n_paths=1, list_key=list_key)
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


def tuple_values_for_paths(extraction: Dict[str, Any], paths: List[JSONPath], list_key: Optional[str] = None) -> List[Tuple[Any, ...]]:
    out: List[Tuple[Any, ...]] = []
    parts_list = [p.split(".") for p in paths]
    products = iter_records(extraction, list_key=list_key)
    for record in products:
        if not isinstance(record, dict):
            continue
        tuple_components: List[Any] = []
        for parts in parts_list:
            current_value: Any = record
            path_resolved = True
            for part in parts:
                if isinstance(current_value, dict) and part in current_value:
                    current_value = current_value[part]
                else:
                    path_resolved = False
                    break
            if not path_resolved or current_value is None or isinstance(current_value, (dict, list)):
                tuple_components = []
                break
            tuple_components.append(normalize_scalar(current_value))
        if tuple_components:
            out.append(tuple(tuple_components))
    return out


def evaluate_composite_key(extractions: List[Dict[str, Any]], paths: List[JSONPath], list_key: Optional[str] = None) -> KeyMetrics:
    per_vals = [tuple_values_for_paths(e, paths, list_key=list_key) for e in extractions]
    m = _evaluate_per_vals(
        extractions, per_vals,
        depth_hint=sum(p.count(".") for p in paths),
        n_paths=len(paths),
        list_key=list_key
    )
    return KeyMetrics(
        path=tuple(paths),
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


# --------------------- Cascade (funnel) ------------------

class CascadeConfig(BaseModel):
    min_coverage: float = 0.0        # e.g., 0.6 if you want
    min_uniqueness: float = 0.0      # e.g., 0.2 to avoid constant keys
    topk_stage1: int = 30            # after stability sort
    topk_stage2: int = 12            # after intra-JSON sort
    topk_stage3: int = 6             # after union filter

    class Config:
        frozen = True


class CascadeReport(BaseModel):
    stage0_kept: List[KeyMetrics]
    stage1_kept: List[KeyMetrics]
    stage2_kept: List[KeyMetrics]
    stage3_kept: List[KeyMetrics]
    final_best: KeyMetrics

    class Config:
        frozen = True


def cascade_select_keys(
    extractions: List[Dict[str, Any]],
    candidates: List[str],
    config: CascadeConfig = CascadeConfig(),
    list_key: Optional[str] = None
) -> CascadeReport:
    # Evaluate singles
    singles: List[KeyMetrics] = [evaluate_single_key(extractions, p, list_key=list_key) for p in candidates]

    # -------- Stage 0: gating (generic, no blacklist)
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
        raise ValueError("No keys pass Stage 0 (require I_ge_2>0, jaccard_min>0, and coverage).")

    # -------- Stage 1: stability-first
    pool1 = sorted(
        pool0,
        key=lambda m: (m.I_E, m.I_E_minus_1, round(m.jaccard_min, 6), round(m.jaccard_mean, 6)),
        reverse=True
    )[:config.topk_stage1]

    # -------- Stage 2: intra-JSON quality
    pool2 = sorted(
        pool1,
        key=lambda m: (round(m.uniqueness_min, 6), round(m.coverage_min, 6)),
        reverse=True
    )[:config.topk_stage2]

    # -------- Stage 3: parsimony/globality
    pool3 = sorted(
        pool2,
        key=lambda m: (m.union_size, ),  # smaller union is better
        reverse=False
    )[:config.topk_stage3]

    # -------- Stage 4: tie-breakers
    final_sorted = sorted(
        pool3,
        key=lambda m: (sum(p.count(".") for p in m.path), -len(m.path)),  # deeper, fewer paths
        reverse=True
    )
    final_best = final_sorted[0]

    return CascadeReport(
        stage0_kept=pool0,
        stage1_kept=pool1,
        stage2_kept=pool2,
        stage3_kept=pool3,
        final_best=final_best
    )


# --------------------- Selection API (wraps cascade) -----

class KeySelectionResult(BaseModel):
    best_single: KeyMetrics
    best_composite: Optional[KeyMetrics]
    candidate_table: List[KeyMetrics]
    min_support_for_autolock: int
    cascade_report: CascadeReport

    class Config:
        frozen = True


def select_best_keys(
    extractions: List[Dict[str, Any]],
    max_candidates_for_composite: int = 20,
    max_k: int = 3,
    min_support_ratio_for_autolock: float = 0.75,
    cascade_cfg: CascadeConfig = CascadeConfig(),
    list_key: Optional[str] = None
) -> KeySelectionResult:
    if not extractions:
        raise ValueError("No extractions provided.")

    E = len(extractions)
    t = max(2, math.ceil(min_support_ratio_for_autolock * E))

    # 1) discover candidates
    candidates = discover_scalar_paths(extractions, list_key=list_key)
    if not candidates:
        raise ValueError("No scalar candidate paths discovered.")

    # 2) cascade funnel on singles
    report = cascade_select_keys(extractions, candidates, cascade_cfg, list_key=list_key)
    best_single = report.final_best

    # Build a sorted candidate table (for diagnostics)
    singles_all: List[KeyMetrics] = [evaluate_single_key(extractions, p, list_key=list_key) for p in candidates]
    singles_all = [m for m in singles_all if (m.I_ge_2 > 0 and m.jaccard_min > 0.0)]
    singles_all.sort(key=lambda m: (round(m.jaccard_min, 6), m.I_E, m.I_E_minus_1, round(m.jaccard_mean, 6),
                                    round(m.uniqueness_min, 6), round(m.coverage_min, 6), -m.union_size), reverse=True)

    # 3) greedy composite search from Stage-3 pool; accept only if stability improves
    topN_paths = [m.path[0] for m in report.stage3_kept][:max_candidates_for_composite]
    def stability_tuple(m: KeyMetrics) -> Tuple:
        return (round(m.jaccard_min, 6), m.I_E, m.I_E_minus_1, round(m.jaccard_mean, 6))

    best_combo: Optional[KeyMetrics] = None
    if topN_paths:
        current = [topN_paths[0]]
        best_combo = evaluate_composite_key(extractions, current, list_key=list_key)
        improved = True
        while improved and len(current) < max_k:
            improved = False
            for cand in (p for p in topN_paths if p not in current):
                trial = evaluate_composite_key(extractions, current + [cand], list_key=list_key)
                if trial.score_tuple > best_combo.score_tuple and stability_tuple(trial) > stability_tuple(best_combo):
                    best_combo = trial
                    current.append(cand)
                    improved = True

        # Brute-force fallback over topN_paths to explore 2..max_k composites
        for r in range(2, min(max_k, len(topN_paths)) + 1):
            for combo in combinations(topN_paths, r):
                trial = evaluate_composite_key(extractions, list(combo), list_key=list_key)
                # Accept if stability strictly improves or score improves
                if stability_tuple(trial) > stability_tuple(best_combo) or trial.score_tuple > best_combo.score_tuple:
                    best_combo = trial

    return KeySelectionResult(
        best_single=best_single,
        best_composite=best_combo,
        candidate_table=singles_all,
        min_support_for_autolock=t,
        cascade_report=report,
    )
