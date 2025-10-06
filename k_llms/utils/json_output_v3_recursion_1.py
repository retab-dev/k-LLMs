# key_selection.py  — cascade selection, stability-first, no blacklist
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel

Scalar = Any
Path = str


# --------------------- Normalization ---------------------

def normalize_scalar(x: Any) -> Any:
    """Normalize scalars for robust matching (lowercase, collapse whitespace)."""
    if isinstance(x, str):
        s = x.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s
    return x


# --------------------- Utilities -------------------------

def freeze(value: Any) -> Any:
    """Make values hashable for set/Counter usage."""
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def count_number_of_products(data: Dict[str, Any]) -> int:
    products = data.get("products", [])
    return len(products) if isinstance(products, list) else 0


def flatten_obj(obj: Any, parent_key: str = "") -> Dict[str, Any]:
    """Flatten dicts into dot paths; lists are kept as a whole (not exploded)."""
    flat: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                flat.update(flatten_obj(v, new_key))
            elif isinstance(v, list):
                flat[new_key] = v
            else:
                flat[new_key] = v
    else:
        if parent_key:
            flat[parent_key] = obj
    return flat


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
    collected: List[Dict[str, Any]] = []
    
    # If list_key is specified, use it directly
    if list_key is not None:
        seq = extraction.get(list_key)
        if isinstance(seq, list):
            for item in seq:
                if isinstance(item, dict):
                    collected.append(item)
        return collected
    
    # Fallback behavior: check RECORD_LIST_KEYS first
    for key in RECORD_LIST_KEYS:
        seq = extraction.get(key)
        if isinstance(seq, list):
            for item in seq:
                if isinstance(item, dict):
                    collected.append(item)
    if collected:
        return collected
    
    # Auto-detect: find first list of dicts
    for v in extraction.values():
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    collected.append(item)
    return collected

def values_for_path(extraction: Dict[str, Any], path: Path, list_key: Optional[str] = None) -> List[Scalar]:
    """Return scalar values for a dot path across all products of one extraction."""
    parts = path.split(".")
    out: List[Scalar] = []
    products = iter_records(extraction, list_key=list_key)
    for prod in products:
        if not isinstance(prod, dict):
            continue
        cur: Any = prod
        ok = True
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok and cur is not None and not isinstance(cur, (dict, list)):
            out.append(normalize_scalar(cur))
    return out


def discover_scalar_paths(extractions: List[Dict[str, Any]], list_key: Optional[str] = None) -> List[Path]:
    """Enumerate candidate dot paths that resolve to scalar values (ignores lists/dicts)."""
    candidates: Set[str] = set()
    for data in extractions:
        products = iter_records(data, list_key=list_key)
        for prod in products:
            if not isinstance(prod, dict):
                continue
            stack: List[Tuple[str, Any]] = [("", prod)]
            while stack:
                base, node = stack.pop()
                if not isinstance(node, dict):
                    continue
                for k, v in node.items():
                    path = f"{base}.{k}" if base else k
                    if isinstance(v, dict):
                        stack.append((path, v))
                    elif isinstance(v, list):
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


def evaluate_single_key(extractions: List[Dict[str, Any]], path: Path, list_key: Optional[str] = None) -> KeyMetrics:
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


def tuple_values_for_paths(extraction: Dict[str, Any], paths: List[Path], list_key: Optional[str] = None) -> List[Tuple[Any, ...]]:
    out: List[Tuple[Any, ...]] = []
    parts_list = [p.split(".") for p in paths]
    products = iter_records(extraction, list_key=list_key)
    for prod in products:
        if not isinstance(prod, dict):
            continue
        t: List[Any] = []
        ok = True
        for parts in parts_list:
            cur: Any = prod
            for part in parts:
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if not ok or cur is None or isinstance(cur, (dict, list)):
                ok = False
                break
            t.append(normalize_scalar(cur))
        if ok:
            out.append(tuple(t))
    return out


def evaluate_composite_key(extractions: List[Dict[str, Any]], paths: List[Path], list_key: Optional[str] = None) -> KeyMetrics:
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


# --------------------- Autolock (support-wise) ----------

def autolock_whitelist_single(
    extractions: List[Dict[str, Any]],
    path: Path,
    min_support: int,
    require_unique_per_extraction: bool = True,
) -> Set[Any]:
    per_vals = [values_for_path(e, path) for e in extractions]
    per_sets = [set(vs) for vs in per_vals]
    support = Counter()
    for s in per_sets:
        for v in s:
            support[v] += 1
    per_counts = [Counter(vs) for vs in per_vals]

    white: Set[Any] = set()
    for v, sup in support.items():
        if sup >= min_support:
            if not require_unique_per_extraction:
                white.add(v)
            else:
                ok = True
                for cnt in per_counts:
                    if v in cnt and cnt[v] > 1:
                        ok = False
                        break
                if ok:
                    white.add(v)
    return white


def autolock_whitelist_composite(
    extractions: List[Dict[str, Any]],
    paths: Tuple[Path, ...],
    min_support: int,
    require_unique_per_extraction: bool = True,
) -> Set[Tuple[Any, ...]]:
    per_vals = [tuple_values_for_paths(e, list(paths)) for e in extractions]
    per_sets = [set(vs) for vs in per_vals]
    support = Counter()
    for s in per_sets:
        for v in s:
            support[v] += 1
    per_counts = [Counter(vs) for vs in per_vals]

    white: Set[Tuple[Any, ...]] = set()
    for v, sup in support.items():
        if sup >= min_support:
            if not require_unique_per_extraction:
                white.add(v)
            else:
                ok = True
                for cnt in per_counts:
                    if v in cnt and cnt[v] > 1:
                        ok = False
                        break
                if ok:
                    white.add(v)
    return white


# --------------------- Alignment helpers ----------------

def key_value(product: Dict[str, Any], paths: Tuple[str, ...]) -> Optional[Tuple[Any, ...]]:
    """Return normalized tuple key for a product; None if any part missing/non-scalar."""
    out = []
    for p in paths:
        cur = product
        ok = True
        for part in p.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if not ok or cur is None or isinstance(cur, (dict, list)):
            return None
        out.append(normalize_scalar(cur))
    return tuple(out)

def index_by_key(extractions: List[Dict[str, Any]], paths: Tuple[str, ...]):
    """Build per-extraction index: key -> list of product indices (can be >1 if duplicates)."""
    indexes = []
    for e in extractions:
        mapping = defaultdict(list)
        products = iter_records(e)
        for i, prod in enumerate(products or []):
            kv = key_value(prod, paths)
            if kv is not None:
                mapping[kv].append(i)
        indexes.append(mapping)
    return indexes

def build_autolock_rows(
    extractions: List[Dict[str, Any]],
    paths: Tuple[str, ...],
    min_support: int,
):
    """Create alignment rows for whitelist values (support >= min_support)."""
    idxs = index_by_key(extractions, paths)
    support = defaultdict(int)
    for m in idxs:
        for kv in m.keys():
            support[kv] += 1
    locked_keys = {kv for kv, sup in support.items() if sup >= min_support}

    rows = []
    for kv in sorted(locked_keys):  # deterministic
        row = []
        for m in idxs:
            # autolock requires uniqueness per extraction; if duplicates, leave None to resolve later
            if kv in m and len(m[kv]) == 1:
                row.append(m[kv][0])
            else:
                row.append(None)
        rows.append((kv, row))
    return rows, locked_keys

def residual_buckets(extractions: List[Dict[str, Any]], paths: Tuple[str, ...], locked_keys: set):
    """Group remaining products by key for second-phase matching."""
    idxs = index_by_key(extractions, paths)
    buckets = defaultdict(list)
    for f_idx, m in enumerate(idxs):
        for kv, lst in m.items():
            if kv in locked_keys:
                continue
            for i in lst:
                buckets[kv].append((f_idx, i))
    return buckets


# --------------------- Support diagnostics ---------------

def compute_I_ge_t(extractions: List[Dict[str, Any]], paths: Tuple[str, ...], t: int) -> int:
    """Count how many key values have support >= t across extractions for given paths."""
    if not paths:
        return 0
    # Build per-extraction sets of values
    per_sets: List[Set[Any]] = []
    if len(paths) == 1:
        for e in extractions:
            per_sets.append(set(values_for_path(e, paths[0])))
    else:
        for e in extractions:
            per_sets.append(set(tuple_values_for_paths(e, list(paths))))
    # Aggregate support
    support = Counter()
    for s in per_sets:
        for v in s:
            support[v] += 1
    return sum(1 for _v, sup in support.items() if sup >= t)


# --------------------- IO / CLI -------------------------

def load_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        # Allow any configured record list keys, so skip strict products-list check
        return data
    except Exception:
        return None


def collect_files(paths_or_dirs: List[str]) -> List[str]:
    files: List[str] = []
    for p in paths_or_dirs:
        if os.path.isdir(p):
            for root, _dirs, fnames in os.walk(p):
                for fn in fnames:
                    if fn.lower().endswith(".json"):
                        files.append(os.path.join(root, fn))
        elif os.path.isfile(p) and p.lower().endswith(".json"):
            files.append(p)
    return sorted(set(files))


def main() -> None:
    ap = argparse.ArgumentParser(description="Select best key (single/composite) for aligning extractions (cascade).")
    ap.add_argument("inputs", nargs="+", help="JSON files or directories containing JSON files.")
    ap.add_argument("--topN", type=int, default=20, help="Top-N single keys (from Stage-3) to explore for composite.")
    ap.add_argument("--maxK", type=int, default=3, help="Max number of keys in composite.")
    ap.add_argument("--min-support-ratio", type=float, default=0.75, help="Support ratio for autolock (ceil(ratio*E)).")
    ap.add_argument("--no-unique-check", action="store_true", help="Do not require uniqueness per extraction for autolock.")
    ap.add_argument("--min-coverage", type=float, default=0.0, help="Optional minimum coverage gate in Stage 0 (e.g., 0.6).")
    ap.add_argument("--min-uniqueness", type=float, default=0.0, help="Optional minimum uniqueness gate in Stage 0 (e.g., 0.2).")
    ap.add_argument("--list-key", action="append", help="Record list key to use (can be repeated). Defaults to 'products'.")
    ap.add_argument("--stage1", type=int, default=30, help="Top-K after stability stage.")
    ap.add_argument("--stage2", type=int, default=12, help="Top-K after intra-JSON stage.")
    ap.add_argument("--stage3", type=int, default=6, help="Top-K after union stage.")
    args = ap.parse_args()

    filepaths = collect_files(args.inputs)
    if not filepaths:
        print("No JSON files found.")
        return

    # configure record list keys
    global RECORD_LIST_KEYS
    if args.list_key:
        RECORD_LIST_KEYS = args.list_key
    else:
        RECORD_LIST_KEYS = ["products"]

    extractions: List[Dict[str, Any]] = []
    for fp in filepaths:
        data = load_json_file(fp)
        if data is None:
            print(f"[skip] {fp} is not a valid extraction (dict with products list).")
            continue
        extractions.append(data)

    if not extractions:
        print("No valid extractions loaded.")
        return

    print(f"Loaded {len(extractions)} extractions from {len(filepaths)} files.")

    cascade_cfg = CascadeConfig(
        min_coverage=args.min_coverage,
        min_uniqueness=args.min_uniqueness,
        topk_stage1=args.stage1,
        topk_stage2=args.stage2,
        topk_stage3=args.stage3,
    )

    # Selection (with cascade + greedy composite)
    try:
        result = select_best_keys(
            extractions,
            max_candidates_for_composite=args.topN,
            max_k=args.maxK,
            min_support_ratio_for_autolock=args.min_support_ratio,
            cascade_cfg=cascade_cfg,
        )
    except ValueError as e:
        print(f"\n[selection failed] {e}")
        return

    # Report cascade stages (compact)
    print("\n=== Cascade report ===")
    print(f"Stage0 kept: {len(result.cascade_report.stage0_kept)}")
    print(f"Stage1 kept: {len(result.cascade_report.stage1_kept)}")
    print(f"Stage2 kept: {len(result.cascade_report.stage2_kept)}")
    print(f"Stage3 kept: {len(result.cascade_report.stage3_kept)}")

    # Report best single
    bs = result.best_single
    print("\n=== Best single key ===")
    print("path:", ".".join(bs.path))
    t_support = result.min_support_for_autolock
    print(f"coverage_min={bs.coverage_min:.3f}, uniqueness_min={bs.uniqueness_min:.3f}, "
          f"jaccard_min={bs.jaccard_min:.3f}, I_E={bs.I_E}, I_ge_{t_support}={compute_I_ge_t(extractions, bs.path, t_support)}, union={bs.union_size}")

    # Report best composite
    if result.best_composite:
        bc = result.best_composite
        print("\n=== Best composite key ===")
        print("paths:", " + ".join(bc.path))
        print(f"coverage_min={bc.coverage_min:.3f}, uniqueness_min={bc.uniqueness_min:.3f}, "
              f"jaccard_min={bc.jaccard_min:.3f}, I_E={bc.I_E}, I_ge_{t_support}={compute_I_ge_t(extractions, bc.path, t_support)}, union={bc.union_size}")
    else:
        print("\n(no composite improved over the best single key)")

    # Autolock whitelists
    t = result.min_support_for_autolock
    require_unique = not args.no_unique_check

    white_single = autolock_whitelist_single(extractions, bs.path[0], min_support=t, require_unique_per_extraction=require_unique)
    print(f"\nAutolock whitelist (single) size={len(white_single)} (min_support={t}, unique_check={require_unique})")

    if result.best_composite and len(result.best_composite.path) > 1:
        white_combo = autolock_whitelist_composite(extractions, result.best_composite.path, min_support=t, require_unique_per_extraction=require_unique)
        print(f"Autolock whitelist (composite) size={len(white_combo)}")
    else:
        print("No composite whitelist (no multi-key composite chosen).")

    # (Optional) example: build initial aligned rows using autolock keys
    rows, locked_keys = build_autolock_rows(extractions, result.best_composite.path if result.best_composite else bs.path, t)
    print(f"\nPre-aligned rows via autolock: {len(rows)}")
    print(f"Residual buckets (keys not autolocked): {len(residual_buckets(extractions, result.best_composite.path if result.best_composite else bs.path, locked_keys))}")


if __name__ == "__main__":
    main()