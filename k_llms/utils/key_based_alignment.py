#!/usr/bin/env python3
"""
Key-Based Alignment Engine

Recursively aligns JSON extractions using intelligent key selection.
Produces aligned outputs with consistent structure across all sources and
key mappings for traceability back to original source paths.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .key_selection import (
    CascadeConfig,
    select_best_keys,
)
from .fuzzy_key_selection import (
    select_best_keys_with_fuzzy_fallback,
)

# JSON-like nested type aliases (kept consistent with consensus_utils.py)
JSONScalar = str | int | float | bool | None
JSONLike = Dict[str, Any] | List[Any] | JSONScalar

# Verbose/file logging (CLI-controlled)
VERBOSE: bool = False
LOG_FILE: Optional[str] = None

def _log(msg: str) -> None:
    should_print = VERBOSE
    should_write = LOG_FILE is not None
    if not (should_print or should_write):
        return
    if should_print:
        print(msg)
    if should_write and LOG_FILE:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        except Exception:
            # best-effort logging; ignore file errors
            pass

# --------------------- Helper Functions ---------------------

def _get_key_tuple(
    obj: Dict[str, Any], paths: Tuple[str, ...]
) -> Optional[Tuple[Any, ...]]:
    """
    Extracts a tuple of values from a dictionary based on a tuple of key paths.
    Returns None if any path cannot be resolved.
    """
    values = []
    for path in paths:
        parts = path.split(".")
        current_value = obj
        path_resolved = True
        for part in parts:
            if isinstance(current_value, dict) and part in current_value:
                current_value = current_value[part]
            else:
                path_resolved = False
                break
        if not path_resolved or current_value is None or isinstance(current_value, (dict, list)):
            return None
        values.append(current_value)
    return tuple(values)


def _align_lists_by_key(
    lists_to_align: Sequence[Optional[List[Dict[str, Any]]]],
    key_paths: Tuple[str, ...],
) -> Tuple[List[List[Optional[Dict[str, Any]]]], List[List[Optional[int]]]]:
    """
    Aligns lists of dictionaries using a selected key.

    Args:
        lists_to_align: A list containing the lists to be aligned (one from each source).
        key_paths: The single or composite key path(s) to use for alignment.

    Returns:
        A tuple containing:
        - aligned_rows: A list of rows, where each row contains corresponding items.
        - original_indices: A mapping from [row_idx][col_idx] to the item's original index
          in its source list.
    """
    if not any(lists_to_align):
        return [], []

    # 1. Build an index for each list: {key_tuple -> original_index}
    #    And collect all unique key tuples across all lists.
    all_key_tuples = set()
    indexes = []
    for source_list in lists_to_align:
        mapping: Dict[Tuple[Any, ...], int] = {}
        if isinstance(source_list, list):
            for i, item in enumerate(source_list):
                if isinstance(item, dict):
                    key_tuple = _get_key_tuple(item, key_paths)
                    if key_tuple is not None and key_tuple not in mapping:
                        mapping[key_tuple] = i
                        all_key_tuples.add(key_tuple)
        indexes.append(mapping)

    # Establish an output order:
    # Prefer the order from the most complete source list, then append any remaining
    # keys in sorted order for determinism.
    def _safe_len(source_list: Optional[List[Dict[str, Any]]]) -> int:
        return len(source_list) if isinstance(source_list, list) else 0

    if indexes:
        best_source_idx = max(range(len(lists_to_align)), key=lambda i: _safe_len(lists_to_align[i]))
    else:
        best_source_idx = 0

    best_source_list = lists_to_align[best_source_idx] if best_source_idx < len(lists_to_align) else None
    ordered_keys: List[Tuple[Any, ...]] = []
    seen_keys: set = set()

    if isinstance(best_source_list, list):
        for item in best_source_list:
            if isinstance(item, dict):
                key_tuple = _get_key_tuple(item, key_paths)
                if key_tuple is not None and key_tuple not in seen_keys:
                    ordered_keys.append(key_tuple)
                    seen_keys.add(key_tuple)

    # Append any remaining keys (not present in best source order) in sorted order
    remaining_keys = sorted(all_key_tuples - seen_keys)
    ordered_keys.extend(remaining_keys)

    # 2. Build the aligned rows and original index mappings.
    aligned_rows: List[List[Optional[Dict[str, Any]]]] = []
    original_indices: List[List[Optional[int]]] = []

    for key_tuple in ordered_keys:
        row: List[Optional[Dict[str, Any]]] = []
        indices_row: List[Optional[int]] = []
        for source_idx, source_list in enumerate(lists_to_align):
            original_idx = indexes[source_idx].get(key_tuple)
            if original_idx is not None and isinstance(source_list, list):
                row.append(source_list[original_idx])
                indices_row.append(original_idx)
            else:
                row.append(None)
                indices_row.append(None)
        aligned_rows.append(row)
        original_indices.append(indices_row)

    return aligned_rows, original_indices


# --------------------- Core Recursive Alignment ---------------------

def _compute_key_aligned_structure(
    values: Sequence[Any],
    original_paths: Sequence[Optional[str]],
    cascade_cfg: CascadeConfig,
) -> Tuple[Any, Dict[str, List[Optional[str]]]]:
    """
    Internal core for recursive alignment using key-based matching.
    Keeps the original behavior and return types.
    """
    if not values or all(v is None for v in values):
        return None, {}

    non_nulls = [v for v in values if v is not None]
    if not non_nulls:
        return None, {}

    # Determine the type of the elements to be aligned
    first_type = type(non_nulls[0])
    is_same_type = all(isinstance(v, first_type) for v in non_nulls)
    key_mappings: Dict[str, List[Optional[str]]] = {}

    # --- Case 1: Base case (scalars, mixed types, or not dict/list) ---
    if not is_same_type or first_type not in (dict, list):
        # Use the first non-null value as the representative aligned value
        aligned_value = deepcopy(non_nulls[0])
        key_mappings[""] = list(original_paths)
        return aligned_value, key_mappings

    # --- Case 2: Aligning dictionaries ---
    if first_type is dict:
        dicts = [v if isinstance(v, dict) else {} for v in values]
        all_keys = sorted(set(key for d in dicts for key in d.keys()))

        aligned_dict: Dict[str, Any] = {}
        for key in all_keys:
            values_for_key = [d.get(key) for d in dicts]
            original_paths_for_key = [
                (f"{p}.{key}" if p else key) if p is not None else None
                for p in original_paths
            ]

            aligned_value, sub_mapping = _compute_key_aligned_structure(
                values_for_key, original_paths_for_key, cascade_cfg
            )
            aligned_dict[key] = aligned_value

            # Prepend the current key to the sub-paths in the mapping
            for sub_key, paths in sub_mapping.items():
                new_key = f"{key}.{sub_key}" if sub_key else key
                key_mappings[new_key] = paths

        return aligned_dict, key_mappings

    # --- Case 3: Aligning lists ---
    if first_type is list:
        lists = [v if isinstance(v, list) else [] for v in values]
        
        # Check if lists contain dicts suitable for key-based alignment
        is_list_of_dicts = all(
            all(isinstance(item, dict) for item in lst) for lst in lists if lst
        )

        if is_list_of_dicts:
            # Use dummy extractions to find the best alignment key for the current lists
            dummy_extractions = [{"items": lst} for lst in lists]
            try:
                # First: try standard selection (with composite support)
                result = select_best_keys(
                    dummy_extractions, list_key="items", cascade_cfg=cascade_cfg
                )
                key_paths_v3 = (
                    result.best_composite.path
                    if result.best_composite
                    and result.best_composite.score_tuple
                    > result.best_single.score_tuple
                    else result.best_single.path
                )
                # Prepare standard-selection metrics for logging
                v3_metrics = result.best_composite if (
                    result.best_composite and result.best_composite.score_tuple > result.best_single.score_tuple
                ) else result.best_single
                _log(
                    f"[KEY-SELECT standard] path={list(key_paths_v3)} jaccard_min={round(v3_metrics.jaccard_min,6)} "
                    f"coverage_min={round(v3_metrics.coverage_min,6)} uniqueness_min={round(v3_metrics.uniqueness_min,6)} "
                    f"I_E={v3_metrics.I_E} union_size={v3_metrics.union_size}"
                )
                
                # Then: try fuzzy and prefer if it improves stability
                try:
                    comp = select_best_keys_with_fuzzy_fallback(
                        dummy_extractions,
                        cascade_cfg=cascade_cfg,
                        list_key="items",
                        fuzzy_numeric_round_decimals=2,
                        enable_fuzzy_fallback=True,
                        prefer_fuzzy_if_better=True,
                    )
                    if comp.chosen == "fuzzy" and comp.fuzzy_best is not None:
                        key_paths = comp.fuzzy_best.path
                        m = comp.fuzzy_best
                        _log(
                            f"[KEY-SELECT fuzzy] chosen=fuzzy path={list(key_paths)} jaccard_min={round(m.jaccard_min,6)} "
                            f"coverage_min={round(m.coverage_min,6)} uniqueness_min={round(m.uniqueness_min,6)} "
                            f"I_E={m.I_E} union_size={m.union_size}"
                        )
                    else:
                        key_paths = key_paths_v3
                        m = v3_metrics
                        _log(
                            f"[KEY-SELECT fuzzy] chosen=standard path={list(key_paths_v3)} jaccard_min={round(m.jaccard_min,6)} "
                            f"coverage_min={round(m.coverage_min,6)} uniqueness_min={round(m.uniqueness_min,6)} "
                            f"I_E={m.I_E} union_size={m.union_size}"
                        )
                except Exception:
                    key_paths = key_paths_v3
                    m = v3_metrics
                    _log(
                        f"[KEY-SELECT fuzzy] error => fallback to standard path={list(key_paths_v3)} jaccard_min={round(m.jaccard_min,6)} "
                        f"coverage_min={round(m.coverage_min,6)} uniqueness_min={round(m.uniqueness_min,6)} "
                        f"I_E={m.I_E} union_size={m.union_size}"
                    )
                    
            except ValueError:
                # Standard failed; try fuzzy as last resort
                try:
                    comp = select_best_keys_with_fuzzy_fallback(
                        dummy_extractions,
                        cascade_cfg=cascade_cfg,
                        list_key="items",
                        fuzzy_numeric_round_decimals=2,
                        enable_fuzzy_fallback=True,
                        prefer_fuzzy_if_better=True,
                    )
                    chosen = comp.fuzzy_best if comp.chosen == "fuzzy" else comp.normal_best
                    key_paths = chosen.path if chosen is not None else None
                    if chosen is not None and key_paths is not None:
                        _log(
                            f"[KEY-SELECT v4-only] chosen={comp.chosen} path={list(key_paths)} jaccard_min={round(chosen.jaccard_min,6)} "
                            f"coverage_min={round(chosen.coverage_min,6)} uniqueness_min={round(chosen.uniqueness_min,6)} "
                            f"I_E={chosen.I_E} union_size={chosen.union_size}"
                        )
                except Exception:
                    key_paths = None # No suitable key found
                    _log("[KEY-SELECT] no key found (v3 failed, v4 failed)")

            if key_paths:
                aligned_rows, original_indices = _align_lists_by_key(lists, key_paths)

                aligned_list = []
                for i, row in enumerate(aligned_rows):
                    original_paths_for_row = [
                        (
                            (f"{p}.{original_indices[i][j]}" if p else str(original_indices[i][j]))
                            if (p is not None and original_indices[i][j] is not None)
                            else None
                        )
                        for j, p in enumerate(original_paths)
                    ]
                    aligned_item, sub_mapping = _compute_key_aligned_structure(
                        row, original_paths_for_row, cascade_cfg
                    )
                    aligned_list.append(aligned_item)
                    # Prepend the aligned list index to the mapping keys
                    for sub_key, paths in sub_mapping.items():
                        new_key = f"{i}.{sub_key}" if sub_key else str(i)
                        key_mappings[new_key] = paths
                return aligned_list, key_mappings

        # Fallback: For lists of scalars or if key selection fails, zip them
        _log("[ALIGN] Fallback zip alignment for lists (scalars or no key)")
        aligned_list = []
        max_len = max(len(lst) for lst in lists) if lists else 0
        for i in range(max_len):
            row = [lst[i] if i < len(lst) else None for lst in lists]
            original_paths_for_row = [
                (
                    (f"{p}.{i}" if p else str(i)) if i < len(values[j]) else None
                ) if p is not None else None
                for j, p in enumerate(original_paths)
            ]
            
            aligned_item, sub_mapping = _compute_key_aligned_structure(
                row, original_paths_for_row, cascade_cfg
            )
            aligned_list.append(aligned_item)
            for sub_key, paths in sub_mapping.items():
                new_key = f"{i}.{sub_key}" if sub_key else str(i)
                key_mappings[new_key] = paths
                
        return aligned_list, key_mappings

    return values, {} # Should not be reached


def recursive_align(
    values: Sequence[JSONLike],
    string_similarity_method: str,
    min_support_ratio: float = 0.5,
    max_novelty_ratio: float = 0.25,
    current_path: str = "",
    reference_idx: Optional[int] = None,
    min_uniqueness: Optional[float] = None,
    min_coverage: Optional[float] = None,
) -> tuple[Sequence[JSONLike], dict[str, list[str | None]]]:
    """
    Key-based recursive alignment with the same API as the similarity-based aligner.

    Returns:
      - per-source aligned outputs (same length/order as `values`)
      - key mappings from the aligned structure to original per-source paths
    """
    # Edge cases
    if not values:
        return list(values), {}
    if all(v is None for v in values):
        return list(values), {current_path: [current_path for _ in values]}

    non_nulls = [v for v in values if v is not None]
    if not non_nulls:
        return list(values), {}

    # Configure the key-selection cascade
    # - Map support->coverage by default
    # - Allow explicit overrides via min_uniqueness/min_coverage when provided
    eff_min_coverage = min_coverage if min_coverage is not None else min_support_ratio
    eff_min_uniqueness = min_uniqueness if min_uniqueness is not None else 0.5
    cascade_cfg = CascadeConfig(min_coverage=eff_min_coverage, min_uniqueness=eff_min_uniqueness)

    # 1) Compute aligned structure and per-path mappings
    original_paths: List[Optional[str]] = [current_path for _ in values]
    aligned_data, raw_key_mappings = _compute_key_aligned_structure(values, original_paths, cascade_cfg)

    # 2) Materialize per-source aligned outputs (relative to each provided value)
    per_source_outputs: List[JSONLike] = []
    for i, src_root in enumerate(values):
        # _materialize_source_view expects a dict source root. Wrap lists in a dict
        # so pathing like "0", "1" can still be addressed in a stable container.
        materialized_root: Dict[str, Any]
        if isinstance(src_root, dict):
            materialized_root = src_root
        elif isinstance(src_root, list):
            materialized_root = {"items": src_root}
            # Rewrite mappings to start from "items" when materializing a list root
            if raw_key_mappings:
                raw_key_mappings = { (f"items.{k}" if k else "items"): v for k, v in raw_key_mappings.items() }
        else:
            materialized_root = {}
        per_source_outputs.append(
            _materialize_source_view(
                aligned_node=aligned_data,
                key_mappings=raw_key_mappings,
                source_idx=i,
                current_path="",
                source_root=materialized_root,
            )
        )

    # 3) Prefix mapping keys and values with `current_path` (for API parity)
    if current_path:
        prefixed_mapping: Dict[str, List[Optional[str]]] = {}
        for key, paths in raw_key_mappings.items():
            # Prefix mapping dictionary keys
            pref_key = f"{current_path}.{key}" if key else current_path
            # Prefix individual source paths when present
            pref_paths: List[Optional[str]] = []
            for p in paths:
                if p is None or p == "":
                    pref_paths.append(current_path if current_path else None)
                else:
                    pref_paths.append(f"{current_path}.{p}" if current_path else p)
            prefixed_mapping[pref_key] = pref_paths
        key_mappings = prefixed_mapping
    else:
        key_mappings = raw_key_mappings

    return per_source_outputs, key_mappings


# --------------------- Main API and CLI ---------------------



def _get_value_by_path(obj: Any, path: Optional[str]) -> Any:
    """Retrieve a value from a nested object using a dot path with optional int indices.

    Examples:
        path "check.amount" -> obj["check"]["amount"]
        path "properties.2.revenue_events.1.price" -> obj["properties"][2]["revenue_events"][1]["price"]
    """
    if path is None:
        return None
    # Empty string denotes the root
    if path == "":
        return obj

    cur = obj
    for token in path.split("."):
        if token == "":
            # tolerate accidental leading dot
            continue
        # try list index
        try:
            idx = int(token)
            if isinstance(cur, list) and 0 <= idx < len(cur):
                cur = cur[idx]
                continue
            else:
                return None
        except ValueError:
            pass

        if isinstance(cur, dict) and token in cur:
            cur = cur[token]
        else:
            return None
    return cur


def _materialize_source_view(
    aligned_node: Any,
    key_mappings: Dict[str, List[Optional[str]]],
    source_idx: int,
    current_path: str = "",
    source_root: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build the per-source aligned structure by projecting values via key_mappings.

    - For dicts/lists: preserve structure and recurse.
    - For scalars: fetch the value from the source using the mapped original path.
    """
    # Initialize source_root holder only once
    if source_root is None:
        # The caller will pass the real root later when invoking
        raise ValueError("source_root must be provided at the top-level call.")

    # Dict: preserve keys, recurse
    if isinstance(aligned_node, dict):
        out: Dict[str, Any] = {}
        for k, v in aligned_node.items():
            next_path = f"{current_path}.{k}" if current_path else k
            out[k] = _materialize_source_view(v, key_mappings, source_idx, next_path, source_root)
        return out

    # List: preserve indices, recurse
    if isinstance(aligned_node, list):
        out_list: List[Any] = []
        for i, v in enumerate(aligned_node):
            next_path = f"{current_path}.{i}" if current_path else str(i)
            out_list.append(
                _materialize_source_view(v, key_mappings, source_idx, next_path, source_root)
            )
        return out_list

    # Scalar / None: pull per-source value via mapping; fallback to aligned value
    mapped_paths = key_mappings.get(current_path)
    if mapped_paths is not None and 0 <= source_idx < len(mapped_paths):
        original_path = mapped_paths[source_idx]
        return _get_value_by_path(source_root, original_path)

    # If we have no mapping entry, just return the aligned leaf value
    return deepcopy(aligned_node)