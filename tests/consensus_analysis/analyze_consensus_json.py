#!/usr/bin/env python3
"""
Minimal end-to-end check: load JSONs, run recursive_list_alignments,
and print BEFORE/AFTER alignment with key mappings.
"""

import json
import os
import sys
from typing import Any, List

import dotenv
from k_llms.utils.consensus_utils import (
    recursive_list_alignments,
)
from openai import OpenAI

# Load environment variables from .env file
dotenv.load_dotenv()


def load_json(filename: str) -> Any:
    """Load a JSON file from the same directory as this script."""
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dummy_embeddings(texts: list[str]) -> list[list[float]]:
    """Deterministic dummy embeddings to avoid external API calls."""
    return [[0.0] * 8 for _ in texts]


def _collect_disagreements(aligned: list[Any]) -> list[str]:
    """Return human-readable diagnostics listing disagreements and missings across JSONs.

    For dicts/lists, recurse shallowly to surface common issues.
    """
    notes: list[str] = []
    n = len(aligned)
    if n == 0:
        return notes
    # All None => nothing to compare
    if all(v is None for v in aligned):
        return notes
    # If primitives
    if all(not isinstance(v, (dict, list)) for v in aligned if v is not None):
        present_vals = [v for v in aligned if v is not None]
        if len(set(map(str, present_vals))) > 1:
            notes.append(f"primitive disagreement: {present_vals}")
        if len(present_vals) < n:
            notes.append(f"missing in {n - len(present_vals)} JSON(s)")
        return notes
    # Dicts
    if all((v is None) or isinstance(v, dict) for v in aligned):
        # union keys
        keys = set()
        for v in aligned:
            if isinstance(v, dict):
                keys.update(v.keys())
        for k in sorted(keys):
            vals_k = [(v.get(k) if isinstance(v, dict) else None) for v in aligned]
            sub = _collect_disagreements(vals_k)
            for s in sub:
                notes.append(f"{k}: {s}")
        return notes
    # Lists
    if all((v is None) or isinstance(v, list) for v in aligned):
        # Compare lengths
        lens = [len(v) if isinstance(v, list) else 0 for v in aligned]
        if len(set(lens)) > 1:
            notes.append(f"list length mismatch: {lens}")
        # Shallow element comparison (first 3)
        max_len = max(lens)
        for i in range(min(3, max_len)):
            vals_i = [(v[i] if isinstance(v, list) and i < len(v) else None) for v in aligned]
            sub = _collect_disagreements(vals_i)
            for s in sub:
                notes.append(f"[{i}]: {s}")
        return notes
    # Mixed types
    notes.append("type mismatch across JSONs at this node")
    return notes


def run_recursive_alignment_demo(files: list[str] | None = None) -> None:
    """Apply recursive_list_alignments on given JSONs and print outputs.

    - If `files` is None, defaults to the small examples in this folder.
    - Accepts any number of JSON file paths relative to this script.
    """
    if not files:
        files = [
            "json1.json",
            "json2.json",
            "json3.json",
            "json4.json",
        ]

    jsons: List[Any] = [load_json(f) for f in files]

    # Use a string similarity method that does not require embeddings
    string_similarity_method = "levenshtein"

    aligned_jsons, key_mappings = recursive_list_alignments(
        jsons,
        string_similarity_method,
        dummy_embeddings,
        client=OpenAI(),
        min_support_ratio=0.5,
    )

    # Print originals as JSON
    print("ORIGINAL_JSONS:")
    for idx, obj in enumerate(jsons, start=1):
        print(f"# JSON {idx}")
        print(json.dumps(obj, ensure_ascii=False, indent=2))

    # Print outputs as JSON
    print("\nALIGNED_JSONS:")
    for idx, obj in enumerate(aligned_jsons, start=1):
        print(f"# JSON {idx}")
        print(json.dumps(obj, ensure_ascii=False, indent=2))

    print("\nKEY_MAPPINGS:")
    print(json.dumps(key_mappings, ensure_ascii=False, indent=2))

    # Diagnostics
    # print("\nDIAGNOSTICS:")
    # diags = _collect_disagreements(aligned_jsons)
    # if not diags:
    #     print("no disagreements detected at shallow scan")
    # else:
    #     for d in diags[:50]:  # limit
    #         print(f"- {d}")


if __name__ == "__main__":
    # Allow passing JSON file names as CLI args
    args = sys.argv[1:]
    run_recursive_alignment_demo(args if args else None)
