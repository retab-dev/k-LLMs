#!/usr/bin/env python3
"""
Minimal end-to-end check: load 4 JSONs, run recursive_list_alignments,
and print BEFORE/AFTER alignment with a robust label extractor.
"""

import json
import os
from typing import Any, List

from k_llms.utils.consensus_utils import (
    recursive_list_alignments,
)
from openai import OpenAI


def load_json(filename: str) -> Any:
    """Load a JSON file from the same directory as this script."""
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_for_item(item: Any) -> Any:
    """Return a readable label for an item (dict/primitive/None)."""
    if item is None:
        return None
    if isinstance(item, dict):
        # Try common key candidates; fall back to a compact dict string
        for k in ("article_code", "id", "sku", "code", "key", "name", "title"):
            if k in item and isinstance(item[k], (str, int)):
                return item[k]
        # Fallback: stable compact representation
        return {k: item[k] for k in sorted(item.keys()) if k in item}
    return item


def dummy_embeddings(texts: list[str]) -> list[list[float]]:
    """Deterministic dummy embeddings to avoid external API calls."""
    return [[0.0] * 8 for _ in texts]


def run_recursive_alignment_demo() -> None:
    """Run recursive_list_alignments on items from 4 JSONs and print results."""
    json1 = load_json("json1.json")
    json2 = load_json("json2.json")
    json3 = load_json("json3.json")
    json4 = load_json("json4.json")

    items_lists: List[List[Any]] = [
        json1.get("items", []),
        json2.get("items", []),
        json3.get("items", []),
        json4.get("items", []),
    ]

    print("BEFORE alignment:")
    for idx, lst in enumerate(items_lists, start=1):
        print(f"  JSON {idx}: {[label_for_item(x) for x in lst][:20]}")

    # Use a string similarity method that does not require embeddings
    string_similarity_method = "levenshtein"

    aligned_lists, key_mappings = recursive_list_alignments(
        items_lists,
        string_similarity_method,
        dummy_embeddings,
        client=OpenAI(),
        min_support_ratio=0.5,
    )

    print("\nAFTER alignment:")
    for idx, lst in enumerate(aligned_lists, start=1):
        print(f"  JSON {idx}: {[label_for_item(x) for x in lst][:20]}")

    print("\nKey mappings (first 10 entries):")
    shown = 0
    for k, v in key_mappings.items():
        print(f"  {k}: {v}")
        shown += 1
        if shown >= 10:
            break


if __name__ == "__main__":
    run_recursive_alignment_demo()
