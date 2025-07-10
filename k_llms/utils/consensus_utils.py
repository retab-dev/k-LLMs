import json
import logging
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import zip_longest
from math import isclose
from threading import Lock
from typing import Any, Awaitable, Callable, Literal, Optional, overload

import numpy as np
from cachetools import TTLCache
from Levenshtein import distance as levenshtein_distance
from .majority_sorting import _original_positions, sort_by_original_majority
from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails
from pydantic import BaseModel, computed_field
from scipy.optimize import linear_sum_assignment  # type: ignore
from retab.utils.json_schema import filter_auxiliary_fields, flatten_dict, unflatten_dict
from retab.types.documents.extractions import RetabParsedChatCompletion, RetabParsedChatCompletionChunk, RetabParsedChoiceChunk, RetabParsedChoiceDeltaChunk
from unidecode import unidecode

NumericalPrimitive = int | float
DataType = NumericalPrimitive | dict | list | tuple
EnumLikeType = str | bool
StringSimilarityMethod = Literal["levenshtein", "jaccard", "hamming", "embeddings"]
StringConsensusMethod = Literal["centroid", "llm-consensus"]
SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE = Callable[[list[str]], list[list[float]]]
ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE = Callable[[list[str]], Awaitable[list[list[float]]]]


""" Open AI Auxiliary Functions"""
max_tokens_per_model = {"text-embedding-3-small": 8191, "text-embedding-3-large": 8191}
pricing = {"text-embedding-3-small": 0.020, "text-embedding-3-large": 0.13}


IGNORED_KEY_PATTERNS = [
    # contains reasoning___
    r"reasoning___",
    # contains quote___
    r"quote___",
]

logger = logging.getLogger(__name__)

if os.getenv("ENV_NAME") == "dev":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


class ConsensusSettings(BaseModel):
    allow_none_as_candidate: bool = False
    # String specific settings
    string_similarity_method: StringSimilarityMethod = "embeddings"
    string_consensus_method: StringConsensusMethod = "centroid"
    # Align objects with a minimum similarity threshold
    minimum_voters_threshold: float = 0.75
    min_support_ratio: float = 0.51  # At least 51% of the voters must agree


T = dict | int | float | str | bool | None


# Type aliases for clarity
Index = tuple[int, int]

SIMILARITY_SCORE_LOWER_BOUND = 1e-8


class SimilarityCache:
    """
    Cache utility for pairwise similarity computations.
    """

    def __init__(self, sim_fn: Callable[[Any, Any], float], list_of_lists: list[list[Any]]):
        self.sim_fn = sim_fn
        self.cache: dict[tuple[Index, Index], float] = {}
        self.list_of_lists = list_of_lists

    def get(self, a_idx: Index, b_idx: Index) -> float:
        key = (a_idx, b_idx)
        reverse_key = (b_idx, a_idx)

        if key in self.cache:
            return self.cache[key]
        if reverse_key in self.cache:
            return self.cache[reverse_key]

        sim = self.sim_fn(
            self.list_of_lists[a_idx[0]][a_idx[1]],  # a_obj
            self.list_of_lists[b_idx[0]][b_idx[1]],  # b_obj
        )
        self.cache[key] = sim
        self.cache[reverse_key] = sim
        return sim


def _prune_low_support_elements(aligned_lists: list[list[T | None]], min_support_ratio: float) -> list[list[T | None]]:
    """Remove columns with support below threshold."""
    if not aligned_lists:
        return aligned_lists

    n_lists = len(aligned_lists)
    n_cols_set = set([len(lst) for lst in aligned_lists])
    if len(n_cols_set) > 1:
        print("Warning: All lists must have the same number of columns")
        return aligned_lists

    if not n_cols_set:  # All lists are empty
        return aligned_lists

    n_cols = n_cols_set.pop()

    if n_cols == 0:  # Empty columns
        return aligned_lists

    # Calculate support for each column
    support = []
    for col_idx in range(n_cols):
        non_none_count = sum(1 for lst in aligned_lists if lst[col_idx] is not None)
        support.append(non_none_count / n_lists)

    # print(f"support: {support}")
    max_support = max(support)
    if max_support < min_support_ratio:
        print(f"All columns below threshold, keeping columns with support equal to {max_support} (the highest support)")
        min_support_ratio = max_support

    # Keep only columns with sufficient support
    keep_cols = [i for i, s in enumerate(support) if s >= min_support_ratio]

    # Create new pruned lists
    pruned_lists = []
    for lst in aligned_lists:
        pruned_lst = [lst[i] if i < len(lst) else None for i in keep_cols]
        pruned_lists.append(pruned_lst)

    return pruned_lists


def low_cutoff_bound(scores) -> float:
    if len(scores) == 0:
        return 0.0

    eps = 0.0001

    # Sort the scores
    scores = np.sort(scores)

    # Define a baseline threshold based on the bottom (no filtering)
    low_cutoff = scores[0]

    # But only apply it if the gap between the low values and the rest is significant
    # Example heuristic: Look for a "jump" in values near the low end
    # If the jump is not significant, we don't apply the cutoff
    diffs = np.diff(scores[: int(0.2 * len(scores))])
    if len(diffs) > 0:
        jump_threshold = np.median(diffs) * 3
        jump_idx = np.argmax(diffs > jump_threshold)
        if diffs[jump_idx] > jump_threshold:
            low_cutoff = scores[jump_idx + 1] + eps  # We add a small epsilon to make the low_cutoff non-inclusive

    return float(low_cutoff)


def remove_outliers(data: list[float]) -> list[float]:
    """
    Remove outliers from a list of floats.
    """
    lower = low_cutoff_bound(data)
    return [el for el in data if el >= lower]


def _compute_dynamic_threshold(sim_cache: SimilarityCache) -> float:
    """
    Computes a dynamic similarity threshold based on the distribution of similarity scores.

    For each element in each list, finds the best matching element across all other lists,
    ensuring that each element is only matched once within its own list.

    Args:
        list_of_lists: All lists to analyze
        sim_cache: Cache for similarity computations

    Returns:
        A dynamically computed similarity threshold
    """
    list_of_lists = sim_cache.list_of_lists
    BASE_THRESHOLD = 0.5
    if not list_of_lists or len(list_of_lists) < 2:
        return BASE_THRESHOLD  # Default threshold if not enough lists

    similarity_scores = []
    total_lists = len(list_of_lists)

    # For each list
    for i in range(total_lists):
        list_i = list_of_lists[i]
        if not list_i:
            continue

        # Track which elements in other lists have been used by previous elements in list_i
        used_elements = {j: set() for j in range(total_lists) if j != i}

        # Process each element in list_i in order
        for k_i in range(len(list_i)):
            best_match_score = BASE_THRESHOLD  # Default lower bound
            best_match = None  # (list_idx, element_idx)

            # Find best match across all other lists among available elements
            for j in range(i + 1, total_lists):
                list_j = list_of_lists[j]
                if not list_j:
                    continue

                # Check all available elements in list_j
                for k_j in range(len(list_j)):
                    if k_j in used_elements[j]:
                        continue  # Skip if this element was already used

                    obj_index_i = (i, k_i)
                    obj_index_j = (j, k_j)

                    sim = sim_cache.get(obj_index_i, obj_index_j)

                    if sim > best_match_score:
                        best_match_score = sim
                        best_match = (j, k_j)

            # If we found a match, record it and mark the element as used
            if best_match is not None and best_match_score > 0:
                # print(f"({i}, {k_i}) -> ({best_match[0]}, {best_match[1]}) with score {best_match_score}")
                similarity_scores.append(best_match_score)
                used_elements[best_match[0]].add(best_match[1])
    # Similarity scores are only the highest ones.
    # We need to get the minimum that is actually representative of this distribution (there might be outliers)
    similarity_scores.sort()
    similarity_scores = remove_outliers(similarity_scores)
    if not similarity_scores:
        return BASE_THRESHOLD
    return max(BASE_THRESHOLD, 0.95 * similarity_scores[0])


def _build_reference_list(sim_cache: SimilarityCache, min_support_ratio: float = 0.5, max_novelty_ratio: float = 0.5, threshold: float = 0.4) -> list[Index]:
    """
    Builds a reference list from the healthiest list, optionally augmented with well-supported elements.

    Args:
        list_of_lists: All lists to analyze.
        sim_cache: Cache for similarity computations.
        max_novelty_ratio: median_length * (1 + max_novelty_ratio) is the maximum number of elements in the reference list.
        min_support_ratio: Minimum proportion of lists that must support an added element.
        augmented: If True, augment the base list with additional well-supported elements.

    Returns:
        A list of tuples (list_idx, obj_pos) referencing elements from the original lists.
    """
    list_of_lists = sim_cache.list_of_lists
    int(np.mean([len(lst) for lst in list_of_lists]))

    # Track which positions in each list are still available for augmentation
    unused_positions = {idx: set(range(len(lst))) for idx, lst in enumerate(list_of_lists)}

    # Collect candidate elements from unused positions
    candidate_elements = [(list_idx, obj_pos) for list_idx, unused_indices in unused_positions.items() for obj_pos in unused_indices]

    # Group similar candidates and count their support
    support_groups: dict[Index, list[Index]] = defaultdict(list)
    support_groups_used_lists: dict[Index, set[int]] = defaultdict(set)

    for list_idx1, obj_pos1 in candidate_elements:
        obj_index1 = (list_idx1, obj_pos1)

        # Find the best matching existing group
        best_sim = -1
        best_group_repr_index = None

        for group_repr_index, group_used_lists in support_groups_used_lists.items():
            if list_idx1 in group_used_lists:
                # All elements in a group must come from different lists
                continue

            # Calculate the similarity between the current element and the group representative
            sim = sim_cache.get(obj_index1, group_repr_index)

            # Keep track of the best match
            if sim >= threshold and sim > best_sim:
                best_sim = sim
                best_group_repr_index = group_repr_index

        # Add to best group if found, otherwise create a new group
        if best_group_repr_index is not None:
            support_groups[best_group_repr_index].append(obj_index1)
            support_groups_used_lists[best_group_repr_index].add(list_idx1)

            # Now we re-elect the group representative
            # Create a dummy function for embeddings since we're only dealing with indices here
            def dummy_embeddings_fn(strings):
                return [[0.0] * 10 for _ in strings]  # Return dummy embeddings

            new_group_repr_index, _ = consensus_as_primitive(
                support_groups[best_group_repr_index], ConsensusSettings(), is_last_chunk=True, sync_get_openai_embeddings_from_text=dummy_embeddings_fn
            )
            if new_group_repr_index != best_group_repr_index:
                # logger.debug(f"Re-electing group representative for group {best_group_repr_index} -> {new_group_repr_index}")
                support_groups[new_group_repr_index] = support_groups[best_group_repr_index]
                support_groups_used_lists[new_group_repr_index] = support_groups_used_lists[best_group_repr_index]
                del support_groups[best_group_repr_index]
                del support_groups_used_lists[best_group_repr_index]
        else:
            support_groups[obj_index1] = [obj_index1]
            support_groups_used_lists[obj_index1] = {list_idx1}

    support_ratios: dict[Index, float] = {k: len(v) / len(list_of_lists) for k, v in support_groups.items()}
    # Sort and filter the support_counter dict
    support_ratios = {k: v for k, v in support_ratios.items() if v >= min_support_ratio}
    support_ratios = dict(sorted(support_ratios.items(), key=lambda x: (-x[1], x[0])))

    # max_elements = int(np.ceil(mean_length * (1 + max_novelty_ratio)))

    # Add well-supported candidates to reference list
    reference_list = list(support_ratios.keys())  # [:max_elements]

    return reference_list


def _align_lists_to_reference_hungarian(
    sim_cache: SimilarityCache,
    reference_indices: list[tuple[int, int]],
    threshold: float = 0.4,
) -> list[list[Any]]:
    list_of_lists = sim_cache.list_of_lists
    n_lists = len(list_of_lists)
    n_refs = len(reference_indices)

    aligned_lists = [[None for _ in range(n_refs)] for _ in range(n_lists)]
    if not reference_indices:
        return aligned_lists

    # Match each list to the reference
    for list_idx, lst in enumerate(list_of_lists):
        n_objs = len(lst)
        if n_objs == 0:
            continue

        sim_matrix = np.full((n_refs, n_objs), -np.inf)

        for ref_pos, (ref_list_idx, ref_obj_pos) in enumerate(reference_indices):
            ref_index = (ref_list_idx, ref_obj_pos)

            for obj_pos, obj in enumerate(lst):
                obj_index = (list_idx, obj_pos)

                if obj_index == ref_index:
                    # This element is part of the reference, skip similarity computation
                    sim_matrix[ref_pos, obj_pos] = 1.0
                    continue

                sim = sim_cache.get(obj_index, ref_index)
                sim_matrix[ref_pos, obj_pos] = sim

        cost_matrix = 1.0 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for ref_pos, obj_pos in zip(row_ind, col_ind):
            sim = sim_matrix[ref_pos, obj_pos]
            if sim >= threshold and aligned_lists[list_idx][ref_pos] is None:
                aligned_lists[list_idx][ref_pos] = lst[obj_pos]

    return aligned_lists


def lists_alignment(
    list_of_lists: list[list[Any]], sim_fn: Callable[[Any, Any], float], min_support_ratio: float = 0.5, max_novelty_ratio: float = 0.25, reference_list_idx: Optional[int] = None
) -> tuple[list[list[Any]], list[list[int | None]]]:
    """
    Master function to align lists based on element similarity.

    Args:
        lists: Lists of objects to align.
        sim_fn: Function that takes two objects and returns similarity in [0, 1].
        min_support_ratio: Minimum fraction of lists that must agree to add elements to reference.
        threshold: Minimum similarity threshold for alignment. Will be used as a fallback if dynamic threshold calculation fails.
        reference_list_idx: Index of the list to use as reference. If None, the reference list will be dynamically computed using all lists.
    Returns:
        List of aligned lists.
    """
    if not list_of_lists or all(not lst for lst in list_of_lists):
        return [[] for _ in list_of_lists], [[None for _ in range(len(lst))] for lst in list_of_lists]

    sim_cache = SimilarityCache(sim_fn, list_of_lists)

    # Actual alignment
    if reference_list_idx is None:
        # Compute the dynamic threshold
        dynamic_threshold = _compute_dynamic_threshold(sim_cache)
        # Get reference list and dynamically computed threshold
        reference_list = _build_reference_list(sim_cache, min_support_ratio, max_novelty_ratio, threshold=dynamic_threshold)

        # Align lists using the final threshold (Reduce again the threshold)
        aligned = _align_lists_to_reference_hungarian(sim_cache, reference_list, threshold=0.95 * dynamic_threshold)

        # Cap the number of elements in the aligned lists to the number of elements in the reference list
        aligned = _prune_low_support_elements(aligned, min_support_ratio)

        # Sort the aligned lists by the average index of the reference indices within the initial list of lists
        aligned, original_list_reference_indices = sort_by_original_majority(aligned, list_of_lists)
    # Known reference (ground truth)
    else:
        # Just use the reference list
        reference_list: list[Index] = [(reference_list_idx, i) for i in range(len(list_of_lists[reference_list_idx]))]

        # Align the lists to the reference list (no threshold)
        aligned = _align_lists_to_reference_hungarian(sim_cache, reference_list, threshold=0.0)

        # We don't prune the lists, because the reference list is the truth and is already sorted
        # just build the original_list_reference_indices
        original_list_reference_indices = _original_positions(aligned, list_of_lists)

    # Return the aligned lists
    return aligned, original_list_reference_indices


def exists_nested_lists(values: list[Any]) -> bool:
    """
    Checks if a list contains any nested lists within it.

    Args:
        values: A list of values to check

    Returns:
        True if any nested structures are found, False otherwise
    """
    if not values:
        return False

    for v in values:
        # Direct nesting (value is a list)
        if isinstance(v, list):
            return True

        # Indirect nesting (value is a dict with nested values)
        elif isinstance(v, dict):
            if exists_nested_lists(list(v.values())):
                return True
    return False


def recursive_list_alignments(
    values: list[Any],
    string_similarity_method: StringSimilarityMethod,
    sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    min_support_ratio: float,
    max_novelty_ratio: float = 0.25,
    current_path: str = "",
    reference_idx: Optional[int] = None,
) -> tuple[list[Any], dict[str, list[str | None]]]:
    """
    This function recursively aligns values in a list of dictionaries or lists.

    The function recursively processes nested structures, aligning items at each level:
    - For dictionaries: aligns values for each key across all dictionaries
    - For lists: aligns items across lists using object similarity

    IMPORTANT ASSUMPTIONS:
    - All non-None values at the same level are expected to have the same type
    - The first non-None item's type is used to determine processing strategy
    - Mixed type collections are not explicitly handled and may produce unexpected results

    Args:
        values: A list of values (typically dictionaries or lists) to align
        string_similarity_method: Method to use for string similarity calculation
        min_support_ratio: Minimum ratio of elements needed to keep a column
        max_novelty_ratio: Maximum ratio of elements needed to keep a column
        current_path: The current path in the nested structure
        reference_idx: The index of the reference list to use for alignment (to specify the ground truth list, if any). Leave None for dynamic alignment.

    Returns:
        A list of aligned values, preserving the original structure
    """
    # Handle empty values list
    if not values:
        return values, {}

    # If all values are None, just return them
    if all(v is None for v in values):
        return values, {current_path: [current_path for _ in values]}

    # Filter out None values for type checking
    non_nulls = [v for v in values if v is not None]

    # Make a defensive copy to avoid modifying the original values
    # Note: Using deepcopy to ensure we don't modify the original data
    values = deepcopy(values)

    # Assumption: All non-null values have the same type
    # We use the first non-null value's type to determine processing
    first_type = type(non_nulls[0])
    same_type = all(isinstance(x, first_type) for x in non_nulls)
    key_mappings: dict[str, list[str | None]] = {}

    if not same_type or first_type not in (dict, list):
        key_mappings[current_path] = [current_path if (v is not None or idx == reference_idx) else None for idx, v in enumerate(values)]
        return values, key_mappings

    # 1) All dict => Now we update each dict with the aligned values
    if first_type is dict:
        dicts_only = [(d if isinstance(d, dict) else {}) for d in values]

        # Find all unique keys across all dictionaries
        all_keys = list(set([k for d in dicts_only for k in d.keys()]))
        all_keys.sort()

        # For each key, align the values across all dictionaries
        for key in all_keys:
            values_for_key = [d.get(key) for d in dicts_only]

            _current_path = f"{current_path}.{key}" if current_path else key
            aligned_values_for_key, sub_key_mapping = recursive_list_alignments(
                values_for_key,
                string_similarity_method,
                sync_get_openai_embeddings_from_text,
                min_support_ratio,
                max_novelty_ratio=max_novelty_ratio,
                current_path=_current_path,
                reference_idx=reference_idx,
            )

            # Update each dictionary with its aligned value
            for _d, aligned_value in zip(dicts_only, aligned_values_for_key):
                _d[key] = aligned_value

            # Update the key mapping
            key_mappings.update(sub_key_mapping)

        # Update the values list with the dicts_only
        values = [{k: _d.get(k) for k in all_keys} for _d in dicts_only]
    # 2) All list => consensus_list
    if first_type is list:
        # Convert any non-list items to empty lists
        lists_only = [(lst if isinstance(lst, list) else []) for lst in values]
        # Initialize the original_list_reference_indices with None for bounding purposes
        original_list_reference_indices: list[list[int | None]] = [[None for _ in lst] for lst in lists_only]

        # Skip alignment for empty lists
        if any(lst for lst in lists_only):
            # Align the lists by similarity
            def sim_fn(a, b):
                return generic_similarity(a, b, string_similarity_method, sync_get_openai_embeddings_from_text)

            aligned_lists_only, original_list_reference_indices = lists_alignment(
                lists_only, sim_fn, min_support_ratio=min_support_ratio, max_novelty_ratio=max_novelty_ratio, reference_list_idx=reference_idx
            )

            # Update each list with its aligned values
            for l_idx, new_lst in enumerate(aligned_lists_only):
                values[l_idx] = new_lst
        else:
            # Make sure all elements are empty lists
            for i in range(len(values)):
                values[i] = []

        # Now, call the align function recursively on each position of the lists (they now have the same length)
        if len(values) > 0:
            list_length = len(values[0])
            if list_length > 0:
                for i in range(list_length):
                    values_i = [lst[i] for lst in values]
                    values_i, sub_key_mapping = recursive_list_alignments(
                        values_i,
                        string_similarity_method,
                        sync_get_openai_embeddings_from_text,
                        min_support_ratio,
                        max_novelty_ratio=max_novelty_ratio,
                        current_path="",
                        reference_idx=reference_idx,
                    )
                    for l_idx, new_lst in enumerate(values_i):
                        values[l_idx][i] = new_lst

                    # Update the key mapping according to the original positions.
                    for key, sub_values in sub_key_mapping.items():
                        # Build the paths correctly
                        ## The key
                        _key_path = f"{current_path}.{i}" if current_path else str(i)
                        _key_path = f"{_key_path}.{key}" if key else _key_path
                        ## The values
                        current_values = []
                        for l_idx, v in enumerate(sub_values):
                            _original_position = original_list_reference_indices[l_idx][i]
                            if _original_position is None or v is None:
                                current_values.append(None)
                            else:
                                _original_value_path = f"{current_path}.{_original_position}" if current_path else _original_position
                                _original_value_path = f"{_original_value_path}.{v}" if v else _original_value_path
                                current_values.append(_original_value_path)
                        key_mappings[_key_path] = current_values
            elif current_path:  # Don't  support empty root paths
                # All lists are empty, let's add to the key_mapping just with the root to the this path
                key_mappings[current_path] = [current_path] * len(values)
    return values, key_mappings


###############################################################################
# 1) String Similarity Helpers
###############################################################################
## --- Caching Setup ---
embeddings_cache = TTLCache(maxsize=1024, ttl=300)
similarity_cache = TTLCache(maxsize=1024, ttl=300)
embeddings_cache_lock = Lock()
similarity_cache_lock = Lock()


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculates cosine similarity between two numpy vectors."""
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)
    if arr1.shape != arr2.shape:
        # Or handle this case as appropriate (e.g., return 0.0)
        raise ValueError("Vectors must have the same shape for cosine similarity")

    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)

    # Handle cases where one or both vectors have zero magnitude
    if norm1 == 0 or norm2 == 0:
        return SIMILARITY_SCORE_LOWER_BOUND

    # Calculate cosine similarity
    similarity = np.dot(arr1, arr2) / (norm1 * norm2)

    similarity = 0.5 * (similarity + 1.0)  # Normalize to [0, 1]

    # Clip result to [-1.0, 1.0] to handle potential floating-point inaccuracies
    similarity_clip = np.clip(similarity, SIMILARITY_SCORE_LOWER_BOUND, 1.0)

    return similarity_clip


def get_embeddings(s: str, sync_get_openai_embeddings_from_text: Callable[[list[str]], list[list[float]]]) -> list[float]:
    # If not in cache, compute embeddings
    logger.debug(f"Cache miss for '{s}'")
    result = sync_get_openai_embeddings_from_text([s])[0]

    return result


def normalize_string(text: str) -> str:
    """
    Normalize a string by removing non-alphanumeric characters and lowercasing.

    Args:
        text: Input string to normalize

    Returns:
        Normalized string with only alphanumeric characters, all lowercase
    """
    if not text:
        return ""
    # Remove all non-alphanumeric characters and convert to lowercase
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def hamming_distance_padded(s: str, t: str) -> int:
    """
    Compute the Hamming distance between two strings, treating spaces as wildcards.

    Args:
        s: The first string
        t: The second string

    Returns:
        The Hamming distance between the two strings
    """
    # Normalize inputs
    s = normalize_string(s)
    t = normalize_string(t)

    return sum(a != b for a, b in zip_longest(s, t, fillvalue=" "))


def hamming_similarity(str_1: str, str_2: str) -> float:
    """
    Compute the Hamming similarity between two strings.

    Args:
        str_1: The first string
        str_2: The second string

    Returns:
        A float between 0 and 1, where 1 means the strings are identical
    """
    # Normalize inputs
    str_1 = normalize_string(str_1)
    str_2 = normalize_string(str_2)

    max_length = max(len(str_1), len(str_2))

    if max_length == 0:
        return 1.0

    dist = hamming_distance_padded(str_1, str_2)

    # Clip the result to the lower bound
    return max(SIMILARITY_SCORE_LOWER_BOUND, 1 - (dist / max_length))


def jaccard_similarity(str_1: str, str_2: str) -> float:
    """
    Compute the Jaccard similarity between two strings.

    Args:
        str_1: The first string
        str_2: The second string

    Returns:
        A float between 0 and 1, where 1 means the strings are identical
    """
    # Normalize inputs
    str_1 = normalize_string(str_1)
    str_2 = normalize_string(str_2)

    set_a = set(str_1)
    set_b = set(str_2)
    intersection = set_a & set_b
    union = set_a | set_b
    if not union:
        return 1.0
    # Clip the result to the lower bound
    return max(SIMILARITY_SCORE_LOWER_BOUND, len(intersection) / len(union))


def levenshtein_similarity(str_1: str, str_2: str) -> float:
    """
    Calculate similarity between two values using Levenshtein distance.
    Returns a similarity score between 0.0 and 1.0.
    """
    # Normalize inputs
    str_1 = normalize_string(str_1)
    str_2 = normalize_string(str_2)

    max_length = max(len(str_1), len(str_2))

    if max_length == 0:
        return 1.0

    dist = levenshtein_distance(str_1, str_2)
    # Clip the result to the lower bound
    return max(SIMILARITY_SCORE_LOWER_BOUND, 1 - (dist / max_length))


def key_normalization(key: str) -> str:
    """This method is useful to compare keys under list indexes (that refers to the same kind of error but on different list index position)"""
    # We will replace all .{i} with .* where i is the index of the list (using regex for this)
    key_parts = key.split(".")
    new_key_parts = []
    for key_part in key_parts:
        if key_part.isdigit():
            new_key_parts.append("*")
        else:
            new_key_parts.append(key_part)
    return ".".join(new_key_parts)


###############################################################################
# 3) Similarity-based comparison between two generic values of (presumably) same type
###############################################################################
def get_cached_similarity(s1: str, s2: str, method: str) -> float | None:
    """Thread-safe getter for similarity cache"""
    key = (min(s1, s2), max(s1, s2), method)
    with similarity_cache_lock:
        try:
            return similarity_cache[key]
        except KeyError:
            return None


def set_cached_similarity(s1: str, s2: str, method: str, value: float) -> None:
    """Thread-safe setter for similarity cache"""
    key = (min(s1, s2), max(s1, s2), method)
    with similarity_cache_lock:
        similarity_cache[key] = value


def string_similarity(s1: str, s2: str, method: StringSimilarityMethod, sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE) -> float:
    """
    Returns a similarity score in [0,1] between two strings using the specified method.
    This version is thread-safe and handles race conditions properly.
    """
    # Check cache first
    cached_result = get_cached_similarity(s1, s2, method)
    if cached_result is not None:
        return cached_result
    result: float | None = None
    # Compute similarity if not in cache
    if method == "jaccard":
        result = jaccard_similarity(s1, s2)
    elif method == "hamming":
        result = hamming_similarity(s1, s2)
    # Only use embeddings if the strings are long enough, otherwise it is overkill and not worth the time
    elif method == "embeddings" and len(s1) > 50 and len(s2) > 50:
        try:
            result = _cosine_similarity(get_embeddings(s1, sync_get_openai_embeddings_from_text), get_embeddings(s2, sync_get_openai_embeddings_from_text))
        except Exception as e:
            logger.error(f"Error getting embeddings for '{s1}' and '{s2}'", exc_info=e)
    if result is None:
        # Fall-back to levenshtein
        result = levenshtein_similarity(s1, s2)

    # Cache the result
    set_cached_similarity(s1, s2, method, result)
    return result


def numerical_similarity(val1: NumericalPrimitive, val2: NumericalPrimitive) -> float:
    """
    Returns a similarity score in [0,1] between two 'primitive' values:
      - None vs None => 1.0
      - Numerics => relative closeness
    """
    # booleans => exact
    if isinstance(val1, bool) and isinstance(val2, bool):
        return 1.0 if val1 == val2 else SIMILARITY_SCORE_LOWER_BOUND
    # numeric => relative closeness (1% tolerance)
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) and isclose(val1, val2, rel_tol=0.01):
        return 1.0

    # Otherwise, we check if they match just for robustness, but this should never happen, the similarity should be SIMILARITY_SCORE_LOWER_BOUND
    return 1.0 if val1 == val2 else SIMILARITY_SCORE_LOWER_BOUND


def dict_similarity(d1: dict, d2: dict, string_similarity_method: StringSimilarityMethod, sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE) -> float:
    """
    Index-based approach for comparing two dictionaries:
      - Union of keys
      - Compare each sub-value
        => if sub-values are dict => recurse
        => if sub-values are list => handle with list_of_dicts_similarity or
           list_of_primitives similarity
        => else => compare_primitives
      - Average result
    """
    all_keys = set(d1.keys()) | set(d2.keys())

    # Remove IGNORED_KEY_PATTERNS from all_keys
    all_keys = [k for k in all_keys if not any(re.match(pattern, k) for pattern in IGNORED_KEY_PATTERNS)]

    if not all_keys:
        return 1.0

    total_score = 0.0
    for k in all_keys:
        v1 = d1.get(k)
        v2 = d2.get(k)
        total_score += generic_similarity(v1, v2, string_similarity_method, sync_get_openai_embeddings_from_text)

    return total_score / len(all_keys)


def list_similarity(
    l1: list[Any] | tuple[Any, ...],
    l2: list[Any] | tuple[Any, ...],
    string_similarity_method: StringSimilarityMethod,
    sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
) -> float:
    """
    Compare two lists, returning a similarity score in [0,1].
    """
    max_len = max(len(l1), len(l2))
    if max_len == 0:
        return 1.0
    total_score = 0.0
    for i in range(max_len):
        v1 = l1[i] if i < len(l1) else None
        v2 = l2[i] if i < len(l2) else None
        total_score += generic_similarity(v1, v2, string_similarity_method, sync_get_openai_embeddings_from_text)
    return total_score / max_len


def generic_similarity(
    v1: Any,
    v2: Any,
    string_similarity_method: StringSimilarityMethod,
    sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
) -> float:
    """
    Compare two values, returning a similarity score in [0,1].
    """

    # both None => perfect
    if not bool(v1) and not bool(v2):
        return 1.0
    # one None => 0
    if v1 is None or v2 is None:
        return SIMILARITY_SCORE_LOWER_BOUND
    if isinstance(v1, str) and isinstance(v2, str):
        return string_similarity(v1, v2, string_similarity_method, sync_get_openai_embeddings_from_text)
    elif isinstance(v1, NumericalPrimitive) and isinstance(v2, NumericalPrimitive):
        return numerical_similarity(v1, v2)
    elif isinstance(v1, dict) and isinstance(v2, dict):
        return dict_similarity(v1, v2, string_similarity_method, sync_get_openai_embeddings_from_text)
    elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
        return list_similarity(v1, v2, string_similarity_method, sync_get_openai_embeddings_from_text)
    else:
        return SIMILARITY_SCORE_LOWER_BOUND


###############################################################################
# 4) Vote-based (enum-like) consensus
###############################################################################


def sanitize_value(v: str | bool) -> str:
    # cast to string and lowercose
    v = str(v).lower()
    # remove spaces
    v = v.replace(" ", "")
    # remove accents
    v = unidecode(v)
    # remove special characters
    return re.sub(r"[^a-zA-Z0-9]", "", v)


def voting_consensus(
    values: list[str | bool | None], consensus_settings: ConsensusSettings, is_last_chunk: bool, verbose: bool = False, parent_valid_frac: float = 1.0
) -> tuple[str | bool | None, float]:
    """
    Vote-based approach: pick most common non-null value.
    Confidence = proportion among total (including None).
    """

    if parent_valid_frac < consensus_settings.minimum_voters_threshold and not is_last_chunk:
        return (None, 0.0)

    total_values = len(values)

    # Handle empty or all-None case
    if not any(v is not None for v in values):
        return (None, parent_valid_frac)

    # Determine if we're dealing with booleans or strings
    first_non_none = next((v for v in values if v is not None), None)
    is_boolean = isinstance(first_non_none, bool)

    if is_boolean:
        # For booleans: treat None as False
        processed_values = [v or False for v in values]
        counts = Counter(processed_values)
        best_val, best_count = counts.most_common(1)[0]
    else:
        # For strings: exclude None values
        if consensus_settings.allow_none_as_candidate:
            valid_values = values
        else:
            valid_values = [v for v in values if v is not None]
        # Normalize string values for comparison
        processed_values = [(sanitize_value(v) if v is not None else None) for v in valid_values]

        counts = Counter(processed_values)
        best_normalized, best_count = counts.most_common(1)[0]
        # Find the original value that corresponds to the best normalized value
        best_val = valid_values[processed_values.index(best_normalized)]

    confidence = parent_valid_frac * (best_count / total_values)

    if verbose:
        print("Voting consensus!")
        print(f"\tInitial values: {values}")
        print(f"\tBest value: {best_val}")
        print(f"\tBest count: {best_count}")
        print(f"\tConfidence: {confidence}")

    return (best_val, round(confidence, 5))


###############################################################################
# 6) Non-enum primitives (or treated as primitives) => pairwise similarity "center"
###############################################################################
def string_consensus_llm(values: list[str]) -> str:
    with OpenAI(api_key="AIzaSyAkHc8uVjIUMlfaTACfcOLYeV7F9PFOtak", base_url="https://generativelanguage.googleapis.com/v1beta/openai/") as client:
        system_prompt = """
You are a helpful assistant that builds a consensus string from a list of strings.
## Context
- We are doing a voting-like document extraction task, this is just a small part of the task.
- We generate multiple response candidates (strings) for a given field, and we need to define the consensus string.

## Instructions
- You will be given a list of strings.
- You need to build a consensus string from the list of strings.
- The consensus string should be a string that is most similar to the majority of the strings in the list.
- On general, the consensus string is meant to capture the "general idea/information" of the list, not the exact wording.
- If the list is too diverse and you cannot elect a consensus string, return "Uncertain" -- But avoid this answer whenever possible.
- If the list is empty, return "Unknown".

## Output
- The output should be a raw string, not a JSON. Not enclosed in quotes.

## Examples
### Example 1
- Input: ["The sky is blue", "The sky is blue", "The sky is blue"]
- Output: The sky is blue

### Example 2
- Input: ["The sky is blue", "The sky is green", "The sky is red"]
- Output: Uncertain

### Example 3
- Input: []
- Output: Unknown

### Example 4
- Input: ["The sky is blue tonight", "The sky is blue today", "The sky is blue"]
- Output: The sky is blue

I think you got the point.
"""
        values_json_dumped = [json.dumps(v) for v in values]
        response = client.chat.completions.create(
            model="gemini-2.0-flash-lite",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Input: {values_json_dumped}\nOutput:"}],
        )
        if response.choices[0].message.content is None:
            return "Unknown"
        consensus_string = str(response.choices[0].message.content).strip()
        return consensus_string


def consensus_as_primitive(
    values: list[Any],
    consensus_settings: ConsensusSettings,
    is_last_chunk: bool,
    sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    parent_valid_frac: float = 1.0,
) -> tuple[Any, float]:
    if parent_valid_frac < consensus_settings.minimum_voters_threshold and not is_last_chunk:
        return (None, 0.0)

    non_none_values = [v for v in values if v is not None]
    if len(non_none_values) == 0:
        return (None, parent_valid_frac)
    if len(non_none_values) == 1:
        return (non_none_values[0], parent_valid_frac * (len(non_none_values) / len(values)))

    # Check if the values are string and we are using the llm-consensus method
    first_val_type = type(non_none_values[0])
    if first_val_type is str and consensus_settings.string_consensus_method == "llm-consensus" and consensus_settings.string_similarity_method == "embeddings":
        # Feed an LLM with the values and ask it to build the consensus string
        consensus_string = string_consensus_llm(non_none_values)
        # Compute the similarity between the consensus string and the values
        similarities = [generic_similarity(consensus_string, v, consensus_settings.string_similarity_method, sync_get_openai_embeddings_from_text) for v in non_none_values]
        confidence = float(np.nanmean(similarities))
        return consensus_string, confidence

    n = len(values)
    if n == 0:
        return (None, 0.0)
    if n == 1:
        return (values[0], parent_valid_frac)
    sim_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            sim = generic_similarity(values[i], values[j], consensus_settings.string_similarity_method, sync_get_openai_embeddings_from_text)
            sim_matrix[i, j] = sim_matrix[j, i] = sim
        # Set the diagonal to NaN to avoid it being considered in the average
        sim_matrix[i, i] = np.nan
    # Compute the average of the non-NaN values for each row
    avg_sims = np.nanmean(sim_matrix, axis=1)
    # Get the index of the maximum average similarity
    best_idx = int(np.argmax(avg_sims))
    # Get the best value
    best_value = values[best_idx]
    # Get the confidence score
    confidence = parent_valid_frac * float(avg_sims[best_idx])
    return (best_value, round(confidence, 5))


###############################################################################
# 7) consensus_list => merges lists or picks the best entire list
###############################################################################
def compute_similarity_scores(
    values: list[Any], consensus_settings: ConsensusSettings, sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE
) -> list[float]:
    """
    Compute similarity scores for each value against all others, without selecting a best value.
    Returns a list of scores, one for each input value.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    sim_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            sim = generic_similarity(values[i], values[j], consensus_settings.string_similarity_method, sync_get_openai_embeddings_from_text)
            sim_matrix[i, j] = sim_matrix[j, i] = sim
        sim_matrix[i, i] = 1.0

    return [float(round(score, 5)) for score in sim_matrix.mean(axis=1)]


###############################################################################
# 8) consensus_dict => merges dictionaries field by field
###############################################################################
def consensus_dict(
    dict_values: list[dict],
    consensus_settings: ConsensusSettings,
    sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    is_last_chunk: bool = False,
    parent_valid_frac: float = 1.0,
) -> tuple[dict, dict[str, float]]:
    """
    Returns (merged_dict, confidence_dict).
      - merged_dict => the final aggregated dictionary
      - confidence_dict => per-field confidence
    """
    seen = set()
    all_keys = [k for d in dict_values for k in d.keys() if k not in seen and not seen.add(k)]

    result = {}
    confs = {}

    special_field_prefix = ["reasoning___", "quote___"]

    for key in all_keys:
        sub_vals = [d.get(key, None) for d in dict_values]
        # Special handling for reasoning___ fields
        if any(prefix in key for prefix in special_field_prefix):
            # We skip reasoning and quote fields on consensus.
            continue
        else:
            val, conf = consensus_values(sub_vals, consensus_settings, sync_get_openai_embeddings_from_text, is_last_chunk=is_last_chunk, parent_valid_frac=parent_valid_frac)
            result[key] = val
            confs[key] = conf

    return (result, confs)


def consensus_list(
    list_values: list[list[Any]],
    consensus_settings: ConsensusSettings,
    sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    is_last_chunk: bool = False,
    parent_valid_frac: float = 1.0,
) -> tuple[list[Any], list[float | dict]]:
    """
    - We do element-wise merging, calling consensus_value on each index.
    - We ignore leftover items if lists differ in length.
    Returns:
      (consensus_list, confidence_list) where confidence_list contains either:
      - For primitive lists: list of float confidences
      - For object lists: list of dict confidences with _consensus_score
    """
    if not list_values:
        return ([], [])

    non_empty_list_values = [lst for lst in list_values if lst]
    if not non_empty_list_values:
        return ([], [])

    # element-wise merging
    lengths = [len(lst) for lst in list_values]
    maximum_len = max(lengths)

    if maximum_len == 0:
        return ([], [])

    final_list = []
    confidences = []
    for i in range(maximum_len):
        items = [(model_list[i] if i < len(model_list) else None) for model_list in list_values]
        val_i, conf_i = consensus_values(items, consensus_settings, sync_get_openai_embeddings_from_text, is_last_chunk=is_last_chunk, parent_valid_frac=parent_valid_frac)
        final_list.append(val_i)
        confidences.append(conf_i)

    return final_list, confidences


def intermediary_consensus_cleanup(obj: Any) -> Any:
    if isinstance(obj, dict):
        new_obj = {k: w for k, v in obj.items() if (w := intermediary_consensus_cleanup(v)) is not None}
        if len(new_obj) > 0:
            return new_obj
        return None
    elif isinstance(obj, (list, tuple)):
        new_obj = [w for v in obj if (w := intermediary_consensus_cleanup(v)) is not None]
        if len(new_obj) > 0:
            return new_obj
        return None
    elif isinstance(obj, str):
        if obj.strip() == "":
            return None
        return obj.strip()
    return obj


###############################################################################
# 9) Main Dispatcher
###############################################################################
def consensus_values(
    values: list[Any],
    consensus_settings: ConsensusSettings,
    sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    is_last_chunk: bool = False,
    parent_valid_frac: float = 1.0,
) -> tuple[Any, float | list[float | dict] | dict[str, float]]:
    """
    Decide which consensus function to call:
      - If all non-null are dict => consensus_dict
      - If all non-null are list => consensus_list
      - Else => check if it's enum => vote-based
      - Else => pairwise similarity for primitives
    Returns:
      (final_value, confidence).
      - If it's dict => (merged_dict, nested_conf_dict)
      - If it's list => (list_of_values, list_of_confidences)
      - If it's primitives => (best_val, float_conf)
    """
    if not values:
        return (None, parent_valid_frac)

    # Remove None values
    non_none_values = [v for v in values if v is not None]

    if not non_none_values:
        return (None, 0.0)

    # For enum-like types, use voting consensus
    if isinstance(non_none_values[0], str) or isinstance(non_none_values[0], bool):
        # Check if they are really enum-like types
        values_as_strings = [str(v).strip() for v in non_none_values]
        is_enum_like = all([len(v.split()) < 3 for v in values_as_strings])
        if is_enum_like:
            # Get the parent_valid_frac for these values (useful for intermediate chunks)
            return voting_consensus(values, consensus_settings, is_last_chunk, parent_valid_frac=parent_valid_frac)

    # If we have a dictionary, recursively process it
    if isinstance(non_none_values[0], dict):
        # Calculate the valid_frac for this dict's values
        dicts_only = [v for v in values if isinstance(v, dict)]
        total_valid = len(dicts_only)
        parent_valid_frac *= total_valid / len(values)

        return consensus_dict(dicts_only, consensus_settings, sync_get_openai_embeddings_from_text, is_last_chunk, parent_valid_frac=parent_valid_frac)

    # If we have a list of values, recursively process them
    if isinstance(non_none_values[0], list):
        # Calculate the valid_frac for this list's values
        lists_only = [v for v in values if isinstance(v, list)]
        total_valid = len(lists_only)
        parent_valid_frac *= total_valid / len(values)

        return consensus_list(lists_only, consensus_settings, sync_get_openai_embeddings_from_text, is_last_chunk, parent_valid_frac=parent_valid_frac)

    # 4) Otherwise => standard primitive consensus
    parent_valid_frac *= len(non_none_values) / len(values)
    if sync_get_openai_embeddings_from_text is None:
        raise ValueError("sync_get_openai_embeddings_from_text is required for primitive consensus")
    consensus_result, confidence = consensus_as_primitive(
        non_none_values, consensus_settings, is_last_chunk, sync_get_openai_embeddings_from_text, parent_valid_frac=parent_valid_frac
    )
    return consensus_result, confidence


# def consolidate_consensus_usage(result_list: list[RetabParsedChatCompletion] | list[RetabParsedChatCompletionStream]) -> CompletionUsage | None:
def consolidate_consensus_usage(result_list: list[RetabParsedChatCompletion]) -> CompletionUsage | None:
    """
    Consolidate the usage of the consensus models.
    """
    if not result_list:
        return None
    consensus_usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    for model_result in result_list:
        if model_result.usage is not None:
            # Handle main token counts
            consensus_usage.prompt_tokens += model_result.usage.prompt_tokens or 0
            consensus_usage.completion_tokens += model_result.usage.completion_tokens or 0
            consensus_usage.total_tokens += model_result.usage.total_tokens or 0

            # Handle prompt_tokens_details
            if model_result.usage.prompt_tokens_details is not None:
                if consensus_usage.prompt_tokens_details is None:
                    consensus_usage.prompt_tokens_details = PromptTokensDetails()

                # Handle audio_tokens in prompt_tokens_details
                if model_result.usage.prompt_tokens_details.audio_tokens is not None:
                    if consensus_usage.prompt_tokens_details.audio_tokens is None:
                        consensus_usage.prompt_tokens_details.audio_tokens = 0
                    consensus_usage.prompt_tokens_details.audio_tokens += model_result.usage.prompt_tokens_details.audio_tokens

                # Handle cached_tokens in prompt_tokens_details
                if model_result.usage.prompt_tokens_details.cached_tokens is not None:
                    if consensus_usage.prompt_tokens_details.cached_tokens is None:
                        consensus_usage.prompt_tokens_details.cached_tokens = 0
                    consensus_usage.prompt_tokens_details.cached_tokens += model_result.usage.prompt_tokens_details.cached_tokens

            # Handle completion_tokens_details
            if model_result.usage.completion_tokens_details is not None:
                if consensus_usage.completion_tokens_details is None:
                    consensus_usage.completion_tokens_details = CompletionTokensDetails()

                # Handle audio_tokens in completion_tokens_details
                if model_result.usage.completion_tokens_details.audio_tokens is not None:
                    if consensus_usage.completion_tokens_details.audio_tokens is None:
                        consensus_usage.completion_tokens_details.audio_tokens = 0
                    consensus_usage.completion_tokens_details.audio_tokens += model_result.usage.completion_tokens_details.audio_tokens

                # Handle other fields in completion_tokens_details
                if model_result.usage.completion_tokens_details.accepted_prediction_tokens is not None:
                    if consensus_usage.completion_tokens_details.accepted_prediction_tokens is None:
                        consensus_usage.completion_tokens_details.accepted_prediction_tokens = 0
                    consensus_usage.completion_tokens_details.accepted_prediction_tokens += model_result.usage.completion_tokens_details.accepted_prediction_tokens

                if model_result.usage.completion_tokens_details.rejected_prediction_tokens is not None:
                    if consensus_usage.completion_tokens_details.rejected_prediction_tokens is None:
                        consensus_usage.completion_tokens_details.rejected_prediction_tokens = 0
                    consensus_usage.completion_tokens_details.rejected_prediction_tokens += model_result.usage.completion_tokens_details.rejected_prediction_tokens

                if model_result.usage.completion_tokens_details.reasoning_tokens is not None:
                    if consensus_usage.completion_tokens_details.reasoning_tokens is None:
                        consensus_usage.completion_tokens_details.reasoning_tokens = 0
                    consensus_usage.completion_tokens_details.reasoning_tokens += model_result.usage.completion_tokens_details.reasoning_tokens

    return consensus_usage


# Add async versions after the existing sync functions


# Add this after the existing get_embeddings function (around line 780)
async def async_get_embeddings(s: str, async_get_openai_embeddings_from_text: Callable[[list[str]], Awaitable[list[list[float]]]]) -> list[float]:
    """Async version of get_embeddings"""
    logger.debug(f"Cache miss for '{s}'")
    result = (await async_get_openai_embeddings_from_text([s]))[0]
    return result


async def async_string_similarity(s1: str, s2: str, method: StringSimilarityMethod, async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE) -> float:
    """
    Async version of string_similarity.
    Returns a similarity score in [0,1] between two strings using the specified method.
    This version is thread-safe and handles race conditions properly.
    """
    # Check cache first
    cached_result = get_cached_similarity(s1, s2, method)
    if cached_result is not None:
        return cached_result

    result: float | None = None
    # Compute similarity if not in cache
    if method == "jaccard":
        result = jaccard_similarity(s1, s2)
    elif method == "hamming":
        result = hamming_similarity(s1, s2)
    # Only use embeddings if the strings are long enough, otherwise it is overkill and not worth the time
    elif method == "embeddings" and len(s1) > 50 and len(s2) > 50:
        try:
            emb1 = await async_get_embeddings(s1, async_get_openai_embeddings_from_text)
            emb2 = await async_get_embeddings(s2, async_get_openai_embeddings_from_text)
            result = _cosine_similarity(emb1, emb2)
        except Exception as e:
            logger.error(f"Error getting embeddings for '{s1}' and '{s2}'", exc_info=e)

    if result is None:
        # Fall-back to levenshtein
        result = levenshtein_similarity(s1, s2)

    # Cache the result
    set_cached_similarity(s1, s2, method, result)
    return result


async def async_dict_similarity(
    d1: dict, d2: dict, string_similarity_method: StringSimilarityMethod, async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE
) -> float:
    """
    Async version of dict_similarity.
    Index-based approach for comparing two dictionaries.
    """
    all_keys = set(d1.keys()) | set(d2.keys())

    # Remove IGNORED_KEY_PATTERNS from all_keys
    all_keys = [k for k in all_keys if not any(re.match(pattern, k) for pattern in IGNORED_KEY_PATTERNS)]

    if not all_keys:
        return 1.0

    total_score = 0.0
    for k in all_keys:
        v1 = d1.get(k)
        v2 = d2.get(k)
        total_score += await async_generic_similarity(v1, v2, string_similarity_method, async_get_openai_embeddings_from_text)

    return total_score / len(all_keys)


async def async_list_similarity(
    l1: list[Any] | tuple[Any, ...],
    l2: list[Any] | tuple[Any, ...],
    string_similarity_method: StringSimilarityMethod,
    async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
) -> float:
    """
    Async version of list_similarity.
    Compare two lists, returning a similarity score in [0,1].
    """
    max_len = max(len(l1), len(l2))
    if max_len == 0:
        return 1.0
    total_score = 0.0
    for i in range(max_len):
        v1 = l1[i] if i < len(l1) else None
        v2 = l2[i] if i < len(l2) else None
        total_score += await async_generic_similarity(v1, v2, string_similarity_method, async_get_openai_embeddings_from_text)
    return total_score / max_len


async def async_generic_similarity(
    v1: Any,
    v2: Any,
    string_similarity_method: StringSimilarityMethod,
    async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
) -> float:
    """
    Async version of generic_similarity.
    Compare two values, returning a similarity score in [0,1].
    """
    # both None => perfect
    if not bool(v1) and not bool(v2):
        return 1.0
    # one None => 0
    if v1 is None or v2 is None:
        return SIMILARITY_SCORE_LOWER_BOUND
    if isinstance(v1, str) and isinstance(v2, str):
        return await async_string_similarity(v1, v2, string_similarity_method, async_get_openai_embeddings_from_text)
    elif isinstance(v1, NumericalPrimitive) and isinstance(v2, NumericalPrimitive):
        return numerical_similarity(v1, v2)
    elif isinstance(v1, dict) and isinstance(v2, dict):
        return await async_dict_similarity(v1, v2, string_similarity_method, async_get_openai_embeddings_from_text)
    elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
        return await async_list_similarity(v1, v2, string_similarity_method, async_get_openai_embeddings_from_text)
    else:
        return SIMILARITY_SCORE_LOWER_BOUND


async def async_consensus_as_primitive(
    values: list[Any],
    consensus_settings: ConsensusSettings,
    is_last_chunk: bool,
    async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    parent_valid_frac: float = 1.0,
) -> tuple[Any, float]:
    """Async version of consensus_as_primitive"""
    if parent_valid_frac < consensus_settings.minimum_voters_threshold and not is_last_chunk:
        return (None, 0.0)

    non_none_values = [v for v in values if v is not None]
    if len(non_none_values) == 0:
        return (None, parent_valid_frac)
    if len(non_none_values) == 1:
        return (non_none_values[0], parent_valid_frac * (len(non_none_values) / len(values)))

    # Check if the values are string and we are using the llm-consensus method
    first_val_type = type(non_none_values[0])
    if first_val_type is str and consensus_settings.string_consensus_method == "llm-consensus" and consensus_settings.string_similarity_method == "embeddings":
        # Feed an LLM with the values and ask it to build the consensus string
        consensus_string = string_consensus_llm(non_none_values)
        # Compute the similarity between the consensus string and the values
        similarities = []
        for v in non_none_values:
            sim = await async_generic_similarity(consensus_string, v, consensus_settings.string_similarity_method, async_get_openai_embeddings_from_text)
            similarities.append(sim)
        confidence = float(np.nanmean(similarities))
        return consensus_string, confidence

    n = len(values)
    if n == 0:
        return (None, 0.0)
    if n == 1:
        return (values[0], parent_valid_frac)

    sim_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            sim = await async_generic_similarity(values[i], values[j], consensus_settings.string_similarity_method, async_get_openai_embeddings_from_text)
            sim_matrix[i, j] = sim_matrix[j, i] = sim
        # Set the diagonal to NaN to avoid it being considered in the average
        sim_matrix[i, i] = np.nan

    # Compute the average of the non-NaN values for each row
    avg_sims = np.nanmean(sim_matrix, axis=1)
    # Get the index of the maximum average similarity
    best_idx = int(np.argmax(avg_sims))
    # Get the best value
    best_value = values[best_idx]
    # Get the confidence score
    confidence = parent_valid_frac * float(avg_sims[best_idx])
    return (best_value, round(confidence, 5))


async def async_consensus_dict(
    dict_values: list[dict],
    consensus_settings: ConsensusSettings,
    async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    is_last_chunk: bool = False,
    parent_valid_frac: float = 1.0,
) -> tuple[dict, dict[str, float]]:
    """
    Async version of consensus_dict.
    Returns (merged_dict, confidence_dict).
      - merged_dict => the final aggregated dictionary
      - confidence_dict => per-field confidence
    """
    seen = set()
    all_keys = [k for d in dict_values for k in d.keys() if k not in seen and not seen.add(k)]

    result = {}
    confs = {}

    special_field_prefix = ["reasoning___", "quote___"]

    for key in all_keys:
        sub_vals = [d.get(key, None) for d in dict_values]
        # Special handling for reasoning___ fields
        if any(prefix in key for prefix in special_field_prefix):
            # We skip reasoning and quote fields on consensus.
            continue
        else:
            val, conf = await async_consensus_values(
                sub_vals, consensus_settings, async_get_openai_embeddings_from_text, is_last_chunk=is_last_chunk, parent_valid_frac=parent_valid_frac
            )
            result[key] = val
            confs[key] = conf

    return (result, confs)


async def async_consensus_list(
    list_values: list[list[Any]],
    consensus_settings: ConsensusSettings,
    async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    is_last_chunk: bool = False,
    parent_valid_frac: float = 1.0,
) -> tuple[list[Any], list[float | dict]]:
    """
    Async version of consensus_list.
    - We do element-wise merging, calling consensus_value on each index.
    - We ignore leftover items if lists differ in length.
    Returns:
      (consensus_list, confidence_list) where confidence_list contains either:
      - For primitive lists: list of float confidences
      - For object lists: list of dict confidences with _consensus_score
    """
    if not list_values:
        return ([], [])

    non_empty_list_values = [lst for lst in list_values if lst]
    if not non_empty_list_values:
        return ([], [])

    # element-wise merging
    lengths = [len(lst) for lst in list_values]
    maximum_len = max(lengths)

    if maximum_len == 0:
        return ([], [])

    final_list = []
    confidences = []
    for i in range(maximum_len):
        items = [(model_list[i] if i < len(model_list) else None) for model_list in list_values]
        val_i, conf_i = await async_consensus_values(
            items, consensus_settings, async_get_openai_embeddings_from_text, is_last_chunk=is_last_chunk, parent_valid_frac=parent_valid_frac
        )
        final_list.append(val_i)
        confidences.append(conf_i)

    return final_list, confidences


async def async_consensus_values(
    values: list[Any],
    consensus_settings: ConsensusSettings,
    async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    is_last_chunk: bool = False,
    parent_valid_frac: float = 1.0,
) -> tuple[Any, float | list[float | dict] | dict[str, float]]:
    """
    Async version of consensus_values.
    Decide which consensus function to call:
      - If all non-null are dict => consensus_dict
      - If all non-null are list => consensus_list
      - Else => check if it's enum => vote-based
      - Else => pairwise similarity for primitives
    Returns:
      (final_value, confidence).
      - If it's dict => (merged_dict, nested_conf_dict)
      - If it's list => (list_of_values, list_of_confidences)
      - If it's primitives => (best_val, float_conf)
    """
    if not values:
        return (None, parent_valid_frac)

    # Remove None values
    non_none_values = [v for v in values if v is not None]

    if not non_none_values:
        return (None, 0.0)

    # For enum-like types, use voting consensus
    if isinstance(non_none_values[0], str) or isinstance(non_none_values[0], bool):
        # Check if they are really enum-like types
        values_as_strings = [str(v).strip() for v in non_none_values]
        is_enum_like = all([len(v.split()) < 3 for v in values_as_strings])
        if is_enum_like:
            # Get the parent_valid_frac for these values (useful for intermediate chunks)
            return voting_consensus(values, consensus_settings, is_last_chunk, parent_valid_frac=parent_valid_frac)

    # If we have a dictionary, recursively process it
    if isinstance(non_none_values[0], dict):
        # Calculate the valid_frac for this dict's values
        dicts_only = [v for v in values if isinstance(v, dict)]
        total_valid = len(dicts_only)
        parent_valid_frac *= total_valid / len(values)

        return await async_consensus_dict(dicts_only, consensus_settings, async_get_openai_embeddings_from_text, is_last_chunk, parent_valid_frac=parent_valid_frac)

    # If we have a list of values, recursively process them
    if isinstance(non_none_values[0], list):
        # Calculate the valid_frac for this list's values
        lists_only = [v for v in values if isinstance(v, list)]
        total_valid = len(lists_only)
        parent_valid_frac *= total_valid / len(values)

        return await async_consensus_list(lists_only, consensus_settings, async_get_openai_embeddings_from_text, is_last_chunk, parent_valid_frac=parent_valid_frac)

    # 4) Otherwise => standard primitive consensus
    parent_valid_frac *= len(non_none_values) / len(values)
    if async_get_openai_embeddings_from_text is None:
        raise ValueError("async_get_openai_embeddings_from_text is required for primitive consensus")
    consensus_result, confidence = await async_consensus_as_primitive(
        non_none_values, consensus_settings, is_last_chunk, async_get_openai_embeddings_from_text, parent_valid_frac=parent_valid_frac
    )
    return consensus_result, confidence


async def async_lists_alignment(
    list_of_lists: list[list[Any]],
    async_sim_fn: Callable[[Any, Any], Awaitable[float]],
    min_support_ratio: float = 0.5,
    max_novelty_ratio: float = 0.25,
    reference_list_idx: Optional[int] = None,
) -> tuple[list[list[Any]], list[list[int | None]]]:
    """
    Async version of lists_alignment.
    Master function to align lists based on element similarity.
    """
    if not list_of_lists or all(not lst for lst in list_of_lists):
        return [[] for _ in list_of_lists], [[None for _ in range(len(lst))] for lst in list_of_lists]

    # Create async similarity cache
    async_sim_cache = AsyncSimilarityCache(async_sim_fn, list_of_lists)

    # Actual alignment
    if reference_list_idx is None:
        # Compute the dynamic threshold
        dynamic_threshold = await _async_compute_dynamic_threshold(async_sim_cache)
        # Get reference list and dynamically computed threshold
        reference_list = await _async_build_reference_list(async_sim_cache, min_support_ratio, max_novelty_ratio, threshold=dynamic_threshold)

        # Align lists using the final threshold (Reduce again the threshold)
        aligned = await _async_align_lists_to_reference_hungarian(async_sim_cache, reference_list, threshold=0.95 * dynamic_threshold)

        # Cap the number of elements in the aligned lists to the number of elements in the reference list
        aligned = _prune_low_support_elements(aligned, min_support_ratio)

        # Sort the aligned lists by the average index of the reference indices within the initial list of lists
        aligned, original_list_reference_indices = sort_by_original_majority(aligned, list_of_lists)
    # Known reference (ground truth)
    else:
        # Just use the reference list
        reference_list: list[Index] = [(reference_list_idx, i) for i in range(len(list_of_lists[reference_list_idx]))]

        # Align the lists to the reference list (no threshold)
        aligned = await _async_align_lists_to_reference_hungarian(async_sim_cache, reference_list, threshold=0.0)

        # We don't prune the lists, because the reference list is the truth and is already sorted
        # just build the original_list_reference_indices
        original_list_reference_indices = _original_positions(aligned, list_of_lists)

    # Return the aligned lists
    return aligned, original_list_reference_indices


async def async_recursive_list_alignments(
    values: list[Any],
    string_similarity_method: StringSimilarityMethod,
    async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    min_support_ratio: float,
    max_novelty_ratio: float = 0.25,
    current_path: str = "",
    reference_idx: Optional[int] = None,
) -> tuple[list[Any], dict[str, list[str | None]]]:
    """
    Async version of recursive_list_alignments.
    This function recursively aligns values in a list of dictionaries or lists.
    """
    # Handle empty values list
    if not values:
        return values, {}

    # If all values are None, just return them
    if all(v is None for v in values):
        return values, {current_path: [current_path for _ in values]}

    # Filter out None values for type checking
    non_nulls = [v for v in values if v is not None]

    # Make a defensive copy to avoid modifying the original values
    # Note: Using deepcopy to ensure we don't modify the original data
    values = deepcopy(values)

    # Assumption: All non-null values have the same type
    # We use the first non-null value's type to determine processing
    first_type = type(non_nulls[0])
    same_type = all(isinstance(x, first_type) for x in non_nulls)
    key_mappings: dict[str, list[str | None]] = {}

    if not same_type or first_type not in (dict, list):
        key_mappings[current_path] = [current_path if (v is not None or idx == reference_idx) else None for idx, v in enumerate(values)]
        return values, key_mappings

    # 1) All dict => Now we update each dict with the aligned values
    if first_type is dict:
        dicts_only = [(d if isinstance(d, dict) else {}) for d in values]

        # Find all unique keys across all dictionaries
        all_keys = list(set([k for d in dicts_only for k in d.keys()]))
        all_keys.sort()

        # For each key, align the values across all dictionaries
        for key in all_keys:
            values_for_key = [d.get(key) for d in dicts_only]

            _current_path = f"{current_path}.{key}" if current_path else key
            aligned_values_for_key, sub_key_mapping = await async_recursive_list_alignments(
                values_for_key,
                string_similarity_method,
                async_get_openai_embeddings_from_text,
                min_support_ratio,
                max_novelty_ratio=max_novelty_ratio,
                current_path=_current_path,
                reference_idx=reference_idx,
            )

            # Update each dictionary with its aligned value
            for _d, aligned_value in zip(dicts_only, aligned_values_for_key):
                _d[key] = aligned_value

            # Update the key mapping
            key_mappings.update(sub_key_mapping)

        # Update the values list with the dicts_only
        values = [{k: _d.get(k) for k in all_keys} for _d in dicts_only]

    # 2) All list => consensus_list
    if first_type is list:
        # Convert any non-list items to empty lists
        lists_only = [(lst if isinstance(lst, list) else []) for lst in values]
        # Initialize the original_list_reference_indices with None for bounding purposes
        original_list_reference_indices: list[list[int | None]] = [[None for _ in lst] for lst in lists_only]

        # Skip alignment for empty lists
        if any(lst for lst in lists_only):
            # Align the lists by similarity
            async def async_sim_fn(a, b):
                return await async_generic_similarity(a, b, string_similarity_method, async_get_openai_embeddings_from_text)

            aligned_lists_only, original_list_reference_indices = await async_lists_alignment(
                lists_only, async_sim_fn, min_support_ratio=min_support_ratio, max_novelty_ratio=max_novelty_ratio, reference_list_idx=reference_idx
            )

            # Update each list with its aligned values
            for l_idx, new_lst in enumerate(aligned_lists_only):
                values[l_idx] = new_lst
        else:
            # Make sure all elements are empty lists
            for i in range(len(values)):
                values[i] = []

        # Now, call the align function recursively on each position of the lists (they now have the same length)
        if len(values) > 0:
            list_length = len(values[0])
            if list_length > 0:
                for i in range(list_length):
                    values_i = [lst[i] for lst in values]
                    values_i, sub_key_mapping = await async_recursive_list_alignments(
                        values_i,
                        string_similarity_method,
                        async_get_openai_embeddings_from_text,
                        min_support_ratio,
                        max_novelty_ratio=max_novelty_ratio,
                        current_path="",
                        reference_idx=reference_idx,
                    )
                    for l_idx, new_lst in enumerate(values_i):
                        values[l_idx][i] = new_lst

                    # Update the key mapping according to the original positions.
                    for key, sub_values in sub_key_mapping.items():
                        # Build the paths correctly
                        ## The key
                        _key_path = f"{current_path}.{i}" if current_path else str(i)
                        _key_path = f"{_key_path}.{key}" if key else _key_path
                        ## The values
                        current_values = []
                        for l_idx, v in enumerate(sub_values):
                            _original_position = original_list_reference_indices[l_idx][i]
                            if _original_position is None or v is None:
                                current_values.append(None)
                            else:
                                _original_value_path = f"{current_path}.{_original_position}" if current_path else _original_position
                                _original_value_path = f"{_original_value_path}.{v}" if v else _original_value_path
                                current_values.append(_original_value_path)
                        key_mappings[_key_path] = current_values
            elif current_path:  # Don't  support empty root paths
                # All lists are empty, let's add to the key_mapping just with the root to the this path
                key_mappings[current_path] = [current_path] * len(values)

    return values, key_mappings


# Helper classes and functions for async operations
class AsyncSimilarityCache:
    """
    Async version of SimilarityCache for pairwise similarity computations.
    """

    def __init__(self, async_sim_fn: Callable[[Any, Any], Awaitable[float]], list_of_lists: list[list[Any]]):
        self.async_sim_fn = async_sim_fn
        self.cache: dict[tuple[Index, Index], float] = {}
        self.list_of_lists = list_of_lists

    async def get(self, a_idx: Index, b_idx: Index) -> float:
        key = (a_idx, b_idx)
        reverse_key = (b_idx, a_idx)

        if key in self.cache:
            return self.cache[key]
        if reverse_key in self.cache:
            return self.cache[reverse_key]

        sim = await self.async_sim_fn(
            self.list_of_lists[a_idx[0]][a_idx[1]],  # a_obj
            self.list_of_lists[b_idx[0]][b_idx[1]],  # b_obj
        )
        self.cache[key] = sim
        self.cache[reverse_key] = sim
        return sim


async def _async_compute_dynamic_threshold(async_sim_cache: AsyncSimilarityCache) -> float:
    """
    Async version of _compute_dynamic_threshold.
    Computes a dynamic similarity threshold based on the distribution of similarity scores.
    """
    list_of_lists = async_sim_cache.list_of_lists
    BASE_THRESHOLD = 0.5
    if not list_of_lists or len(list_of_lists) < 2:
        return BASE_THRESHOLD  # Default threshold if not enough lists

    similarity_scores = []
    total_lists = len(list_of_lists)

    # For each list
    for i in range(total_lists):
        list_i = list_of_lists[i]
        if not list_i:
            continue

        # Track which elements in other lists have been used by previous elements in list_i
        used_elements = {j: set() for j in range(total_lists) if j != i}

        # Process each element in list_i in order
        for k_i in range(len(list_i)):
            best_match_score = BASE_THRESHOLD  # Default lower bound
            best_match = None  # (list_idx, element_idx)

            # Find best match across all other lists among available elements
            for j in range(i + 1, total_lists):
                list_j = list_of_lists[j]
                if not list_j:
                    continue

                # Check all available elements in list_j
                for k_j in range(len(list_j)):
                    if k_j in used_elements[j]:
                        continue  # Skip if this element was already used

                    obj_index_i = (i, k_i)
                    obj_index_j = (j, k_j)

                    sim = await async_sim_cache.get(obj_index_i, obj_index_j)

                    if sim > best_match_score:
                        best_match_score = sim
                        best_match = (j, k_j)

            # If we found a match, record it and mark the element as used
            if best_match is not None and best_match_score > 0:
                similarity_scores.append(best_match_score)
                used_elements[best_match[0]].add(best_match[1])

    # Similarity scores are only the highest ones.
    # We need to get the minimum that is actually representative of this distribution (there might be outliers)
    similarity_scores.sort()
    similarity_scores = remove_outliers(similarity_scores)
    if not similarity_scores:
        return BASE_THRESHOLD
    return max(BASE_THRESHOLD, 0.95 * similarity_scores[0])


async def _async_build_reference_list(async_sim_cache: AsyncSimilarityCache, min_support_ratio: float = 0.5, max_novelty_ratio: float = 0.5, threshold: float = 0.4) -> list[Index]:
    """
    Async version of _build_reference_list.
    Builds a reference list from the healthiest list, optionally augmented with well-supported elements.
    """
    list_of_lists = async_sim_cache.list_of_lists

    # Track which positions in each list are still available for augmentation
    unused_positions = {idx: set(range(len(lst))) for idx, lst in enumerate(list_of_lists)}

    # Collect candidate elements from unused positions
    candidate_elements = [(list_idx, obj_pos) for list_idx, unused_indices in unused_positions.items() for obj_pos in unused_indices]

    # Group similar candidates and count their support
    support_groups: dict[Index, list[Index]] = defaultdict(list)
    support_groups_used_lists: dict[Index, set[int]] = defaultdict(set)

    for list_idx1, obj_pos1 in candidate_elements:
        obj_index1 = (list_idx1, obj_pos1)

        # Find the best matching existing group
        best_sim = -1
        best_group_repr_index = None

        for group_repr_index, group_used_lists in support_groups_used_lists.items():
            if list_idx1 in group_used_lists:
                # All elements in a group must come from different lists
                continue

            # Calculate the similarity between the current element and the group representative
            sim = await async_sim_cache.get(obj_index1, group_repr_index)

            # Keep track of the best match
            if sim >= threshold and sim > best_sim:
                best_sim = sim
                best_group_repr_index = group_repr_index

        # Add to best group if found, otherwise create a new group
        if best_group_repr_index is not None:
            support_groups[best_group_repr_index].append(obj_index1)
            support_groups_used_lists[best_group_repr_index].add(list_idx1)

            # Now we re-elect the group representative
            # Create a dummy function for embeddings since we're only dealing with indices here
            async def dummy_embeddings_fn(strings):
                return [[0.0] * 10 for _ in strings]  # Return dummy embeddings

            new_group_repr_index, _ = await async_consensus_as_primitive(
                support_groups[best_group_repr_index], ConsensusSettings(), is_last_chunk=True, async_get_openai_embeddings_from_text=dummy_embeddings_fn
            )
            if new_group_repr_index != best_group_repr_index:
                support_groups[new_group_repr_index] = support_groups[best_group_repr_index]
                support_groups_used_lists[new_group_repr_index] = support_groups_used_lists[best_group_repr_index]
                del support_groups[best_group_repr_index]
                del support_groups_used_lists[best_group_repr_index]
        else:
            support_groups[obj_index1] = [obj_index1]
            support_groups_used_lists[obj_index1] = {list_idx1}

    support_ratios: dict[Index, float] = {k: len(v) / len(list_of_lists) for k, v in support_groups.items()}
    # Sort and filter the support_counter dict
    support_ratios = {k: v for k, v in support_ratios.items() if v >= min_support_ratio}
    support_ratios = dict(sorted(support_ratios.items(), key=lambda x: (-x[1], x[0])))

    # Add well-supported candidates to reference list
    reference_list = list(support_ratios.keys())

    return reference_list


async def _async_align_lists_to_reference_hungarian(
    async_sim_cache: AsyncSimilarityCache,
    reference_indices: list[tuple[int, int]],
    threshold: float = 0.4,
) -> list[list[Any]]:
    """Async version of _align_lists_to_reference_hungarian"""
    list_of_lists = async_sim_cache.list_of_lists
    n_lists = len(list_of_lists)
    n_refs = len(reference_indices)

    aligned_lists = [[None for _ in range(n_refs)] for _ in range(n_lists)]
    if not reference_indices:
        return aligned_lists

    # Match each list to the reference
    for list_idx, lst in enumerate(list_of_lists):
        n_objs = len(lst)
        if n_objs == 0:
            continue

        sim_matrix = np.full((n_refs, n_objs), -np.inf)

        for ref_pos, (ref_list_idx, ref_obj_pos) in enumerate(reference_indices):
            ref_index = (ref_list_idx, ref_obj_pos)

            for obj_pos, obj in enumerate(lst):
                obj_index = (list_idx, obj_pos)

                if obj_index == ref_index:
                    # This element is part of the reference, skip similarity computation
                    sim_matrix[ref_pos, obj_pos] = 1.0
                    continue

                sim = await async_sim_cache.get(obj_index, ref_index)
                sim_matrix[ref_pos, obj_pos] = sim

        cost_matrix = 1.0 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for ref_pos, obj_pos in zip(row_ind, col_ind):
            sim = sim_matrix[ref_pos, obj_pos]
            if sim >= threshold and aligned_lists[list_idx][ref_pos] is None:
                aligned_lists[list_idx][ref_pos] = lst[obj_pos]

    return aligned_lists
