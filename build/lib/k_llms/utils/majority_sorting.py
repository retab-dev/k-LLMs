import heapq
from typing import Any, Optional


# ---------------------------------------------------------------------------#
# 1.  Build, for every aligned cell, its original index (or None)            #
# ---------------------------------------------------------------------------#
def _original_positions(
    aligned: list[list[Any]],
    originals: list[list[Any]],
) -> list[list[Optional[int]]]:
    pos: list[list[Optional[int]]] = [[None] * len(aligned[0]) for _ in aligned]
    for r, (row_al, row_orig) in enumerate(zip(aligned, originals)):
        lookup = {id(obj): k for k, obj in enumerate(row_orig)}
        for c, x in enumerate(row_al):
            if x is not None:
                k = lookup.get(id(x))
                if k is not None:
                    pos[r][c] = k
    return pos


# ---------------------------------------------------------------------------#
# 2.  Pair-wise majority wins, based on original indices                     #
# ---------------------------------------------------------------------------#
def _pairwise_wins(pos: list[list[Optional[int]]]) -> list[list[int]]:
    n_cols = len(pos[0])
    wins = [[0] * n_cols for _ in range(n_cols)]
    for row in pos:
        present = [(c, k) for c, k in enumerate(row) if k is not None]
        for i, ki in present:
            for j, kj in present:
                if ki < kj:
                    wins[i][j] += 1
    return wins


def _majority_graph(wins: list[list[int]]) -> tuple[list[set[int]], list[int]]:
    n = len(wins)
    adj: list[set[int]] = [set() for _ in range(n)]
    indeg: list[int] = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and wins[i][j] > wins[j][i]:
                adj[i].add(j)
                indeg[j] += 1
    return adj, indeg


def _avg_original_pos(pos: list[list[Optional[int]]]) -> list[float]:
    n_cols = len(pos[0])
    s, cnt = [0.0] * n_cols, [0] * n_cols
    for row in pos:
        for c, k in enumerate(row):
            if k is not None:
                s[c] += k
                cnt[c] += 1
    return [s[c] / cnt[c] if cnt[c] else float("inf") for c in range(n_cols)]


def _toposort(adj: list[set[int]], indeg: list[int], key: list[float]) -> list[int]:
    heap = [(key[c], c) for c, d in enumerate(indeg) if d == 0]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (key[v], v))
    return order


# ---------------------------------------------------------------------------#
# 3.  Public function — plug it in after alignment                           #
# ---------------------------------------------------------------------------#
def sort_by_original_majority(
    aligned_list_of_lists: list[list[Any]],
    initial_list_of_lists: list[list[Any]],
) -> tuple[list[list[Any]], list[list[Optional[int]]]]:
    """
    Re-order the columns of *aligned_list_of_lists* so they follow the pair-wise
    majority order observed in *initial_list_of_lists*.

    Returns
    -------
    sorted_aligned_lists
    sorted_original_indices   (same shape, ints or None)
    """
    if not aligned_list_of_lists:
        return aligned_list_of_lists, [[None for _ in row] for row in aligned_list_of_lists]

    # 1) map every cell back to its native position
    pos = _original_positions(aligned_list_of_lists, initial_list_of_lists)

    # 2) build majority graph
    wins = _pairwise_wins(pos)
    adj, indeg = _majority_graph(wins)
    tie_key = _avg_original_pos(pos)
    col_order = _toposort(adj, indeg, tie_key)

    #   …append any columns trapped in a Condorcet cycle
    if len(col_order) < len(aligned_list_of_lists[0]):
        left = [c for c in range(len(aligned_list_of_lists[0])) if c not in col_order]
        col_order.extend(sorted(left, key=lambda c: tie_key[c]))

    # 3) apply the permutation to both tables
    sorted_lists = [[row[c] for c in col_order] for row in aligned_list_of_lists]
    sorted_original_indices = [[row[c] for c in col_order] for row in pos]

    return sorted_lists, sorted_original_indices
