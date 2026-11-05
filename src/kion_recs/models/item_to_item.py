import numpy as np
from scipy.sparse import csr_matrix


def build_item_profiles(matrix: csr_matrix) -> csr_matrix:
    item_users = matrix.T.tocsr()

    squared_norms = np.asarray(item_users.multiply(item_users).sum(axis=1)).ravel()
    norms = np.sqrt(squared_norms)

    inverse_norms = np.divide(
        np.ones_like(norms),
        norms,
        out=np.zeros_like(norms),
        where=norms > 0,
    )

    return item_users.multiply(inverse_norms[:, None]).tocsr()


def find_similar_items(
    item_profiles: csr_matrix,
    item_index: int,
    k: int = 10,
) -> tuple[list[int], list[float]]:
    if k <= 0:
        raise ValueError("k must be positive")

    if not 0 <= item_index < item_profiles.shape[0]:
        raise ValueError("item_index is out of range")

    scores = (item_profiles[item_index] @ item_profiles.T).toarray().ravel()

    scores[item_index] = 0.0

    indices = np.arange(len(scores))
    ranked_indices = np.lexsort((indices, -scores))
    positive_indices = ranked_indices[scores[ranked_indices] > 0][:k]

    return positive_indices.tolist(), scores[positive_indices].tolist()


def recommend_from_neighbors(
    neighbors: csr_matrix,
    seen_indices: set[int],
    k: int = 10,
) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive")

    if neighbors.shape[0] != neighbors.shape[1]:
        raise ValueError("neighbors must be a square matrix")

    if not seen_indices:
        return []

    if any(index < 0 or index >= neighbors.shape[0] for index in seen_indices):
        raise ValueError("seen_indices contain an invalid index")

    history = sorted(seen_indices)
    scores = np.asarray(neighbors[history].sum(axis=0)).ravel()

    scores[history] = 0.0

    indices = np.arange(len(scores))
    ranked_indices = np.lexsort((indices, -scores))
    positive_indices = ranked_indices[scores[ranked_indices] > 0][:k]

    return positive_indices.tolist()
