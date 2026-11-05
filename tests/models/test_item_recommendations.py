import numpy as np
import pytest
from scipy.sparse import csr_matrix

from kion_recs.models.item_to_item import recommend_from_neighbors


def test_recommendations_sum_similarities_and_exclude_seen() -> None:
    neighbors = csr_matrix(
        np.array(
            [
                [0.0, 0.9, 0.3, 0.6],
                [0.9, 0.0, 0.4, 0.0],
                [0.3, 0.4, 0.0, 0.0],
                [0.6, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    recommendations = recommend_from_neighbors(
        neighbors,
        seen_indices={0, 1},
        k=2,
    )

    assert recommendations == [2, 3]


def test_recommendations_break_ties_by_index() -> None:
    neighbors = csr_matrix(
        np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    recommendations = recommend_from_neighbors(
        neighbors,
        seen_indices={0},
        k=1,
    )

    assert recommendations == [1]


def test_recommendations_return_empty_without_history() -> None:
    neighbors = csr_matrix((3, 3), dtype=np.float32)

    assert recommend_from_neighbors(neighbors, set()) == []


def test_recommendations_return_empty_without_positive_candidates() -> None:
    neighbors = csr_matrix((3, 3), dtype=np.float32)

    assert recommend_from_neighbors(neighbors, {0}) == []


@pytest.mark.parametrize("k", [0, -1])
def test_recommendations_reject_non_positive_k(k: int) -> None:
    neighbors = csr_matrix((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="k must be positive"):
        recommend_from_neighbors(neighbors, {0}, k=k)


@pytest.mark.parametrize("invalid_index", [-1, 3])
def test_recommendations_reject_invalid_history(invalid_index: int) -> None:
    neighbors = csr_matrix((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="invalid index"):
        recommend_from_neighbors(neighbors, {invalid_index})


def test_recommendations_reject_non_square_matrix() -> None:
    neighbors = csr_matrix((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="square matrix"):
        recommend_from_neighbors(neighbors, {0})
