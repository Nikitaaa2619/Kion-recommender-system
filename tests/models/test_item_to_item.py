import numpy as np
import pytest
from scipy.sparse import csr_matrix

from kion_recs.models.item_to_item import (
    build_item_profiles,
    find_similar_items,
)


def test_similar_items_use_cosine_similarity() -> None:
    matrix = csr_matrix(
        np.array(
            [
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
            ],
            dtype=np.float32,
        )
    )

    profiles = build_item_profiles(matrix)
    indices, scores = find_similar_items(profiles, item_index=0, k=3)

    assert profiles.shape == (4, 3)
    assert indices == [1]
    assert scores == pytest.approx([1 / np.sqrt(2)])
    assert np.isfinite(profiles.data).all()


def test_similar_items_break_ties_by_index() -> None:
    matrix = csr_matrix(
        np.array(
            [
                [1, 1, 1],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )
    )

    profiles = build_item_profiles(matrix)
    indices, scores = find_similar_items(profiles, item_index=0, k=2)

    assert indices == [1, 2]
    assert scores == pytest.approx([1.0, 1.0])


def test_similar_items_return_empty_for_item_without_viewers() -> None:
    matrix = csr_matrix(
        np.array(
            [
                [1, 0],
                [1, 0],
            ],
            dtype=np.float32,
        )
    )

    profiles = build_item_profiles(matrix)
    indices, scores = find_similar_items(profiles, item_index=1)

    assert indices == []
    assert scores == []


@pytest.mark.parametrize("k", [0, -1])
def test_similar_items_reject_non_positive_k(k: int) -> None:
    profiles = csr_matrix(np.eye(2, dtype=np.float32))

    with pytest.raises(ValueError, match="k must be positive"):
        find_similar_items(profiles, item_index=0, k=k)


@pytest.mark.parametrize("item_index", [-1, 2])
def test_similar_items_reject_invalid_index(item_index: int) -> None:
    profiles = csr_matrix(np.eye(2, dtype=np.float32))

    with pytest.raises(ValueError, match="item_index is out of range"):
        find_similar_items(profiles, item_index=item_index)
