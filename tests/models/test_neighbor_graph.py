import numpy as np
import pytest
from scipy.sparse import csr_matrix

from kion_recs.models.item_to_item import build_item_profiles
from kion_recs.train_item_to_item import build_neighbor_graph


@pytest.mark.parametrize("batch_size", [1, 2, 128])
def test_neighbor_graph_selects_top_neighbors(batch_size: int) -> None:
    matrix = csr_matrix(
        np.array(
            [
                [1, 1, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
    )
    profiles = build_item_profiles(matrix)

    neighbors = build_neighbor_graph(
        profiles,
        k=1,
        batch_size=batch_size,
    )

    expected = np.array(
        [
            [0.0, 1 / np.sqrt(2), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert isinstance(neighbors, csr_matrix)
    assert neighbors.shape == (4, 4)
    assert neighbors.nnz == 3
    np.testing.assert_allclose(neighbors.toarray(), expected, rtol=1e-6)


@pytest.mark.parametrize("k", [0, -1])
def test_neighbor_graph_rejects_non_positive_k(k: int) -> None:
    profiles = csr_matrix(np.eye(2, dtype=np.float32))

    with pytest.raises(ValueError, match="k must be positive"):
        build_neighbor_graph(profiles, k=k)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_neighbor_graph_rejects_non_positive_batch_size(
    batch_size: int,
) -> None:
    profiles = csr_matrix(np.eye(2, dtype=np.float32))

    with pytest.raises(ValueError, match="batch_size must be positive"):
        build_neighbor_graph(profiles, batch_size=batch_size)


def test_neighbor_graph_supports_empty_catalog() -> None:
    profiles = csr_matrix((0, 3), dtype=np.float32)

    neighbors = build_neighbor_graph(profiles)

    assert neighbors.shape == (0, 0)
    assert neighbors.nnz == 0
