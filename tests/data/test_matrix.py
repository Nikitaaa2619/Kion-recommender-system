import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from kion_recs.data.matrix import build_interaction_matrix


def test_matrix_preserves_user_item_relationships() -> None:
    train = pd.DataFrame(
        {
            "user_id": [200, 100, 200],
            "item_id": [900, 500, 500],
        }
    )

    matrix, user_ids, item_ids = build_interaction_matrix(train)

    assert isinstance(matrix, csr_matrix)
    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 2)
    assert user_ids.tolist() == [100, 200]
    assert item_ids.tolist() == [500, 900]

    np.testing.assert_array_equal(
        matrix.toarray(),
        [
            [1.0, 0.0],
            [1.0, 1.0],
        ],
    )

    assert user_ids[1] == 200
    assert item_ids[1] == 900


def test_matrix_counts_duplicate_pairs_once() -> None:
    train = pd.DataFrame(
        {
            "user_id": [100, 100],
            "item_id": [500, 500],
        }
    )

    matrix, _, _ = build_interaction_matrix(train)

    assert matrix.nnz == 1
    assert matrix[0, 0] == 1.0


def test_matrix_rejects_empty_train() -> None:
    train = pd.DataFrame(columns=["user_id", "item_id"])

    with pytest.raises(ValueError, match="train must not be empty"):
        build_interaction_matrix(train)


@pytest.mark.parametrize("column", ["user_id", "item_id"])
def test_matrix_rejects_missing_ids(column: str) -> None:
    train = pd.DataFrame(
        {
            "user_id": [100],
            "item_id": [500],
        }
    )
    train[column] = None

    with pytest.raises(ValueError, match="must not contain missing values"):
        build_interaction_matrix(train)
