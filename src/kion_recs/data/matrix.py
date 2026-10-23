import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_interaction_matrix(
    train: pd.DataFrame,
) -> tuple[csr_matrix, pd.Index, pd.Index]:
    if train.empty:
        raise ValueError("train must not be empty")

    if train[["user_id", "item_id"]].isna().any().any():
        raise ValueError("user_id and item_id must not contain missing values")

    user_ids = pd.Index(sorted(train["user_id"].unique()), name="user_id")
    item_ids = pd.Index(sorted(train["item_id"].unique()), name="item_id")

    row_indices = user_ids.get_indexer(train["user_id"])
    column_indices = item_ids.get_indexer(train["item_id"])

    values = np.ones(len(train), dtype=np.float32)

    matrix = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )

    matrix.data.fill(1.0)

    return matrix, user_ids, item_ids
