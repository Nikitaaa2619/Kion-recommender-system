from pathlib import Path

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares

from kion_recs.data.matrix import build_interaction_matrix
from kion_recs.models.als import fit_als


def test_als_filters_seen_items_and_survives_save_load(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "user_id": [100, 100, 200, 200, 300, 300],
            "item_id": [10, 20, 20, 30, 30, 40],
        }
    )
    matrix, user_ids, item_ids = build_interaction_matrix(train)

    model = fit_als(matrix)

    assert model.user_factors.shape == (3, 32)
    assert model.item_factors.shape == (4, 32)
    assert np.isfinite(model.user_factors).all()
    assert np.isfinite(model.item_factors).all()

    user_index = user_ids.get_loc(100)
    indices, _ = model.recommend(
        userid=user_index,
        user_items=matrix[user_index],
        N=2,
        filter_already_liked_items=True,
    )
    recommendations = item_ids.take(indices).tolist()

    assert len(recommendations) == 2
    assert len(set(recommendations)) == 2
    assert set(recommendations) == {30, 40}

    model_path = tmp_path / "model.npz"
    model.save(model_path)
    restored = AlternatingLeastSquares.load(model_path)

    np.testing.assert_array_equal(
        restored.user_factors,
        model.user_factors,
    )
    np.testing.assert_array_equal(
        restored.item_factors,
        model.item_factors,
    )
