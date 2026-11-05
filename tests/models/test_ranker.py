from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRanker

from kion_recs.ranker_data import FEATURE_COLUMNS
from kion_recs.train_ranker import fit_ranker, select_informative_groups


def make_training_data() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "user_id": [2, 1, 2, 1, 3, 3, 4, 4],
            "item_id": [30, 10, 40, 20, 50, 60, 70, 80],
            "target": [1, 0, 0, 1, 0, 0, 1, 1],
        }
    )

    for column in FEATURE_COLUMNS:
        data[column] = np.ones(len(data), dtype=np.float32)

    data["rank_popularity"] = np.array(
        [2, 2, 1, 1, 1, 2, 1, 2],
        dtype=np.float32,
    )
    data["rank_als"] = np.nan
    data["rank_item_to_item"] = np.nan

    return data


def test_ranker_keeps_only_groups_with_both_labels() -> None:
    data = make_training_data()

    result = select_informative_groups(data)

    assert result["user_id"].tolist() == [1, 1, 2, 2]
    assert result["item_id"].tolist() == [10, 20, 30, 40]
    assert result["target"].tolist() == [0, 1, 1, 0]
    assert len(data) == 8


@pytest.mark.parametrize("label", [0, 1])
def test_ranker_rejects_only_constant_groups(label: int) -> None:
    data = make_training_data()
    data["target"] = label

    with pytest.raises(ValueError, match="both positive and negative"):
        select_informative_groups(data)


def test_ranker_rejects_non_binary_targets() -> None:
    data = make_training_data()
    data.loc[0, "target"] = 2

    with pytest.raises(ValueError, match="Targets must be binary"):
        select_informative_groups(data)


def test_ranker_predicts_and_survives_save_load(tmp_path: Path) -> None:
    training = select_informative_groups(make_training_data())

    model = fit_ranker(training, iterations=3)
    predictions = model.predict(training[FEATURE_COLUMNS])

    assert model.tree_count_ == 3
    assert model.feature_names_ == FEATURE_COLUMNS
    assert predictions.shape == (len(training),)
    assert np.isfinite(predictions).all()

    model_path = tmp_path / "ranker.cbm"
    model.save_model(str(model_path))

    restored = CatBoostRanker()
    restored.load_model(str(model_path))
    restored_predictions = restored.predict(training[FEATURE_COLUMNS])

    np.testing.assert_allclose(
        restored_predictions,
        predictions,
        rtol=1e-7,
        atol=1e-7,
    )
