import numpy as np
import pandas as pd
import pytest

from kion_recs.bootstrap_ranker import pair_recall_metrics


def make_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [2, 1, 2, 1],
            "group": ["cold", "warm", "cold", "warm"],
            "model": ["ranker", "item_to_item", "item_to_item", "ranker"],
            "recall": [0.6, 0.2, 0.5, 0.4],
        }
    )


def test_pairing_matches_users_instead_of_row_positions() -> None:
    result = pair_recall_metrics(make_metrics())

    expected = pd.DataFrame(
        {
            "user_id": [1, 2],
            "group": ["warm", "cold"],
            "recall_baseline": [0.2, 0.5],
            "recall_ranker": [0.4, 0.6],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_pairing_is_independent_of_input_order() -> None:
    metrics = make_metrics()
    shuffled = metrics.sample(frac=1, random_state=7)

    original_result = pair_recall_metrics(metrics)
    shuffled_result = pair_recall_metrics(shuffled)

    pd.testing.assert_frame_equal(original_result, shuffled_result)


def test_pairing_rejects_different_user_sets() -> None:
    metrics = make_metrics().drop(index=0)

    with pytest.raises(ValueError, match="exactly the same users"):
        pair_recall_metrics(metrics)


def test_pairing_rejects_different_user_groups() -> None:
    metrics = make_metrics()
    metrics.loc[0, "group"] = "warm"

    with pytest.raises(ValueError, match="groups must match"):
        pair_recall_metrics(metrics)


def test_pairing_rejects_duplicate_model_user_rows() -> None:
    metrics = make_metrics()
    metrics = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="one metric row per user"):
        pair_recall_metrics(metrics)


@pytest.mark.parametrize("invalid_recall", [-0.1, 1.1, np.nan, np.inf])
def test_pairing_rejects_invalid_recall(invalid_recall: float) -> None:
    metrics = make_metrics()
    metrics.loc[0, "recall"] = invalid_recall

    with pytest.raises(
        ValueError,
        match="missing values|between 0 and 1",
    ):
        pair_recall_metrics(metrics)


def test_pairing_requires_both_models() -> None:
    metrics = make_metrics()
    metrics = metrics.loc[metrics["model"] == "item_to_item"]

    with pytest.raises(ValueError, match="Both models"):
        pair_recall_metrics(metrics)
