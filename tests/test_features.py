import numpy as np
import pandas as pd
import pytest

from kion_recs.features import add_history_features


def test_features_use_unique_counts_and_handle_cold_users() -> None:
    history = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "item_id": [10, 20, 20, 10, 30],
            "last_watch_dt": pd.to_datetime(
                [
                    "2021-07-25",
                    "2021-07-26",
                    "2021-08-01",
                    "2021-07-26",
                    "2021-07-25",
                ]
            ),
        }
    )
    candidates = pd.DataFrame(
        {
            "user_id": [2, 1, 99],
            "item_id": [20, 10, 30],
            "rank_popularity": pd.array([None, 1, 5], dtype="Int32"),
            "target": [1, 0, 0],
        }
    )

    result = add_history_features(
        candidates,
        history,
        cutoff="2021-08-02",
    )

    expected = candidates.copy()
    expected["user_n_items"] = pd.Series([2, 2, 0], dtype="int32")
    expected["user_days_since_last_watch"] = pd.Series(
        [7.0, 1.0, np.nan],
        dtype="float32",
    )
    expected["item_n_users"] = pd.Series([1, 2, 1], dtype="int32")
    expected["item_n_users_7d"] = pd.Series([1, 1, 0], dtype="int32")
    expected["is_cold_user"] = pd.Series([0, 0, 1], dtype="int8")

    expected = expected[result.columns]

    pd.testing.assert_frame_equal(result, expected)
    assert "user_n_items" not in candidates.columns


@pytest.mark.parametrize("future_date", ["2021-08-02", "2021-08-03"])
def test_features_reject_history_from_target_period(future_date: str) -> None:
    history = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "last_watch_dt": pd.to_datetime([future_date]),
        }
    )
    candidates = pd.DataFrame({"user_id": [1], "item_id": [20]})

    with pytest.raises(ValueError, match="earlier than cutoff"):
        add_history_features(candidates, history, cutoff="2021-08-02")


def test_features_reject_missing_history_dates() -> None:
    history = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "last_watch_dt": pd.to_datetime([None]),
        }
    )
    candidates = pd.DataFrame({"user_id": [1], "item_id": [20]})

    with pytest.raises(ValueError, match="dates must be present"):
        add_history_features(candidates, history, cutoff="2021-08-02")


def test_features_reject_text_history_dates() -> None:
    history = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "last_watch_dt": ["2021-08-01"],
        }
    )
    candidates = pd.DataFrame({"user_id": [1], "item_id": [20]})

    with pytest.raises(ValueError, match="datetime dtype"):
        add_history_features(candidates, history, cutoff="2021-08-02")


def test_features_reject_empty_history() -> None:
    history = pd.DataFrame(
        {
            "user_id": [],
            "item_id": [],
            "last_watch_dt": pd.to_datetime([]),
        }
    )
    candidates = pd.DataFrame({"user_id": [1], "item_id": [20]})

    with pytest.raises(ValueError, match="History must not be empty"):
        add_history_features(candidates, history, cutoff="2021-08-02")
