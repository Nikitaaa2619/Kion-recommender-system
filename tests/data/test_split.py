import pandas as pd
import pytest

from kion_recs.data.split import temporal_split


def test_temporal_split_assigns_boundary_dates_correctly() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [101, 102, 103, 104, 105],
            "last_watch_dt": pd.to_datetime(
                [
                    "2021-08-08",
                    "2021-08-09",
                    "2021-08-15",
                    "2021-08-16",
                    "2021-08-22",
                ]
            ),
        }
    )

    train, validation, test = temporal_split(interactions)

    assert train["item_id"].tolist() == [101]
    assert validation["item_id"].tolist() == [102, 103]
    assert test["item_id"].tolist() == [104, 105]
    assert len(train) + len(validation) + len(test) == len(interactions)


def test_temporal_split_rejects_reversed_boundaries() -> None:
    interactions = pd.DataFrame({"last_watch_dt": pd.to_datetime(["2021-08-10"])})

    with pytest.raises(ValueError, match="validation_start"):
        temporal_split(
            interactions,
            validation_start="2021-08-16",
            test_start="2021-08-09",
        )


def test_temporal_split_rejects_text_dates() -> None:
    interactions = pd.DataFrame({"last_watch_dt": ["2021-08-10"]})

    with pytest.raises(ValueError, match="datetime dtype"):
        temporal_split(interactions)


def test_temporal_split_rejects_missing_dates() -> None:
    interactions = pd.DataFrame({"last_watch_dt": pd.to_datetime(["2021-08-10", None])})

    with pytest.raises(ValueError, match="missing values"):
        temporal_split(interactions)
