import pandas as pd
import pytest

from kion_recs.data.ranker_split import split_ranker_training


def test_ranker_split_assigns_boundary_dates() -> None:
    interactions = pd.DataFrame(
        {
            "item_id": [10, 20, 30],
            "last_watch_dt": pd.to_datetime(
                [
                    "2021-08-01",
                    "2021-08-02",
                    "2021-08-08",
                ]
            ),
        }
    )

    history, targets = split_ranker_training(interactions)

    assert history["item_id"].tolist() == [10]
    assert targets["item_id"].tolist() == [20, 30]
    assert len(history) + len(targets) == len(interactions)


@pytest.mark.parametrize("future_date", ["2021-08-09", "2021-08-16"])
def test_ranker_split_rejects_future_interactions(future_date: str) -> None:
    interactions = pd.DataFrame(
        {"last_watch_dt": pd.to_datetime(["2021-08-01", "2021-08-02", future_date])}
    )

    with pytest.raises(ValueError, match="must not contain dates"):
        split_ranker_training(interactions)


@pytest.mark.parametrize(
    "dates",
    [
        ["2021-08-01"],
        ["2021-08-02"],
    ],
)
def test_ranker_split_rejects_empty_period(dates: list[str]) -> None:
    interactions = pd.DataFrame({"last_watch_dt": pd.to_datetime(dates)})

    with pytest.raises(ValueError, match="must both contain interactions"):
        split_ranker_training(interactions)
