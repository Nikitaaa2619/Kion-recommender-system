import pandas as pd
import pytest

from kion_recs.data.audit import validate_interactions


def make_valid_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [101],
            "last_watch_dt": ["2021-08-01"],
            "total_dur": [3600],
            "watched_pct": [100.0],
        }
    )


def test_validate_interactions_accepts_valid_schema() -> None:
    interactions = make_valid_interactions()

    assert validate_interactions(interactions) is None


def test_validate_interactions_rejects_missing_column() -> None:
    interactions = make_valid_interactions().drop(columns=["watched_pct"])

    with pytest.raises(ValueError, match="watched_pct"):
        validate_interactions(interactions)
