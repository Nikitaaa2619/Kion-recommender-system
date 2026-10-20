import pandas as pd
import pytest

from kion_recs.models.popularity import fit_popularity, recommend_popular


def test_fit_popularity_counts_unique_users_and_breaks_ties() -> None:
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 1, 3, 4],
            "item_id": [20, 20, 20, 10, 10, 30],
        }
    )

    popularity = fit_popularity(train)

    assert popularity["item_id"].tolist() == [10, 20, 30]
    assert popularity["n_users"].tolist() == [2, 2, 1]


def test_recommend_popular_excludes_seen_items() -> None:
    popularity = pd.DataFrame(
        {
            "item_id": [10, 20, 30],
            "n_users": [5, 4, 3],
        }
    )

    recommendations = recommend_popular(popularity, seen_items={10}, k=2)

    assert recommendations == [20, 30]


def test_recommend_popular_supports_new_users() -> None:
    popularity = pd.DataFrame(
        {
            "item_id": [10, 20, 30],
            "n_users": [5, 4, 3],
        }
    )

    recommendations = recommend_popular(popularity, seen_items=set(), k=2)

    assert recommendations == [10, 20]


def test_recommend_popular_returns_available_items_when_fewer_than_k() -> None:
    popularity = pd.DataFrame(
        {
            "item_id": [10, 20],
            "n_users": [5, 4],
        }
    )

    recommendations = recommend_popular(popularity, seen_items={10}, k=10)

    assert recommendations == [20]


@pytest.mark.parametrize("k", [0, -1])
def test_recommend_popular_rejects_non_positive_k(k: int) -> None:
    popularity = pd.DataFrame(
        {
            "item_id": [10],
            "n_users": [5],
        }
    )

    with pytest.raises(ValueError, match="k must be positive"):
        recommend_popular(popularity, seen_items=set(), k=k)
