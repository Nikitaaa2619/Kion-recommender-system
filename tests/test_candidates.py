import pandas as pd

from kion_recs.candidates import merge_candidate_lists


def test_candidates_merge_sources_and_exclude_seen() -> None:
    result = merge_candidate_lists(
        popularity=[10, 20, 30],
        als=[20, 40],
        item_to_item=[40, 50, 10],
        seen_items={10},
    )

    expected = pd.DataFrame(
        {
            "item_id": [20, 30, 40, 50],
            "rank_popularity": [2, 3, None, None],
            "rank_als": [1, None, 2, None],
            "rank_item_to_item": [None, None, 1, 2],
            "n_sources": [2, 1, 2, 1],
        }
    ).astype(
        {
            "item_id": "int64",
            "rank_popularity": "Int32",
            "rank_als": "Int32",
            "rank_item_to_item": "Int32",
            "n_sources": "int8",
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_candidates_keep_first_rank_for_duplicate_item() -> None:
    result = merge_candidate_lists(
        popularity=[20, 20, 30],
        als=[],
        item_to_item=[],
        seen_items=set(),
    )

    assert result["item_id"].tolist() == [20, 30]
    assert result["rank_popularity"].tolist() == [1, 3]
    assert result["n_sources"].tolist() == [1, 1]


def test_candidates_support_popularity_only() -> None:
    result = merge_candidate_lists(
        popularity=[30, 10],
        als=[],
        item_to_item=[],
        seen_items=set(),
    )

    assert result["item_id"].tolist() == [10, 30]
    assert result["rank_popularity"].tolist() == [2, 1]
    assert result["rank_als"].isna().all()
    assert result["rank_item_to_item"].isna().all()


def test_candidates_support_empty_lists() -> None:
    result = merge_candidate_lists(
        popularity=[],
        als=[],
        item_to_item=[],
        seen_items=set(),
    )

    assert result.empty
    assert result.columns.tolist() == [
        "item_id",
        "rank_popularity",
        "rank_als",
        "rank_item_to_item",
        "n_sources",
    ]
    assert str(result["rank_als"].dtype) == "Int32"


def test_candidates_return_empty_when_all_items_seen() -> None:
    result = merge_candidate_lists(
        popularity=[10],
        als=[20],
        item_to_item=[10, 20],
        seen_items={10, 20},
    )

    assert result.empty
