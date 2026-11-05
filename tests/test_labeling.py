import pandas as pd
import pytest

from kion_recs.labeling import label_candidates


def test_labeling_preserves_candidates_and_matches_both_ids() -> None:
    candidates = pd.DataFrame(
        {
            "user_id": [2, 1, 1],
            "item_id": [10, 20, 10],
            "rank_popularity": pd.array([1, None, 3], dtype="Int32"),
        }
    )
    targets = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 3],
            "item_id": [20, 20, 10, 99],
        }
    )

    result = label_candidates(candidates, targets)

    expected = candidates.copy()
    expected["target"] = pd.Series([0, 1, 1], dtype="int8")

    pd.testing.assert_frame_equal(result, expected)
    assert "target" not in candidates.columns


def test_labeling_assigns_zero_when_targets_are_empty() -> None:
    candidates = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
        }
    )
    targets = pd.DataFrame(
        {"user_id": [], "item_id": []},
        dtype="int64",
    )

    result = label_candidates(candidates, targets)

    assert result["target"].tolist() == [0, 0]


def test_labeling_supports_empty_candidates() -> None:
    candidates = pd.DataFrame(
        {"user_id": [], "item_id": []},
        dtype="int64",
    )
    targets = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
        }
    )

    result = label_candidates(candidates, targets)

    assert result.empty
    assert result.columns.tolist() == ["user_id", "item_id", "target"]
    assert str(result["target"].dtype) == "int8"


def test_labeling_rejects_duplicate_candidates() -> None:
    candidates = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 10],
        }
    )
    targets = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
        }
    )

    with pytest.raises(ValueError, match="unique user-item pairs"):
        label_candidates(candidates, targets)


@pytest.mark.parametrize("column", ["user_id", "item_id"])
def test_labeling_rejects_missing_candidate_ids(column: str) -> None:
    candidates = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
        }
    )
    candidates[column] = None

    targets = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
        }
    )

    with pytest.raises(ValueError, match="missing IDs"):
        label_candidates(candidates, targets)
