import pytest

from kion_recs.metrics import recall_at_k


@pytest.mark.parametrize(
    ("recommendations", "relevant_items", "k", "expected"),
    [
        ([10, 20, 30], {10, 40}, 3, 0.5),
        ([10, 20], {10, 20}, 2, 1.0),
        ([10, 20], {30}, 2, 0.0),
        ([], {10}, 10, 0.0),
        ([20, 10], {10}, 1, 0.0),
        ([10, 10, 20], {10, 20}, 2, 0.5),
    ],
)
def test_recall_at_k(
    recommendations: list[int],
    relevant_items: set[int],
    k: int,
    expected: float,
) -> None:
    result = recall_at_k(recommendations, relevant_items, k)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k([10], {10}, k=k)


def test_recall_at_k_rejects_empty_relevant_items() -> None:
    with pytest.raises(ValueError, match="relevant_items must not be empty"):
        recall_at_k([10], set())
