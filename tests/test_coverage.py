import pytest

from kion_recs.coverage import catalog_coverage


@pytest.mark.parametrize(
    ("recommended_items", "catalog_items", "expected"),
    [
        ({10, 20}, {10, 20, 30, 40}, 0.5),
        ({10, 20}, {10, 20}, 1.0),
        (set(), {10, 20}, 0.0),
    ],
)
def test_catalog_coverage(
    recommended_items: set[int],
    catalog_items: set[int],
    expected: float,
) -> None:
    result = catalog_coverage(recommended_items, catalog_items)

    assert result == pytest.approx(expected)


def test_catalog_coverage_rejects_empty_catalog() -> None:
    with pytest.raises(ValueError, match="catalog_items must not be empty"):
        catalog_coverage(set(), set())


def test_catalog_coverage_rejects_unknown_items() -> None:
    with pytest.raises(ValueError, match="must belong to catalog_items"):
        catalog_coverage({10, 99}, {10, 20})
