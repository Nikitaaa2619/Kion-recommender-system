import numpy as np
import pytest

from kion_recs.statistics import paired_bootstrap_mean


def test_bootstrap_returns_zero_for_identical_systems() -> None:
    result = paired_bootstrap_mean(
        np.zeros(10),
        n_bootstrap=100,
    )

    assert result == pytest.approx((0.0, 0.0, 0.0))


def test_bootstrap_preserves_constant_improvement() -> None:
    result = paired_bootstrap_mean(
        np.full(10, 0.1),
        n_bootstrap=100,
    )

    assert result == pytest.approx((0.1, 0.1, 0.1))


def test_bootstrap_is_reproducible_with_fixed_seed() -> None:
    differences = np.array([-0.5, 0.0, 0.1, 0.2, 0.7])

    first = paired_bootstrap_mean(
        differences,
        n_bootstrap=200,
        random_state=42,
    )
    second = paired_bootstrap_mean(
        differences,
        n_bootstrap=200,
        random_state=42,
    )

    assert first == second
    assert first[0] == pytest.approx(0.1)
    assert first[1] < first[2]


def test_bootstrap_does_not_change_input() -> None:
    differences = np.array([-0.2, 0.1, 0.4])
    original = differences.copy()

    paired_bootstrap_mean(differences, n_bootstrap=100)

    np.testing.assert_array_equal(differences, original)


@pytest.mark.parametrize(
    "differences",
    [
        np.array([]),
        np.array([0.1]),
    ],
)
def test_bootstrap_rejects_too_few_observations(
    differences: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="At least two"):
        paired_bootstrap_mean(differences)


def test_bootstrap_rejects_non_vector_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        paired_bootstrap_mean(np.array([[0.1, 0.2]]))


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_bootstrap_rejects_non_finite_values(invalid_value: float) -> None:
    with pytest.raises(ValueError, match="finite values"):
        paired_bootstrap_mean(np.array([0.1, invalid_value]))


@pytest.mark.parametrize("n_bootstrap", [0, -1])
def test_bootstrap_rejects_non_positive_repeat_count(n_bootstrap: int) -> None:
    with pytest.raises(ValueError, match="n_bootstrap must be positive"):
        paired_bootstrap_mean(
            np.array([0.1, 0.2]),
            n_bootstrap=n_bootstrap,
        )


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1])
def test_bootstrap_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must be between"):
        paired_bootstrap_mean(
            np.array([0.1, 0.2]),
            confidence=confidence,
        )
