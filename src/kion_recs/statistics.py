import numpy as np


def paired_bootstrap_mean(
    differences: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("differences must be one-dimensional")

    if len(values) < 2:
        raise ValueError("At least two paired observations are required")

    if not np.isfinite(values).all():
        raise ValueError("differences must contain only finite values")

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)

    for iteration in range(n_bootstrap):
        indices = rng.integers(
            low=0,
            high=len(values),
            size=len(values),
        )
        bootstrap_means[iteration] = values[indices].mean()

    tail_probability = (1 - confidence) / 2

    lower, upper = np.quantile(
        bootstrap_means,
        [tail_probability, 1 - tail_probability],
    )

    return float(values.mean()), float(lower), float(upper)
