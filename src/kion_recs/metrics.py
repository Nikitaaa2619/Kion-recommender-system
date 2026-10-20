def recall_at_k(
    recommendations: list[int],
    relevant_items: set[int],
    k: int = 10,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant_items:
        raise ValueError("relevant_items must not be empty")

    recommended_items = set(recommendations[:k])
    hits = recommended_items.intersection(relevant_items)

    return len(hits) / len(relevant_items)