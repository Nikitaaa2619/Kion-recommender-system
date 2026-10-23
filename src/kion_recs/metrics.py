from math import log2


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


def average_precision_at_k(
    recommendations: list[int],
    relevant_items: set[int],
    k: int = 10,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant_items:
        raise ValueError("relevant_items must not be empty")

    hits = 0
    precision_sum = 0.0
    seen_items = set()

    for rank, item_id in enumerate(recommendations[:k], start=1):
        if item_id in relevant_items and item_id not in seen_items:
            hits += 1
            precision_sum += hits / rank

        seen_items.add(item_id)

    denominator = min(len(relevant_items), k)

    return precision_sum / denominator


def ndcg_at_k(
    recommendations: list[int],
    relevant_items: set[int],
    k: int = 10,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant_items:
        raise ValueError("relevant_items must not be empty")

    dcg = 0.0
    seen_items = set()

    for rank, item_id in enumerate(recommendations[:k], start=1):
        if item_id in relevant_items and item_id not in seen_items:
            dcg += 1 / log2(rank + 1)

        seen_items.add(item_id)

    ideal_hits = min(len(relevant_items), k)
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / ideal_dcg
