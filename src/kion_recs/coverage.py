def catalog_coverage(
    recommended_items: set[int],
    catalog_items: set[int],
) -> float:
    if not catalog_items:
        raise ValueError("catalog_items must not be empty")

    unknown_items = recommended_items.difference(catalog_items)

    if unknown_items:
        raise ValueError("recommended_items must belong to catalog_items")

    return len(recommended_items) / len(catalog_items)