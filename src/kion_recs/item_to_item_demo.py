import pandas as pd

from kion_recs.data.matrix import build_interaction_matrix
from kion_recs.models.item_to_item import (
    build_item_profiles,
    find_similar_items,
)


def main() -> None:
    source_item_id = 10440

    train = pd.read_parquet(
        "data/processed/train.parquet",
        columns=["user_id", "item_id"],
    )
    items = pd.read_csv(
        "data/raw/data_original/items.csv",
        usecols=["item_id", "title"],
    )

    matrix, _, item_ids = build_interaction_matrix(train)
    profiles = build_item_profiles(matrix)

    source_index = item_ids.get_indexer([source_item_id])[0]

    if source_index < 0:
        raise ValueError("Source item is not present in train")

    similar_indices, scores = find_similar_items(
        profiles,
        item_index=source_index,
        k=10,
    )

    titles = items.set_index("item_id")["title"]
    source_title = titles.get(source_item_id, "Unknown title")

    result = pd.DataFrame(
        {
            "item_id": item_ids.take(similar_indices),
            "similarity": scores,
        }
    )
    result = result.merge(
        items,
        on="item_id",
        how="left",
        validate="one_to_one",
    )
    result = result.sort_values(
        ["similarity", "item_id"],
        ascending=[False, True],
    )

    print(f"Source: {source_item_id} — {source_title}")
    print("\nSimilar items:")
    print(
        result.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
