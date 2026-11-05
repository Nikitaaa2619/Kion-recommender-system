import pandas as pd


def merge_candidate_lists(
    popularity: list[int],
    als: list[int],
    item_to_item: list[int],
    seen_items: set[int],
) -> pd.DataFrame:
    candidates = {}

    sources = {
        "rank_popularity": popularity,
        "rank_als": als,
        "rank_item_to_item": item_to_item,
    }

    for rank_column, item_ids in sources.items():
        for rank, item_id in enumerate(item_ids, start=1):
            if item_id in seen_items:
                continue

            if item_id not in candidates:
                candidates[item_id] = {"item_id": item_id}

            if rank_column not in candidates[item_id]:
                candidates[item_id][rank_column] = rank

    rank_columns = list(sources)
    columns = ["item_id", *rank_columns]

    result = pd.DataFrame(
        list(candidates.values()),
        columns=columns,
    )

    result["item_id"] = result["item_id"].astype("int64")

    for column in rank_columns:
        result[column] = result[column].astype("Int32")

    result["n_sources"] = result[rank_columns].notna().sum(axis=1).astype("int8")

    return result.sort_values("item_id").reset_index(drop=True)
