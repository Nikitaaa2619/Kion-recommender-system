from pathlib import Path

import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import load_npz

from kion_recs.metrics import recall_at_k
from kion_recs.models.item_to_item import recommend_from_neighbors
from kion_recs.models.popularity import fit_popularity, recommend_popular


def main() -> None:
    limit = 100
    batch_size = 256

    als_dir = Path("artifacts/als")
    neighbors_dir = Path("artifacts/item_to_item_k50")

    model = AlternatingLeastSquares.load(als_dir / "model.npz")
    model.num_threads = 4

    matrix = load_npz(neighbors_dir / "train_interactions.npz")
    neighbors = load_npz(neighbors_dir / "neighbors.npz")

    user_ids = pd.Index(pd.read_parquet(neighbors_dir / "user_ids.parquet")["user_id"])
    item_ids = pd.Index(pd.read_parquet(neighbors_dir / "item_ids.parquet")["item_id"])
    als_user_ids = pd.Index(pd.read_parquet(als_dir / "user_ids.parquet")["user_id"])
    als_item_ids = pd.Index(pd.read_parquet(als_dir / "item_ids.parquet")["item_id"])

    if not user_ids.equals(als_user_ids) or not item_ids.equals(als_item_ids):
        raise ValueError("ALS and item-to-item ID mappings must match")

    als_matrix = load_npz(als_dir / "train_interactions.npz")

    if als_matrix.shape != matrix.shape:
        raise ValueError("Training matrix shapes must match")

    if (als_matrix != matrix).nnz:
        raise ValueError("ALS and item-to-item must use the same interactions")

    train = pd.read_parquet(
        "data/processed/train.parquet",
        columns=["user_id", "item_id"],
    )
    validation = pd.read_parquet(
        "data/processed/validation.parquet",
        columns=["user_id", "item_id"],
    )

    popularity = fit_popularity(train)
    cold_candidates = popularity["item_id"].head(limit).tolist()
    validation_history = validation.groupby("user_id")["item_id"].agg(set).to_dict()

    if not validation_history:
        raise ValueError("Validation must contain at least one user")

    evaluation_users = list(validation_history)
    results = []

    print(f"Candidates per source: {limit}")
    print(f"Evaluating {len(evaluation_users):,} users...")

    for start in range(0, len(evaluation_users), batch_size):
        batch_users = evaluation_users[start : start + batch_size]
        batch_indices = user_ids.get_indexer(batch_users)
        warm_indices = batch_indices[batch_indices >= 0]

        als_by_index = {}

        if len(warm_indices):
            recommended_indices, _ = model.recommend(
                userid=warm_indices,
                user_items=matrix[warm_indices],
                N=limit,
                filter_already_liked_items=True,
            )

            for user_index, indices in zip(
                warm_indices,
                recommended_indices,
                strict=True,
            ):
                als_by_index[user_index] = item_ids.take(indices).tolist()

        for user_id, user_index in zip(
            batch_users,
            batch_indices,
            strict=True,
        ):
            is_warm = user_index >= 0

            if is_warm:
                seen_indices = set(matrix[user_index].indices.tolist())
                seen_items = set(item_ids.take(sorted(seen_indices)).tolist())

                popular_items = recommend_popular(
                    popularity,
                    seen_items=seen_items,
                    k=limit,
                )
                als_items = als_by_index[user_index]

                neighbor_indices = recommend_from_neighbors(
                    neighbors,
                    seen_indices,
                    k=limit,
                )
                neighbor_items = item_ids.take(neighbor_indices).tolist()
            else:
                popular_items = cold_candidates
                als_items = []
                neighbor_items = []

            source_sets = {
                "popularity": set(popular_items),
                "als": set(als_items),
                "item_to_item": set(neighbor_items),
            }
            source_sets["without_als"] = source_sets["popularity"] | source_sets["item_to_item"]
            source_sets["union"] = source_sets["without_als"] | source_sets["als"]

            relevant_items = validation_history[user_id]
            row = {
                "group": "warm" if is_warm else "cold",
                "n_candidates": len(source_sets["union"]),
            }

            for name, candidate_set in source_sets.items():
                row[name] = recall_at_k(
                    list(candidate_set),
                    relevant_items,
                    k=max(1, len(candidate_set)),
                )

            results.append(row)

        if (start // batch_size) % 100 == 0:
            completed = start + len(batch_users)
            print(f"Processed: {completed:,} / {len(evaluation_users):,}")

    results_table = pd.DataFrame(results)

    for group_name, group_results in [
        ("All", results_table),
        ("Warm", results_table.loc[results_table["group"] == "warm"]),
        ("Cold", results_table.loc[results_table["group"] == "cold"]),
    ]:
        if group_results.empty:
            continue

        print(f"\n{group_name}: {len(group_results):,} users")

        for name in ["popularity", "als", "item_to_item", "without_als", "union"]:
            print(f"Candidate Recall ({name}): {group_results[name].mean():.6f}")

        mean_size = group_results["n_candidates"].mean()
        max_size = group_results["n_candidates"].max()
        als_gain = (group_results["union"] - group_results["without_als"]).mean()

        print(f"Mean candidate count: {mean_size:.2f}")
        print(f"Maximum candidate count: {max_size}")
        print(f"Additional recall from ALS: {als_gain:.6f}")


if __name__ == "__main__":
    main()
