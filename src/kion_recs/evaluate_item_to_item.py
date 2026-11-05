from pathlib import Path

import pandas as pd
from scipy.sparse import load_npz

from kion_recs.coverage import catalog_coverage
from kion_recs.metrics import average_precision_at_k, ndcg_at_k, recall_at_k
from kion_recs.models.item_to_item import recommend_from_neighbors
from kion_recs.models.popularity import fit_popularity


def main() -> None:
    k = 10
    artifact_dir = Path("artifacts/item_to_item_k50")

    neighbors = load_npz(artifact_dir / "neighbors.npz")
    matrix = load_npz(artifact_dir / "train_interactions.npz")

    user_ids = pd.Index(pd.read_parquet(artifact_dir / "user_ids.parquet")["user_id"])
    item_ids = pd.Index(pd.read_parquet(artifact_dir / "item_ids.parquet")["item_id"])

    if matrix.shape != (len(user_ids), len(item_ids)):
        raise ValueError("Matrix shape does not match ID mappings")

    if neighbors.shape != (len(item_ids), len(item_ids)):
        raise ValueError("Neighbors shape does not match item IDs")

    train = pd.read_parquet(
        "data/processed/train.parquet",
        columns=["user_id", "item_id"],
    )
    validation = pd.read_parquet(
        "data/processed/validation.parquet",
        columns=["user_id", "item_id"],
    )

    popularity = fit_popularity(train)
    ranked_items = popularity["item_id"].tolist()

    validation_history = validation.groupby("user_id")["item_id"].agg(set).to_dict()

    if not validation_history:
        raise ValueError("Validation must contain at least one user")

    evaluation_users = list(validation_history)
    user_indices = user_ids.get_indexer(evaluation_users)

    results = []
    all_recommended_items = set()
    warm_users_with_fallback = 0

    print(f"Model: {artifact_dir}")
    print(f"Evaluating {len(evaluation_users):,} users...")

    for position, (user_id, user_index) in enumerate(
        zip(evaluation_users, user_indices, strict=True),
        start=1,
    ):
        is_warm = user_index >= 0

        if is_warm:
            seen_indices = set(matrix[user_index].indices.tolist())
            recommended_indices = recommend_from_neighbors(
                neighbors,
                seen_indices,
                k=k,
            )
            recommendations = item_ids.take(recommended_indices).tolist()
            seen_items = set(item_ids.take(sorted(seen_indices)).tolist())

            if len(recommendations) < k:
                warm_users_with_fallback += 1
        else:
            recommendations = []
            seen_items = set()

        if len(recommendations) < k:
            excluded_items = seen_items.union(recommendations)

            for item_id in ranked_items:
                if item_id not in excluded_items:
                    recommendations.append(item_id)
                    excluded_items.add(item_id)

                if len(recommendations) == k:
                    break

        relevant_items = validation_history[user_id]
        all_recommended_items.update(recommendations)

        results.append(
            {
                "group": "warm" if is_warm else "cold",
                "recall": recall_at_k(recommendations, relevant_items, k=k),
                "ap": average_precision_at_k(
                    recommendations,
                    relevant_items,
                    k=k,
                ),
                "ndcg": ndcg_at_k(recommendations, relevant_items, k=k),
            }
        )

        if position % 20000 == 0 or position == len(evaluation_users):
            print(f"Processed: {position:,} / {len(evaluation_users):,}")

    results_table = pd.DataFrame(results)

    for group_name, group_results in [
        ("All", results_table),
        ("Warm", results_table.loc[results_table["group"] == "warm"]),
        ("Cold", results_table.loc[results_table["group"] == "cold"]),
    ]:
        if group_results.empty:
            continue

        print(f"\n{group_name}: {len(group_results):,} users")
        print(f"Recall@{k}: {group_results['recall'].mean():.6f}")
        print(f"MAP@{k}: {group_results['ap'].mean():.6f}")
        print(f"NDCG@{k}: {group_results['ndcg'].mean():.6f}")

    catalog_items = set(item_ids)
    coverage = catalog_coverage(all_recommended_items, catalog_items)

    print(f"\nWarm users with popularity supplementation: {warm_users_with_fallback:,}")
    print(f"Recommended unique items: {len(all_recommended_items):,}")
    print(f"Train catalog items: {len(catalog_items):,}")
    print(f"Coverage@{k}: {coverage:.6f} ({coverage:.2%})")


if __name__ == "__main__":
    main()
