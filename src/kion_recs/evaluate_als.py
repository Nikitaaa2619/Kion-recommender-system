from pathlib import Path

import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import load_npz

from kion_recs.coverage import catalog_coverage
from kion_recs.metrics import average_precision_at_k, ndcg_at_k, recall_at_k
from kion_recs.models.popularity import fit_popularity


def main() -> None:
    k = 10
    batch_size = 256
    artifact_dir = Path("artifacts/als_f64")

    model = AlternatingLeastSquares.load(artifact_dir / "model.npz")
    model.num_threads = 4

    matrix = load_npz(artifact_dir / "train_interactions.npz")
    user_ids = pd.Index(pd.read_parquet(artifact_dir / "user_ids.parquet")["user_id"])
    item_ids = pd.Index(pd.read_parquet(artifact_dir / "item_ids.parquet")["item_id"])

    if matrix.shape != (len(user_ids), len(item_ids)):
        raise ValueError("Matrix shape does not match ID mappings")

    if model.user_factors.shape[0] != len(user_ids) or model.item_factors.shape[0] != len(item_ids):
        raise ValueError("Model shape does not match ID mappings")

    train = pd.read_parquet(
        "data/processed/train.parquet",
        columns=["user_id", "item_id"],
    )
    validation = pd.read_parquet(
        "data/processed/validation.parquet",
        columns=["user_id", "item_id"],
    )

    popularity = fit_popularity(train)
    cold_recommendations = popularity["item_id"].head(k).tolist()

    validation_history = validation.groupby("user_id")["item_id"].agg(set).to_dict()

    if not validation_history:
        raise ValueError("Validation must contain at least one user")

    evaluation_users = list(validation_history)
    results = []
    all_recommended_items = set()

    print(f"Model: {artifact_dir}")
    print(f"Evaluating {len(evaluation_users):,} users...")

    for start in range(0, len(evaluation_users), batch_size):
        batch_users = evaluation_users[start : start + batch_size]
        batch_indices = user_ids.get_indexer(batch_users)
        warm_indices = batch_indices[batch_indices >= 0]

        warm_recommendations = {}

        if len(warm_indices):
            recommended_indices, _ = model.recommend(
                userid=warm_indices,
                user_items=matrix[warm_indices],
                N=k,
                filter_already_liked_items=True,
            )

            for user_index, item_indices in zip(
                warm_indices,
                recommended_indices,
                strict=True,
            ):
                warm_recommendations[user_index] = item_ids.take(item_indices).tolist()

        for user_id, user_index in zip(
            batch_users,
            batch_indices,
            strict=True,
        ):
            is_warm = user_index >= 0

            if is_warm:
                recommendations = warm_recommendations[user_index]
            else:
                recommendations = cold_recommendations

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
        print(f"Recall@{k}: {group_results['recall'].mean():.6f}")
        print(f"MAP@{k}: {group_results['ap'].mean():.6f}")
        print(f"NDCG@{k}: {group_results['ndcg'].mean():.6f}")

    catalog_items = set(item_ids)
    coverage = catalog_coverage(all_recommended_items, catalog_items)

    print(f"\nRecommended unique items: {len(all_recommended_items):,}")
    print(f"Train catalog items: {len(catalog_items):,}")
    print(f"Coverage@{k}: {coverage:.6f} ({coverage:.2%})")


if __name__ == "__main__":
    main()
