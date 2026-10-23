import pandas as pd

from kion_recs.coverage import catalog_coverage
from kion_recs.metrics import average_precision_at_k, ndcg_at_k, recall_at_k
from kion_recs.models.popularity import fit_popularity


def main() -> None:
    k = 10

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
    catalog_items = set(ranked_items)

    train_history = train.groupby("user_id")["item_id"].agg(set).to_dict()
    validation_history = validation.groupby("user_id")["item_id"].agg(set).to_dict()

    if not validation_history:
        raise ValueError("Validation must contain at least one user")

    results = []
    all_recommended_items = set()

    print(f"Evaluating {len(validation_history):,} users...")

    for user_id, relevant_items in validation_history.items():
        seen_items = train_history.get(user_id, set())
        recommendations = []

        for item_id in ranked_items:
            if item_id not in seen_items:
                recommendations.append(item_id)

            if len(recommendations) == k:
                break

        all_recommended_items.update(recommendations)

        results.append(
            {
                "user_id": user_id,
                "group": "warm" if seen_items else "cold",
                "recall": recall_at_k(recommendations, relevant_items, k=k),
                "ap": average_precision_at_k(
                    recommendations,
                    relevant_items,
                    k=k,
                ),
                "ndcg": ndcg_at_k(recommendations, relevant_items, k=k),
            }
        )

    results_table = pd.DataFrame(results)

    for group_name, group_results in [
        ("All", results_table),
        ("Warm", results_table.loc[results_table["group"] == "warm"]),
        ("Cold", results_table.loc[results_table["group"] == "cold"]),
    ]:
        if group_results.empty:
            continue

        mean_recall = group_results["recall"].mean()
        mean_ap = group_results["ap"].mean()
        mean_ndcg = group_results["ndcg"].mean()

        print(f"\n{group_name}: {len(group_results):,} users")
        print(f"Recall@{k}: {mean_recall:.6f}")
        print(f"MAP@{k}: {mean_ap:.6f}")
        print(f"NDCG@{k}: {mean_ndcg:.6f}")

    coverage = catalog_coverage(all_recommended_items, catalog_items)

    print(f"\nRecommended unique items: {len(all_recommended_items):,}")
    print(f"Train catalog items: {len(catalog_items):,}")
    print(f"Coverage@{k}: {coverage:.6f} ({coverage:.2%})")


if __name__ == "__main__":
    main()
