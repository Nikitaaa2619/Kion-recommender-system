import pandas as pd

from kion_recs.metrics import recall_at_k
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

    train_history = train.groupby("user_id")["item_id"].agg(set).to_dict()
    validation_history = validation.groupby("user_id")["item_id"].agg(set).to_dict()

    scores = []
    warm_scores = []
    cold_scores = []

    print(f"Evaluating {len(validation_history):,} users...")

    for user_id, relevant_items in validation_history.items():
        seen_items = train_history.get(user_id, set())
        recommendations = []

        for item_id in ranked_items:
            if item_id not in seen_items:
                recommendations.append(item_id)

            if len(recommendations) == k:
                break

        score = recall_at_k(recommendations, relevant_items, k=k)
        scores.append(score)

        if seen_items:
            warm_scores.append(score)
        else:
            cold_scores.append(score)

    if not scores:
        raise ValueError("Validation must contain at least one user")

    mean_recall = sum(scores) / len(scores)

    print(f"Users evaluated: {len(scores):,}")
    print(f"Cold-start users: {len(cold_scores):,}")
    print(f"Recall@{k}: {mean_recall:.6f}")

    for group_name, group_scores in [
        ("Warm", warm_scores),
        ("Cold", cold_scores),
    ]:
        if group_scores:
            group_mean = sum(group_scores) / len(group_scores)
            print(f"{group_name} Recall@{k}: {group_mean:.6f} ({len(group_scores):,} users)")


if __name__ == "__main__":
    main()
