from pathlib import Path

import pandas as pd


def fit_popularity(train: pd.DataFrame) -> pd.DataFrame:
    popularity = train.groupby("item_id")["user_id"].nunique().rename("n_users").reset_index()

    return popularity.sort_values(
        ["n_users", "item_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def recommend_popular(
    popularity: pd.DataFrame,
    seen_items: set[int],
    k: int = 10,
) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive")

    candidates = popularity.loc[
        ~popularity["item_id"].isin(seen_items),
        "item_id",
    ]

    return candidates.head(k).tolist()


def main() -> None:
    train = pd.read_parquet(
        "data/processed/train.parquet",
        columns=["user_id", "item_id"],
    )
    popularity = fit_popularity(train)

    print("Top 10 popular items:")
    print(popularity.head(10).to_string(index=False))

    recommendations = recommend_popular(
        popularity,
        seen_items=set(),
        k=10,
    )
    print(f"\nRecommendations for a new user: {recommendations}")

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "popularity.parquet"
    popularity.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
