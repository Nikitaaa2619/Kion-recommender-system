from pathlib import Path

import pandas as pd

INTERACTIONS_PATH = Path("data/raw/data_original/interactions.csv")

INTERACTIONS_DTYPES = {
    "user_id": "int32",
    "item_id": "int32",
    "total_dur": "int32",
    "watched_pct": "float32",
}

REQUIRED_INTERACTIONS_COLUMNS = frozenset(
    {
        "user_id",
        "item_id",
        "last_watch_dt",
        "total_dur",
        "watched_pct",
    }
)


def validate_interactions(interactions: pd.DataFrame) -> None:
    missing_columns = REQUIRED_INTERACTIONS_COLUMNS.difference(interactions.columns)

    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {columns}")


def load_interactions(path: Path = INTERACTIONS_PATH) -> pd.DataFrame:
    interactions = pd.read_csv(
        path,
        dtype=INTERACTIONS_DTYPES,
        parse_dates=["last_watch_dt"],
    )
    validate_interactions(interactions)
    return interactions


def main() -> None:
    interactions = load_interactions()

    print("Schema: OK")
    print(f"Rows: {len(interactions):,}")
    print(f"Users: {interactions['user_id'].nunique():,}")
    print(f"Items: {interactions['item_id'].nunique():,}")
    print(
        "Date range: "
        f"{interactions['last_watch_dt'].min().date()} — "
        f"{interactions['last_watch_dt'].max().date()}"
    )

    missing_values = interactions.isna().sum()
    duplicate_pairs = interactions.duplicated(subset=["user_id", "item_id"]).sum()
    invalid_watched_pct = (
        ~interactions["watched_pct"].dropna().between(0, 100)
    ).sum()
    non_positive_duration = (interactions["total_dur"] <= 0).sum()

    user_activity = interactions.groupby("user_id").size()

    print("\nMissing values:")
    print(missing_values.to_string())
    print(f"\nDuplicate user-item pairs: {duplicate_pairs:,}")
    print(f"Watched percentages outside 0–100: {invalid_watched_pct:,}")
    print(f"Non-positive durations: {non_positive_duration:,}")
    print("\nInteractions per user:")
    print(f"Median: {user_activity.median():.0f}")
    print(f"95th percentile: {user_activity.quantile(0.95):.0f}")
    print(f"Maximum: {user_activity.max():,}")


if __name__ == "__main__":
    main()