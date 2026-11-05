from pathlib import Path

import pandas as pd


def add_history_features(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    cutoff: str,
) -> pd.DataFrame:
    boundary = pd.Timestamp(cutoff)
    dates = history["last_watch_dt"]

    if history.empty:
        raise ValueError("History must not be empty")

    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise ValueError("History dates must have a datetime dtype")

    if dates.isna().any() or (dates >= boundary).any():
        raise ValueError("History dates must be present and earlier than cutoff")

    user_features = history.groupby("user_id").agg(
        user_n_items=("item_id", "nunique"),
        user_last_watch=("last_watch_dt", "max"),
    )
    user_features["user_days_since_last_watch"] = (
        boundary - user_features["user_last_watch"]
    ).dt.days
    user_features = user_features.drop(columns=["user_last_watch"]).reset_index()

    item_features = (
        history.groupby("item_id")["user_id"].nunique().rename("item_n_users").reset_index()
    )

    recent_history = history.loc[dates >= boundary - pd.Timedelta(days=7)]
    recent_item_features = (
        recent_history.groupby("item_id")["user_id"]
        .nunique()
        .rename("item_n_users_7d")
        .reset_index()
    )

    result = candidates.merge(
        user_features,
        on="user_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    result = result.merge(
        item_features,
        on="item_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    result = result.merge(
        recent_item_features,
        on="item_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    result["is_cold_user"] = result["user_n_items"].isna().astype("int8")

    for column in ["user_n_items", "item_n_users", "item_n_users_7d"]:
        result[column] = result[column].fillna(0).astype("int32")

    result["user_days_since_last_watch"] = result["user_days_since_last_watch"].astype("float32")

    return result


def main() -> None:
    data_dir = Path("data/processed/ranker_train")

    candidates = pd.read_parquet(data_dir / "candidates_sample.parquet")
    history = pd.read_parquet(data_dir / "history.parquet")

    result = add_history_features(
        candidates,
        history,
        cutoff="2021-08-02",
    )

    feature_columns = [
        "user_n_items",
        "user_days_since_last_watch",
        "item_n_users",
        "item_n_users_7d",
        "is_cold_user",
    ]

    print(f"Rows before: {len(candidates):,}")
    print(f"Rows after: {len(result):,}")
    print("\nExample features:")
    print(result[["user_id", "item_id", *feature_columns]].head(10).to_string(index=False))

    output_path = data_dir / "candidates_sample_features.parquet"
    result.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
