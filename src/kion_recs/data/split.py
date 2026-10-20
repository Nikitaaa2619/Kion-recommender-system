from pathlib import Path

import pandas as pd

from kion_recs.data.audit import load_interactions


def temporal_split(
    interactions: pd.DataFrame,
    validation_start: str = "2021-08-09",
    test_start: str = "2021-08-16",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_boundary = pd.Timestamp(validation_start)
    test_boundary = pd.Timestamp(test_start)

    if validation_boundary >= test_boundary:
        raise ValueError("validation_start must be earlier than test_start")

    dates = interactions["last_watch_dt"]

    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise ValueError("last_watch_dt must have a datetime dtype")

    if dates.isna().any():
        raise ValueError("last_watch_dt must not contain missing values")

    train = interactions.loc[dates < validation_boundary].copy()

    validation = interactions.loc[
        (dates >= validation_boundary) & (dates < test_boundary)
    ].copy()

    test = interactions.loc[dates >= test_boundary].copy()

    return train, validation, test


def main() -> None:
    interactions = load_interactions()
    train, validation, test = temporal_split(interactions)

    parts = [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]

    for name, part in parts:
        print(
            f"{name}: "
            f"{len(part):,} rows, "
            f"{part['user_id'].nunique():,} users, "
            f"{part['item_id'].nunique():,} items"
        )
        print(
            f"  Dates: {part['last_watch_dt'].min().date()} — "
            f"{part['last_watch_dt'].max().date()}"
        )

    total_rows = len(train) + len(validation) + len(test)
    print(f"\nRows preserved: {total_rows == len(interactions)}")

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, part in parts:
        output_path = output_dir / f"{name}.parquet"
        part.to_parquet(output_path, index=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()