from pathlib import Path

import pandas as pd

from kion_recs.data.split import temporal_split


def split_ranker_training(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history, targets, future = temporal_split(
        interactions,
        validation_start="2021-08-02",
        test_start="2021-08-09",
    )

    if not future.empty:
        raise ValueError("Ranker training input must not contain dates from 2021-08-09")

    if history.empty or targets.empty:
        raise ValueError("History and target periods must both contain interactions")

    return history, targets


def main() -> None:
    train = pd.read_parquet("data/processed/train.parquet")
    history, targets = split_ranker_training(train)

    for name, part in [
        ("Generator history", history),
        ("Ranker targets", targets),
    ]:
        print(f"\n{name}")
        print(f"Rows: {len(part):,}")
        print(f"Users: {part['user_id'].nunique():,}")
        print(f"Dates: {part['last_watch_dt'].min().date()} — {part['last_watch_dt'].max().date()}")

    print(f"\nRows preserved: {len(history) + len(targets) == len(train)}")

    output_dir = Path("data/processed/ranker_train")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, part in [
        ("history", history),
        ("targets", targets),
    ]:
        output_path = output_dir / f"{name}.parquet"
        part.to_parquet(output_path, index=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
