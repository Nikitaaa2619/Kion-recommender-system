import json
from pathlib import Path

import pandas as pd

from kion_recs.features import build_history_feature_tables


def main() -> None:
    cutoff = "2021-08-09"
    history_path = Path("data/processed/train.parquet")
    output_dir = Path("data/processed/ranker_validation")

    history = pd.read_parquet(
        history_path,
        columns=["user_id", "item_id", "last_watch_dt"],
    )

    user_features, item_features = build_history_feature_tables(
        history,
        cutoff=cutoff,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    user_features.to_parquet(
        output_dir / "user_features.parquet",
        index=False,
    )
    item_features.to_parquet(
        output_dir / "item_features.parquet",
        index=False,
    )

    metadata = {
        "cutoff_exclusive": cutoff,
        "history_path": str(history_path),
        "history_min_date": str(history["last_watch_dt"].min().date()),
        "history_max_date": str(history["last_watch_dt"].max().date()),
        "history_rows": len(history),
        "user_feature_rows": len(user_features),
        "item_feature_rows": len(item_features),
        "targets_path": "data/processed/validation.parquet",
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Feature cutoff: {cutoff}")
    print(f"History rows: {len(history):,}")
    print(f"History last date: {metadata['history_max_date']}")
    print(f"User feature rows: {len(user_features):,}")
    print(f"Item feature rows: {len(item_features):,}")
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
