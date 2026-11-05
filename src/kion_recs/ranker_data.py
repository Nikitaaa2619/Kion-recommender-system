import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

FEATURE_COLUMNS = [
    "rank_popularity",
    "rank_als",
    "rank_item_to_item",
    "n_sources",
    "user_n_items",
    "user_days_since_last_watch",
    "item_n_users",
    "item_n_users_7d",
    "is_cold_user",
]


def load_ranker_training(
    run_dir: Path,
    max_users: int = 20000,
    random_state: int = 42,
) -> pd.DataFrame:
    if max_users <= 0:
        raise ValueError("max_users must be positive")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    if manifest.get("status") != "complete":
        raise ValueError("Dataset run must have complete status")

    parts = sorted((run_dir / "parts").glob("part_*.parquet"))

    if not parts or len(parts) != manifest["parts"]:
        raise ValueError("Parquet part count does not match manifest")

    all_users = set()
    users_by_part = {}
    total_rows = 0

    print("Checking dataset parts...")

    for part in parts:
        part_users = set(pd.read_parquet(part, columns=["user_id"])["user_id"].tolist())

        if all_users.intersection(part_users):
            raise ValueError("A user group is split across multiple parts")

        all_users.update(part_users)
        users_by_part[part] = part_users
        total_rows += pq.read_metadata(part).num_rows

    if len(all_users) != manifest["selected_users"]:
        raise ValueError("User count does not match manifest")

    if total_rows != manifest["candidate_rows"]:
        raise ValueError("Row count does not match manifest")

    selected_users = set(
        pd.Series(sorted(all_users))
        .sample(
            n=min(max_users, len(all_users)),
            random_state=random_state,
        )
        .tolist()
    )

    columns = ["user_id", "item_id", "target", *FEATURE_COLUMNS]
    frames = []

    print(f"Loading groups for {len(selected_users):,} users...")

    for part in parts:
        selected_in_part = selected_users.intersection(users_by_part[part])

        if not selected_in_part:
            continue

        frame = pd.read_parquet(
            part,
            columns=columns,
            filters=[("user_id", "in", sorted(selected_in_part))],
        )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)

    if data["user_id"].nunique() != len(selected_users):
        raise ValueError("Some selected user groups were not loaded")

    if data[["user_id", "item_id", "target"]].isna().any().any():
        raise ValueError("IDs and targets must not contain missing values")

    if not data["target"].isin([0, 1]).all():
        raise ValueError("Targets must be binary")

    if data.duplicated(subset=["user_id", "item_id"]).any():
        raise ValueError("Candidate user-item pairs must be unique")

    for column in FEATURE_COLUMNS:
        data[column] = data[column].to_numpy(
            dtype="float32",
            na_value=float("nan"),
        )

    return data.sort_values(["user_id", "item_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=20000)
    args = parser.parse_args()

    data = load_ranker_training(
        args.run_dir,
        max_users=args.max_users,
    )

    positives_by_user = data.groupby("user_id")["target"].sum()
    memory_mib = data.memory_usage(deep=True).sum() / 1024**2

    print(f"\nLoaded users: {data['user_id'].nunique():,}")
    print(f"Loaded rows: {len(data):,}")
    print(f"Positive rows: {int(data['target'].sum()):,}")
    print(f"Users with positive candidates: {(positives_by_user > 0).sum():,}")
    print(f"Users without positive candidates: {(positives_by_user == 0).sum():,}")
    print(f"DataFrame memory: {memory_mib:.2f} MiB")
    print(f"Features: {FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
