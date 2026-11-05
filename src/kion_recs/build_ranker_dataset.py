import argparse
import json
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import load_npz

from kion_recs.candidates import merge_candidate_lists
from kion_recs.features import (
    attach_history_features,
    build_history_feature_tables,
)
from kion_recs.labeling import label_candidates
from kion_recs.models.item_to_item import recommend_from_neighbors
from kion_recs.models.popularity import recommend_popular


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()

    if args.max_users is not None and args.max_users <= 0:
        raise ValueError("--max-users must be positive")

    limit = 100
    batch_size = 256
    cutoff = "2021-08-02"

    artifact_dir = Path("artifacts/generators_before_2021-08-02")
    data_dir = Path("data/processed/ranker_train")

    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    if metadata["cutoff_exclusive"] != cutoff:
        raise ValueError("Generator cutoff does not match dataset cutoff")

    if pd.Timestamp(metadata["history_max_date"]) >= pd.Timestamp(cutoff):
        raise ValueError("Generator history contains target-period dates")

    model = AlternatingLeastSquares.load(artifact_dir / "als.npz")
    model.num_threads = 4

    matrix = load_npz(artifact_dir / "train_interactions.npz")
    neighbors = load_npz(artifact_dir / "neighbors.npz")
    popularity = pd.read_parquet(artifact_dir / "popularity.parquet")

    user_ids = pd.Index(pd.read_parquet(artifact_dir / "user_ids.parquet")["user_id"])
    item_ids = pd.Index(pd.read_parquet(artifact_dir / "item_ids.parquet")["item_id"])

    if matrix.shape != (len(user_ids), len(item_ids)):
        raise ValueError("Matrix shape does not match ID mappings")

    if neighbors.shape != (len(item_ids), len(item_ids)):
        raise ValueError("Neighbors shape does not match item IDs")

    if model.user_factors.shape[0] != len(user_ids) or model.item_factors.shape[0] != len(item_ids):
        raise ValueError("ALS shape does not match ID mappings")

    history = pd.read_parquet(data_dir / "history.parquet")
    targets = pd.read_parquet(data_dir / "targets.parquet")
    target_dates = targets["last_watch_dt"]

    if (
        target_dates.isna().any()
        or (target_dates < pd.Timestamp(cutoff)).any()
        or (target_dates >= pd.Timestamp("2021-08-09")).any()
    ):
        raise ValueError("Targets must belong to 2021-08-02 through 2021-08-08")

    user_features, item_features = build_history_feature_tables(
        history,
        cutoff=cutoff,
    )

    available_users = pd.Series(sorted(targets["user_id"].unique()))

    if args.max_users is None:
        selected_users = available_users.tolist()
    else:
        selected_users = (
            available_users.sample(
                n=min(args.max_users, len(available_users)),
                random_state=42,
            )
            .sort_values()
            .tolist()
        )

    if not selected_users:
        raise ValueError("Targets must contain users")

    target_pairs = targets.loc[
        targets["user_id"].isin(selected_users),
        ["user_id", "item_id"],
    ].drop_duplicates()
    target_counts = target_pairs.groupby("user_id").size()

    output_root = data_dir / "candidate_runs"
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix="run_", dir=str(output_root)))
    parts_dir = run_dir / "parts"
    parts_dir.mkdir()

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "status": "building",
        "generator_metadata": metadata,
        "candidates_per_source": limit,
        "batch_size": batch_size,
        "max_users": args.max_users,
        "selected_users": len(selected_users),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    total_rows = 0
    total_positives = 0
    positive_users = 0
    warm_users = 0
    recall_sum = 0.0
    part_count = 0

    print(f"Users to process: {len(selected_users):,}")
    print(f"Run directory: {run_dir}", flush=True)

    for start in range(0, len(selected_users), batch_size):
        batch_users = selected_users[start : start + batch_size]
        batch_indices = user_ids.get_indexer(batch_users)
        warm_indices = batch_indices[batch_indices >= 0]
        warm_users += len(warm_indices)
        als_by_index = {}

        if len(warm_indices):
            recommended_indices, _ = model.recommend(
                userid=warm_indices,
                user_items=matrix[warm_indices],
                N=limit,
                filter_already_liked_items=True,
            )

            for user_index, indices in zip(
                warm_indices,
                recommended_indices,
                strict=True,
            ):
                als_by_index[user_index] = item_ids.take(indices).tolist()

        frames = []

        for user_id, user_index in zip(
            batch_users,
            batch_indices,
            strict=True,
        ):
            if user_index >= 0:
                seen_indices = set(matrix[user_index].indices.tolist())
                seen_items = set(item_ids.take(sorted(seen_indices)).tolist())
                als_items = als_by_index[user_index]

                neighbor_indices = recommend_from_neighbors(
                    neighbors,
                    seen_indices,
                    k=limit,
                )
                neighbor_items = item_ids.take(neighbor_indices).tolist()
            else:
                seen_items = set()
                als_items = []
                neighbor_items = []

            popular_items = recommend_popular(
                popularity,
                seen_items=seen_items,
                k=limit,
            )
            candidates = merge_candidate_lists(
                popularity=popular_items,
                als=als_items,
                item_to_item=neighbor_items,
                seen_items=seen_items,
            )
            candidates.insert(0, "user_id", user_id)
            frames.append(candidates)

        batch = pd.concat(frames, ignore_index=True)
        batch_targets = target_pairs.loc[target_pairs["user_id"].isin(batch_users)]
        batch = label_candidates(batch, batch_targets)
        batch = attach_history_features(batch, user_features, item_features)

        positives_by_user = batch.groupby("user_id")["target"].sum()
        positive_users += int((positives_by_user > 0).sum())
        total_positives += int(batch["target"].sum())
        total_rows += len(batch)

        batch_recalls = (
            positives_by_user.reindex(batch_users, fill_value=0) / target_counts.loc[batch_users]
        )
        recall_sum += float(batch_recalls.sum())

        part_path = parts_dir / f"part_{part_count:05d}.parquet"
        batch.to_parquet(
            part_path,
            index=False,
            compression="zstd",
        )
        part_count += 1

        if part_count % 10 == 0 or start + len(batch_users) == len(selected_users):
            completed = start + len(batch_users)
            print(
                f"Processed: {completed:,} / {len(selected_users):,}; rows saved: {total_rows:,}",
                flush=True,
            )

    manifest.update(
        {
            "status": "complete",
            "parts": part_count,
            "warm_users": warm_users,
            "cold_users": len(selected_users) - warm_users,
            "candidate_rows": total_rows,
            "positive_rows": total_positives,
            "positive_share": total_positives / total_rows,
            "users_without_positive_candidates": len(selected_users) - positive_users,
            "target_pairs": len(target_pairs),
            "retrieved_target_pair_share": total_positives / len(target_pairs),
            "candidate_recall_macro": recall_sum / len(selected_users),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nDataset complete")
    print(json.dumps(manifest, indent=2))
    print(f"\nSaved: {run_dir}")


if __name__ == "__main__":
    main()
