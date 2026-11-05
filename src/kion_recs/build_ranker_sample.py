import json
from pathlib import Path

import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import load_npz

from kion_recs.candidates import merge_candidate_lists
from kion_recs.labeling import label_candidates
from kion_recs.models.item_to_item import recommend_from_neighbors
from kion_recs.models.popularity import recommend_popular


def main() -> None:
    limit = 100
    sample_size = 256
    artifact_dir = Path("artifacts/generators_before_2021-08-02")

    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    if metadata["cutoff_exclusive"] != "2021-08-02":
        raise ValueError("Generators must use the 2021-08-02 cutoff")

    if pd.Timestamp(metadata["history_max_date"]) >= pd.Timestamp("2021-08-02"):
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

    targets = pd.read_parquet("data/processed/ranker_train/targets.parquet")
    dates = targets["last_watch_dt"]

    if (
        dates.isna().any()
        or (dates < pd.Timestamp("2021-08-02")).any()
        or (dates >= pd.Timestamp("2021-08-09")).any()
    ):
        raise ValueError("Targets must belong to 2021-08-02 through 2021-08-08")

    available_users = pd.Series(sorted(targets["user_id"].unique()))
    selected_users = (
        available_users.sample(
            n=min(sample_size, len(available_users)),
            random_state=42,
        )
        .sort_values()
        .tolist()
    )

    if not selected_users:
        raise ValueError("Targets must contain users")

    selected_indices = user_ids.get_indexer(selected_users)
    warm_indices = selected_indices[selected_indices >= 0]
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
        selected_users,
        selected_indices,
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

    candidates = pd.concat(frames, ignore_index=True)
    selected_targets = targets.loc[targets["user_id"].isin(selected_users)]
    labeled = label_candidates(candidates, selected_targets)

    positive_rows = int(labeled["target"].sum())
    positive_users = int(labeled.groupby("user_id")["target"].max().sum())
    target_pairs = len(selected_targets[["user_id", "item_id"]].drop_duplicates())

    print(f"Sample users: {len(selected_users):,}")
    print(f"Warm users: {len(warm_indices):,}")
    print(f"Cold users: {(selected_indices < 0).sum():,}")
    print(f"Candidate rows: {len(labeled):,}")
    print(f"Positive rows: {positive_rows:,}")
    print(f"Positive share: {labeled['target'].mean():.2%}")
    print(f"Users without positive candidates: {len(selected_users) - positive_users}")
    print(f"Target pairs: {target_pairs:,}")
    print(f"Retrieved target-pair share: {positive_rows / target_pairs:.6f}")

    output_path = Path("data/processed/ranker_train/candidates_sample.parquet")
    labeled.to_parquet(output_path, index=False)

    print("\nExample rows:")
    print(labeled.head(10).to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
