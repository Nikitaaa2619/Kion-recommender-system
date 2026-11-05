import json
from pathlib import Path
from time import perf_counter

import pandas as pd
from scipy.sparse import save_npz

from kion_recs.data.matrix import build_interaction_matrix
from kion_recs.models.als import fit_als
from kion_recs.models.item_to_item import build_item_profiles
from kion_recs.models.popularity import fit_popularity
from kion_recs.train_item_to_item import build_neighbor_graph


def main() -> None:
    cutoff = pd.Timestamp("2021-08-02")
    history_path = Path("data/processed/ranker_train/history.parquet")
    output_dir = Path("artifacts/generators_before_2021-08-02")

    history = pd.read_parquet(history_path)
    dates = history["last_watch_dt"]

    if dates.isna().any() or (dates >= cutoff).any():
        raise ValueError("Generator history must contain only dates before 2021-08-02")

    print(f"History rows: {len(history):,}")
    print(f"History dates: {dates.min().date()} — {dates.max().date()}")

    matrix, user_ids, item_ids = build_interaction_matrix(history)

    print(f"Users: {len(user_ids):,}")
    print(f"Items: {len(item_ids):,}")

    started_at = perf_counter()

    print("\nBuilding popularity...")
    popularity = fit_popularity(history)

    print("\nTraining ALS with 32 factors...")
    model = fit_als(matrix, factors=32)

    print("\nBuilding item-to-item neighbors...")
    profiles = build_item_profiles(matrix)
    neighbors = build_neighbor_graph(
        profiles,
        k=50,
        batch_size=128,
    )

    elapsed = perf_counter() - started_at

    output_dir.mkdir(parents=True, exist_ok=True)

    popularity.to_parquet(
        output_dir / "popularity.parquet",
        index=False,
    )
    model.save(output_dir / "als.npz")
    save_npz(output_dir / "neighbors.npz", neighbors)
    save_npz(output_dir / "train_interactions.npz", matrix)

    pd.DataFrame({"user_id": user_ids}).to_parquet(
        output_dir / "user_ids.parquet",
        index=False,
    )
    pd.DataFrame({"item_id": item_ids}).to_parquet(
        output_dir / "item_ids.parquet",
        index=False,
    )

    metadata = {
        "history_path": str(history_path),
        "cutoff_exclusive": str(cutoff.date()),
        "history_min_date": str(dates.min().date()),
        "history_max_date": str(dates.max().date()),
        "n_interactions": len(history),
        "n_users": len(user_ids),
        "n_items": len(item_ids),
        "als_factors": int(model.factors),
        "als_regularization": float(model.regularization),
        "als_alpha": float(model.alpha),
        "als_iterations": int(model.iterations),
        "als_random_state": 42,
        "neighbors_per_item": 50,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nGenerator build time: {elapsed:.2f} seconds")
    print(f"Stored neighbor links: {neighbors.nnz:,}")
    print(f"Saved generators and metadata: {output_dir}")


if __name__ == "__main__":
    main()
