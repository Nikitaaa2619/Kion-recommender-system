from pathlib import Path

import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import load_npz

from kion_recs.candidates import merge_candidate_lists
from kion_recs.models.item_to_item import recommend_from_neighbors
from kion_recs.models.popularity import fit_popularity, recommend_popular


def main() -> None:
    user_id = 0
    candidates_per_source = 100

    als_dir = Path("artifacts/als")
    neighbors_dir = Path("artifacts/item_to_item_k50")

    model = AlternatingLeastSquares.load(als_dir / "model.npz")
    model.num_threads = 4

    matrix = load_npz(neighbors_dir / "train_interactions.npz")
    neighbors = load_npz(neighbors_dir / "neighbors.npz")

    user_ids = pd.Index(pd.read_parquet(neighbors_dir / "user_ids.parquet")["user_id"])
    item_ids = pd.Index(pd.read_parquet(neighbors_dir / "item_ids.parquet")["item_id"])

    als_user_ids = pd.Index(pd.read_parquet(als_dir / "user_ids.parquet")["user_id"])
    als_item_ids = pd.Index(pd.read_parquet(als_dir / "item_ids.parquet")["item_id"])

    if not user_ids.equals(als_user_ids) or not item_ids.equals(als_item_ids):
        raise ValueError("ALS and item-to-item ID mappings must match")

    als_matrix = load_npz(als_dir / "train_interactions.npz")

    if als_matrix.shape != matrix.shape:
        raise ValueError("Training matrix shapes must match")

    if (als_matrix != matrix).nnz:
        raise ValueError("ALS and item-to-item must use the same interactions")

    train = pd.read_parquet(
        "data/processed/train.parquet",
        columns=["user_id", "item_id"],
    )
    popularity = fit_popularity(train)

    user_index = user_ids.get_indexer([user_id])[0]

    if user_index < 0:
        raise ValueError("Demo user must have training history")

    seen_indices = set(matrix[user_index].indices.tolist())
    seen_items = set(item_ids.take(sorted(seen_indices)).tolist())

    popular_candidates = recommend_popular(
        popularity,
        seen_items=seen_items,
        k=candidates_per_source,
    )

    als_indices, _ = model.recommend(
        userid=user_index,
        user_items=matrix[user_index],
        N=candidates_per_source,
        filter_already_liked_items=True,
    )
    als_candidates = item_ids.take(als_indices).tolist()

    neighbor_indices = recommend_from_neighbors(
        neighbors,
        seen_indices=seen_indices,
        k=candidates_per_source,
    )
    neighbor_candidates = item_ids.take(neighbor_indices).tolist()

    candidates = merge_candidate_lists(
        popularity=popular_candidates,
        als=als_candidates,
        item_to_item=neighbor_candidates,
        seen_items=seen_items,
    )
    candidates.insert(0, "user_id", user_id)

    print(f"User: {user_id}")
    print(f"Seen items: {len(seen_items):,}")
    print(f"Popularity candidates: {len(popular_candidates)}")
    print(f"ALS candidates: {len(als_candidates)}")
    print(f"Item-to-item candidates: {len(neighbor_candidates)}")
    print(f"Unique merged candidates: {len(candidates)}")

    print("\nCandidates by number of sources:")
    print(candidates["n_sources"].value_counts().sort_index().to_string())

    print("\nExample rows, sorted by item_id:")
    print(candidates.head(10).to_string(index=False))

    output_dir = Path("artifacts/candidate_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"user_{user_id}.parquet"

    candidates.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
