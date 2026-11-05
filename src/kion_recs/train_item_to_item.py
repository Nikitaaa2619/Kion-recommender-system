from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from kion_recs.data.matrix import build_interaction_matrix
from kion_recs.models.item_to_item import build_item_profiles


def build_neighbor_graph(
    item_profiles: csr_matrix,
    k: int = 50,
    batch_size: int = 128,
) -> csr_matrix:
    if k <= 0:
        raise ValueError("k must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    n_items = item_profiles.shape[0]

    if n_items == 0:
        return csr_matrix((0, 0), dtype=np.float32)

    rows = []
    columns = []
    values = []
    item_indices = np.arange(n_items)

    for start in range(0, n_items, batch_size):
        stop = min(start + batch_size, n_items)

        similarities = (item_profiles[start:stop] @ item_profiles.T).toarray()

        for local_index, item_index in enumerate(range(start, stop)):
            scores = similarities[local_index]
            scores[item_index] = 0.0

            ranked_indices = np.lexsort((item_indices, -scores))
            selected = ranked_indices[scores[ranked_indices] > 0][:k]

            rows.append(np.full(len(selected), item_index, dtype=np.int32))
            columns.append(selected.astype(np.int32))
            values.append(scores[selected].astype(np.float32))

        if (start // batch_size) % 10 == 0 or stop == n_items:
            print(f"Processed items: {stop:,} / {n_items:,}")

    return csr_matrix(
        (
            np.concatenate(values),
            (
                np.concatenate(rows),
                np.concatenate(columns),
            ),
        ),
        shape=(n_items, n_items),
        dtype=np.float32,
    )


def main() -> None:
    neighbors_per_item = 50

    train = pd.read_parquet(
        "data/processed/train.parquet",
        columns=["user_id", "item_id"],
    )
    matrix, user_ids, item_ids = build_interaction_matrix(train)
    profiles = build_item_profiles(matrix)

    print(f"Building neighbors for {len(item_ids):,} items...")
    started_at = perf_counter()

    neighbors = build_neighbor_graph(
        profiles,
        k=neighbors_per_item,
        batch_size=128,
    )

    elapsed = perf_counter() - started_at
    print(f"\nBuild time: {elapsed:.2f} seconds")
    print(f"Stored neighbor links: {neighbors.nnz:,}")

    output_dir = Path(f"artifacts/item_to_item_k{neighbors_per_item}")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Saved neighbors, matrix and ID mappings: {output_dir}")


if __name__ == "__main__":
    main()
