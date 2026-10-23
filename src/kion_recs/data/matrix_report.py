import pandas as pd

from kion_recs.data.matrix import build_interaction_matrix


def main() -> None:
    train = pd.read_parquet(
        "data/processed/train.parquet",
        columns=["user_id", "item_id"],
    )

    matrix, user_ids, item_ids = build_interaction_matrix(train)

    total_cells = matrix.shape[0] * matrix.shape[1]
    density = matrix.nnz / total_cells

    sparse_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    dense_bytes = total_cells * matrix.dtype.itemsize

    print(f"Users: {len(user_ids):,}")
    print(f"Items: {len(item_ids):,}")
    print(f"Non-zero interactions: {matrix.nnz:,}")
    print(f"Density: {density:.6%}")
    print(f"CSR memory: {sparse_bytes / 1024**2:.2f} MiB")
    print(f"Dense equivalent: {dense_bytes / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
