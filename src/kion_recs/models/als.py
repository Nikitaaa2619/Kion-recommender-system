from pathlib import Path
from time import perf_counter

import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, save_npz

from kion_recs.data.matrix import build_interaction_matrix


def fit_als(
    matrix: csr_matrix,
    factors: int = 32,
) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=0.1,
        iterations=15,
        alpha=20.0,
        random_state=42,
        num_threads=4,
    )
    model.fit(matrix, show_progress=True)

    return model


def main() -> None:
    factors = 64

    train = pd.read_parquet(
        "data/processed/train.parquet",
        columns=["user_id", "item_id"],
    )
    matrix, user_ids, item_ids = build_interaction_matrix(train)

    print(f"Training ALS: {matrix.shape[0]:,} users, {matrix.shape[1]:,} items")
    print(f"Factors: {factors}")

    started_at = perf_counter()
    model = fit_als(matrix, factors=factors)
    elapsed = perf_counter() - started_at

    print(f"\nTraining time: {elapsed:.2f} seconds")

    user_index = 0
    recommended_indices, _ = model.recommend(
        userid=user_index,
        user_items=matrix[user_index],
        N=10,
        filter_already_liked_items=True,
    )
    recommendations = item_ids.take(recommended_indices).tolist()

    print(f"Example user_id: {user_ids[user_index]}")
    print(f"Recommendations: {recommendations}")

    output_dir = Path(f"artifacts/als_f{factors}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / "model.npz")
    save_npz(output_dir / "train_interactions.npz", matrix)

    pd.DataFrame({"user_id": user_ids}).to_parquet(
        output_dir / "user_ids.parquet",
        index=False,
    )
    pd.DataFrame({"item_id": item_ids}).to_parquet(
        output_dir / "item_ids.parquet",
        index=False,
    )

    print(f"Saved model, matrix and ID mappings: {output_dir}")


if __name__ == "__main__":
    main()
