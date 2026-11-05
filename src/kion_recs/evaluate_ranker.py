import argparse
import json
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import load_npz

from kion_recs.candidates import merge_candidate_lists
from kion_recs.coverage import catalog_coverage
from kion_recs.data.matrix import build_interaction_matrix
from kion_recs.features import attach_history_features
from kion_recs.metrics import average_precision_at_k, ndcg_at_k, recall_at_k
from kion_recs.models.item_to_item import recommend_from_neighbors
from kion_recs.models.popularity import fit_popularity, recommend_popular
from kion_recs.ranker_data import FEATURE_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranker-dir", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()

    if args.max_users is not None and args.max_users <= 0:
        raise ValueError("--max-users must be positive")

    k = 10
    limit = 100
    batch_size = 256

    als_dir = Path("artifacts/als")
    neighbors_dir = Path("artifacts/item_to_item_k50")
    feature_dir = Path("data/processed/ranker_validation")

    ranker = CatBoostRanker()
    ranker.load_model(str(args.ranker_dir / "model.cbm"))

    if ranker.feature_names_ != FEATURE_COLUMNS:
        raise ValueError("Ranker feature names or order do not match")

    feature_metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))

    if feature_metadata["cutoff_exclusive"] != "2021-08-09":
        raise ValueError("Validation features must use the 2021-08-09 cutoff")

    user_features = pd.read_parquet(feature_dir / "user_features.parquet")
    item_features = pd.read_parquet(feature_dir / "item_features.parquet")

    history = pd.read_parquet(
        "data/processed/train.parquet",
        columns=["user_id", "item_id", "last_watch_dt"],
    )

    if (
        history["last_watch_dt"].isna().any()
        or (history["last_watch_dt"] >= pd.Timestamp("2021-08-09")).any()
    ):
        raise ValueError("Validation history must end before 2021-08-09")

    matrix, user_ids, item_ids = build_interaction_matrix(history)
    popularity = fit_popularity(history)

    for directory in [als_dir, neighbors_dir]:
        saved_user_ids = pd.Index(pd.read_parquet(directory / "user_ids.parquet")["user_id"])
        saved_item_ids = pd.Index(pd.read_parquet(directory / "item_ids.parquet")["item_id"])
        saved_matrix = load_npz(directory / "train_interactions.npz")

        if not user_ids.equals(saved_user_ids) or not item_ids.equals(saved_item_ids):
            raise ValueError("Generator ID mappings do not match validation history")

        if saved_matrix.shape != matrix.shape or (saved_matrix != matrix).nnz:
            raise ValueError("Generator interactions do not match validation history")

    als = AlternatingLeastSquares.load(als_dir / "model.npz")
    als.num_threads = 4
    neighbors = load_npz(neighbors_dir / "neighbors.npz")

    validation = pd.read_parquet("data/processed/validation.parquet")
    dates = validation["last_watch_dt"]

    if (
        dates.isna().any()
        or (dates < pd.Timestamp("2021-08-09")).any()
        or (dates >= pd.Timestamp("2021-08-16")).any()
    ):
        raise ValueError("Validation dates must belong to 2021-08-09 through 2021-08-15")

    relevant_by_user = validation.groupby("user_id")["item_id"].agg(set).to_dict()
    available_users = pd.Series(sorted(relevant_by_user))

    if available_users.empty:
        raise ValueError("Validation must contain users")

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

    records = []
    recommended_catalogs = {
        "item_to_item": set(),
        "ranker": set(),
    }

    print(f"Ranker: {args.ranker_dir}")
    print(f"Evaluating users: {len(selected_users):,}")

    for start in range(0, len(selected_users), batch_size):
        batch_users = selected_users[start : start + batch_size]
        batch_indices = user_ids.get_indexer(batch_users)
        warm_indices = batch_indices[batch_indices >= 0]
        als_by_index = {}

        if len(warm_indices):
            recommended_indices, _ = als.recommend(
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
        baseline_by_user = {}
        group_by_user = {}

        for user_id, user_index in zip(batch_users, batch_indices, strict=True):
            is_warm = user_index >= 0
            group_by_user[user_id] = "warm" if is_warm else "cold"

            if is_warm:
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

            baseline = neighbor_items[:k].copy()

            for item_id in popular_items:
                if len(baseline) == k:
                    break
                if item_id not in baseline:
                    baseline.append(item_id)

            baseline_by_user[user_id] = baseline

            candidates = merge_candidate_lists(
                popularity=popular_items,
                als=als_items,
                item_to_item=neighbor_items,
                seen_items=seen_items,
            )
            candidates.insert(0, "user_id", user_id)
            frames.append(candidates)

        batch = pd.concat(frames, ignore_index=True)
        batch = attach_history_features(batch, user_features, item_features)

        for column in FEATURE_COLUMNS:
            batch[column] = batch[column].to_numpy(
                dtype="float32",
                na_value=float("nan"),
            )

        predictions = ranker.predict(
            batch[FEATURE_COLUMNS],
            thread_count=4,
        )

        if not np.isfinite(predictions).all():
            raise ValueError("Ranker predictions must be finite")

        batch["score"] = predictions

        for user_id, user_candidates in batch.groupby("user_id", sort=False):
            ranked = (
                user_candidates.sort_values(
                    ["score", "item_id"],
                    ascending=[False, True],
                )
                .head(k)["item_id"]
                .tolist()
            )

            for model_name, recommendations in [
                ("item_to_item", baseline_by_user[user_id]),
                ("ranker", ranked),
            ]:
                relevant_items = relevant_by_user[user_id]
                recommended_catalogs[model_name].update(recommendations)

                records.append(
                    {
                        "user_id": int(user_id),
                        "group": group_by_user[user_id],
                        "model": model_name,
                        "recall": recall_at_k(recommendations, relevant_items, k=k),
                        "ap": average_precision_at_k(
                            recommendations,
                            relevant_items,
                            k=k,
                        ),
                        "ndcg": ndcg_at_k(recommendations, relevant_items, k=k),
                    }
                )

        completed = start + len(batch_users)

        if (start // batch_size) % 100 == 0 or completed == len(selected_users):
            print(f"Processed: {completed:,} / {len(selected_users):,}", flush=True)

    results = pd.DataFrame(records)
    summary = {
        "ranker_dir": str(args.ranker_dir),
        "max_users": args.max_users,
        "selected_users": len(selected_users),
        "results": {},
    }

    for model_name in ["item_to_item", "ranker"]:
        model_results = results.loc[results["model"] == model_name]
        summary["results"][model_name] = {}

        print(f"\nModel: {model_name}")

        for group_name, group_results in [
            ("All", model_results),
            ("Warm", model_results.loc[model_results["group"] == "warm"]),
            ("Cold", model_results.loc[model_results["group"] == "cold"]),
        ]:
            if group_results.empty:
                continue

            metrics = {
                "users": len(group_results),
                "recall_at_10": float(group_results["recall"].mean()),
                "map_at_10": float(group_results["ap"].mean()),
                "ndcg_at_10": float(group_results["ndcg"].mean()),
            }
            summary["results"][model_name][group_name] = metrics

            print(f"\n{group_name}: {metrics['users']:,} users")
            print(f"Recall@10: {metrics['recall_at_10']:.6f}")
            print(f"MAP@10: {metrics['map_at_10']:.6f}")
            print(f"NDCG@10: {metrics['ndcg_at_10']:.6f}")

        coverage = catalog_coverage(
            recommended_catalogs[model_name],
            set(item_ids),
        )
        summary["results"][model_name]["coverage_at_10"] = coverage
        print(f"\nCoverage@10: {coverage:.6f} ({coverage:.2%})")

    output_root = Path("artifacts/ranker_evaluations")
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(mkdtemp(prefix="run_", dir=str(output_root)))

    results.to_parquet(output_dir / "user_metrics.parquet", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved evaluation: {output_dir}")


if __name__ == "__main__":
    main()
