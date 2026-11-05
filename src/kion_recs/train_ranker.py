import argparse
import json
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

import pandas as pd
from catboost import CatBoostRanker, Pool

from kion_recs.ranker_data import FEATURE_COLUMNS, load_ranker_training


def select_informative_groups(data: pd.DataFrame) -> pd.DataFrame:
    if not data["target"].isin([0, 1]).all():
        raise ValueError("Targets must be binary")

    label_counts = data.groupby("user_id")["target"].nunique()
    informative_users = label_counts.index[label_counts == 2]

    if len(informative_users) == 0:
        raise ValueError("No user groups contain both positive and negative targets")

    return (
        data.loc[data["user_id"].isin(informative_users)]
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )


def fit_ranker(
    training: pd.DataFrame,
    iterations: int = 100,
) -> CatBoostRanker:
    pool = Pool(
        data=training[FEATURE_COLUMNS],
        label=training["target"],
        group_id=training["user_id"],
    )

    model = CatBoostRanker(
        loss_function="PairLogit:max_pairs=1000",
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        thread_count=4,
        task_type="CPU",
        use_best_model=False,
        allow_writing_files=False,
        verbose=20,
    )
    model.fit(pool)

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    data = load_ranker_training(
        args.run_dir,
        max_users=args.max_users,
        random_state=42,
    )
    selected_user_count = data["user_id"].nunique()

    training = select_informative_groups(data)
    training_user_count = training["user_id"].nunique()

    print(f"\nSelected users: {selected_user_count:,}")
    print(f"Training users: {training_user_count:,}")
    print(f"Excluded constant-target groups: {selected_user_count - training_user_count:,}")
    print(f"Training rows: {len(training):,}")
    print(f"Training positives: {int(training['target'].sum()):,}")
    print(f"Features: {FEATURE_COLUMNS}")

    del data

    started_at = perf_counter()
    model = fit_ranker(training, iterations=args.iterations)
    elapsed = perf_counter() - started_at

    output_root = Path("artifacts/ranker_runs")
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(mkdtemp(prefix="run_", dir=str(output_root)))

    model.save_model(str(output_dir / "model.cbm"))

    training[["user_id"]].drop_duplicates().to_parquet(
        output_dir / "training_users.parquet",
        index=False,
    )

    metadata = {
        "status": "complete",
        "dataset_run": str(args.run_dir),
        "max_users": args.max_users,
        "selection_random_state": 42,
        "selected_users": int(selected_user_count),
        "training_users": int(training_user_count),
        "excluded_constant_target_groups": int(selected_user_count - training_user_count),
        "training_rows": len(training),
        "positive_rows": int(training["target"].sum()),
        "feature_columns": FEATURE_COLUMNS,
        "model_parameters": model.get_params(),
        "training_seconds": elapsed,
        "validation_evaluated": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nTraining time: {elapsed:.2f} seconds")
    print(f"Trees: {model.tree_count_}")
    print(f"Saved: {output_dir}")
    print("Validation quality has not been evaluated yet.")


if __name__ == "__main__":
    main()
