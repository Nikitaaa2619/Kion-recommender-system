import argparse
import json
from pathlib import Path

import pandas as pd

from kion_recs.statistics import paired_bootstrap_mean


def pair_recall_metrics(results: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "group", "model", "recall"}
    missing = required.difference(results.columns)

    if missing:
        raise ValueError(f"Missing metric columns: {', '.join(sorted(missing))}")

    selected = results.loc[
        results["model"].isin(["item_to_item", "ranker"]),
        ["user_id", "group", "model", "recall"],
    ].copy()

    if selected.isna().any().any():
        raise ValueError("Metric rows must not contain missing values")

    if not selected["group"].isin(["warm", "cold"]).all():
        raise ValueError("User groups must be warm or cold")

    selected["recall"] = pd.to_numeric(selected["recall"], errors="raise")

    if not selected["recall"].between(0, 1).all():
        raise ValueError("Recall values must be between 0 and 1")

    if selected.duplicated(subset=["user_id", "model"]).any():
        raise ValueError("Each model must have one metric row per user")

    baseline = selected.loc[selected["model"] == "item_to_item"].drop(columns=["model"])
    ranker = selected.loc[selected["model"] == "ranker"].drop(columns=["model"])

    if baseline.empty or ranker.empty:
        raise ValueError("Both models must have metric rows")

    paired = baseline.merge(
        ranker,
        on="user_id",
        how="inner",
        suffixes=("_baseline", "_ranker"),
        validate="one_to_one",
    )

    if len(paired) != len(baseline) or len(paired) != len(ranker):
        raise ValueError("Models must be evaluated on exactly the same users")

    if not paired["group_baseline"].equals(paired["group_ranker"]):
        raise ValueError("User groups must match between models")

    return (
        paired.rename(columns={"group_baseline": "group"})
        .drop(columns=["group_ranker"])
        .sort_values("user_id")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive")

    results = pd.read_parquet(args.evaluation_dir / "user_metrics.parquet")
    paired = pair_recall_metrics(results)

    evaluation_summary = json.loads(
        (args.evaluation_dir / "summary.json").read_text(encoding="utf-8")
    )

    if len(paired) != evaluation_summary["selected_users"]:
        raise ValueError("Paired user count does not match evaluation summary")

    report = {
        "evaluation_dir": str(args.evaluation_dir),
        "ranker_dir": evaluation_summary["ranker_dir"],
        "evaluation_max_users": evaluation_summary["max_users"],
        "metric": "Recall@10",
        "confidence": 0.95,
        "n_bootstrap": args.n_bootstrap,
        "random_state": 42,
        "groups": {},
    }

    for group_name, group_results in [
        ("All", paired),
        ("Warm", paired.loc[paired["group"] == "warm"]),
        ("Cold", paired.loc[paired["group"] == "cold"]),
    ]:
        if len(group_results) < 2:
            print(f"\n{group_name}: skipped, fewer than two users")
            continue

        print(
            f"\nBootstrapping {group_name}: {len(group_results):,} users...",
            flush=True,
        )

        differences = (group_results["recall_ranker"] - group_results["recall_baseline"]).to_numpy()

        mean_difference, lower, upper = paired_bootstrap_mean(
            differences,
            n_bootstrap=args.n_bootstrap,
            confidence=0.95,
            random_state=42,
        )

        baseline_mean = float(group_results["recall_baseline"].mean())
        ranker_mean = float(group_results["recall_ranker"].mean())

        report["groups"][group_name] = {
            "users": len(group_results),
            "baseline_mean": baseline_mean,
            "ranker_mean": ranker_mean,
            "mean_difference": mean_difference,
            "ci_lower": lower,
            "ci_upper": upper,
        }

        print(f"Baseline Recall@10: {baseline_mean:.6f}")
        print(f"Ranker Recall@10: {ranker_mean:.6f}")
        print(f"Mean difference: {mean_difference:.6f}")
        print(f"95% interval: [{lower:.6f}, {upper:.6f}]")
        print(
            f"In percentage points: "
            f"{100 * mean_difference:.3f} "
            f"[{100 * lower:.3f}, {100 * upper:.3f}]"
        )

    output_path = args.evaluation_dir / "bootstrap_recall.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
