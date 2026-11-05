import pandas as pd


def label_candidates(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["user_id", "item_id"]

    for name, table in [
        ("candidates", candidates),
        ("targets", targets),
    ]:
        missing_columns = set(keys).difference(table.columns)

        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            raise ValueError(f"{name} is missing columns: {columns}")

        if table[keys].isna().any().any():
            raise ValueError(f"{name} must not contain missing IDs")

    if "target" in candidates.columns:
        raise ValueError("candidates must not already contain a target column")

    if candidates.duplicated(subset=keys).any():
        raise ValueError("candidates must contain unique user-item pairs")

    positive_pairs = targets[keys].drop_duplicates().copy()
    positive_pairs["target"] = 1

    result = candidates.merge(
        positive_pairs,
        on=keys,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    result["target"] = result["target"].fillna(0).astype("int8")

    return result
