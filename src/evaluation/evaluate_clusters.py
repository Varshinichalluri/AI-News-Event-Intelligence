
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "processed" / "clustered_articles.csv"
)

EXPECTED_GROUPS = {
    "A001": "DEMO-AI",
    "A002": "DEMO-AI",
    "A003": "DEMO-AI",
    "A004": "DEMO-AI",
    "A005": "DEMO-METRO",
    "A006": "DEMO-METRO",
    "A007": "DEMO-AI",
    "A008": "DEMO-ENERGY",
}


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Clustered data is missing. Run clustering first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = {"article_id", "event_id"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Required columns: {sorted(required)}"
        )

    if df["article_id"].duplicated().any():
        raise ValueError("Duplicate article IDs found.")

    df["expected_group"] = df["article_id"].map(
        EXPECTED_GROUPS
    )

    if df["expected_group"].isna().any():
        missing = df.loc[
            df["expected_group"].isna(), "article_id"
        ].tolist()
        raise ValueError(
            f"Missing expected labels for: {missing}"
        )

    if set(df["article_id"]) != set(EXPECTED_GROUPS):
        raise ValueError(
            "Dataset article IDs do not match the demo labels."
        )

    ari = adjusted_rand_score(
        df["expected_group"], df["event_id"]
    )
    nmi = normalized_mutual_info_score(
        df["expected_group"], df["event_id"]
    )

    print("\nCLUSTER EVALUATION REPORT")
    print("=" * 45)
    print(f"Articles evaluated: {len(df)}")
    print(f"Predicted clusters: {df['event_id'].nunique()}")
    print(f"Expected groups: {df['expected_group'].nunique()}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")

    print("\nEXPECTED VS PREDICTED")
    print(
        df[[
            "article_id",
            "expected_group",
            "event_id",
        ]].sort_values("article_id").to_string(index=False)
    )


if __name__ == "__main__":
    main()