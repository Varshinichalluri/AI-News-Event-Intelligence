
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "gdelt_news_articles.csv"

REQUIRED_COLUMNS = [
    "article_id",
    "title",
    "source",
    "published_at",
    "url",
    "content",
]


def load_news_data(path: Path) -> pd.DataFrame:
    """Load news articles and validate their basic structure."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["published_at"] = pd.to_datetime(
        df["published_at"], errors="coerce", utc=True
    )

    for column in ["title", "content", "source"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["article_id"] = df["article_id"].astype(str).str.strip()

    return df


def main() -> None:
    df = load_news_data(DATA_PATH)

    print("\nDATASET VALIDATION REPORT")
    print("=" * 40)
    print(f"Total articles: {len(df)}")
    print(f"Unique article IDs: {df['article_id'].nunique()}")
    print(f"Missing titles: {(df['title'] == '').sum()}")
    print(f"Missing content: {(df['content'] == '').sum()}")
    print(f"Invalid publication dates: {df['published_at'].isna().sum()}")
    print(f"Duplicate URLs: {df['url'].duplicated().sum()}")
    print("\nSample articles:")
    print(df[["article_id", "title", "published_at"]].head().to_string(index=False))


if __name__ == "__main__":
    main()