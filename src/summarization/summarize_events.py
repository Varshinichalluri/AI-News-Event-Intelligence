
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT_DIR / "data" / "processed" / "clustered_articles.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "event_summaries.csv"


def make_summary(group):
    """Create a simple extractive summary from an event's article titles."""

    titles = (
        group["title"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    titles = titles[titles != ""]

    if titles.empty:
        return "No article titles are available for this event."

    unique_titles = titles.drop_duplicates()

    if len(unique_titles) == 1:
        return unique_titles.iloc[0]

    first_title = unique_titles.iloc[0]
    second_title = unique_titles.iloc[1]

    return (
        f"This event is covered in {len(group)} collected article(s). "
        f"One report: {first_title} "
        f"Additional coverage: {second_title}"
    )


def main():
    print("Loading clustered articles...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find input file: {INPUT_FILE}\n"
            "Run the clustering module first."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"event_id", "title"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(
            df["published_at"], errors="coerce", utc=True
        )
    else:
        df["published_at"] = pd.NaT

    if "source" not in df.columns:
        df["source"] = "Unknown source"

    df["source"] = df["source"].fillna("Unknown source")
    df["title"] = df["title"].fillna("").astype(str)

    summaries = []

    for event_id, group in df.groupby("event_id", sort=False):
        valid_dates = group["published_at"].dropna()

        sources = sorted(
            set(group["source"].astype(str).str.strip())
            - {"", "Unknown source"}
        )

        summaries.append({
            "event_id": event_id,
            "event_title": make_summary(group),
            "article_count": len(group),
            "source_count": len(sources),
            "sources": ", ".join(sources),
            "first_seen": (
                valid_dates.min().isoformat()
                if not valid_dates.empty else ""
            ),
            "last_seen": (
                valid_dates.max().isoformat()
                if not valid_dates.empty else ""
            ),
        })

    summary_df = pd.DataFrame(summaries)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSUMMARIZATION REPORT")
    print("=" * 40)
    print(f"Articles processed: {len(df)}")
    print(f"Event summaries created: {len(summary_df)}")
    print(f"Output saved to:\n{OUTPUT_FILE}")

    print("\nSUMMARY PREVIEW")
    print("=" * 40)

    for _, row in summary_df.head(5).iterrows():
        print(f"\n{row['event_id']} | Articles: {row['article_count']}")
        print(f"Summary: {row['event_title']}")
        print(f"Sources: {row['source_count']}")


if __name__ == "__main__":
    main()