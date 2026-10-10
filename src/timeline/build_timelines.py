
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT_DIR / "data" / "processed" / "clustered_articles.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "event_timelines.csv"


def main():
    print("Loading clustered news articles...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run the clustering module first."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"event_id", "title"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(
            df["published_at"], errors="coerce", utc=True
        )
    else:
        df["published_at"] = pd.NaT

    if "url" not in df.columns:
        df["url"] = ""

    if "source" not in df.columns:
        df["source"] = ""

    df["title"] = df["title"].fillna("Untitled article")
    df["url"] = df["url"].fillna("")
    df["source"] = df["source"].fillna("Unknown source")

    df = df.sort_values(
        ["event_id", "published_at"],
        ascending=[True, True],
        na_position="last",
    )

    timeline_rows = []

    for event_id, group in df.groupby("event_id", sort=False):
        valid_dates = group["published_at"].dropna()

        if not valid_dates.empty:
            first_seen = valid_dates.min()
            last_seen = valid_dates.max()
            duration_hours = (
                last_seen - first_seen
            ).total_seconds() / 3600
        else:
            first_seen = pd.NaT
            last_seen = pd.NaT
            duration_hours = None

        articles = []

        for _, row in group.iterrows():
            articles.append({
                "title": row["title"],
                "published_at": (
                    row["published_at"].isoformat()
                    if pd.notna(row["published_at"])
                    else None
                ),
                "source": row["source"],
                "url": row["url"],
            })

        timeline_rows.append({
            "event_id": event_id,
            "article_count": len(group),
            "first_seen": (
                first_seen.isoformat()
                if pd.notna(first_seen)
                else None
            ),
            "last_seen": (
                last_seen.isoformat()
                if pd.notna(last_seen)
                else None
            ),
            "duration_hours": (
                round(duration_hours, 2)
                if duration_hours is not None
                else None
            ),
            "event_title": group.iloc[0]["title"],
            "articles": articles,
        })

    timeline_df = pd.DataFrame(timeline_rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    timeline_df.to_csv(OUTPUT_FILE, index=False)

    print("\nTIMELINE REPORT")
    print("=" * 40)
    print(f"Articles processed: {len(df)}")
    print(f"Events processed: {len(timeline_df)}")
    print(f"Events with dates: {timeline_df['first_seen'].notna().sum()}")
    print(f"Output saved to:\n{OUTPUT_FILE}")

    print("\nTIMELINE PREVIEW")
    print("=" * 40)

    for _, event in timeline_df.head(5).iterrows():
        print(f"\n{event['event_id']} | {event['article_count']} articles")
        print(f"Event title: {event['event_title']}")
        print(f"First seen: {event['first_seen']}")
        print(f"Last seen:  {event['last_seen']}")


if __name__ == "__main__":
    main()