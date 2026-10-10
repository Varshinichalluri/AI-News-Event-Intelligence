
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "processed" / "clustered_articles.csv"
)

df = pd.read_csv(DATA_PATH)

print("\nCLUSTER QUALITY INSPECTION")
print("=" * 55)

for event_id, group in df.groupby("event_id", sort=True):
    print(f"\n{event_id} — {len(group)} article(s)")

    for _, row in group.iterrows():
        print(f"  ID: {row['article_id']}")
        print(f"  Title: {row['title']}")
        print(f"  Source: {row['source']}")
        print()

print("\nINSPECTION COMPLETE")