
from pathlib import Path
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT_DIR / "data" / "processed" / "articles_with_embeddings.csv"
EMBEDDINGS_FILE = ROOT_DIR / "data" / "processed" / "article_embeddings.npy"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "clustered_articles.csv"

# Lower cosine distance means articles must be more similar.
DISTANCE_THRESHOLD = 0.35

# Generic words are excluded so they don't dominate title overlap.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "for", "to", "of", "in",
    "on", "at", "by", "from", "with", "as", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "their", "they", "them", "you", "your", "we", "our",
    "what", "why", "how", "who", "when", "where", "which", "could",
    "would", "should", "will", "can", "may", "might", "new", "says",
    "said", "say", "after", "before", "about", "into", "over", "more",
    "than", "all", "any", "here", "there", "have", "has", "had",
    "artificial", "intelligence", "ai", "model", "models"
}


def meaningful_tokens(title):
    """Extract informative title words and ignore generic terms."""
    words = re.findall(r"[a-z0-9]+", str(title).lower())

    return {
        word for word in words
        if len(word) > 2 and word not in STOP_WORDS
    }


def title_overlap(tokens_a, tokens_b):
    """Calculate Jaccard overlap between meaningful title words."""
    union = tokens_a | tokens_b

    if not union:
        return 0.0

    return len(tokens_a & tokens_b) / len(union)


def main():
    print("Loading article metadata and embeddings...")

    df = pd.read_csv(INPUT_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)

    if len(df) != len(embeddings):
        raise ValueError(
            f"Article count ({len(df)}) does not match "
            f"embedding count ({len(embeddings)})."
        )

    if len(df) < 2:
        raise ValueError("At least two articles are required.")

    if "title" not in df.columns:
        raise ValueError("Input CSV must contain a 'title' column.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values.")

    embeddings = normalize(embeddings)
    tokens = [meaningful_tokens(title) for title in df["title"].fillna("")]

    print(f"Articles loaded: {len(df)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print("\nCalculating semantic similarities...")

    # Cosine distance between article embeddings.
    similarity = embeddings @ embeddings.T
    similarity = np.clip(similarity, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)

    # Adjust distances using meaningful title-word overlap.
    # This is a heuristic: inspect results before treating clusters as final.
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            overlap = title_overlap(tokens[i], tokens[j])

            
            
        if overlap >= 0.40:
        # Strong title overlap: substantially favor grouping.
         adjusted_distance = distance[i, j] * 0.65
        elif overlap >= 0.15:
        # Some shared event-specific terms: modestly favor grouping.
         adjusted_distance = distance[i, j] * 0.85
        elif overlap == 0:
        # No shared meaningful words: require stronger semantic evidence.
         adjusted_distance = distance[i, j] + 0.12
    else:
        # Very weak title overlap.
        adjusted_distance = distance[i, j] + 0.05
 


        distance[i, j] = adjusted_distance
        distance[j, i] = adjusted_distance

    distance = np.clip(distance, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)

    print("Clustering articles...")

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=DISTANCE_THRESHOLD,
    )

    labels = model.fit_predict(distance)
    df["cluster_label"] = labels

    # Put the largest groups first and assign readable event IDs.
    cluster_sizes = df["cluster_label"].value_counts()
    ordered_labels = cluster_sizes.index.tolist()

    event_ids = {
        label: f"EVT-{index:03d}"
        for index, label in enumerate(ordered_labels, start=1)
    }

    df["event_id"] = df["cluster_label"].map(event_ids)

    # Flag exact normalized-title duplicates for review.
    normalized_titles = (
        df["title"]
        .fillna("")
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )

    df["duplicate_headline"] = (
        normalized_titles.ne("")
        & normalized_titles.duplicated(keep=False)
    )

    number_of_clusters = df["cluster_label"].nunique()

    score = None
    if 1 < number_of_clusters < len(df):
        try:
            score = silhouette_score(
                distance,
                labels,
                metric="precomputed",
            )
        except ValueError:
            pass

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCLUSTERING REPORT")
    print("=" * 45)
    print(f"Articles processed: {len(df)}")
    print(f"Number of clusters: {number_of_clusters}")
    print(f"Largest cluster: {cluster_sizes.max()} articles")
    print(f"Smallest cluster: {cluster_sizes.min()} articles")

    if score is not None:
        print(f"Adjusted-distance silhouette score: {score:.4f}")
    else:
        print("Silhouette score: not available")

    print(f"\nOutput saved to:\n{OUTPUT_FILE}")

    print("\nCLUSTER PREVIEW")
    print("=" * 45)

    for label in ordered_labels[:10]:
        group = df[df["cluster_label"] == label]

        print(f"\n{event_ids[label]} | Articles: {len(group)}")

        for title in group["title"].head(5):
            print(f"  - {title}")

        if len(group) > 5:
            print(f"  ... and {len(group) - 5} more articles")


if __name__ == "__main__":
    main()