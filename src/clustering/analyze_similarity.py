
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]

ARTICLES_FILE = ROOT / "data" / "processed" / "articles_with_embeddings.csv"
EMBEDDINGS_FILE = ROOT / "data" / "processed" / "article_embeddings.npy"
CLUSTERED_FILE = ROOT / "data" / "processed" / "clustered_articles.csv"

STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are",
    "was", "were", "has", "have", "its", "into", "how", "what",
    "why", "who", "says", "said", "new", "after", "over", "their",
    "artificial", "intelligence", "model", "models"
}


def meaningful_tokens(title):
    words = re.findall(r"[a-z0-9]+", str(title).lower())
    return {word for word in words if len(word) > 2 and word not in STOP_WORDS}


def title_overlap(tokens_a, tokens_b):
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


print("\nPAIRWISE SIMILARITY DIAGNOSTICS")
print("=" * 65)

articles = pd.read_csv(ARTICLES_FILE)
embeddings = np.load(EMBEDDINGS_FILE)
clustered = pd.read_csv(CLUSTERED_FILE)

if len(articles) != len(embeddings) or len(clustered) != len(articles):
    raise ValueError(
        f"Row count mismatch: articles={len(articles)}, "
        f"embeddings={len(embeddings)}, clustered={len(clustered)}"
    )

articles = articles.reset_index(drop=True)
clustered = clustered.reset_index(drop=True)

title_col = next(
    (c for c in articles.columns if c.lower() in ["title", "headline"]),
    None,
)
cluster_title_col = next(
    (c for c in clustered.columns if c.lower() in ["title", "headline"]),
    None,
)
cluster_col = next(
    (c for c in clustered.columns
     if "cluster" in c.lower() or "event_id" in c.lower()),
    None,
)

if title_col is None or cluster_title_col is None or cluster_col is None:
    raise ValueError(
        "Required columns not found.\n"
        f"Article columns: {articles.columns.tolist()}\n"
        f"Clustered columns: {clustered.columns.tolist()}"
    )

# Confirm that the two files have matching article order.
if not (
    articles[title_col].fillna("").astype(str).to_numpy()
    == clustered[cluster_title_col].fillna("").astype(str).to_numpy()
).all():
    raise ValueError(
        "Article order differs between metadata and clustered files. "
        "Align the rows before interpreting these diagnostics."
    )

similarity_matrix = cosine_similarity(embeddings)
token_sets = [
    meaningful_tokens(title) for title in articles[title_col]
]

missed_pairs = []
questionable_pairs = []

for i in range(len(articles)):
    for j in range(i + 1, len(articles)):
        similarity = float(similarity_matrix[i, j])
        same_cluster = (
            clustered.loc[i, cluster_col]
            == clustered.loc[j, cluster_col]
        )
        overlap = title_overlap(token_sets[i], token_sets[j])

        item = {
            "similarity": round(similarity, 3),
            "title_overlap": round(overlap, 3),
            "cluster_1": str(clustered.loc[i, cluster_col]),
            "cluster_2": str(clustered.loc[j, cluster_col]),
            "article_1": str(articles.loc[i, title_col]),
            "article_2": str(articles.loc[j, title_col]),
        }

        # Stronger candidates: similar meaning plus shared title terms.
        if not same_cluster and similarity >= 0.55 and overlap >= 0.10:
            missed_pairs.append(item)

        # Pairs grouped together despite weak semantic similarity.
        if same_cluster and similarity < 0.35:
            questionable_pairs.append(item)

missed_pairs.sort(
    key=lambda x: (x["title_overlap"], x["similarity"]),
    reverse=True,
)
questionable_pairs.sort(key=lambda x: x["similarity"])


def print_pairs(heading, pairs, limit=20):
    print(f"\n{heading}")
    print("-" * 65)

    if not pairs:
        print("No pairs matched these diagnostic rules.")
        return

    for pair in pairs[:limit]:
        print(
            f"\nSimilarity: {pair['similarity']} | "
            f"Title overlap: {pair['title_overlap']}"
        )
        print(f"Cluster 1: {pair['cluster_1']}")
        print(f"Article 1: {pair['article_1']}")
        print(f"Cluster 2: {pair['cluster_2']}")
        print(f"Article 2: {pair['article_2']}")


print_pairs("POSSIBLE MISSED EVENT MATCHES", missed_pairs)
print_pairs("POSSIBLE INCORRECT MERGES", questionable_pairs)

print(f"\nArticles checked: {len(articles)}")
print(f"Possible missed matches: {len(missed_pairs)}")
print(f"Possible incorrect merges: {len(questionable_pairs)}")
print("Diagnostic complete.")
