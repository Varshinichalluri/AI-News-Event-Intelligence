
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "gdelt_news_articles_enriched.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_PATH = OUTPUT_DIR / "article_embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "articles_with_embeddings.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["title"] = df["title"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)

    # Combine headline and article body for richer context.
    texts = (
        "Title: " + df["title"].str.strip()
        + "\nContent: " + df["content"].str.strip()
    ).tolist()

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Generating embeddings for {len(texts)} articles...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Save numerical vectors separately from article metadata.
    np.save(EMBEDDINGS_PATH, embeddings)
    df.to_csv(METADATA_PATH, index=False)

    print("\nEMBEDDING REPORT")
    print("=" * 40)
    print(f"Articles processed: {len(df)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"All embeddings finite: {np.isfinite(embeddings).all()}")
    print(f"All embeddings normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)}")
    print(f"Saved vectors: {EMBEDDINGS_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")

    # Show the most similar article pairs for inspection.
    similarity_matrix = cosine_similarity(embeddings)
    pairs = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            pairs.append(
                (
                    similarity_matrix[i, j],
                    df.iloc[i]["article_id"],
                    df.iloc[j]["article_id"],
                    df.iloc[i]["title"],
                    df.iloc[j]["title"],
                )
            )

    pairs.sort(reverse=True)

    print("\nTOP 5 MOST SIMILAR ARTICLE PAIRS")
    for score, id_a, id_b, title_a, title_b in pairs[:5]:
        print(f"\nSimilarity: {score:.4f} | {id_a} vs {id_b}")
        print(f"  A: {title_a}")
        print(f"  B: {title_b}")


if __name__ == "__main__":
    main()