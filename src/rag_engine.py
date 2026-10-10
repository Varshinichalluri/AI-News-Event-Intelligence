
import json
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

# Load news articles and saved embeddings
ARTICLES_PATH = "data/processed/articles_with_embeddings.csv"
EMBEDDINGS_PATH = "data/processed/article_embeddings.npy"


def retrieve_articles(question, articles, embeddings, model, top_k=5):
    """Retrieve news articles most relevant to the user's question."""
    question_embedding = model.encode([question])
    scores = cosine_similarity(question_embedding, embeddings)[0]
    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for i in indices:
        article = articles.iloc[i]

        results.append({
            "title": str(article.get("title", "Untitled")),
            "source": str(article.get("source", "Unknown")),
            "url": str(article.get("url", "")),
            "content": str(article.get("content", ""))[:1200],
            "score": float(scores[i]),
        })

    return results


def generate_answer(question, articles):
    """Generate an answer using Ollama and retrieved news articles."""
    context = ""

    for i, article in enumerate(articles, start=1):
        context += (
            f"\n[{i}] Title: {article['title']}\n"
            f"Publisher: {article['source']}\n"
            f"Content: {article['content']}\n"
        )

    prompt = f"""
You are an AI News Event Intelligence assistant.

Answer the question using only the news evidence provided below.
Do not invent facts or details.
If the evidence does not answer the question, say so clearly.
Keep the answer clear and concise.
Add citations such as [1] or [2] to relevant statements.
The numbers must match the supplied articles.

NEWS EVIDENCE:
{context}

QUESTION:
{question}

ANSWER:
"""

    payload = json.dumps({
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result.get("response", "").strip()


def main():
    print("Loading news articles and embeddings...")

    articles = pd.read_csv(ARTICLES_PATH).fillna("")
    embeddings = np.load(EMBEDDINGS_PATH)

    if len(articles) != len(embeddings):
        print("Error: Article count and embedding count do not match.")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")

    question = input("\nEnter your AI news question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    print("\nSearching relevant news articles...")
    results = retrieve_articles(question, articles, embeddings, model)

    print("\nGenerating answer with Llama 3.2...\n")

    try:
        answer = generate_answer(question, results)
        print("=" * 60)
        print("AI-GENERATED ANSWER")
        print("=" * 60)
        print(answer)

    except urllib.error.URLError as exc:
        print("Could not connect to Ollama.")
        print("Make sure Ollama is running.")
        print("Details:", exc)
        return

    print("\n" + "=" * 60)
    print("SOURCE ARTICLES")
    print("=" * 60)

    for i, article in enumerate(results, start=1):
        print(f"\n[{i}] {article['title']}")
        print(f"Publisher: {article['source']}")
        print(f"Relevance score: {article['score']:.3f}")
        print(f"URL: {article['url']}")


if __name__ == "__main__":
    main()
