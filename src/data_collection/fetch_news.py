
import time
from pathlib import Path
from html import unescape
from newspaper import Article

import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from googlenewsdecoder import gnewsdecoder


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gdelt_news_articles_enriched.csv"
)

# Google News RSS settings
RSS_URL = "https://news.google.com/rss/search"

RSS_PARAMS = {
    "q": '"artificial intelligence" OR "AI model"',
    "hl": "en-IN",
    "gl": "IN",
    "ceid": "IN:en",
}

MAX_ARTICLES = 100
MIN_ARTICLE_LENGTH = 200
MIN_RSS_LENGTH = 80
REQUEST_TIMEOUT = 15


def clean_html(value):
    """Convert HTML snippets into readable text."""
    if not value:
        return ""

    soup = BeautifulSoup(
        unescape(str(value)),
        "html.parser",
    )

    return " ".join(soup.get_text(" ", strip=True).split())

        # Third attempt: Newspaper article extraction
    try:
            article = Article(url, request_timeout=REQUEST_TIMEOUT)
            article.download()
            article.parse()

            newspaper_text = " ".join(
                article.text.split()
            )

            if len(newspaper_text) >= MIN_ARTICLE_LENGTH:
                return newspaper_text

    except Exception as exc:
            print(f"  Newspaper extraction failed: {exc}")
def get_rss_content(item, title):
    """Collect the best available text from an RSS item."""
    candidates = []

    # Standard RSS description
    description_tag = item.find("description")

    if description_tag:
        description = clean_html(
            description_tag.get_text()
        )

        if description and description.lower() != title.lower():
            candidates.append(description)

    # Optional content:encoded field
    for tag in item.find_all(
        lambda element: (
            element.name
            and element.name.split(":")[-1] == "encoded"
        )
    ):
        encoded_text = clean_html(tag.get_text())

        if encoded_text and encoded_text.lower() != title.lower():
            candidates.append(encoded_text)

    # Prefer the longest available RSS text
    if candidates:
        return max(candidates, key=len)

    return ""


def decode_article_url(google_url):
    """Resolve a Google News URL to its publisher URL."""
    if not google_url:
        return ""

    try:
        result = gnewsdecoder(
            google_url,
            interval=1,
        )

        if result and result.get("success"):
            return result.get("decoded_url") or google_url

    except Exception as exc:
        print(f"  URL decoding failed: {exc}")

    return google_url


def extract_article_text(url):
    """Download and extract readable publisher article text."""
    if not url or not url.startswith(("http://", "https://")):
        return ""

    try:
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return ""

        # First attempt: Trafilatura article extraction
        text = trafilatura.extract(
            downloaded,
            favor_precision=True,
            include_comments=False,
            include_tables=False,
        )

        if text and len(text.strip()) >= MIN_ARTICLE_LENGTH:
            return " ".join(text.split())

        # Second attempt: BeautifulSoup fallback
        soup = BeautifulSoup(
            downloaded,
            "html.parser",
        )

        for element in soup([
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "form",
            "noscript",
        ]):
            element.decompose()

        candidates = []

        # Prefer article-specific content
        for element in soup.find_all(["article", "main"]):
            candidate = " ".join(
                element.get_text(" ", strip=True).split()
            )

            if len(candidate) >= MIN_ARTICLE_LENGTH:
                candidates.append(candidate)

        if candidates:
            return max(candidates, key=len)

    except Exception as exc:
        print(f"  Text extraction failed: {exc}")

    return ""


def main():
    """Fetch, enrich, and save AI news articles."""
    print("Fetching recent AI news from Google News RSS...")

    try:
        response = requests.get(
            RSS_URL,
            params=RSS_PARAMS,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": "Mozilla/5.0",
            },
        )

        response.raise_for_status()

        feed = BeautifulSoup(
            response.content,
            "xml",
        )

        items = feed.find_all("item")[:MAX_ARTICLES]

    except Exception as exc:
        print(f"ERROR: Could not retrieve RSS feed: {exc}")
        return

    if not items:
        print("ERROR: No RSS articles were found.")
        return

    articles = []

    for index, item in enumerate(items, start=1):

        # Article title
        title = (
            clean_html(item.title.get_text())
            if item.title
            else ""
        )

        # Original Google News URL
        google_url = (
            item.link.get_text(strip=True)
            if item.link
            else ""
        )

        # Publisher
        source_tag = item.find("source")

        source = (
            source_tag.get_text(strip=True)
            if source_tag
            else "Unknown"
        )

        # Publication date
        published_tag = item.find("pubDate")

        published_at = (
            published_tag.get_text(strip=True)
            if published_tag
            else ""
        )

        # Best available RSS description/content
        rss_description = get_rss_content(
            item,
            title,
        )

        print(
            f"[{index}/{len(items)}] "
            f"Processing: {title[:90]}"
        )

        # Resolve the publisher article URL
        article_url = decode_article_url(google_url)

        # Try to extract the full article
        content = extract_article_text(article_url)

        if content:
            content_type = "article_text"

        else:
            # Fall back to RSS content
            if len(rss_description.strip()) >= MIN_RSS_LENGTH:
                content = rss_description
                content_type = "rss_description"

            elif rss_description:
                # Keep shorter descriptions if available
                content = rss_description
                content_type = "rss_description"

            else:
                # Last resort: use the headline
                content = title
                content_type = "headline_only"

        # Save the collected article
        articles.append({
            "article_id": f"RSS-{index:04d}",
            "title": title,
            "source": source,
            "published_at": published_at,
            "url": article_url,
            "content": content,
            "content_type": content_type,
        })

        print(
            f"  Type: {content_type} | "
            f"Content length: {len(content)}"
        )

        # Avoid rapid consecutive publisher requests
        time.sleep(0.5)

    # Convert collected articles into a DataFrame
    df = pd.DataFrame(articles)

    # Create output directory if needed
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the enriched dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # Final report
    print("\nNews collection completed!")
    print(f"Articles collected: {len(df)}")
    print(f"Unique sources: {df['source'].nunique()}")

    print("\nContent type counts:")
    print(df["content_type"].value_counts().to_string())

    print(
        "\nAverage content length: "
        f"{df['content'].str.len().mean():.0f}"
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
