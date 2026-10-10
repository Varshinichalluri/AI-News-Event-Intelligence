# AI News Event Intelligence

**An NLP-powered application for discovering related AI news events, clustering articles, and exploring event timelines.**

## Overview

AI News Event Intelligence is an end-to-end Python project that collects AI-related news from Google News RSS, extracts available article content, generates semantic embeddings, and groups similar articles into events.

The project generates event timelines and summaries and presents the results through an interactive Streamlit dashboard.

## Key Features

- **Automated News Collection** — Collects AI-related news using Google News RSS.
- **Article Text Extraction** — Retrieves article text where accessible, with RSS-description and headline fallbacks.
- **Semantic Embeddings** — Represents article text using the Sentence Transformers `all-MiniLM-L6-v2` model.
- **Event Clustering** — Groups similar articles using embedding similarity and clustering algorithms.
- **Event Identification** — Assigns event IDs to grouped articles.
- **Timeline Generation** — Organizes related news coverage chronologically.
- **Event Summaries** — Creates template-based summaries from available event information.
- **Interactive Dashboard** — Explore events, timelines, and articles through Streamlit.
- **Original Source Links** — Access the original news articles from the dashboard.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Natural Language Processing | Sentence Transformers |
| Machine Learning | Scikit-learn |
| News Collection | Google News RSS |
| Text Extraction | Trafilatura, BeautifulSoup, Newspaper4k |
| Dashboard | Streamlit |

## System Architecture

```text
Google News RSS
      |
      v
News Collection & Article Extraction
      |
      v
Text Preparation
      |
      v
Semantic Embedding Generation
      |
      v
Similarity Analysis & Event Clustering
      |
      v
Event Identification
      |
      +-------------------+
      |                   |
      v                   v
Timeline Generation   Event Summaries
      |                   |
      +---------+---------+
                |
                v
       Streamlit Dashboard
                |
                v
       Explore Events & News
```

## Project Structure

```text
AI-News-Event-Intelligence/
│
├── app/
│   └── dashboard.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_collection/
│   │   └── fetch_news.py
│   ├── preprocessing/
│   │   └── validate_data.py
│   ├── embeddings/
│   │   └── generate_embeddings.py
│   ├── clustering/
│   │   ├── cluster_articles.py
│   │   ├── inspect_clusters.py
│   │   └── analyze_similarity.py
│   ├── timeline/
│   │   └── build_timelines.py
│   └── summarization/
│       └── summarize_events.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

*The structure above reflects the current project organization; adjust any filenames if yours differ.*

## Installation and Setup

### Prerequisites

- Python 3.10 or later recommended
- Git
- Internet access for news collection and initial model download

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-News-Event-Intelligence
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Ensure `requirements.txt` includes all packages used by the current project.

## Running the Project

Run each command separately from the project root.

### Step 1: Collect News

```bash
python -m src.data_collection.fetch_news
```

### Step 2: Generate Embeddings

```bash
python -m src.embeddings.generate_embeddings
```

### Step 3: Cluster Related Articles

```bash
python -m src.clustering.cluster_articles
```

### Step 4: Build Event Timelines

```bash
python -m src.timeline.build_timelines
```

### Step 5: Generate Event Summaries

```bash
python -m src.summarization.summarize_events
```

### Step 6: Launch the Dashboard

```bash
python -m streamlit run app/dashboard.py
```

The dashboard opens in your browser, allowing you to explore detected events, related articles, and timelines.

## Sample Results

In one recorded execution, the dashboard displayed:

| Metric | Result |
|---|---:|
| Articles collected | 100 |
| Detected events | 72 |
| Distinct news sources | 83 |

These values describe one execution and may change when the collection and analysis pipeline is rerun.

## Screenshots

Add screenshots of your actual running dashboard here.

Recommended screenshots:

1. Dashboard overview showing article, event, and source counts.
2. Event Explorer displaying a detected event and its related articles.
3. Timelines tab.
4. All Articles tab.

Save the images in a `screenshots/` directory and embed them in this section once available.

## Current Limitations

- Article text extraction depends on publisher accessibility.
- Some articles may contain only RSS descriptions or headlines.
- Event clustering depends on embedding similarity and selected clustering thresholds.
- Some articles may be incorrectly grouped or remain in separate events.
- Event summaries are template-based and are not currently generated by an LLM.

## Future Enhancements

- LLM-powered event summarization.
- Improved event clustering and quantitative evaluation.
- Additional news sources and topic filters.
- Sentiment analysis and event trend monitoring.
- Dashboard deployment.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Developed as an AI/NLP portfolio project using Python, semantic embeddings, clustering, and Streamlit.**