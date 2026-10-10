# AI News Event Intelligence

An AI-powered news analytics and intelligence platform that collects AI-related news, groups similar articles into events, builds event timelines, and enables evidence-based question answering through a Retrieval-Augmented Generation (RAG) chatbot.

## Features

- Collect AI-related news using Google News RSS.
- Extract and preprocess article text.
- Generate semantic embeddings using Sentence Transformers.
- Cluster similar articles into events.
- Assign event IDs and analyze article similarity.
- Build event timelines and event summaries.
- Explore news articles and events through an interactive Streamlit dashboard.
- Ask natural-language questions about collected news using a RAG chatbot.
- Retrieve relevant article content using semantic similarity search.
- Generate answers using a locally running Ollama language model.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Machine Learning:** Scikit-learn
- **Dashboard:** Streamlit
- **News Collection and Extraction:** Google News RSS, BeautifulSoup, Trafilatura
- **Vector Search:** ChromaDB and/or NumPy-based cosine similarity, depending on the implemented retrieval path
- **LLM:** Ollama with `llama3.2:3b`

## Project Structure

```text
AI-News-Event-Intelligence/
├── app/
│   └── dashboard.py
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_collection/
│   ├── preprocessing/
│   ├── embeddings/
│   ├── clustering/
│   ├── timeline/
│   ├── summarization/
│   └── rag_engine.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Varshinichalluri/AI-News-Event-Intelligence.git
cd AI-News-Event-Intelligence
```

### 2. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Dashboard

Start the local Ollama service and ensure the required model is available:

```bash
ollama pull llama3.2:3b
```

Run the Streamlit dashboard from the project root:

```bash
python -m streamlit run app/dashboard.py
```

Open the local URL displayed in the terminal.

**Note:** The RAG chatbot requires the local Ollama service and the configured model. If your repository already includes processed article data and embeddings, you can test the dashboard without rerunning the entire collection pipeline.

## How the RAG Chatbot Works

1. **Question:** The user enters a natural-language question.
2. **Retrieval:** The system finds relevant article content using semantic similarity.
3. **Context building:** The most relevant retrieved articles are supplied as context.
4. **Generation:** The local Ollama language model generates an answer grounded in the retrieved evidence.
5. **Source display:** Retrieved article references help users inspect the supporting news sources.

## Current Results

In one recorded run, the project processed 100 articles and identified 72 event clusters from 83 sources. Results may vary depending on the collected articles and clustering configuration.

## Limitations

- Article extraction depends on publisher accessibility and article availability.
- Clustering quality depends on the collected data, embeddings, and similarity threshold.
- RAG answer quality depends on the relevance and completeness of the indexed articles.
- The chatbot may return insufficient evidence when the dataset does not cover a question.
- Local LLM inference requires Ollama and sufficient system resources.

## Future Improvements

- Add automated news collection and scheduled updates.
- Improve clustering evaluation and event deduplication.
- Add keyword extraction and named entity recognition.
- Expand source coverage and evaluate RAG retrieval quality.
- Deploy the dashboard for broader access.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
