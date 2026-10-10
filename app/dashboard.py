
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"

ARTICLES_FILE = DATA_DIR / "clustered_articles.csv"
SUMMARIES_FILE = DATA_DIR / "event_summaries.csv"
TIMELINES_FILE = DATA_DIR / "event_timelines.csv"


st.set_page_config(
    page_title="AI News Event Intelligence",
    page_icon="📰",
    layout="wide",
)

st.title("📰 AI News Event Intelligence")
st.caption(
    "Collect, group, and explore AI-related news coverage "
    "from multiple sources."
)


@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


articles = load_csv(ARTICLES_FILE)
summaries = load_csv(SUMMARIES_FILE)
timelines = load_csv(TIMELINES_FILE)

if articles.empty:
    st.error(
        "No clustered articles found. Run the data collection, "
        "embedding, and clustering pipeline first."
    )
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

if "event_id" in articles.columns:
    event_ids = sorted(articles["event_id"].dropna().unique().tolist())
    selected_events = st.sidebar.multiselect(
        "Filter events",
        options=event_ids,
        default=event_ids,
    )
    filtered_articles = articles[
        articles["event_id"].isin(selected_events)
    ].copy()
else:
    filtered_articles = articles.copy()

# Overview metrics
event_count = (
    filtered_articles["event_id"].nunique()
    if "event_id" in filtered_articles.columns
    else 0
)

col1, col2, col3 = st.columns(3)
col1.metric("Articles", len(filtered_articles))
col2.metric("Events", event_count)
col3.metric(
    "News sources",
    filtered_articles["source"].nunique()
    if "source" in filtered_articles.columns
    else "N/A",
)

st.divider()

tab1, tab2, tab3 = st.tabs(
    ["🗞️ Event Explorer", "🕒 Timelines", "📄 All Articles"]
)

with tab1:
    st.subheader("Explore news events")

    if not summaries.empty and "event_id" in summaries.columns:
        available_summary_ids = summaries["event_id"].dropna().tolist()
        matching_ids = [
            event_id for event_id in available_summary_ids
            if event_id in selected_events
        ]

        if matching_ids:
            selected_summary_id = st.selectbox(
                "Choose an event",
                matching_ids,
            )
            selected_summary = summaries[
                summaries["event_id"] == selected_summary_id
            ].iloc[0]

            st.markdown(f"### {selected_summary_id}")
            st.write(selected_summary.get("event_title", "No summary available"))
            st.caption(
                f"Articles: {selected_summary.get('article_count', 'N/A')} "
                f" · Sources: {selected_summary.get('source_count', 'N/A')}"
            )

            event_articles = filtered_articles[
                filtered_articles["event_id"] == selected_summary_id
            ]

            st.markdown("#### Related articles")
            for _, article in event_articles.iterrows():
                title = article.get("title", "Untitled article")
                source = article.get("source", "Unknown source")
                url = article.get("url", "")

                st.markdown(f"**{title}**")
                st.caption(str(source))

                if isinstance(url, str) and url.startswith(("https://", "http://")):
                    st.markdown(f"[Read article]({url})")

                st.divider()
        else:
            st.info("No event summaries match the selected filters.")
    else:
        st.warning("The event summaries file is missing or empty.")

with tab2:
    st.subheader("Event timelines")

    if not timelines.empty and "event_id" in timelines.columns:
        filtered_timelines = timelines[
            timelines["event_id"].isin(selected_events)
        ].copy()

        display_columns = [
            column for column in [
                "event_id",
                "event_title",
                "article_count",
                "first_seen",
                "last_seen",
                "duration_hours",
            ]
            if column in filtered_timelines.columns
        ]

        st.dataframe(
            filtered_timelines[display_columns],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("The event timelines file is missing or empty.")

with tab3:
    st.subheader("All collected articles")

    search_text = st.text_input(
        "Search article titles",
        placeholder="Enter a keyword, company, or model name...",
    )

    display_articles = filtered_articles.copy()

    if search_text and "title" in display_articles.columns:
        display_articles = display_articles[
            display_articles["title"].str.contains(
                search_text,
                case=False,
                na=False,
                regex=False,
            )
        ]

    columns = [
        column for column in [
            "event_id",
            "title",
            "source",
            "published_at",
            "url",
        ]
        if column in display_articles.columns
    ]

    st.dataframe(
        display_articles[columns],
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Prototype note: event grouping and title-based summaries may "
    "contain errors. Review clusters before relying on them."
)