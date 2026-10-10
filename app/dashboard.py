from pathlib import Path
from html import escape
import json
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AI News Event Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


# =========================================================
# DATA LOADING AND HELPERS
# =========================================================
@st.cache_data
def load_csv(filename):
    path = DATA / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return pd.DataFrame()


articles = load_csv("clustered_articles.csv")
summaries = load_csv("event_summaries.csv")
timelines = load_csv("event_timelines.csv")

if articles.empty:
    st.error(
        "Article data not found. Expected "
        "data/processed/clustered_articles.csv"
    )
    st.stop()


def find_column(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def safe(value, fallback="Not available"):
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return fallback if not text or text.lower() == "nan" else text


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def clean_export(df):
    return df.drop(
        columns=["_event", "_source", "_title", "_date", "_day"],
        errors="ignore",
    )


# =========================================================
# COLUMN DETECTION AND NORMALIZATION
# =========================================================
event_col = find_column(articles, ["event_id", "cluster_id"])
title_col = find_column(articles, ["title", "headline", "article_title"])
source_col = find_column(articles, ["source", "source_name", "publisher"])
date_col = find_column(
    articles, ["published_at", "published_date", "publish_date", "date"]
)
url_col = find_column(articles, ["url", "link", "article_url"])
description_col = find_column(
    articles, ["description", "summary", "article_summary", "text", "content"]
)

if event_col is None:
    articles["event_id"] = "EVT-UNKNOWN"
    event_col = "event_id"

if title_col is None:
    articles["title"] = "Untitled article"
    title_col = "title"

if source_col is None:
    articles["source"] = "Unknown source"
    source_col = "source"

articles["_event"] = articles[event_col].fillna("EVT-UNKNOWN").astype(str)
articles["_source"] = articles[source_col].fillna("Unknown source").astype(str)
articles["_title"] = articles[title_col].fillna("Untitled article").astype(str)

if date_col:
    articles["_date"] = pd.to_datetime(
        articles[date_col], errors="coerce", utc=True
    )
else:
    articles["_date"] = pd.NaT


# =========================================================
# DARK PLOTLY THEME
# =========================================================
CHART_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "color": "#DCE8F8",
        "family": "Inter, sans-serif",
        "size": 12,
    },
    "margin": {"l": 15, "r": 15, "t": 20, "b": 15},
    "hoverlabel": {
        "bgcolor": "#17263B",
        "bordercolor": "#3A5572",
        "font": {"color": "#FFFFFF"},
    },
}


# =========================================================
# GLOBAL STYLING
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080F1E;
    --panel: #111E31;
    --border: #263B55;
    --text: #EAF2FF;
    --muted: #91A4BD;
    --cyan: #4DE2D0;
    --blue: #66A8FF;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"] {
    background: #080F1E !important;
    color: var(--text) !important;
}

[data-testid="stHeader"] {
    background: rgba(8,15,30,.96) !important;
}

[data-testid="stMain"] .block-container {
    max-width: 1600px;
    padding: 1.8rem 2.4rem 3rem;
}

[data-testid="stSidebar"] {
    background: #0C1729 !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #16253B !important;
    color: white !important;
    border-color: #344861 !important;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    letter-spacing: -.4px;
}

.hero {
    padding: 34px;
    border: 1px solid #284A63;
    border-radius: 22px;
    background:
      radial-gradient(ellipse at 85% 10%, rgba(77,226,208,.17), transparent 38%),
      radial-gradient(ellipse at 10% 100%, rgba(102,168,255,.14), transparent 42%),
      linear-gradient(120deg, #101B30, #102C40 70%, #123D45);
    margin-bottom: 24px;
}

.hero-label {
    color: var(--cyan);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2.5px;
    margin-bottom: 13px;
}

.hero h1 {
    color: white !important;
    font-size: clamp(27px, 3vw, 39px);
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 12px;
}

.hero p {
    color: #B5C6D9;
    font-size: 14px;
    max-width: 800px;
    line-height: 1.8;
    margin: 0;
}

.status-pill {
    display: inline-block;
    color: #8EF0DC;
    background: rgba(77,226,208,.09);
    border: 1px solid rgba(77,226,208,.25);
    border-radius: 30px;
    padding: 6px 11px;
    font-size: 11px;
    font-weight: 700;
    margin-top: 19px;
}

.section-label {
    font-size: 11px;
    font-weight: 800;
    color: var(--cyan);
    letter-spacing: 2px;
    margin: 20px 0 6px;
}

.section-sub {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 17px;
}

.metric-card {
    background: linear-gradient(145deg, #132239, #101B2D);
    border: 1px solid #263B55;
    border-radius: 17px;
    padding: 19px;
    min-height: 125px;
    box-shadow: 0 8px 25px rgba(0,0,0,.12);
}

.metric-top {
    color: #9CB0C8;
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 15px;
}

.metric-value {
    color: #F2F7FF;
    font-size: 31px;
    font-weight: 800;
    line-height: 1.2;
}

.metric-foot {
    color: var(--cyan);
    font-size: 11px;
    margin-top: 9px;
}

.panel {
    background: linear-gradient(145deg, #111F33, #0E192A);
    border: 1px solid #243650;
    border-radius: 17px;
    padding: 18px;
    margin-bottom: 16px;
}

.panel-title {
    color: #EDF4FF;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 5px;
}

.panel-caption {
    color: #91A4BD;
    font-size: 12px;
    margin-bottom: 12px;
}

.event-card {
    background: linear-gradient(130deg, #14243A, #101B2D);
    border: 1px solid #2A3F59;
    border-radius: 15px;
    padding: 18px;
    margin: 10px 0;
}

.event-id {
    color: var(--cyan);
    font-size: 10px;
    letter-spacing: 1.5px;
    font-weight: 800;
    margin-bottom: 8px;
}

.event-title {
    color: #EFF6FF;
    font-size: 15px;
    line-height: 1.6;
    font-weight: 700;
    margin-bottom: 12px;
}

.event-info {
    color: #9CB0C8;
    font-size: 11px;
    line-height: 1.8;
}

.article-card {
    background: #111E31;
    border: 1px solid #253952;
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
}

.article-card a, a {
    color: #70C7FF !important;
}

.article-source {
    color: var(--cyan);
    font-size: 11px;
    margin-top: 7px;
}

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 7px;
    background: #0C1728;
    border: 1px solid #20324A;
    border-radius: 13px;
    padding: 6px;
}

.stTabs [data-baseweb="tab"] {
    color: #A7B9CE;
    background: transparent;
    border-radius: 9px;
    padding: 10px 15px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: #164653 !important;
    color: #76F0DD !important;
}

.stButton button, .stDownloadButton button {
    background: #164653;
    border: 1px solid #267A7C;
    color: #E9FFFC;
    border-radius: 9px;
    font-weight: 600;
}

.stButton button:hover, .stDownloadButton button:hover {
    background: #1C6268;
    border-color: var(--cyan);
    color: white;
}

[data-testid="stDataFrame"] {
    border: 1px solid #253952;
    border-radius: 12px;
}

div[data-testid="stMetric"] {
    background: #111E31;
    border: 1px solid #293E58;
    border-radius: 15px;
    padding: 17px;
}

div[data-testid="stMetricLabel"] {
    color: #A7B9CE !important;
}

div[data-testid="stMetricValue"] {
    color: #EFF6FF !important;
    font-weight: 800;
}

hr {
    border-color: #263750;
}

footer {
    visibility: hidden;
}

@media (max-width: 800px) {
    [data-testid="stMain"] .block-container {
        padding: 1rem;
    }

    .hero {
        padding: 24px;
    }
}/* Your existing CSS remains here */

/* Fix typed question visibility in AI News Chat */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
    background-color: #111E31 !important;
    color: #EAF2FF !important;
    -webkit-text-fill-color: #EAF2FF !important;
    caret-color: #4DE2D0 !important;
    border: 1px solid #4DE2D0 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #91A4BD !important;
    -webkit-text-fill-color: #91A4BD !important;
    opacity: 1 !important;
}

[data-testid="stChatMessage"] {
    color: #EAF2FF !important;
}

[data-testid="stChatMessage"] p {
    color: #EAF2FF !important;
}
</style>
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.markdown("## 🌐 NEWS INTELLIGENCE")
st.sidebar.caption("EVENT ANALYTICS PLATFORM")
st.sidebar.divider()
st.sidebar.markdown("### Refine your coverage")

all_sources = sorted(articles["_source"].unique().tolist())
all_events = sorted(articles["_event"].unique().tolist())

selected_sources = st.sidebar.multiselect(
    "Publishers", all_sources, default=[]
)
selected_events = st.sidebar.multiselect(
    "Event clusters", all_events, default=[]
)

use_dates = False
start_date = None
end_date = None
valid_dates = articles["_date"].dropna()

if not valid_dates.empty:
    use_dates = st.sidebar.checkbox(
        "Limit publication dates", value=False
    )

    if use_dates:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        chosen_range = st.sidebar.date_input(
            "Publication period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(chosen_range, (tuple, list)) and len(chosen_range) == 2:
            start_date, end_date = chosen_range
        elif chosen_range:
            start_date = end_date = chosen_range

st.sidebar.divider()
st.sidebar.markdown("### About this project")
st.sidebar.caption("Collect · Embed · Cluster · Explore")
st.sidebar.caption(
    "Discover related news articles and investigate event coverage."
)
st.sidebar.caption(
    "Python · Pandas · Sentence Transformers · Plotly · Streamlit"
)


# =========================================================
# APPLY FILTERS
# =========================================================
df = articles.copy()

if selected_sources:
    df = df[df["_source"].isin(selected_sources)]

if selected_events:
    df = df[df["_event"].isin(selected_events)]

if use_dates and start_date and end_date:
    day_values = df["_date"].dt.date
    df = df[
        day_values.between(start_date, end_date) | day_values.isna()
    ]

if df.empty:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-label">NEWS ANALYTICS</div>
          <h1>No matching articles</h1>
          <p>Adjust the filters in the sidebar to view your dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# =========================================================
# HERO HEADER
# =========================================================
st.markdown(
    """
<div class="hero">
  <div class="hero-label">● &nbsp; AI-POWERED NEWS ANALYTICS</div>
  <h1>AI News Event Intelligence</h1>
  <p>
    From headlines to connected stories. Explore emerging events,
    compare publisher coverage, and investigate how news unfolds
    over time in one intelligence workspace.
  </p>
  <div class="status-pill">● &nbsp; ANALYTICS WORKSPACE &nbsp; / &nbsp; DATASET LOADED</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# KPI CARDS
# =========================================================
total_articles = len(df)
total_events = df["_event"].nunique()
total_sources = df["_source"].nunique()
average_size = total_articles / max(total_events, 1)

metrics = [
    ("TOTAL ARTICLES", f"{total_articles:,}", "▤", "News records in current view"),
    ("DISCOVERED EVENTS", f"{total_events:,}", "◈", "Unique event clusters"),
    ("PUBLISHERS", f"{total_sources:,}", "◎", "Distinct news sources"),
    ("AVG. EVENT SIZE", f"{average_size:.1f}", "⌁", "Articles per event"),
]

metric_columns = st.columns(4)

for metric_col, (label, value, symbol, foot) in zip(metric_columns, metrics):
    with metric_col:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-top">{symbol} &nbsp; {label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-foot">↗ &nbsp; {foot}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="section-label">INTELLIGENCE CENTER</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-sub">Monitor coverage, investigate clusters, '
    'and explore your article collection.</div>',
    unsafe_allow_html=True,
)

overview_tab, explorer_tab, timeline_tab, library_tab, chat_tab = st.tabs(
    [
        "◫  Overview",
        "◈  Event Explorer",
        "⌁  Timelines",
        "▤  Article Library",
        "💬  AI News Chat",
    ]
)


# =========================================================
# OVERVIEW TAB
# =========================================================
with overview_tab:
    source_counts = df["_source"].value_counts().head(10)
    event_counts = df["_event"].value_counts().head(10)

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="panel"><div class="panel-title">Publisher Coverage</div>'
            '<div class="panel-caption">Publishers with the most articles</div>',
            unsafe_allow_html=True,
        )

        if not source_counts.empty:
            source_plot = pd.DataFrame(
                {
                    "Publisher": source_counts.index.astype(str),
                    "Articles": source_counts.values,
                }
            )

            fig_sources = px.bar(
                source_plot,
                x="Articles",
                y="Publisher",
                orientation="h",
                color="Articles",
                color_continuous_scale=["#155E75", "#4DE2D0"],
                hover_data={"Publisher": True, "Articles": True},
            )

            fig_sources.update_layout(
                **CHART_LAYOUT,
                height=350,
                showlegend=False,
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
            )
            fig_sources.update_xaxes(
                showgrid=True, gridcolor="#263750", zeroline=False
            )
            fig_sources.update_yaxes(showgrid=False)

            st.plotly_chart(
                fig_sources,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="panel"><div class="panel-title">Most Covered Events</div>'
            '<div class="panel-caption">Clusters with the most articles</div>',
            unsafe_allow_html=True,
        )

        if not event_counts.empty:
            event_plot = pd.DataFrame(
                {
                    "Event": event_counts.index.astype(str),
                    "Articles": event_counts.values,
                }
            )

            fig_events = px.bar(
                event_plot,
                x="Articles",
                y="Event",
                orientation="h",
                color="Articles",
                color_continuous_scale=["#3730A3", "#66A8FF"],
                hover_data={"Event": True, "Articles": True},
            )

            fig_events.update_layout(
                **CHART_LAYOUT,
                height=350,
                showlegend=False,
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
            )
            fig_events.update_xaxes(
                showgrid=True, gridcolor="#263750", zeroline=False
            )
            fig_events.update_yaxes(showgrid=False)

            st.plotly_chart(
                fig_events,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Publication activity and daily breakdown
    st.markdown(
        '<div class="panel"><div class="panel-title">News Publication Activity</div>'
        '<div class="panel-caption">Number of articles by publication date. '
        'This chart does not represent collection time.</div>',
        unsafe_allow_html=True,
    )

    dated = df.dropna(subset=["_date"]).copy()

    if not dated.empty:
        dated["_day"] = dated["_date"].dt.floor("D")
        daily = (
            dated.groupby("_day")
            .size()
            .rename("Articles")
            .sort_index()
        )

        full_days = pd.date_range(
            start=daily.index.min(),
            end=daily.index.max(),
            freq="D",
            tz="UTC",
        )

        daily = daily.reindex(full_days, fill_value=0)
        daily.index.name = "_day"
        daily_table = daily.rename("Articles").reset_index()
        total_dated = int(daily_table["Articles"].sum())

        daily_table["Share of dated articles (%)"] = (
            daily_table["Articles"]
            .div(max(total_dated, 1))
            .mul(100)
            .round(1)
        )
        daily_table["Publication date"] = daily_table["_day"].dt.strftime(
            "%d %b %Y"
        )

        fig_activity = px.area(
            daily_table,
            x="_day",
            y="Articles",
            markers=True,
            color_discrete_sequence=["#4DE2D0"],
            labels={
                "_day": "Publication date (UTC)",
                "Articles": "Article count",
            },
            custom_data=["Publication date"],
        )

        fig_activity.update_traces(
            line={"color": "#4DE2D0", "width": 2.5},
            marker={
                "size": 6,
                "color": "#4DE2D0",
                "line": {"width": 1, "color": "#EAF2FF"},
            },
            fillcolor="rgba(77,226,208,0.12)",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Articles published: %{y}<extra></extra>"
            ),
        )

        yaxis_config = {
            "showgrid": True,
            "gridcolor": "#263750",
            "title": "Number of articles",
            "rangemode": "tozero",
        }

        if daily_table["Articles"].max() <= 15:
            yaxis_config["dtick"] = 1

        fig_activity.update_layout(
            **CHART_LAYOUT,
            height=360,
            hovermode="x unified",
            xaxis={
                "showgrid": False,
                "title": "Publication date (UTC)",
                "tickformat": "%d %b",
                "tickangle": -30,
                "automargin": True,
            },
            yaxis=yaxis_config,
        )

        st.plotly_chart(
            fig_activity,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        peak_row = daily_table.loc[daily_table["Articles"].idxmax()]

        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("Articles with valid dates", f"{total_dated:,}")
        stat2.metric(
            "Days with published articles",
            f"{int((daily_table['Articles'] > 0).sum()):,}",
        )
        stat3.metric(
            "Busiest publication day",
            peak_row["Publication date"],
            help=f"{int(peak_row['Articles']):,} articles have this publication date.",
        )

        with st.expander("View daily article-count breakdown", expanded=True):
            display_daily = daily_table[
                [
                    "Publication date",
                    "Articles",
                    "Share of dated articles (%)",
                ]
            ].copy()

            display_daily = display_daily.iloc[::-1].reset_index(drop=True)

            st.dataframe(
                display_daily,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Articles": st.column_config.NumberColumn(
                        "Articles", format="%d"
                    ),
                    "Share of dated articles (%)": st.column_config.NumberColumn(
                        "Share of dated articles", format="%.1f%%"
                    ),
                },
            )

        st.caption(
            "Dates are interpreted in UTC. Articles without a valid publication date "
            "are excluded from this chart and breakdown. A publication-date spike "
            "does not necessarily mean the collector fetched all those articles that day."
        )
    else:
        st.info("No valid publication dates are available for the activity chart.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Recent event cards
    st.markdown("### Recent Event Discoveries")

    recent = (
        df.sort_values("_date", ascending=False, na_position="last")
        .drop_duplicates("_event")
        .head(6)
    )

    card_columns = st.columns(2)

    for i, (_, row) in enumerate(recent.iterrows()):
        event_id = escape(safe(row["_event"]))
        headline = escape(safe(row["_title"]))
        publisher = escape(safe(row["_source"]))
        cluster_size = int((df["_event"] == row["_event"]).sum())

        with card_columns[i % 2]:
            st.markdown(
                f"""
                <div class="event-card">
                  <div class="event-id">{event_id}</div>
                  <div class="event-title">{headline}</div>
                  <div class="event-info">◉ &nbsp; {publisher}<br>
                    ◈ &nbsp; {cluster_size} article(s) in this cluster
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.download_button(
        "↓  Export filtered dataset",
        csv_bytes(clean_export(df)),
        file_name="news_intelligence_export.csv",
        mime="text/csv",
    )


# =========================================================
# EVENT EXPLORER TAB
# =========================================================
with explorer_tab:
    st.markdown("### Explore connected news stories")
    st.caption(
        "Select an event cluster to inspect related articles and coverage."
    )

    event_options = sorted(df["_event"].unique().tolist())
    chosen_event = st.selectbox("Select event cluster", event_options)
    subset = df[df["_event"] == chosen_event].copy()
    display_title = safe(subset.iloc[0]["_title"])
    summary_text = ""

    summary_event_col = find_column(
        summaries, ["event_id", "cluster_id"]
    )
    summary_title_col = find_column(
        summaries, ["event_title", "title", "headline"]
    )
    summary_body_col = find_column(
        summaries, ["event_summary", "summary", "summary_text", "description"]
    )

    if not summaries.empty and summary_event_col:
        matching = summaries[
            summaries[summary_event_col].astype(str) == chosen_event
        ]

        if not matching.empty:
            summary_row = matching.iloc[0]

            if summary_title_col:
                display_title = safe(
                    summary_row[summary_title_col], display_title
                )

            if summary_body_col:
                summary_text = safe(summary_row[summary_body_col], "")

    st.markdown(
        f"""
        <div class="event-card">
          <div class="event-id">SELECTED EVENT / {escape(chosen_event)}</div>
          <div class="event-title" style="font-size:21px;">{escape(display_title)}</div>
          <div class="event-info">{len(subset)} articles &nbsp; · &nbsp;
          {subset["_source"].nunique()} publishers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if summary_text:
        st.markdown("#### Available event summary")
        st.write(summary_text)
    else:
        st.info(
            "No separate summary text was found. The event headline is shown instead; "
            "it is not presented as an AI-generated summary."
        )

    st.markdown("#### Related coverage")

    for _, row in subset.head(30).iterrows():
        headline = escape(safe(row["_title"]))
        publisher = escape(safe(row["_source"]))
        published = (
            escape(safe(row[date_col])) if date_col else "Date unavailable"
        )
        link = safe(row[url_col], "") if url_col else ""

        st.markdown(
            f"""
            <div class="article-card">
              <div class="event-title">{headline}</div>
              <div class="article-source">{publisher}</div>
              <div class="event-info">{published}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if link.startswith(("https://", "http://")):
            st.markdown(f"[↗ Read original article]({link})")

    st.download_button(
        "↓  Export event articles",
        csv_bytes(clean_export(subset)),
        file_name=f"{chosen_event}_articles.csv",
        mime="text/csv",
    )


# =========================================================
# TIMELINES TAB
# =========================================================
with timeline_tab:
    st.markdown("### Event development over time")
    st.caption(
        "Explore the timeline output created by your project pipeline."
    )

    if timelines.empty:
        st.warning(
            "Timeline data is not available. Run your existing timeline-building script."
        )
    else:
        timeline_event_col = find_column(
            timelines, ["event_id", "cluster_id"]
        )
        timeline_view = timelines.copy()

        if timeline_event_col:
            timeline_view = timeline_view[
                timeline_view[timeline_event_col]
                .astype(str)
                .isin(set(df["_event"].unique()))
            ]

        st.dataframe(
            timeline_view,
            use_container_width=True,
            hide_index=True,
        )

        first_seen_col = find_column(
            timeline_view,
            ["first_seen", "start_time", "start_date", "first_published"],
        )

        if first_seen_col and not timeline_view.empty:
            first_seen = pd.to_datetime(
                timeline_view[first_seen_col],
                errors="coerce",
                utc=True,
            ).dropna()

            if not first_seen.empty:
                activity = (
                    first_seen.dt.floor("D")
                    .value_counts()
                    .sort_index()
                    .rename_axis("Date")
                    .reset_index(name="Events")
                )

                fig_timeline = px.line(
                    activity,
                    x="Date",
                    y="Events",
                    markers=True,
                    color_discrete_sequence=["#A78BFA"],
                    labels={
                        "Date": "First publication date (UTC)",
                        "Events": "Events",
                    },
                )

                fig_timeline.update_layout(**CHART_LAYOUT, height=330)
                fig_timeline.update_xaxes(
                    showgrid=False,
                    tickformat="%d %b",
                    tickangle=-30,
                )
                fig_timeline.update_yaxes(
                    showgrid=True,
                    gridcolor="#263750",
                    rangemode="tozero",
                )

                st.plotly_chart(
                    fig_timeline,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        st.download_button(
            "↓  Export timelines",
            csv_bytes(timeline_view),
            file_name="event_timelines.csv",
            mime="text/csv",
        )


# =========================================================
# ARTICLE LIBRARY TAB
# =========================================================
with library_tab:
    st.markdown("### Search the article collection")

    search_text = st.text_input(
        "Search headlines and descriptions",
        placeholder="Try AI, model, company, research...",
    )

    results = df.copy()

    if search_text.strip():
        query = search_text.strip()

        match_mask = results["_title"].str.contains(
            query,
            case=False,
            na=False,
            regex=False,
        )

        if description_col:
            match_mask |= (
                results[description_col]
                .fillna("")
                .astype(str)
                .str.contains(
                    query,
                    case=False,
                    na=False,
                    regex=False,
                )
            )

        results = results[match_mask]

    st.caption(f"{len(results):,} matching articles")

    display_columns = [
        c for c in [event_col, title_col, source_col, date_col, url_col]
        if c and c in results.columns
    ]

    if display_columns:
        column_config = {}

        if url_col and url_col in display_columns:
            column_config[url_col] = st.column_config.LinkColumn(
                "Original article"
            )

        st.dataframe(
            results[display_columns],
            use_container_width=True,
            hide_index=True,
            height=470,
            column_config=column_config,
        )
    else:
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "↓  Export article search results",
        csv_bytes(clean_export(results)),
        file_name="article_search_results.csv",
        mime="text/csv",
    )


# =========================================================
# AI NEWS CHAT — RAG WITH LOCAL OLLAMA
# =========================================================
with chat_tab:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">● AI-POWERED RESEARCH</div>
            <h1>AI News Assistant</h1>
            <p>
                Ask questions about your indexed news articles.
                Answers are grounded in retrieved articles.
            </p>
            <div class="status-pill">● LOCAL RAG · LLAMA 3.2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def load_rag_resources():
        articles_path = (
            ROOT / "data" / "processed" / "articles_with_embeddings.csv"
        )
        embeddings_path = (
            ROOT / "data" / "processed" / "article_embeddings.npy"
        )

        if not articles_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(
                "RAG data files are missing. Expected "
                "articles_with_embeddings.csv and article_embeddings.npy "
                "inside data/processed."
            )

        rag_articles = pd.read_csv(articles_path).fillna("")
        rag_embeddings = np.load(embeddings_path)
        rag_model = SentenceTransformer("all-MiniLM-L6-v2")

        if len(rag_articles) != len(rag_embeddings):
            raise ValueError(
                f"Article count ({len(rag_articles)}) does not match "
                f"embedding count ({len(rag_embeddings)})."
            )

        return rag_articles, rag_embeddings, rag_model

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            for source in message.get("sources", []):
                title = source.get("title", "Untitled article")
                url = source.get("url", "")

                if url.startswith(("https://", "http://")):
                    st.markdown(f"- [{title}]({url})")
                else:
                    st.markdown(f"- {title}")

    question = st.chat_input(
        "Ask about AI news, companies, research, or trends..."
    )

    if question:
        st.session_state.rag_messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                rag_articles, rag_embeddings, rag_model = (
                    load_rag_resources()
                )

                query_embedding = rag_model.encode([question])

                scores = cosine_similarity(
                    query_embedding,
                    rag_embeddings,
                )[0]

                indices = [
                    int(i)
                    for i in np.argsort(scores)[::-1][:5]
                    if scores[i] >= 0.35
                ]

                if not indices:
                    answer = (
                        "I couldn't find sufficiently relevant articles "
                        "in the indexed dataset. Try asking about a topic "
                        "covered by your news collection."
                    )

                    st.markdown(answer)

                    st.session_state.rag_messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": [],
                        }
                    )

                else:
                    retrieved = []

                    for idx in indices:
                        row = rag_articles.iloc[idx]

                        retrieved.append(
                            {
                                "title": str(
                                    row.get("title", "Untitled")
                                ),
                                "source": str(
                                    row.get("source", "Unknown")
                                ),
                                "url": str(row.get("url", "")),
                                "content": str(
                                    row.get(
                                        "content",
                                        row.get("description", ""),
                                    )
                                )[:1400],
                            }
                        )

                    context = "\n\n".join(
                        f"[{i}] {item['title']} | {item['source']}\n"
                        f"{item['content']}\nURL: {item['url']}"
                        for i, item in enumerate(retrieved, start=1)
                    )

                    prompt = (
                        "You are an AI news research assistant. "
                        "Answer using only the article evidence below. "
                        "Do not invent facts or claim the dataset is real-time. "
                        "Cite claims using [1], [2], etc. If the evidence "
                        "does not answer the question, say so.\n\n"
                        f"ARTICLE EVIDENCE:\n{context}\n\n"
                        f"QUESTION: {question}\n\nANSWER:"
                    )

                    payload = json.dumps(
                        {
                            "model": "llama3.2:3b",
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 0.2},
                        }
                    ).encode("utf-8")

                    request = urllib.request.Request(
                        "http://localhost:11434/api/generate",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )

                    with st.spinner(
                        "Searching articles and generating answer..."
                    ):
                        with urllib.request.urlopen(
                            request,
                            timeout=180,
                        ) as response:
                            result = json.loads(
                                response.read().decode("utf-8")
                            )

                    answer = (
                        result.get("response", "").strip()
                        or "The local model returned an empty answer."
                    )

                    st.markdown(answer)
                    st.markdown("**Retrieved source articles**")

                    for i, item in enumerate(retrieved, start=1):
                        if item["url"].startswith(("https://", "http://")):
                            st.markdown(
                                f"[{i}] [{item['title']}]({item['url']})"
                            )
                        else:
                            st.markdown(f"[{i}] {item['title']}")

                    st.session_state.rag_messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved,
                        }
                    )

            except urllib.error.URLError as exc:
                answer = (
                    "Cannot connect to Ollama at http://localhost:11434. "
                    "Start Ollama and confirm `llama3.2:3b` is installed. "
                    f"Details: {exc}"
                )

                st.error(answer)

                st.session_state.rag_messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": [],
                    }
                )

            except Exception as exc:
                answer = f"Could not generate an answer: {exc}"
                st.error(answer)

                st.session_state.rag_messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": [],
                    }
                )

    if st.session_state.rag_messages:
        if st.button("Clear chat history", key="clear_rag_chat"):
            st.session_state.rag_messages = []
            st.rerun()


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
<div style="text-align:center;padding:15px 0 22px;">
  <div style="color:#4DE2D0;font-weight:800;letter-spacing:2px;font-size:11px;">
    AI NEWS EVENT INTELLIGENCE
  </div>
  <div style="color:#7187A2;font-size:11px;margin-top:8px;">
    Python · Pandas · Sentence Transformers · Plotly · Streamlit
  </div>
  <div style="color:#7187A2;font-size:10px;margin-top:5px;">
    Clusters and summaries can be imperfect. Verify important claims against the original articles.
  </div>
</div>/* Fix AI News Chat input and message text */
.stApp [data-testid="stChatInput"] textarea {
    background-color: #111E31 !important;
    color: #EAF2FF !important;
    -webkit-text-fill-color: #EAF2FF !important;
    caret-color: #4DE2D0 !important;
}

.stApp [data-testid="stChatInput"] textarea::placeholder {
    color: #91A4BD !important;
    -webkit-text-fill-color: #91A4BD !important;
    opacity: 1 !important;
}

.stApp [data-testid="stChatMessage"] {
    color: #EAF2FF !important;
}

.stApp [data-testid="stChatMessage"] p,
.stApp [data-testid="stChatMessage"] div {
    color: #EAF2FF;
}
""",
    unsafe_allow_html=True,
)