"""WIN Semiconductors Earnings Call Dashboard — MVP

Streamlit app showing:
1. Company overview + latest quarter snapshot
2. Financial trends (revenue, margins, EPS, utilization)
3. Sentiment over time
4. Guidance vs actual tracking
5. Per-quarter transcript browser

Run: streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WIN Semi Earnings Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve data directory — works whether run from project root or dashboard/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not (_PROJECT_ROOT / "data").exists():
    _PROJECT_ROOT = Path.cwd()
DATA_DIR = _PROJECT_ROOT / "data" / "processed" / "3105"
DASHBOARD_FILE = DATA_DIR / "dashboard.json"


@st.cache_data
def load_dashboard() -> dict:
    with open(DASHBOARD_FILE) as f:
        return json.load(f)


@st.cache_data
def load_transcript(ticker: str, quarter: str, year: int) -> dict | None:
    """Load the preprocessed transcript + translation for a quarter."""
    fname = f"{ticker}_{quarter}_{year}.json"
    path = DATA_DIR / fname
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)

    # Load translation if available
    translation_path = DATA_DIR / f"{ticker}_{quarter}_{year}_translation.txt"
    if translation_path.exists():
        with open(translation_path) as f:
            data["transcript_en"] = f.read()

    return data


def build_dataframe(quarters: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(quarters)
    # Ensure numeric columns
    numeric_cols = [
        "revenue_ntd_m", "revenue_qoq_pct", "revenue_yoy_pct",
        "gross_margin_pct", "operating_margin_pct", "net_margin_pct",
        "eps_ntd", "utilization_pct", "capex_ntd_m", "depreciation_ntd_m",
        "sentiment_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    data = load_dashboard()
    company = data["company"]
    df = build_dataframe(data["quarters"])

    # Sidebar
    st.sidebar.title(f"{company['name']}")
    st.sidebar.caption(f"{company['name_zh']} ({company['ticker']})")
    st.sidebar.markdown(f"**Exchange:** {company['exchange']}")
    st.sidebar.markdown(f"**Sector:** {company['sector']}")
    st.sidebar.markdown(f"---")
    st.sidebar.markdown(f"📊 {data['total_quarters']} quarters of data")
    st.sidebar.markdown(f"🕐 Generated: {data.get('generated_at', 'N/A')[:10]}")

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Financial Trends", "Guidance Tracker", "Transcript Browser"],
    )

    if page == "Overview":
        render_overview(company, df)
    elif page == "Financial Trends":
        render_trends(df)
    elif page == "Guidance Tracker":
        render_guidance(data)
    elif page == "Transcript Browser":
        render_transcripts(company, df)


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def render_overview(company: dict, df: pd.DataFrame):
    st.title(f"{company['name']} ({company['ticker']})")
    st.caption(company["description"])

    # Latest quarter metrics
    latest = df.iloc[-1] if len(df) > 0 else None
    if latest is not None:
        st.subheader(f"Latest: {latest['period']}")
        cols = st.columns(5)
        cols[0].metric("Revenue", f"NTD {latest.get('revenue_ntd_m', 'N/A'):,.0f}M"
                       if pd.notna(latest.get("revenue_ntd_m")) else "N/A",
                       f"{latest.get('revenue_qoq_pct', '')}% QoQ"
                       if pd.notna(latest.get("revenue_qoq_pct")) else None)
        cols[1].metric("Gross Margin", f"{latest.get('gross_margin_pct', 'N/A'):.1f}%"
                       if pd.notna(latest.get("gross_margin_pct")) else "N/A")
        cols[2].metric("Operating Margin", f"{latest.get('operating_margin_pct', 'N/A'):.1f}%"
                       if pd.notna(latest.get("operating_margin_pct")) else "N/A")
        cols[3].metric("EPS", f"NTD {latest.get('eps_ntd', 'N/A'):.2f}"
                       if pd.notna(latest.get("eps_ntd")) else "N/A")
        cols[4].metric("Sentiment", f"{int(latest.get('sentiment_score', 0))}/10"
                       if pd.notna(latest.get("sentiment_score")) else "N/A")

        # Highlights
        if latest.get("highlights"):
            st.subheader("Key Highlights")
            for h in latest["highlights"]:
                st.markdown(f"- {h}")

        # Thesis impact
        if latest.get("thesis_impact"):
            st.subheader("Thesis Impact")
            st.info(latest["thesis_impact"])

    # Quick trend sparklines
    st.subheader("Historical Snapshot")
    col1, col2 = st.columns(2)

    with col1:
        rev_df = df[df["revenue_ntd_m"].notna()]
        if not rev_df.empty:
            fig = px.bar(rev_df, x="period", y="revenue_ntd_m",
                        title="Quarterly Revenue (NTD M)",
                        labels={"revenue_ntd_m": "Revenue (NTD M)", "period": ""})
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sent_df = df[df["sentiment_score"].notna()]
        if not sent_df.empty:
            fig = px.line(sent_df, x="period", y="sentiment_score",
                         title="Management Sentiment Score (1-10)",
                         labels={"sentiment_score": "Score", "period": ""},
                         markers=True)
            fig.update_layout(height=300, yaxis_range=[0, 10])
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Financial Trends
# ---------------------------------------------------------------------------
def render_trends(df: pd.DataFrame):
    st.title("Financial Trends")

    # Revenue chart
    rev_df = df[df["revenue_ntd_m"].notna()]
    if not rev_df.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=rev_df["period"], y=rev_df["revenue_ntd_m"],
                   name="Revenue (NTD M)", marker_color="#4A90D9"),
            secondary_y=False,
        )
        if rev_df["revenue_yoy_pct"].notna().any():
            fig.add_trace(
                go.Scatter(x=rev_df["period"], y=rev_df["revenue_yoy_pct"],
                          name="YoY Growth %", mode="lines+markers",
                          line=dict(color="#E74C3C", width=2)),
                secondary_y=True,
            )
        fig.update_layout(title="Revenue & YoY Growth", height=400)
        fig.update_yaxes(title_text="Revenue (NTD M)", secondary_y=False)
        fig.update_yaxes(title_text="YoY %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # Margins chart
    margin_cols = ["gross_margin_pct", "operating_margin_pct", "net_margin_pct"]
    margin_df = df[df[margin_cols].notna().any(axis=1)]
    if not margin_df.empty:
        fig = go.Figure()
        colors = {"gross_margin_pct": "#27AE60", "operating_margin_pct": "#F39C12", "net_margin_pct": "#8E44AD"}
        names = {"gross_margin_pct": "Gross Margin", "operating_margin_pct": "Operating Margin", "net_margin_pct": "Net Margin"}
        for col in margin_cols:
            valid = margin_df[margin_df[col].notna()]
            if not valid.empty:
                fig.add_trace(go.Scatter(
                    x=valid["period"], y=valid[col],
                    mode="lines+markers", name=names[col],
                    line=dict(color=colors[col], width=2),
                ))
        fig.update_layout(title="Margin Trends (%)", height=400,
                         yaxis_title="Margin %")
        st.plotly_chart(fig, use_container_width=True)

    # EPS + Utilization
    col1, col2 = st.columns(2)
    with col1:
        eps_df = df[df["eps_ntd"].notna()]
        if not eps_df.empty:
            fig = px.bar(eps_df, x="period", y="eps_ntd",
                        title="EPS (NTD)",
                        color="eps_ntd",
                        color_continuous_scale=["#E74C3C", "#F39C12", "#27AE60"])
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        util_df = df[df["utilization_pct"].notna()]
        if not util_df.empty:
            fig = px.line(util_df, x="period", y="utilization_pct",
                         title="Capacity Utilization (%)",
                         markers=True)
            fig.update_layout(height=350, yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

    # Sentiment trend
    sent_df = df[df["sentiment_score"].notna()]
    if not sent_df.empty:
        fig = go.Figure()
        colors = ["#E74C3C" if s <= 4 else "#F39C12" if s <= 6 else "#27AE60"
                  for s in sent_df["sentiment_score"]]
        fig.add_trace(go.Bar(
            x=sent_df["period"], y=sent_df["sentiment_score"],
            marker_color=colors,
            text=sent_df["sentiment_score"],
            textposition="outside",
        ))
        fig.update_layout(title="Management Sentiment Score (1-10)", height=350,
                         yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Guidance Tracker
# ---------------------------------------------------------------------------
def render_guidance(data: dict):
    st.title("Guidance vs Actual")
    st.caption("Did management deliver on their guidance? Compare what they said vs what happened.")

    tracking = data.get("guidance_tracking", [])
    if not tracking:
        st.warning("No guidance tracking data available.")
        return

    # Table
    rows = []
    for t in tracking:
        rows.append({
            "Guided For": t.get("actual_period", "N/A"),
            "Revenue Guidance": t.get("guidance_revenue", "—"),
            "Actual Revenue (NTD M)": f"{t['actual_revenue_ntd_m']:,.0f}" if t.get("actual_revenue_ntd_m") else "—",
            "Actual QoQ %": f"{t['actual_revenue_qoq_pct']:.1f}%" if t.get("actual_revenue_qoq_pct") is not None else "—",
            "GM Guidance": t.get("guidance_gross_margin", "—"),
            "Actual GM %": f"{t['actual_gross_margin_pct']:.1f}%" if t.get("actual_gross_margin_pct") is not None else "—",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Per-quarter detail
    st.subheader("Quarter Detail")
    for q in data["quarters"]:
        if q.get("guidance_next_q_revenue") or q.get("guidance_next_q_gross_margin"):
            with st.expander(f"{q['period']} — Guidance Given"):
                if q.get("guidance_next_q_revenue"):
                    st.markdown(f"**Next Q Revenue:** {q['guidance_next_q_revenue']}")
                if q.get("guidance_next_q_gross_margin"):
                    st.markdown(f"**Next Q Gross Margin:** {q['guidance_next_q_gross_margin']}")
                if q.get("guidance_full_year"):
                    st.markdown(f"**Full Year:** {q['guidance_full_year']}")


# ---------------------------------------------------------------------------
# Page: Transcript Browser
# ---------------------------------------------------------------------------
def render_transcripts(company: dict, df: pd.DataFrame):
    st.title("Transcript Browser")

    # Quarter selector
    periods = df["period"].tolist()
    selected = st.selectbox("Select Quarter", periods[::-1])  # Most recent first

    if not selected:
        return

    # Find row
    row = df[df["period"] == selected].iloc[0]
    quarter = row["quarter"]
    year = int(row["year"])

    # Load transcript
    transcript_data = load_transcript(company["ticker"], quarter, year)

    # Summary panel
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Key Metrics")
        metrics = {
            "Revenue": f"NTD {row['revenue_ntd_m']:,.0f}M" if pd.notna(row.get("revenue_ntd_m")) else "N/A",
            "Gross Margin": f"{row['gross_margin_pct']:.1f}%" if pd.notna(row.get("gross_margin_pct")) else "N/A",
            "Op Margin": f"{row['operating_margin_pct']:.1f}%" if pd.notna(row.get("operating_margin_pct")) else "N/A",
            "EPS": f"NTD {row['eps_ntd']:.2f}" if pd.notna(row.get("eps_ntd")) else "N/A",
            "Utilization": f"{row['utilization_pct']:.0f}%" if pd.notna(row.get("utilization_pct")) else "N/A",
            "Sentiment": f"{int(row['sentiment_score'])}/10" if pd.notna(row.get("sentiment_score")) else "N/A",
        }
        for k, v in metrics.items():
            st.markdown(f"**{k}:** {v}")

        if row.get("sentiment_rationale"):
            st.markdown("---")
            st.markdown(f"**Sentiment:** {row['sentiment_rationale']}")

    with col1:
        st.subheader("Highlights")
        if row.get("highlights"):
            for h in row["highlights"]:
                st.markdown(f"- {h}")
        else:
            st.caption("No highlights available")

        if row.get("thesis_impact"):
            st.markdown("---")
            st.markdown(f"**Thesis Impact:** {row['thesis_impact']}")

    # Key topics
    if row.get("key_topics"):
        st.subheader("Key Topics")
        for topic in row["key_topics"]:
            if isinstance(topic, dict):
                st.markdown(f"- **{topic.get('topic', '')}** — {topic.get('significance', '')}")

    # Margin analysis
    col1, col2 = st.columns(2)
    with col1:
        if row.get("margin_drivers"):
            st.subheader("Margin Drivers")
            for d in row["margin_drivers"]:
                st.markdown(f"- ✅ {d}")
    with col2:
        if row.get("margin_headwinds"):
            st.subheader("Margin Headwinds")
            for h in row["margin_headwinds"]:
                st.markdown(f"- ⚠️ {h}")

    # Full transcript
    st.markdown("---")
    st.subheader("Full Transcript")

    if transcript_data:
        tab1, tab2 = st.tabs(["English Translation", "Chinese Original"])

        with tab1:
            en = transcript_data.get("transcript_en", "")
            if en:
                st.text_area("", en, height=500, label_visibility="collapsed")
            else:
                st.caption("English translation not available for this quarter")

        with tab2:
            zh = transcript_data.get("transcript_zh", "")
            if zh:
                st.text_area("", zh, height=500, label_visibility="collapsed")
            else:
                st.caption("Chinese transcript not available")
    else:
        st.caption("Transcript data not found for this quarter")


import sys
print(f"[dashboard] Script executing, __name__={__name__}", file=sys.stderr, flush=True)
try:
    main()
    print("[dashboard] main() completed", file=sys.stderr, flush=True)
except Exception as e:
    import traceback
    print(f"[dashboard] ERROR: {e}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    st.error(f"Dashboard error: {e}")
    st.exception(e)
