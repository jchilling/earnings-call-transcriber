#!/usr/bin/env python3
"""Aggregate per-quarter analysis JSONs into a single dashboard-ready file.

Reads all *_analysis.json files from data/processed/{ticker}/ and builds:
1. dashboard.json — time series of financials, sentiment, guidance for charts
2. guidance_tracking.json — guidance vs actual comparison table

Usage:
    python scripts/build_dashboard_data.py --ticker 3105
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Quarter ordering for sorting
QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def quarter_sort_key(period: str) -> tuple[int, int]:
    """Sort key for 'Q3 2020' format strings."""
    parts = period.split()
    if len(parts) != 2:
        return (0, 0)
    q = QUARTER_ORDER.get(parts[0], 0)
    try:
        y = int(parts[1])
    except ValueError:
        y = 0
    return (y, q)


def load_analyses(ticker: str) -> list[dict]:
    """Load all analysis JSONs for a ticker, sorted chronologically."""
    analysis_dir = PROCESSED_DIR / ticker
    files = sorted(analysis_dir.glob(f"{ticker}_*_analysis.json"))

    quarters = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        meta = data.get("_metadata", {})
        quarter = meta.get("quarter", "")
        year = meta.get("year", 0)

        if not quarter or not year:
            # Try parsing from filename: 3105_Q3_2020_analysis.json
            parts = f.stem.replace("_analysis", "").split("_")
            if len(parts) >= 3:
                quarter = parts[1]
                try:
                    year = int(parts[2])
                except ValueError:
                    continue

        period = f"{quarter} {year}"
        quarters.append({
            "period": period,
            "quarter": quarter,
            "year": year,
            "data": data,
            "filename": f.stem,
        })

    quarters.sort(key=lambda x: quarter_sort_key(x["period"]))
    return quarters


def load_preprocessed(ticker: str, stem: str) -> dict | None:
    """Load the preprocessed transcript JSON for additional metadata."""
    base_stem = stem.replace("_analysis", "")
    path = PROCESSED_DIR / ticker / f"{base_stem}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def build_dashboard(ticker: str, company_name: str, company_name_zh: str) -> dict:
    """Build the full dashboard JSON."""
    analyses = load_analyses(ticker)

    if not analyses:
        print(f"No analysis files found for {ticker}")
        sys.exit(1)

    print(f"Loaded {len(analyses)} quarterly analyses")

    # Build quarterly time series
    quarters_data = []
    for entry in analyses:
        data = entry["data"]
        fin = data.get("financials", {})
        guidance = data.get("guidance", {})
        sentiment = data.get("sentiment", {})

        # Load preprocessed for duration
        preprocessed = load_preprocessed(ticker, entry["filename"])
        duration = preprocessed.get("duration_seconds") if preprocessed else None
        source = preprocessed.get("source", "unknown") if preprocessed else "unknown"

        q = {
            "period": entry["period"],
            "quarter": entry["quarter"],
            "year": entry["year"],

            # Financials
            "revenue_ntd_m": fin.get("revenue_ntd_m"),
            "revenue_qoq_pct": fin.get("revenue_qoq_pct"),
            "revenue_yoy_pct": fin.get("revenue_yoy_pct"),
            "gross_margin_pct": fin.get("gross_margin_pct"),
            "operating_margin_pct": fin.get("operating_margin_pct"),
            "net_margin_pct": fin.get("net_margin_pct"),
            "eps_ntd": fin.get("eps_ntd"),
            "utilization_pct": fin.get("utilization_pct"),
            "capex_ntd_m": fin.get("capex_ntd_m"),
            "depreciation_ntd_m": fin.get("depreciation_ntd_m"),

            # Guidance
            "guidance_next_q_revenue": guidance.get("next_q_revenue"),
            "guidance_next_q_gross_margin": guidance.get("next_q_gross_margin"),
            "guidance_full_year": guidance.get("full_year_outlook"),

            # Sentiment
            "sentiment_score": sentiment.get("score"),
            "sentiment_rationale": sentiment.get("rationale"),
            "sentiment_tone_shift": sentiment.get("tone_shift"),

            # Summary
            "highlights": data.get("highlights", []),
            "key_topics": data.get("key_topics", []),
            "thesis_impact": data.get("thesis_impact"),
            "margin_drivers": data.get("margin_analysis", {}).get("drivers", []),
            "margin_headwinds": data.get("margin_analysis", {}).get("headwinds", []),

            # Meta
            "duration_seconds": duration,
            "source": source,
        }
        quarters_data.append(q)

    # Build guidance vs actual tracking
    guidance_tracking = build_guidance_tracking(quarters_data)

    dashboard = {
        "company": {
            "ticker": ticker,
            "name": company_name,
            "name_zh": company_name_zh,
            "exchange": "TPEx",
            "sector": "Semiconductors — III-V Compound",
            "sub_sector": "GaAs Foundry",
            "description": "World's largest pure-play GaAs/compound semiconductor foundry. "
                         "Key products: PA (power amplifiers) for 5G, Wi-Fi, infrastructure; "
                         "3D sensing VCSELs; defense/aerospace components.",
        },
        "quarters": quarters_data,
        "guidance_tracking": guidance_tracking,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "total_quarters": len(quarters_data),
    }

    return dashboard


def build_guidance_tracking(quarters: list[dict]) -> list[dict]:
    """Compare each quarter's guidance to the next quarter's actual results.

    Enables 'did management deliver?' analysis.
    """
    tracking = []

    for i, q in enumerate(quarters):
        if not q.get("guidance_next_q_revenue"):
            continue

        # Find next quarter
        next_q = quarters[i + 1] if i + 1 < len(quarters) else None

        entry = {
            "guidance_period": q["period"],
            "guidance_revenue": q.get("guidance_next_q_revenue"),
            "guidance_gross_margin": q.get("guidance_next_q_gross_margin"),
            "actual_period": next_q["period"] if next_q else None,
            "actual_revenue_ntd_m": next_q.get("revenue_ntd_m") if next_q else None,
            "actual_revenue_qoq_pct": next_q.get("revenue_qoq_pct") if next_q else None,
            "actual_gross_margin_pct": next_q.get("gross_margin_pct") if next_q else None,
        }
        tracking.append(entry)

    return tracking


def main():
    parser = argparse.ArgumentParser(description="Build dashboard data from analyses")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--company-name", default="WIN Semiconductors")
    parser.add_argument("--company-name-zh", default="穩懋半導體")
    args = parser.parse_args()

    print(f"=== Building dashboard data for {args.ticker} ===\n")

    dashboard = build_dashboard(args.ticker, args.company_name, args.company_name_zh)

    out_path = PROCESSED_DIR / args.ticker / "dashboard.json"
    with open(out_path, "w") as f:
        json.dump(dashboard, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {out_path}")
    print(f"  Quarters: {dashboard['total_quarters']}")
    print(f"  Guidance tracking entries: {len(dashboard['guidance_tracking'])}")

    # Summary stats
    revenues = [q["revenue_ntd_m"] for q in dashboard["quarters"] if q.get("revenue_ntd_m")]
    margins = [q["gross_margin_pct"] for q in dashboard["quarters"] if q.get("gross_margin_pct")]
    sentiments = [q["sentiment_score"] for q in dashboard["quarters"] if q.get("sentiment_score")]

    if revenues:
        print(f"  Revenue range: NTD {min(revenues):,.0f}M — {max(revenues):,.0f}M")
    if margins:
        print(f"  Gross margin range: {min(margins):.1f}% — {max(margins):.1f}%")
    if sentiments:
        print(f"  Sentiment range: {min(sentiments)} — {max(sentiments)}")


if __name__ == "__main__":
    main()
