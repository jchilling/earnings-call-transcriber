#!/usr/bin/env python3
"""Translation + structured analysis pipeline for preprocessed transcripts.

Architecture:
  Phase A: Translate ALL transcripts in parallel (DeepSeek, 5 concurrent)
  Phase B: Analyze ALL transcripts in parallel (Claude Sonnet, 3 concurrent)

This is ~4x faster than the old sequential approach because:
- Translations don't depend on each other → full parallelism
- Analysis can use Chinese directly → no need to wait for translation
- Both APIs handle concurrent requests fine

Usage:
    python scripts/analyze_transcripts.py --ticker 3105
    python scripts/analyze_transcripts.py --ticker 3105 --translate-only
    python scripts/analyze_transcripts.py --ticker 3105 --analyze-only
    python scripts/analyze_transcripts.py --ticker 3105 --quarters Q3_2025 Q4_2025
    python scripts/analyze_transcripts.py --ticker 3105 --concurrency 5
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import anthropic
from openai import AsyncOpenAI, OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Translation prompt (DeepSeek)
# ---------------------------------------------------------------------------
TRANSLATION_SYSTEM = """You are a professional translator specializing in Taiwan semiconductor industry earnings calls.

Translate the following Mandarin Chinese earnings call transcript into fluent English.

Rules:
- Preserve all financial figures exactly (NTD amounts, percentages, growth rates)
- Use standard finance terminology:
  毛利率 → gross margin, 營業淨利率 → operating margin, 稼動率 → utilization rate,
  營收 → revenue, 每股盈餘 → EPS, 法人說明會 → investor conference,
  產能利用率 → capacity utilization, 資本支出 → capex, 折舊 → depreciation
- Keep speaker labels (names) as-is — do not translate Chinese names
- Keep timestamp markers as-is
- Translate naturally — do not translate word by word. Restructure sentences for English fluency
- For company-specific terms, keep the Chinese in parentheses on first use: e.g., "utilization rate (稼動率)"
- Q&A section: translate analyst questions and management answers faithfully

Return the translated text only, no explanations."""

# ---------------------------------------------------------------------------
# Analysis prompt (Claude Sonnet)
# ---------------------------------------------------------------------------
ANALYSIS_SYSTEM = """You are a senior equity research analyst covering the Asia semiconductors sector, specializing in III-V compound semiconductor foundries. You have 15 years of experience writing investment notes for institutional investors.

Analyze the following earnings call transcript. Focus on:
1. Key changes to guidance — compare to prior quarter if mentioned
2. Margin drivers and headwinds — what moved gross margin, what's the outlook
3. Management tone on forward outlook — are they confident, cautious, hedging?
4. Anything that would change the investment thesis

CRITICAL RULES — read carefully:
- ONLY extract numbers that are EXPLICITLY stated in the transcript. If a figure is not mentioned, use null.
- Do NOT infer, calculate, or estimate numbers. If management says "revenue grew" without giving a percentage, revenue_qoq_pct must be null.
- For every financial figure you extract, you MUST include a "source" field with the verbatim quote from the transcript that contains that number.
- For sentiment, base your score ONLY on language actually used in the call. Quote specific phrases.
- If the transcript is low quality or too short to extract reliable data, say so in thesis_impact and use null liberally.

Return ONLY valid JSON matching the schema below — no markdown, no code fences, no explanation text.

JSON Schema:
{
  "highlights": ["string — 3-7 bullet points of the most important takeaways"],
  "financials": {
    "revenue_ntd_m": number or null,
    "revenue_qoq_pct": number or null,
    "revenue_yoy_pct": number or null,
    "gross_margin_pct": number or null,
    "operating_margin_pct": number or null,
    "net_margin_pct": number or null,
    "eps_ntd": number or null,
    "utilization_pct": number or null,
    "capex_ntd_m": number or null,
    "depreciation_ntd_m": number or null,
    "sources": {"field_name": "verbatim quote from transcript supporting this number"}
  },
  "guidance": {
    "next_q_revenue": "string or null",
    "next_q_gross_margin": "string or null",
    "full_year_outlook": "string or null",
    "sources": {"field_name": "verbatim quote from transcript"}
  },
  "margin_analysis": {
    "drivers": ["string — factors that helped margins"],
    "headwinds": ["string — factors that hurt margins"]
  },
  "sentiment": {
    "score": "integer 1-10",
    "rationale": "string — why this score, with specific quoted phrases from management",
    "tone_shift": "string — vs prior quarter if detectable, null if first quarter or not detectable",
    "key_phrases": ["exact quotes from management that informed the sentiment score"]
  },
  "key_topics": [
    {"topic": "string", "significance": "string"}
  ],
  "thesis_impact": "string — overall impact on investment thesis"
}"""

ANALYSIS_EXAMPLE = """\
Here is an example of the analysis I expect, based on WIN Semiconductors Q3 2025:

EXAMPLE INPUT (abbreviated):
"2025年第三季合併營收新台幣44.88億元，QOQ成長19%，YOY成長3%...
產能利用率從45%提升至60%...合併毛利率26.9%...EPS 2.52元...
第四季合併營收預計較上一季成長低個位數百分比..."

EXAMPLE OUTPUT:
{
  "highlights": [
    "Revenue NTD 4,488M, +19% QoQ / +3% YoY — beat prior guidance",
    "Gross margin 26.9% (+8.4pp QoQ) driven by utilization recovery to 60%",
    "Ended 3 consecutive quarters of operating losses — operating margin turned positive at 8.6%",
    "EPS NTD 2.52 vs -0.99 last quarter",
    "Q4 guidance: low single-digit QoQ revenue growth, gross margin in high-20s%"
  ],
  "financials": {
    "revenue_ntd_m": 4488,
    "revenue_qoq_pct": 19.0,
    "revenue_yoy_pct": 3.0,
    "gross_margin_pct": 26.9,
    "operating_margin_pct": 8.6,
    "net_margin_pct": 15.2,
    "eps_ntd": 2.52,
    "utilization_pct": 60,
    "capex_ntd_m": 933,
    "depreciation_ntd_m": 936,
    "sources": {
      "revenue_ntd_m": "第三季合併營收新台幣44.88億元",
      "revenue_qoq_pct": "QOQ成長19%",
      "revenue_yoy_pct": "YOY成長3%",
      "gross_margin_pct": "合併毛利率26.9%",
      "utilization_pct": "產能利用率從45%提升至60%",
      "eps_ntd": "EPS 2.52元"
    }
  },
  "guidance": {
    "next_q_revenue": "low single-digit % QoQ growth",
    "next_q_gross_margin": "high 20s %",
    "full_year_outlook": null,
    "sources": {
      "next_q_revenue": "第四季合併營收預計較上一季成長低個位數百分比"
    }
  },
  "margin_analysis": {
    "drivers": [
      "Utilization rate recovery from 45% to 60%",
      "Product mix improvement: Infrastructure grew double digits"
    ],
    "headwinds": [
      "Optical 3D sensing revenue declining",
      "China low-end cellular PA market lost to domestic competitors"
    ]
  },
  "sentiment": {
    "score": 7,
    "rationale": "Management notably more confident. First use of 'back on growth trajectory'. Cautious on China geopolitics.",
    "tone_shift": "positive — shifted from defensive to growth narrative",
    "key_phrases": ["回到成長軌道", "所有產品線都在成長", "對IDM整合帶來的外包機會持樂觀態度"]
  },
  "key_topics": [
    {"topic": "IDM consolidation", "significance": "Potential major catalyst — only qualified III-V foundry in iOS supply chain"},
    {"topic": "Infrastructure as growth pillar", "significance": "Revenue approaching cellular PA levels"},
    {"topic": "Wi-Fi 7 tailwind", "significance": "Full-year Wi-Fi revenue significantly exceeded prior year"}
  ],
  "thesis_impact": "Positive. Earnings inflection point has arrived. Infrastructure diversification reduces smartphone cyclicality."
}

---

"""


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------
def get_async_deepseek_client() -> AsyncOpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")
    return AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def get_claude_client() -> anthropic.AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return anthropic.AsyncAnthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Translation (async)
# ---------------------------------------------------------------------------
async def translate_one(
    filepath: Path, ds_client: AsyncOpenAI, sem: asyncio.Semaphore
) -> tuple[str, float]:
    """Translate a single preprocessed transcript. Returns (stem, elapsed_sec)."""
    stem = filepath.stem
    out_dir = filepath.parent
    translation_path = out_dir / f"{stem}_translation.txt"

    # Skip if already exists
    if translation_path.exists():
        print(f"  [skip] {stem} translation exists")
        return stem, 0.0

    with open(filepath) as f:
        data = json.load(f)

    # Use AlphaMemo English if available
    existing_en = data.get("transcript_en", "").strip()
    if existing_en and len(existing_en) > 100:
        with open(translation_path, "w") as f:
            f.write(existing_en)
        print(f"  [alphamemo] {stem} — used existing English")
        return stem, 0.0

    transcript_zh = data.get("transcript_zh", "")
    if not transcript_zh or len(transcript_zh) < 50:
        print(f"  [skip] {stem} — no Chinese text")
        return stem, 0.0

    # Truncate for DeepSeek context limit (~64k tokens, ~40k Chinese chars)
    max_chars = 40000
    if len(transcript_zh) > max_chars:
        transcript_zh = transcript_zh[:max_chars] + "\n\n[... truncated ...]"

    t0 = time.time()
    async with sem:
        try:
            response = await ds_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": TRANSLATION_SYSTEM},
                    {"role": "user", "content": transcript_zh},
                ],
                temperature=0.2,
                max_tokens=8192,
            )
            english = response.choices[0].message.content
            with open(translation_path, "w") as f:
                f.write(english)
            elapsed = time.time() - t0
            print(f"  [done] {stem} translated ({elapsed:.0f}s, {len(transcript_zh)} zh chars → {len(english)} en chars)")
            return stem, elapsed
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [error] {stem} translation failed ({elapsed:.0f}s): {e}")
            return stem, elapsed


# ---------------------------------------------------------------------------
# Analysis (async)
# ---------------------------------------------------------------------------
async def analyze_one(
    filepath: Path, ticker: str, company_name: str,
    claude_client: anthropic.AsyncAnthropic, sem: asyncio.Semaphore
) -> tuple[str, float]:
    """Analyze a single preprocessed transcript. Returns (stem, elapsed_sec)."""
    stem = filepath.stem
    out_dir = filepath.parent
    analysis_path = out_dir / f"{stem}_analysis.json"

    if analysis_path.exists():
        print(f"  [skip] {stem} analysis exists")
        return stem, 0.0

    with open(filepath) as f:
        data = json.load(f)

    quarter = data.get("quarter", "")
    year = data.get("year", 0)

    # Use English translation if available, otherwise Chinese
    translation_path = out_dir / f"{stem}_translation.txt"
    analysis_input = ""
    if translation_path.exists():
        with open(translation_path) as f:
            analysis_input = f.read()
    if not analysis_input or len(analysis_input) < 100:
        analysis_input = data.get("transcript_zh", "")

    if not analysis_input:
        print(f"  [skip] {stem} — no transcript text")
        return stem, 0.0

    user_prompt = (
        ANALYSIS_EXAMPLE
        + f"Now analyze this transcript for {company_name} ({ticker}) {quarter} {year}:\n\n"
        + analysis_input[:50000]
    )

    t0 = time.time()
    async with sem:
        try:
            response = await claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=ANALYSIS_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            try:
                result = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"  [warn] {stem} JSON parse failed: {e}")
                result = {"_raw_response": raw, "_parse_error": str(e)}

            result["_metadata"] = {
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "source": data.get("source", "unknown"),
                "analysis_model": "claude-sonnet-4-20250514",
            }

            with open(analysis_path, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            elapsed = time.time() - t0
            print(f"  [done] {stem} analyzed ({elapsed:.0f}s)")
            return stem, elapsed

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [error] {stem} analysis failed ({elapsed:.0f}s): {e}")
            return stem, elapsed


# ---------------------------------------------------------------------------
# Main async pipeline
# ---------------------------------------------------------------------------
async def run_pipeline(
    files: list[Path],
    ticker: str,
    company_name: str,
    do_translate: bool,
    do_analyze: bool,
    translate_concurrency: int,
    analyze_concurrency: int,
):
    """Run translation and analysis in parallel phases."""
    pipeline_start = time.time()

    # Phase A: Parallel translations
    if do_translate:
        print(f"\n{'='*60}")
        print(f"Phase A: Translating {len(files)} transcripts (concurrency={translate_concurrency})")
        print(f"{'='*60}")
        ds_client = get_async_deepseek_client()
        sem = asyncio.Semaphore(translate_concurrency)

        t0 = time.time()
        results = await asyncio.gather(
            *[translate_one(f, ds_client, sem) for f in files],
            return_exceptions=True,
        )
        translate_time = time.time() - t0

        completed = sum(1 for r in results if isinstance(r, tuple) and r[1] > 0)
        skipped = sum(1 for r in results if isinstance(r, tuple) and r[1] == 0)
        errors = sum(1 for r in results if isinstance(r, Exception))
        print(f"\nTranslation done: {completed} translated, {skipped} skipped, {errors} errors in {translate_time:.0f}s")
        if completed:
            print(f"  Avg {translate_time/max(completed,1):.0f}s wall / {translate_time/max(completed,1)/translate_concurrency:.0f}s effective per file")

    # Phase B: Parallel analysis
    if do_analyze:
        print(f"\n{'='*60}")
        print(f"Phase B: Analyzing {len(files)} transcripts (concurrency={analyze_concurrency})")
        print(f"{'='*60}")
        claude_client = get_claude_client()
        sem = asyncio.Semaphore(analyze_concurrency)

        t0 = time.time()
        results = await asyncio.gather(
            *[analyze_one(f, ticker, company_name, claude_client, sem) for f in files],
            return_exceptions=True,
        )
        analyze_time = time.time() - t0

        completed = sum(1 for r in results if isinstance(r, tuple) and r[1] > 0)
        skipped = sum(1 for r in results if isinstance(r, tuple) and r[1] == 0)
        errors = sum(1 for r in results if isinstance(r, Exception))
        print(f"\nAnalysis done: {completed} analyzed, {skipped} skipped, {errors} errors in {analyze_time:.0f}s")
        if completed:
            print(f"  Avg {analyze_time/max(completed,1):.0f}s wall / {analyze_time/max(completed,1)/analyze_concurrency:.0f}s effective per file")

    total = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete: {len(files)} files in {total/60:.1f} min ({total:.0f}s)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Translate + analyze preprocessed transcripts")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--translate-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--quarters", nargs="*",
                       help="Process specific quarters (e.g., Q3_2020 Q4_2025)")
    parser.add_argument("--force", action="store_true",
                       help="Re-process even if output exists")
    parser.add_argument("--company-name", default="WIN Semiconductors (穩懋半導體)")
    parser.add_argument("--concurrency", type=int, default=5,
                       help="Max concurrent API calls for translation (default: 5)")
    parser.add_argument("--analyze-concurrency", type=int, default=3,
                       help="Max concurrent API calls for analysis (default: 3)")
    args = parser.parse_args()

    ticker = args.ticker
    do_translate = not args.analyze_only
    do_analyze = not args.translate_only

    print(f"=== Analyzing transcripts for {ticker} ===")
    print(f"  Translate: {do_translate} (concurrency={args.concurrency})")
    print(f"  Analyze: {do_analyze} (concurrency={args.analyze_concurrency})")

    # Find preprocessed files
    processed_dir = PROCESSED_DIR / ticker
    files = sorted(processed_dir.glob(f"{ticker}_Q*.json"))
    files = [f for f in files if "_translation" not in f.name and "_analysis" not in f.name]

    if not files:
        print(f"No preprocessed files found in {processed_dir}")
        sys.exit(1)

    if args.quarters:
        quarter_set = set(args.quarters)
        files = [f for f in files if "_".join(f.stem.split("_")[1:]) in quarter_set]

    print(f"  Found {len(files)} preprocessed transcripts\n")

    # Delete existing outputs if --force
    if args.force:
        deleted = 0
        for f in files:
            stem = f.stem
            for suffix in ["_translation.txt", "_analysis.json"]:
                p = f.parent / f"{stem}{suffix}"
                if p.exists():
                    p.unlink()
                    deleted += 1
        if deleted:
            print(f"  Deleted {deleted} existing output files\n")

    # Verify API keys before starting
    if do_translate:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            print("[error] DEEPSEEK_API_KEY not set")
            sys.exit(1)
    if do_analyze:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[error] ANTHROPIC_API_KEY not set")
            sys.exit(1)

    asyncio.run(run_pipeline(
        files, ticker, args.company_name,
        do_translate, do_analyze,
        args.concurrency, args.analyze_concurrency,
    ))


if __name__ == "__main__":
    main()
