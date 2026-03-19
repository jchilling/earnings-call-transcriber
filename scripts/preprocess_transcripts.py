#!/usr/bin/env python3
"""Preprocessing pipeline for Whisper transcripts.

Steps:
1. Scrape official management names from Yahoo Finance TW
2. Dictionary-based error correction (company name + speaker names)
3. LLM post-correction via DeepSeek for remaining errors
4. Integrate AlphaMemo transcripts for recent quarters
5. Output corrected JSON to data/processed/{ticker}/

Usage:
    python scripts/preprocess_transcripts.py --ticker 3105
    python scripts/preprocess_transcripts.py --ticker 3105 --skip-llm   # dict correction only
    python scripts/preprocess_transcripts.py --ticker 3105 --dry-run    # preview changes
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
ALPHAMEMO_DIR = PROJECT_ROOT / "data" / "audio" / "alphamemo"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Company name correction (applies to all tickers)
# ---------------------------------------------------------------------------
COMPANY_NAME_FIXES: dict[str, dict[str, str]] = {
    "3105": {
        "文貌": "穩懋",
        "文贸": "穩懋",
        "文貿": "穩懋",
        "稳懋": "穩懋",
        "穏懋": "穩懋",
        "文貌半導體": "穩懋半導體",
        "文贸半导体": "穩懋半導體",
        "文貿半導體": "穩懋半導體",
    },
}

# ---------------------------------------------------------------------------
# Speaker name corrections per ticker
# Built from comparing Whisper output against official Yahoo Finance names.
# ---------------------------------------------------------------------------
SPEAKER_NAME_FIXES: dict[str, dict[str, str]] = {
    "3105": {
        # Spokesperson: 曾經洲 (Joe Tseng)
        "曾金舟": "曾經洲",
        "曾金周": "曾經洲",
        "曾经周": "曾經洲",
        "曾慶洲": "曾經洲",
        "曾庆周": "曾經洲",
        "曾庆洲": "曾經洲",
        "鄭敬周": "曾經洲",
        "鄭慶洲": "曾經洲",
        "鄭敬洲": "曾經洲",
        "鄭敬舟": "曾經洲",
        "鄭慶舟": "曾經洲",
        "郑敬周": "曾經洲",
        "郑庆洲": "曾經洲",
        # CEO: 陳國樺 (Kyle Chen)
        "陳國華": "陳國樺",
        "陈国华": "陳國樺",
        "陈国桦": "陳國樺",
        # GM Management: 陳舜平 (Steve Chen)
        "陳順平": "陳舜平",
        "陈顺平": "陳舜平",
        "陈舜平": "陳舜平",
    },
}

# Title canonicalization
TITLE_FIXES: dict[str, dict[str, str]] = {
    "3105": {
        "曾經洲": "財務處協理兼發言人",
        "陳國樺": "總經理",
        "陳舜平": "管理總處總經理",
        "陳進財": "董事長",
    },
}


# ---------------------------------------------------------------------------
# Step 1: Official name lookup from Yahoo Finance TW
# ---------------------------------------------------------------------------
def lookup_official_names(ticker: str, cache_dir: Path | None = None) -> dict[str, Any]:
    """Scrape Yahoo Finance TW profile for official management names.

    Returns dict with keys: chairman, ceo, spokesperson, deputy_spokesperson.
    Caches to data/processed/{ticker}/officials.json.
    """
    if cache_dir is None:
        cache_dir = PROCESSED_DIR / ticker
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "officials.json"

    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        print(f"  [cache] Loaded official names from {cache_file}")
        return cached

    # Try TWO (OTC) first, then TW (TWSE)
    suffixes = [".TWO", ".TW"]
    html = None
    used_url = None

    for suffix in suffixes:
        url = f"https://tw.stock.yahoo.com/quote/{ticker}{suffix}/profile"
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=15,
                           headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200 and "董事長" in resp.text:
                html = resp.text
                used_url = url
                break
        except httpx.HTTPError:
            continue

    if html is None:
        print(f"  [warn] Could not fetch Yahoo Finance profile for {ticker}")
        return {}

    print(f"  [web] Fetched profile from {used_url}")
    soup = BeautifulSoup(html, "html.parser")

    officials: dict[str, dict[str, str]] = {}

    # Parse the structured profile data.
    # Yahoo Finance TW uses a pattern like: 發言人\n曾經洲\n英文簡稱\n...
    # We extract by finding the role label followed by the name (2-4 Chinese chars).
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    role_map = {
        "董事長": "chairman",
        "總經理": "ceo",
        "發言人": "spokesperson",
        "代理發言人": "deputy_spokesperson",
    }

    for i, line in enumerate(lines):
        for zh_title, en_key in role_map.items():
            if line == zh_title and i + 1 < len(lines):
                # The name is on the next line — should be 2-4 Chinese chars
                candidate = lines[i + 1]
                if re.match(r"^[\u4e00-\u9fff]{2,4}$", candidate):
                    officials[en_key] = {
                        "name_zh": candidate,
                        "title_zh": zh_title,
                    }

    with open(cache_file, "w") as f:
        json.dump(officials, f, ensure_ascii=False, indent=2)
    print(f"  [cache] Saved official names to {cache_file}")

    return officials


# ---------------------------------------------------------------------------
# Step 2: Dictionary-based correction
# ---------------------------------------------------------------------------
def apply_dict_corrections(text: str, ticker: str) -> str:
    """Apply company name and speaker name corrections via find-and-replace."""
    company_fixes = COMPANY_NAME_FIXES.get(ticker, {})
    speaker_fixes = SPEAKER_NAME_FIXES.get(ticker, {})

    all_fixes = {**company_fixes, **speaker_fixes}

    # Sort by length descending so longer matches take priority
    for wrong, correct in sorted(all_fixes.items(), key=lambda x: -len(x[0])):
        text = text.replace(wrong, correct)

    return text


def _fix_speaker_name(name: str | None, speaker_fixes: dict[str, str]) -> tuple[str | None, str | None]:
    """Fix a speaker name, handling formats like '曾金周 (Jill Tseng)'.

    Returns (fixed_name, original_name) or (None, None) if no fix needed.
    """
    if not name:
        return None, None

    # Try exact match first
    if name in speaker_fixes:
        return speaker_fixes[name], name

    # Try matching the Chinese portion (before parentheses or space+latin)
    for wrong, correct in speaker_fixes.items():
        if wrong in name:
            return name.replace(wrong, correct), name

    return None, None


def correct_transcript_json(data: dict, ticker: str) -> dict:
    """Apply dictionary corrections to a transcript JSON structure."""
    speaker_fixes = SPEAKER_NAME_FIXES.get(ticker, {})
    title_fixes = TITLE_FIXES.get(ticker, {})

    # Fix speaker names in the speakers dict
    new_speakers = {}
    name_remap = {}  # old_name -> new_name for segment updates

    for spk_id, spk_info in data.get("speakers", {}).items():
        old_name = spk_info.get("name")
        fixed, orig = _fix_speaker_name(old_name, speaker_fixes)
        if fixed:
            name_remap[old_name] = fixed
            spk_info["name"] = fixed
            spk_info["name_original_whisper"] = orig

        # Fix titles
        current_name = spk_info.get("name")
        if current_name:
            # Check title by the canonical Chinese name (strip parenthetical English)
            zh_name = re.split(r"\s*[\(（]", current_name)[0]
            if zh_name in title_fixes:
                spk_info["title"] = title_fixes[zh_name]

        new_speakers[spk_id] = spk_info

    data["speakers"] = new_speakers

    # Fix segments
    for seg in data.get("segments", []):
        old_name = seg.get("speaker_name")
        if old_name:
            # Check remap first (exact match from speakers pass)
            if old_name in name_remap:
                seg["speaker_name"] = name_remap[old_name]
            else:
                fixed, _ = _fix_speaker_name(old_name, speaker_fixes)
                if fixed:
                    seg["speaker_name"] = fixed

        # Fix text content
        if seg.get("text"):
            seg["text"] = apply_dict_corrections(seg["text"], ticker)

    return data


# ---------------------------------------------------------------------------
# Step 3: LLM post-correction (DeepSeek)
# ---------------------------------------------------------------------------
def get_deepseek_client() -> OpenAI:
    """Create DeepSeek client using OpenAI-compatible API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def llm_correct_transcript(text: str, ticker: str, company_name: str,
                           client: OpenAI | None = None) -> str:
    """Use DeepSeek to fix remaining Whisper transcription errors."""
    if client is None:
        client = get_deepseek_client()

    system_prompt = f"""You are correcting a Whisper-transcribed earnings call for {company_name} (ticker: {ticker}).

Fix obvious transcription errors:
- Broken/fragmented sentences (merge fragments that clearly belong together)
- Wrong homophones (common in Mandarin ASR)
- Garbled technical/financial terms
- Inconsistent simplified/traditional Chinese (normalize to Traditional Chinese)

Do NOT change:
- Speaker labels or formatting
- Financial figures or numbers
- Meaning or intent of any statement
- Timestamp markers

Return the corrected transcript text only, no explanations."""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
        max_tokens=16000,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Step 4: AlphaMemo integration
# ---------------------------------------------------------------------------
def load_alphamemo(ticker: str) -> dict[str, dict]:
    """Load AlphaMemo files for a ticker, keyed by quarter string (e.g., 'Q4_2025').

    Returns dict mapping quarter -> parsed AlphaMemo content.
    """
    alphamemo_dir = ALPHAMEMO_DIR / ticker
    if not alphamemo_dir.exists():
        return {}

    result = {}
    for f in sorted(alphamemo_dir.glob(f"{ticker}_*.json")):
        if f.suffix != ".json":
            continue

        try:
            with open(f) as fh:
                raw = json.load(fh)

            metadata = raw.get("metadata", {})
            audio_date = metadata.get("audio_date", "")

            # Content can be either:
            # - A JSON string (newer format, has speaker_name_map + memo + processed_transcript)
            # - A markdown string (older format, has ## Memo + ## Transcript sections)
            content_str = raw.get("content", "")
            if isinstance(content_str, str):
                try:
                    content = json.loads(content_str)
                    content["_format"] = "json"
                except (json.JSONDecodeError, ValueError):
                    # Markdown format — parse into structured dict
                    content = _parse_alphamemo_markdown(content_str)
                    content["_format"] = "markdown"
            else:
                content = content_str
                content["_format"] = "json"

            # Map audio_date to quarter
            # Convention: Q1 reported ~Apr, Q2 ~Jul, Q3 ~Oct, Q4 ~Feb next year
            quarter = _date_to_quarter(audio_date)
            if quarter:
                result[quarter] = {
                    "metadata": metadata,
                    "content": content,
                    "source_file": str(f),
                }
                print(f"  [alphamemo] Loaded {f.name} -> {quarter}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [warn] Failed to parse AlphaMemo {f.name}: {e}")

    return result


def _date_to_quarter(audio_date: str) -> str | None:
    """Map an earnings call date to its fiscal quarter.

    Taiwan earnings calls happen ~2 months after quarter end:
    Q1 (Jan-Mar) -> reported Apr-May
    Q2 (Apr-Jun) -> reported Jul-Aug
    Q3 (Jul-Sep) -> reported Oct-Nov
    Q4 (Oct-Dec) -> reported Jan-Mar next year
    """
    if not audio_date:
        return None

    try:
        year, month, _ = audio_date.split("-")
        year, month = int(year), int(month)
    except (ValueError, AttributeError):
        return None

    if month in (4, 5):
        return f"Q1_{year}"
    elif month in (7, 8):
        return f"Q2_{year}"
    elif month in (10, 11):
        return f"Q3_{year}"
    elif month in (1, 2, 3):
        return f"Q4_{year - 1}"

    return None


def _parse_alphamemo_markdown(md_text: str) -> dict:
    """Parse AlphaMemo markdown format into a structured dict.

    Markdown format has:
    - ## Memo section with ### subsections containing bullet points
    - ## Transcript section with raw transcript text
    """
    result: dict[str, Any] = {"speaker_name_map": {}, "memo": [], "processed_transcript": []}

    # Split into memo and transcript sections
    memo_match = re.search(r"## Memo\s*\n(.*?)(?=## Transcript|$)", md_text, re.DOTALL)
    transcript_match = re.search(r"## Transcript\s*\n(.*?)$", md_text, re.DOTALL)

    if memo_match:
        memo_text = memo_match.group(1)
        # Parse ### subsections
        sections = re.split(r"### (.+)\n", memo_text)
        # sections[0] is text before first ###, then alternating heading/content
        for i in range(1, len(sections), 2):
            heading = sections[i].strip()
            content = sections[i + 1].strip() if i + 1 < len(sections) else ""
            items = []
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    items.append({"text": line[2:].strip(), "translate": {}})
            if items:
                result["memo"].append({"heading": heading, "items": items})

    if transcript_match:
        transcript_text = transcript_match.group(1).strip()
        # Store as a single session with the raw text
        result["processed_transcript"] = [{
            "session": "Full",
            "segments": [{"start_sec": 0, "end_sec": 0, "text": transcript_text, "speaker": ""}],
        }]
        result["_raw_transcript"] = transcript_text

    return result


def build_transcript_from_alphamemo(am_data: dict, ticker: str) -> dict:
    """Build a preprocessed transcript JSON from AlphaMemo data."""
    content = am_data["content"]
    metadata = am_data["metadata"]
    fmt = content.get("_format", "json")

    memo = content.get("memo", [])

    # Apply speaker name corrections to AlphaMemo data too
    speaker_name_fixes = SPEAKER_NAME_FIXES.get(ticker, {})

    if fmt == "json":
        # Newer structured JSON format with speaker_name_map + processed_transcript
        speaker_map = content.get("speaker_name_map", {})
        # Fix names in the speaker map
        for spk_id, spk_info in speaker_map.items():
            name = spk_info.get("name", "")
            fixed, _ = _fix_speaker_name(name, speaker_name_fixes)
            if fixed:
                spk_info["name"] = fixed

        speakers = {}
        for spk_id, spk_info in speaker_map.items():
            speakers[spk_id] = {
                "id": spk_id,
                "name": spk_info.get("name"),
                "title": spk_info.get("title"),
                "title_en": spk_info.get("translate", {}).get("en", {}).get("title"),
                "name_en": spk_info.get("translate", {}).get("en", {}).get("name"),
            }

        segments = []
        for session in content.get("processed_transcript", []):
            session_name = session.get("session", "")
            for seg in session.get("segments", []):
                segments.append({
                    "text": seg.get("text", ""),
                    "text_en": seg.get("translate", {}).get("en", ""),
                    "start_time": seg.get("start_sec", 0),
                    "end_time": seg.get("end_sec", 0),
                    "speaker_id": seg.get("speaker"),
                    "speaker_name": speaker_map.get(seg.get("speaker"), {}).get("name"),
                    "session": session_name,
                })

        transcript_zh = apply_dict_corrections("\n".join(
            f"[{_fmt_time(s['start_time'])}] {s.get('speaker_name', 'Unknown')}: {s['text']}"
            for s in segments if s.get("text")
        ), ticker)
        transcript_en = "\n".join(
            f"[{_fmt_time(s['start_time'])}] {s.get('speaker_name', 'Unknown')}: {s.get('text_en', '')}"
            for s in segments if s.get("text_en")
        )

    else:
        # Older markdown format — transcript is raw text, no speaker segmentation
        speakers = {}
        segments = []
        raw_transcript = content.get("_raw_transcript", "")
        transcript_zh = apply_dict_corrections(raw_transcript, ticker)
        transcript_en = ""  # Will be translated in Phase 2

    return {
        "ticker": ticker,
        "language": metadata.get("language", "zh"),
        "duration_seconds": metadata.get("audio_length_ceil_sec"),
        "speakers": speakers,
        "transcript_zh": transcript_zh,
        "transcript_en": transcript_en,
        "segments": segments,
        "memo": memo,
        "source": "alphamemo",
        "source_file": am_data.get("source_file"),
        "deck_zh": metadata.get("deck_zh"),
        "deck_en": metadata.get("deck_en"),
    }


def collate_segments(segments: list[dict]) -> str:
    """Merge consecutive same-speaker segments into paragraph blocks.

    Transforms fragmented Whisper output:
        [00:00:00] Joe: 各位法人
        [00:00:02] Joe: 及媒體先進大家好
        [00:00:06] Joe: 歡迎參加

    Into collated AlphaMemo-style blocks:
        Joe — 財務處協理兼發言人

        (00:00:00) 各位法人及媒體先進大家好，歡迎參加...

    This produces cleaner text for translation (fewer tokens, more coherent)
    and matches the AlphaMemo format users are already familiar with.
    """
    if not segments:
        return ""

    blocks: list[str] = []
    current_speaker = None
    current_texts: list[str] = []
    current_start = 0.0
    prev_end = 0.0

    for seg in segments:
        speaker = seg.get("speaker_name") or seg.get("speaker_id") or "Unknown"
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        if speaker != current_speaker:
            # Flush previous block
            if current_texts:
                merged = "".join(current_texts)
                merged = re.sub(r"\s+", " ", merged).strip()
                blocks.append(f"\n{current_speaker}\n\n({_fmt_time(current_start)}) {merged}")

            current_speaker = speaker
            current_texts = [text]
            current_start = seg.get("start_time", 0)
        else:
            # Same speaker — check for long pause indicating new paragraph
            gap = seg.get("start_time", 0) - prev_end

            if gap > 3.0 and current_texts:
                # Long pause → new paragraph within same speaker
                merged = "".join(current_texts)
                merged = re.sub(r"\s+", " ", merged).strip()
                blocks.append(f"\n{current_speaker}\n\n({_fmt_time(current_start)}) {merged}")
                current_texts = [text]
                current_start = seg.get("start_time", 0)
            else:
                # Merge with connector
                last_char = current_texts[-1][-1] if current_texts[-1] else ""
                if last_char in "。！？!?.，,、；;：:":
                    current_texts.append(text)
                else:
                    current_texts.append("，" + text if text[0] not in "，。！？!?.,、" else text)

        prev_end = seg.get("end_time", seg.get("start_time", 0))

    # Flush last block
    if current_texts:
        merged = "".join(current_texts)
        merged = re.sub(r"\s+", " ", merged).strip()
        blocks.append(f"\n{current_speaker}\n\n({_fmt_time(current_start)}) {merged}")

    return "\n".join(blocks).strip()


def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------
def process_transcript(json_path: Path, ticker: str, alphamemo_data: dict,
                       skip_llm: bool = False, dry_run: bool = False,
                       ds_client: OpenAI | None = None) -> Path | None:
    """Process a single transcript file through the full pipeline.

    Returns the output path, or None if skipped.
    """
    fname = json_path.stem  # e.g., 3105_Q3_2020
    parts = fname.split("_")
    if len(parts) != 3:
        print(f"  [skip] Unexpected filename format: {fname}")
        return None

    _, quarter, year = parts
    quarter_key = f"{quarter}_{year}"

    out_dir = PROCESSED_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fname}.json"

    if out_path.exists() and not dry_run:
        print(f"  [skip] Already processed: {out_path.name}")
        return out_path

    # Check if AlphaMemo data is available for this quarter
    if quarter_key in alphamemo_data:
        print(f"  [alphamemo] Using AlphaMemo for {quarter_key}")
        result = build_transcript_from_alphamemo(alphamemo_data[quarter_key], ticker)
        result["quarter"] = quarter
        result["year"] = int(year)
    else:
        # Load Whisper transcript
        print(f"  [whisper] Processing {json_path.name}")
        with open(json_path) as f:
            data = json.load(f)

        # Step 2: Dictionary correction on JSON structure
        data = correct_transcript_json(data, ticker)

        # Collate consecutive same-speaker segments into paragraph blocks
        transcript_zh = collate_segments(data.get("segments", []))

        # Apply dict corrections to full text too
        transcript_zh = apply_dict_corrections(transcript_zh, ticker)

        # Step 3: LLM post-correction
        if not skip_llm and ds_client:
            print(f"    [llm] Running DeepSeek correction...")
            try:
                transcript_zh = llm_correct_transcript(
                    transcript_zh, ticker, "穩懋半導體 WIN Semiconductors",
                    client=ds_client,
                )
                source = "whisper+deepseek"
            except Exception as e:
                print(f"    [warn] LLM correction failed: {e}")
                source = "whisper+dict"
        else:
            source = "whisper+dict"

        result = {
            "ticker": ticker,
            "quarter": quarter,
            "year": int(year),
            "language": data.get("language", "zh"),
            "duration_seconds": data.get("duration_seconds"),
            "speakers": data.get("speakers", {}),
            "transcript_zh": transcript_zh,
            "segments": data.get("segments", []),
            "source": source,
        }

    if dry_run:
        print(f"    [dry-run] Would write to {out_path}")
        # Show sample corrections
        sample = result.get("transcript_zh", "")[:500]
        print(f"    [preview] {sample}...")
        return None

    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  [done] Wrote {out_path.name} ({result['source']})")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess Whisper transcripts")
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g., 3105)")
    parser.add_argument("--skip-llm", action="store_true",
                       help="Skip LLM post-correction (dict only)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview changes without writing files")
    parser.add_argument("--force", action="store_true",
                       help="Re-process even if output exists")
    parser.add_argument("--quarters", nargs="*",
                       help="Only process specific quarters (e.g., Q3_2020 Q1_2025)")
    args = parser.parse_args()

    ticker = args.ticker
    print(f"=== Preprocessing transcripts for {ticker} ===\n")

    # Step 1: Lookup official names
    print("[Step 1] Looking up official management names...")
    officials = lookup_official_names(ticker)
    if officials:
        for role, info in officials.items():
            print(f"  {role}: {info.get('name_zh', '?')} ({info.get('title_zh', '?')})")
    print()

    # Load AlphaMemo data
    print("[Step 2] Loading AlphaMemo data...")
    alphamemo_data = load_alphamemo(ticker)
    print(f"  Found {len(alphamemo_data)} quarters with AlphaMemo data")
    print()

    # Find transcript files
    print("[Step 3] Finding transcript files...")
    transcript_files = sorted(TRANSCRIPTS_DIR.glob(f"{ticker}_*.json"))
    if not transcript_files:
        print(f"  No transcript files found for {ticker} in {TRANSCRIPTS_DIR}")
        sys.exit(1)
    print(f"  Found {len(transcript_files)} JSON transcript files")

    # Filter by quarters if specified
    if args.quarters:
        quarter_set = set(args.quarters)
        transcript_files = [
            f for f in transcript_files
            if f"{'_'.join(f.stem.split('_')[1:])}" in quarter_set
        ]
        print(f"  Filtered to {len(transcript_files)} files for quarters: {args.quarters}")
    print()

    # Initialize DeepSeek client if needed
    ds_client = None
    if not args.skip_llm and not args.dry_run:
        try:
            ds_client = get_deepseek_client()
            print("[Step 4] DeepSeek client initialized")
        except ValueError as e:
            print(f"[Step 4] {e} — falling back to dict-only correction")
            args.skip_llm = True
    print()

    # Process each transcript
    print("[Processing transcripts]")
    processed = 0
    skipped = 0
    errors = 0

    for json_path in transcript_files:
        try:
            # Remove existing output if --force
            if args.force:
                out_path = PROCESSED_DIR / ticker / f"{json_path.stem}.json"
                if out_path.exists():
                    out_path.unlink()

            result = process_transcript(
                json_path, ticker, alphamemo_data,
                skip_llm=args.skip_llm, dry_run=args.dry_run,
                ds_client=ds_client,
            )
            if result:
                processed += 1
            else:
                skipped += 1

            # Rate limit for LLM calls
            if not args.skip_llm and ds_client and result:
                time.sleep(1)

        except Exception as e:
            print(f"  [error] {json_path.name}: {e}")
            errors += 1

    print(f"\n=== Done ===")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
