#!/usr/bin/env python3
"""Re-run DeepSeek speaker ID on existing transcript JSONs (no GPU needed).

Loads JSON → rebuilds TranscriptSegments → runs two-pass DeepSeek speaker ID
→ overwrites JSON + TXT with identified speakers.

Usage:
    python scripts/reprocess_speaker_id.py --dir data/transcripts/
    python scripts/reprocess_speaker_id.py --file data/transcripts/3105_Q1_2014.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transcription import SpeakerInfo, TranscriptSegment
from src.transcription.speaker_identification import extract_speaker_names
from src.transcription.speaker_identification_llm import (
    correct_speaker_assignments,
    enhance_speaker_names,
)
from src.transcription.pipeline import _merge_speakers_by_name, _apply_speaker_names


def _format_timestamp(seconds: float) -> str:
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _load_segments(data: dict) -> tuple[list[TranscriptSegment], dict[str, SpeakerInfo]]:
    """Rebuild TranscriptSegment list and speaker dict from JSON."""
    segments = []
    for seg in data["segments"]:
        segments.append(TranscriptSegment(
            text=seg["text"],
            start_time=seg["start_time"],
            end_time=seg["end_time"],
            speaker_id=seg.get("speaker_id"),
            speaker_name=None,  # reset — we're re-identifying
            language=seg.get("language"),
            confidence=seg.get("confidence", 0.0),
        ))

    speaker_ids = {seg.speaker_id for seg in segments if seg.speaker_id}
    speakers = {
        sid: SpeakerInfo(
            id=sid,
            segments_count=sum(1 for s in segments if s.speaker_id == sid),
        )
        for sid in speaker_ids
    }
    return segments, speakers


def _save(data: dict, segments: list[TranscriptSegment], speakers: dict[str, SpeakerInfo], json_path: Path) -> None:
    """Update JSON and TXT with new speaker info."""
    # Update segments in JSON
    for i, seg in enumerate(segments):
        data["segments"][i]["speaker_id"] = seg.speaker_id
        data["segments"][i]["speaker_name"] = seg.speaker_name

    # Update speakers dict
    data["speakers"] = {
        sid: {"id": info.id, "name": info.name, "title": info.title,
              "segments_count": info.segments_count}
        for sid, info in speakers.items()
    }

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # Rewrite TXT
    txt_path = json_path.with_suffix(".txt")
    lines = []
    for seg in segments:
        speaker = seg.speaker_name or seg.speaker_id or "UNKNOWN"
        ts = _format_timestamp(seg.start_time)
        lines.append(f"[{ts}] {speaker}: {seg.text}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


async def process_one(json_path: Path) -> bool:
    """Run speaker ID on one JSON file. Returns True if speakers were identified."""
    data = json.loads(json_path.read_text())
    segments, speakers = _load_segments(data)

    # Check if already has identified speakers
    existing_names = sum(1 for s in data.get("speakers", {}).values() if s.get("name"))
    if existing_names > 0:
        print(f"  Skipping (already has {existing_names} identified speakers)")
        return True

    # Pass 1: regex extraction
    speakers = extract_speaker_names(segments)

    # Pass 2: DeepSeek chat identification
    segments, speakers = await enhance_speaker_names(segments, speakers)

    # Merge duplicate names
    segments, speakers = _merge_speakers_by_name(segments, speakers)

    # Apply names to segments
    segments = _apply_speaker_names(segments, speakers)

    # Pass 3: DeepSeek reasoner correction
    segments, speakers = await correct_speaker_assignments(segments, speakers)

    n_id = sum(1 for s in speakers.values() if s.name)
    print(f"  Identified: {n_id}/{len(speakers)} speakers")

    if n_id > 0:
        names = [f"{info.name} ({info.segments_count})" for info in sorted(speakers.values(), key=lambda x: -x.segments_count) if info.name]
        print(f"  Speakers: {', '.join(names)}")

    _save(data, segments, speakers, json_path)
    return n_id > 0


async def main():
    parser = argparse.ArgumentParser(description="Re-run speaker ID on existing transcripts")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dir", help="Directory of transcript JSONs")
    source.add_argument("--file", help="Single JSON file")
    parser.add_argument("--force", action="store_true", help="Re-process even if speakers already identified")
    args = parser.parse_args()

    if args.file:
        json_files = [Path(args.file)]
    else:
        json_files = sorted(Path(args.dir).glob("*.json"))

    if not json_files:
        print("No JSON files found")
        sys.exit(1)

    print(f"Speaker ID reprocessing: {len(json_files)} files\n")
    t0 = time.time()
    success = 0

    for i, jf in enumerate(json_files):
        print(f"[{i+1}/{len(json_files)}] {jf.name}")
        try:
            if await process_one(jf):
                success += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\nDone: {success}/{len(json_files)} files with identified speakers ({elapsed:.0f}s)")


if __name__ == "__main__":
    asyncio.run(main())
