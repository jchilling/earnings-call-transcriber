"""Isolated test for LLM speaker identification + reasoner correction.

Loads cached transcript JSON (from a previous pipeline run) and exercises:
1. Regex identification
2. DeepSeek-chat identification
3. Merge by name
4. DeepSeek-reasoner correction (fixes mid-block speaker handoffs)

No Whisper, no audio processing — instant iteration on prompt engineering.

Usage:
    poetry run python scripts/test_speaker_id.py                    # uses cached JSON
    poetry run python scripts/test_speaker_id.py my_transcript.json # custom file
    poetry run python scripts/test_speaker_id.py --correction-only  # skip ID, only run correction
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import SpeakerInfo, TranscriptSegment
from src.transcription.speaker_identification import extract_speaker_names
from src.transcription.speaker_identification_llm import (
    correct_speaker_assignments,
    enhance_speaker_names,
)


def _load_segments_from_json(
    json_path: Path,
    reset_names: bool = True,
) -> list[TranscriptSegment]:
    """Load transcript segments from a pipeline JSON output file."""
    with open(json_path) as f:
        data = json.load(f)

    segments = []
    for seg in data["segments"]:
        segments.append(
            TranscriptSegment(
                text=seg["text"],
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                speaker_id=seg.get("speaker_id"),
                speaker_name=None if reset_names else seg.get("speaker_name"),
                language=data.get("language"),
                confidence=seg.get("confidence", 0.0),
            )
        )
    return segments


def _load_speakers_from_json(json_path: Path) -> dict[str, SpeakerInfo]:
    """Load speaker info from a pipeline JSON output file."""
    with open(json_path) as f:
        data = json.load(f)

    speakers = {}
    for sid, info in data.get("speakers", {}).items():
        speakers[sid] = SpeakerInfo(
            id=sid,
            name=info.get("name"),
            title=info.get("title"),
            segments_count=info.get("segments_count", 0),
        )
    return speakers


def _merge_speakers_by_name(
    segments: list[TranscriptSegment],
    speakers: dict[str, SpeakerInfo],
) -> tuple[list[TranscriptSegment], dict[str, SpeakerInfo]]:
    """Merge speaker IDs that were identified as the same person."""
    from collections import Counter

    name_to_sids: dict[str, list[str]] = {}
    for sid, info in speakers.items():
        if info.name:
            name_to_sids.setdefault(info.name, []).append(sid)

    merge_map: dict[str, str] = {}
    for name, sids in name_to_sids.items():
        if len(sids) <= 1:
            continue
        canonical = max(sids, key=lambda s: speakers[s].segments_count)
        for sid in sids:
            if sid != canonical:
                merge_map[sid] = canonical

    if not merge_map:
        return segments, speakers

    merged_segments = [
        TranscriptSegment(
            text=seg.text,
            start_time=seg.start_time,
            end_time=seg.end_time,
            speaker_id=merge_map.get(seg.speaker_id, seg.speaker_id),
            speaker_name=seg.speaker_name,
            language=seg.language,
            confidence=seg.confidence,
        )
        for seg in segments
    ]

    counts = Counter(seg.speaker_id for seg in merged_segments if seg.speaker_id)
    merged_speakers: dict[str, SpeakerInfo] = {}
    for sid, count in counts.items():
        info = speakers.get(sid, SpeakerInfo(id=sid))
        merged_speakers[sid] = SpeakerInfo(
            id=sid, name=info.name, title=info.title, segments_count=count,
        )

    return merged_segments, merged_speakers


def _apply_speaker_names(
    segments: list[TranscriptSegment],
    speakers: dict[str, SpeakerInfo],
) -> list[TranscriptSegment]:
    """Apply resolved speaker names to transcript segments."""
    return [
        TranscriptSegment(
            text=seg.text,
            start_time=seg.start_time,
            end_time=seg.end_time,
            speaker_id=seg.speaker_id,
            speaker_name=speakers.get(seg.speaker_id, SpeakerInfo(id="")).name
            if seg.speaker_id else None,
            language=seg.language,
            confidence=seg.confidence,
        )
        for seg in segments
    ]


def _format_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _write_txt(
    segments: list[TranscriptSegment],
    speakers: dict[str, SpeakerInfo],
    duration: float,
    path: Path,
) -> None:
    """Write clean human-readable transcript."""
    lines: list[str] = []
    duration_m, duration_s = divmod(int(duration), 60)
    lines.append("TSMC Q3 2025 Earnings Call Transcript")
    lines.append(f"Duration: {duration_m}:{duration_s:02d}")

    speaker_names = []
    for info in speakers.values():
        if info.name:
            label = info.name
            if info.title:
                label += f" ({info.title})"
            speaker_names.append(label)
    if speaker_names:
        lines.append(f"Speakers: {', '.join(speaker_names)}")

    lines.extend(["", "---", ""])

    prev_speaker: str | None = None
    for seg in segments:
        speaker = seg.speaker_name or seg.speaker_id or "Unknown"
        ts = _format_ts(seg.start_time)
        if speaker != prev_speaker:
            if prev_speaker is not None:
                lines.append("")
            lines.append(f"[{ts}] {speaker}:")
            lines.append(seg.text)
        else:
            lines.append(seg.text)
        prev_speaker = speaker

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_speakers(speakers: dict[str, SpeakerInfo]) -> None:
    for sid, info in sorted(speakers.items(), key=lambda x: -x[1].segments_count):
        name = info.name or "(unknown)"
        title = f" — {info.title}" if info.title else ""
        print(f"  {name}{title} ({info.segments_count} segments)")


async def main() -> None:
    correction_only = "--correction-only" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    default_path = Path(__file__).parent.parent / "tsmc_q3_2025_transcript.json"
    json_path = Path(args[0]) if args else default_path

    if not json_path.exists():
        print(f"Transcript not found: {json_path}")
        print("Run the full pipeline first: poetry run python scripts/test_pipeline.py base")
        return

    print(f"Loading transcript from: {json_path}")

    if correction_only:
        # Load with existing names — only run correction pass
        segments = _load_segments_from_json(json_path, reset_names=False)
        speakers = _load_speakers_from_json(json_path)
        print(f"Loaded {len(segments)} segments, {len(speakers)} speakers")
        print("\n--- Before correction ---")
        _print_speakers(speakers)
    else:
        # Full pipeline: regex → LLM ID → merge → correction
        segments = _load_segments_from_json(json_path, reset_names=True)
        print(f"Loaded {len(segments)} segments")

        # Step 1: Regex
        print("\n--- Regex identification ---")
        speakers = extract_speaker_names(segments)
        regex_identified = sum(1 for s in speakers.values() if s.name)
        print(f"Regex identified: {regex_identified}/{len(speakers)}")

        # Step 2: LLM identification (deepseek-chat)
        print("\n--- LLM identification (deepseek-chat) ---")
        segments, speakers = await enhance_speaker_names(segments, speakers)
        llm_identified = sum(1 for s in speakers.values() if s.name)
        print(f"LLM identified: {llm_identified}/{len(speakers)}")

        # Step 3: Merge by name
        print("\n--- Merge by name ---")
        segments, speakers = _merge_speakers_by_name(segments, speakers)
        print(f"After merge: {len(speakers)} unique speakers")
        _print_speakers(speakers)

        # Step 4: Apply names to segments
        segments = _apply_speaker_names(segments, speakers)

    # Step 5: Reasoner correction (deepseek-reasoner)
    print("\n--- Reasoner correction (deepseek-reasoner) ---")
    segments, speakers = await correct_speaker_assignments(segments, speakers)

    print(f"\n--- Final result: {len(speakers)} unique speakers ---")
    _print_speakers(speakers)

    # Get duration from JSON
    with open(json_path) as f:
        duration = json.load(f).get("duration_seconds", 0)

    # Write outputs
    txt_path = json_path.with_suffix(".corrected.txt")
    _write_txt(segments, speakers, duration, txt_path)
    print(f"\nTXT written to: {txt_path}")

    # Write JSON for inspection
    output_path = json_path.with_suffix(".corrected.json")
    output = {
        "duration_seconds": duration,
        "speakers": {
            sid: {"name": info.name, "title": info.title, "segments_count": info.segments_count}
            for sid, info in speakers.items()
        },
        "segments": [
            {
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "speaker_id": seg.speaker_id,
                "speaker_name": seg.speaker_name,
                "text": seg.text,
            }
            for seg in segments
        ],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"JSON written to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
