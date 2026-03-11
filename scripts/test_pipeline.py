"""Test script: run the full transcription pipeline on TSMC Q3 2025 audio."""

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription import TranscriptionResult
from src.transcription.pipeline import transcribe_with_diarization


def _format_ts(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _write_txt(result: TranscriptionResult, path: Path) -> None:
    """Write a clean, human-readable transcript to a TXT file."""
    lines: list[str] = []

    # Header
    duration_m, duration_s = divmod(int(result.duration_seconds), 60)
    lines.append("TSMC Q3 2025 Earnings Call Transcript")
    lines.append(f"Duration: {duration_m}:{duration_s:02d}")

    # Speaker list
    speaker_names = []
    for info in result.speakers.values():
        if info.name:
            label = info.name
            if info.title:
                label += f" ({info.title})"
            speaker_names.append(label)
    if speaker_names:
        lines.append(f"Speakers: {', '.join(speaker_names)}")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Segments — merge consecutive segments from the same speaker
    prev_speaker: str | None = None
    for seg in result.segments:
        speaker = seg.speaker_name or seg.speaker_id or "Unknown"
        ts = _format_ts(seg.start_time)

        if speaker != prev_speaker:
            if prev_speaker is not None:
                lines.append("")  # blank line between speakers
            lines.append(f"[{ts}] {speaker}:")
            lines.append(seg.text)
        else:
            # Same speaker continues — just append text
            lines.append(seg.text)

        prev_speaker = speaker

    lines.append("")  # trailing newline
    path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    audio_path = Path(__file__).parent.parent / "tsmc_q3_2025.mp3"

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return

    print(f"Starting pipeline on: {audio_path}")
    print(f"File size: {audio_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Use 'base' model for fast validation; swap to 'large-v3' for production quality
    model = sys.argv[1] if len(sys.argv) > 1 else "base"
    print(f"Using model: {model}")

    result = await transcribe_with_diarization(
        audio_path,
        language="en",
        initial_prompt="TSMC Taiwan Semiconductor earnings call. Jeff Su, C.C. Wei, Wendell Huang.",
        model_name=model,
    )

    # Build JSON-serializable output
    output = {
        "language": result.language,
        "model_used": result.model_used,
        "duration_seconds": result.duration_seconds,
        "total_segments": len(result.segments),
        "speakers": {
            sid: asdict(info) for sid, info in result.speakers.items()
        },
        "segments": [
            {
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "speaker_id": seg.speaker_id,
                "speaker_name": seg.speaker_name,
                "text": seg.text,
                "confidence": seg.confidence,
            }
            for seg in result.segments
        ],
    }

    # Write JSON output
    json_path = Path(__file__).parent.parent / "tsmc_q3_2025_transcript.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Write clean TXT output
    txt_path = Path(__file__).parent.parent / "tsmc_q3_2025_transcript.txt"
    _write_txt(result, txt_path)

    print(f"\nJSON written to: {json_path}")
    print(f"TXT  written to: {txt_path}")
    print(f"Duration: {result.duration_seconds:.0f}s")
    print(f"Segments: {len(result.segments)}")
    print(f"Speakers found: {len(result.speakers)}")

    for sid, info in result.speakers.items():
        name = info.name or "(unknown)"
        title = info.title or ""
        print(f"  {sid}: {name} — {title} ({info.segments_count} segments)")

    # Print first 10 segments as preview
    print("\n--- First 10 segments ---")
    for seg in result.segments[:10]:
        speaker = seg.speaker_name or seg.speaker_id or "?"
        print(f"[{seg.start_time:.1f}-{seg.end_time:.1f}] {speaker}: {seg.text[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
