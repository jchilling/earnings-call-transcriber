#!/usr/bin/env python3
"""Re-run vocal analysis on existing transcript JSONs (CPU only).

Loads JSON → finds matching audio → runs librosa/parselmouth vocal analysis
→ updates JSON with per-segment vocal_metrics, audio_quality, vocal_profiles.

Usage:
    python scripts/reprocess_vocal.py --dir data/transcripts/ --audio-dir /path/to/audio/3105_win_semiconductors/
    python scripts/reprocess_vocal.py --file data/transcripts/3105_Q1_2014.json --audio-dir /path/to/audio/3105_win_semiconductors/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modal_app.vocal_analysis import analyze_vocals, compute_audio_quality, build_speaker_profiles


def _find_audio(stem: str, audio_dir: Path) -> Path | None:
    """Find matching MP3 for a transcript stem."""
    mp3 = audio_dir / f"{stem}.mp3"
    if mp3.exists():
        return mp3
    # Try case-insensitive or partial match
    for f in audio_dir.glob("*.mp3"):
        if f.stem.lower() == stem.lower():
            return f
    return None


def process_one(json_path: Path, audio_dir: Path) -> bool:
    """Run vocal analysis on one JSON file. Returns True on success."""
    data = json.loads(json_path.read_text())

    # Check if already has vocal metrics
    has_vocal = any(seg.get("vocal_metrics") for seg in data["segments"])
    if has_vocal:
        print(f"  Skipping (already has vocal metrics)")
        return True

    # Find audio file
    stem = json_path.stem
    audio_path = _find_audio(stem, audio_dir)
    if not audio_path:
        print(f"  ERROR: No audio found for {stem} in {audio_dir}")
        return False

    # Convert to WAV
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"  ERROR: ffmpeg failed: {result.stderr.decode()[:200]}")
            return False

        # Build segments list in the format vocal_analysis expects
        segments_for_vocal = [
            {
                "text": seg["text"],
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
            }
            for seg in data["segments"]
        ]

        # Build diarization segments from speaker_id info
        diar_segments = [
            {
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "speaker_id": seg.get("speaker_id", "UNKNOWN"),
            }
            for seg in data["segments"]
        ]

        t0 = time.time()
        vocal_metrics = analyze_vocals(wav_path, segments_for_vocal)
        audio_quality = compute_audio_quality(wav_path)
        vocal_profiles = build_speaker_profiles(segments_for_vocal, vocal_metrics, diar_segments)
        elapsed = time.time() - t0

    # Update JSON
    for i, seg in enumerate(data["segments"]):
        seg["vocal_metrics"] = vocal_metrics[i] if i < len(vocal_metrics) else None

    data["audio_quality"] = audio_quality
    data["vocal_profiles"] = vocal_profiles

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    n_with_metrics = sum(1 for m in vocal_metrics if m)
    print(f"  Done: {n_with_metrics}/{len(data['segments'])} segments with metrics, "
          f"{len(vocal_profiles)} speaker profiles ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Re-run vocal analysis on existing transcripts")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dir", help="Directory of transcript JSONs")
    source.add_argument("--file", help="Single JSON file")
    parser.add_argument("--audio-dir", required=True, help="Directory containing source MP3 files")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"ERROR: Audio directory not found: {audio_dir}")
        sys.exit(1)

    if args.file:
        json_files = [Path(args.file)]
    else:
        json_files = sorted(Path(args.dir).glob("*.json"))

    if not json_files:
        print("No JSON files found")
        sys.exit(1)

    print(f"Vocal analysis reprocessing: {len(json_files)} files\n")
    t0 = time.time()
    success = 0

    for i, jf in enumerate(json_files):
        print(f"[{i+1}/{len(json_files)}] {jf.name}")
        try:
            if process_one(jf, audio_dir):
                success += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\nDone: {success}/{len(json_files)} files ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
