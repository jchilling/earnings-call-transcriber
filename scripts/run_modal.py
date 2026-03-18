#!/usr/bin/env python3
"""CLI entry point for Modal GPU transcription pipeline.

Usage (single file):
    python scripts/run_modal.py --local-path data/audio/3105_win_semiconductors/3105_Q1_2020.mp3
    python scripts/run_modal.py --hf-path alphamemo/3105/3105_Q4_20260211.mp3

Usage (batch — all MP3s in a folder):
    python scripts/run_modal.py --local-dir data/audio/3105_win_semiconductors/

GPU does: Whisper + pyannote diarization + speaker embeddings.
Local does: vocal analysis (librosa/parselmouth) + DeepSeek speaker ID + save JSON/TXT.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transcription import SpeakerInfo, TranscriptionResult, TranscriptSegment


def _deserialize_segments(gpu_result: dict) -> list[TranscriptSegment]:
    """Convert GPU pipeline output to TranscriptSegment list."""
    whisper_segments = gpu_result["whisper"]["segments"]
    diar_segments = gpu_result.get("diarization", [])

    segments = []
    for i, wseg in enumerate(whisper_segments):
        speaker_id = None
        if diar_segments:
            best_overlap = 0.0
            seg_mid = (wseg["start_time"] + wseg["end_time"]) / 2
            for diar in diar_segments:
                overlap_start = max(wseg["start_time"], diar["start_time"])
                overlap_end = min(wseg["end_time"], diar["end_time"])
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    speaker_id = diar["speaker_id"]
            if speaker_id is None:
                speaker_id = min(
                    diar_segments,
                    key=lambda d: abs((d["start_time"] + d["end_time"]) / 2 - seg_mid),
                )["speaker_id"]

        segments.append(TranscriptSegment(
            text=wseg["text"],
            start_time=wseg["start_time"],
            end_time=wseg["end_time"],
            speaker_id=speaker_id,
            language=wseg.get("language"),
            confidence=wseg.get("confidence", 0.0),
        ))

    return segments


def _run_local_vocal_analysis(audio_path: Path, gpu_result: dict) -> tuple[list[dict], dict, list[dict]]:
    """Run vocal analysis locally on CPU. Returns (vocal_metrics, audio_quality, vocal_profiles)."""
    from src.modal_app.vocal_analysis import analyze_vocals, compute_audio_quality, build_speaker_profiles

    # Convert to WAV locally for analysis
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
            capture_output=True, check=True,
        )

        vocal_metrics = analyze_vocals(wav_path, gpu_result["whisper"]["segments"])
        audio_quality = compute_audio_quality(wav_path)
        vocal_profiles = build_speaker_profiles(
            gpu_result["whisper"]["segments"],
            vocal_metrics,
            gpu_result.get("diarization", []),
        )

    return vocal_metrics, audio_quality, vocal_profiles


async def _run_speaker_id(
    segments: list[TranscriptSegment],
    speakers: dict[str, SpeakerInfo],
) -> tuple[list[TranscriptSegment], dict[str, SpeakerInfo]]:
    """Run DeepSeek two-pass speaker identification."""
    from src.transcription.speaker_identification import extract_speaker_names
    from src.transcription.speaker_identification_llm import (
        correct_speaker_assignments,
        enhance_speaker_names,
    )
    from src.transcription.pipeline import _merge_speakers_by_name, _apply_speaker_names

    speakers = extract_speaker_names(segments)
    segments, speakers = await enhance_speaker_names(segments, speakers)
    segments, speakers = _merge_speakers_by_name(segments, speakers)
    segments = _apply_speaker_names(segments, speakers)
    segments, speakers = await correct_speaker_assignments(segments, speakers)
    return segments, speakers


def _format_timestamp(seconds: float) -> str:
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _save_outputs(
    segments: list[TranscriptSegment],
    speakers: dict[str, SpeakerInfo],
    gpu_result: dict,
    vocal_metrics: list[dict] | None,
    audio_quality: dict | None,
    vocal_profiles: list[dict] | None,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    """Save JSON and TXT outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}.json"
    json_data = {
        "language": gpu_result["whisper"].get("language", ""),
        "model_used": "large-v3",
        "duration_seconds": gpu_result["whisper"].get("duration", 0.0),
        "speakers": {
            sid: {"id": info.id, "name": info.name, "title": info.title,
                  "segments_count": info.segments_count}
            for sid, info in speakers.items()
        },
        "segments": [
            {
                "text": seg.text,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "speaker_id": seg.speaker_id,
                "speaker_name": seg.speaker_name,
                "language": seg.language,
                "confidence": seg.confidence,
                "vocal_metrics": vocal_metrics[i] if vocal_metrics and i < len(vocal_metrics) else None,
            }
            for i, seg in enumerate(segments)
        ],
        "vocal_profiles": vocal_profiles or [],
        "speaker_embeddings": {
            k: (v[:5] + ["..."]) if len(v) > 5 else v
            for k, v in gpu_result.get("speaker_embeddings", {}).items()
        },
        "audio_quality": audio_quality,
        "timing": gpu_result.get("timing", {}),
    }
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2))

    txt_path = output_dir / f"{stem}.txt"
    lines = []
    for seg in segments:
        speaker = seg.speaker_name or seg.speaker_id or "UNKNOWN"
        ts = _format_timestamp(seg.start_time)
        lines.append(f"[{ts}] {speaker}: {seg.text}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    return json_path, txt_path


def _get_hf_token() -> str:
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        try:
            from src.config import settings
            hf_token = settings.hf_token
        except Exception:
            pass
    return hf_token


async def process_single(
    audio_path: Path | None,
    hf_path: str | None,
    hf_token: str,
    language: str | None,
    skip_speaker_id: bool,
    skip_vocal: bool,
    output_dir: Path,
) -> Path:
    """Process a single audio file end-to-end. Returns JSON output path."""
    import modal

    t0 = time.time()

    # --- GPU phase ---
    if hf_path:
        process_fn = modal.Function.from_name("earnings-gpu", "process_earnings_call")
        stem = Path(hf_path).stem
        print(f"  GPU dispatching (HF): {hf_path}")
        gpu_result = await process_fn.remote.aio(
            hf_path=hf_path, hf_token=hf_token, language=language,
        )
    else:
        process_fn = modal.Function.from_name("earnings-gpu", "process_earnings_call_bytes")
        stem = audio_path.stem
        audio_bytes = audio_path.read_bytes()
        size_mb = len(audio_bytes) / 1024 / 1024
        print(f"  GPU dispatching (local): {audio_path.name} ({size_mb:.1f}MB)")
        gpu_result = await process_fn.remote.aio(
            audio_bytes=audio_bytes, filename=audio_path.name,
            hf_token=hf_token, language=language,
        )

    t_gpu = time.time() - t0
    print(f"  GPU done: {gpu_result['timing']} | "
          f"{len(gpu_result['whisper']['segments'])} segs, "
          f"{len(gpu_result.get('diarization', []))} diar turns")

    # --- Local vocal analysis ---
    vocal_metrics = None
    audio_quality = None
    vocal_profiles = None
    if not skip_vocal and audio_path:
        print(f"  Local vocal analysis...")
        t_vocal_start = time.time()
        vocal_metrics, audio_quality, vocal_profiles = _run_local_vocal_analysis(audio_path, gpu_result)
        t_vocal = time.time() - t_vocal_start
        print(f"  Vocal analysis done: {t_vocal:.1f}s")

    # --- Deserialize + speaker ID ---
    segments = _deserialize_segments(gpu_result)
    speaker_ids = {seg.speaker_id for seg in segments if seg.speaker_id}
    speakers = {
        sid: SpeakerInfo(id=sid, segments_count=sum(1 for s in segments if s.speaker_id == sid))
        for sid in speaker_ids
    }

    if not skip_speaker_id:
        print(f"  DeepSeek speaker ID...")
        segments, speakers = await _run_speaker_id(segments, speakers)
        n_id = sum(1 for s in speakers.values() if s.name)
        print(f"  Identified: {n_id}/{len(speakers)} speakers")

    # --- Save ---
    json_path, txt_path = _save_outputs(
        segments, speakers, gpu_result, vocal_metrics, audio_quality, vocal_profiles,
        output_dir, stem,
    )
    t_total = time.time() - t0
    print(f"  Done: {stem} ({t_total:.1f}s total) → {json_path}")
    return json_path


async def main():
    parser = argparse.ArgumentParser(description="Modal GPU transcription pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-path", help="HF dataset path")
    source.add_argument("--local-path", help="Local audio file")
    source.add_argument("--local-dir", help="Local directory — process all MP3s")
    parser.add_argument("--language", default=None)
    parser.add_argument("--skip-speaker-id", action="store_true")
    parser.add_argument("--skip-vocal", action="store_true", help="Skip local vocal analysis")
    parser.add_argument("--output-dir", default="data/transcripts")
    args = parser.parse_args()

    hf_token = _get_hf_token()
    if not hf_token:
        print("ERROR: HF_TOKEN not set (needed for pyannote).")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    if args.local_dir:
        # Batch mode
        mp3s = sorted(Path(args.local_dir).glob("*.mp3"))
        if not mp3s:
            print(f"No MP3 files found in {args.local_dir}")
            sys.exit(1)

        # Skip already-processed files
        existing = {p.stem for p in output_dir.glob("*.json")}
        todo = [p for p in mp3s if p.stem not in existing]

        print(f"Batch: {len(mp3s)} MP3s, {len(existing)} already done, {len(todo)} to process")
        t_batch_start = time.time()

        for i, mp3_path in enumerate(todo):
            print(f"\n[{i+1}/{len(todo)}] {mp3_path.name}")
            try:
                await process_single(
                    audio_path=mp3_path, hf_path=None,
                    hf_token=hf_token, language=args.language,
                    skip_speaker_id=args.skip_speaker_id,
                    skip_vocal=args.skip_vocal,
                    output_dir=output_dir,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        t_batch = time.time() - t_batch_start
        done = len(todo) - len([1 for p in todo if p.stem not in {pp.stem for pp in output_dir.glob("*.json")}])
        print(f"\nBatch complete: {done}/{len(todo)} processed in {t_batch:.0f}s")

    else:
        # Single file mode
        audio_path = Path(args.local_path) if args.local_path else None
        if audio_path and not audio_path.exists():
            print(f"ERROR: File not found: {audio_path}")
            sys.exit(1)

        await process_single(
            audio_path=audio_path,
            hf_path=args.hf_path,
            hf_token=hf_token,
            language=args.language,
            skip_speaker_id=args.skip_speaker_id,
            skip_vocal=args.skip_vocal,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    asyncio.run(main())
