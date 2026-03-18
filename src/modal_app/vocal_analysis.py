"""Local CPU vocal analysis — runs after GPU results come back.

Computes per-segment vocal metrics (F0, RMS, jitter, shimmer, HNR)
and aggregated speaker profiles from the original audio file.
"""

from __future__ import annotations

import math
from pathlib import Path


def analyze_vocals(wav_path: Path, segments: list[dict]) -> list[dict]:
    """Compute vocal/prosodic metrics per segment using librosa + parselmouth."""
    import librosa
    import numpy as np
    import parselmouth

    y, sr = librosa.load(str(wav_path), sr=16000)
    snd = parselmouth.Sound(str(wav_path))

    results = []
    for seg in segments:
        start = seg["start_time"]
        end = seg["end_time"]
        duration = end - start
        if duration < 0.1:
            results.append({})
            continue

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        y_seg = y[start_sample:end_sample]

        if len(y_seg) < sr * 0.1:
            results.append({})
            continue

        # F0 (pitch)
        f0, voiced_flag, _ = librosa.pyin(y_seg, fmin=50, fmax=500, sr=sr)
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        # RMS energy
        rms = librosa.feature.rms(y=y_seg, frame_length=512, hop_length=256)[0]

        # Speech rate
        word_count = len(seg.get("text", "").split())
        speech_rate = word_count / duration if duration > 0 else 0.0

        # Pause detection (silence > 200ms)
        rms_threshold = np.mean(rms) * 0.1 if len(rms) > 0 else 0
        is_silent = rms < rms_threshold
        hop_sec = 256 / sr
        min_frames = int(0.2 / hop_sec)
        pause_count = 0
        pause_total = 0.0
        silent_run = 0
        for is_sil in is_silent:
            if is_sil:
                silent_run += 1
            else:
                if silent_run >= min_frames:
                    pause_count += 1
                    pause_total += silent_run * hop_sec
                silent_run = 0
        if silent_run >= min_frames:
            pause_count += 1
            pause_total += silent_run * hop_sec

        # Parselmouth: jitter, shimmer, HNR
        snd_seg = snd.extract_part(start, end, parselmouth.WindowShape.HANNING, 1.0, False)
        point_process = parselmouth.praat.call(
            snd_seg, "To PointProcess (periodic, cc)", 50, 500,
        )

        try:
            jitter = parselmouth.praat.call(
                point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3,
            )
        except Exception:
            jitter = 0.0

        try:
            shimmer = parselmouth.praat.call(
                [snd_seg, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6,
            )
        except Exception:
            shimmer = 0.0

        try:
            harmonicity = parselmouth.praat.call(
                snd_seg, "To Harmonicity (cc)", 0.01, 50, 0.1, 1.0,
            )
            hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        except Exception:
            hnr = 0.0

        metrics = {
            "f0_mean": round(float(np.mean(f0_valid)), 2) if len(f0_valid) > 0 else 0.0,
            "f0_std": round(float(np.std(f0_valid)), 2) if len(f0_valid) > 0 else 0.0,
            "rms_energy_mean": round(float(np.mean(rms)), 6) if len(rms) > 0 else 0.0,
            "rms_energy_std": round(float(np.std(rms)), 6) if len(rms) > 0 else 0.0,
            "speech_rate": round(speech_rate, 2),
            "pause_count": pause_count,
            "pause_total_sec": round(pause_total, 2),
            "jitter_local": round(float(jitter) * 100, 4) if jitter else 0.0,
            "shimmer_local": round(float(shimmer) * 100, 4) if shimmer else 0.0,
            "hnr": round(float(hnr), 2) if hnr else 0.0,
        }
        results.append(metrics)

    return results


def compute_audio_quality(wav_path: Path) -> dict:
    """Compute audio quality metrics."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(wav_path), sr=16000)
    duration = len(y) / sr

    rms = librosa.feature.rms(y=y)[0]
    signal_rms = np.mean(rms)
    noise_floor = np.percentile(rms, 5)
    snr_db = 20 * np.log10(signal_rms / (noise_floor + 1e-10)) if noise_floor > 0 else 0.0

    clipping_threshold = 0.99 * np.max(np.abs(y)) if len(y) > 0 else 1.0
    clipping_ratio = float(np.mean(np.abs(y) > clipping_threshold))

    return {
        "snr_db": round(float(snr_db), 2),
        "clipping_ratio": round(clipping_ratio, 6),
        "sample_rate": sr,
        "duration_seconds": round(duration, 2),
    }


def build_speaker_profiles(
    segments: list[dict], vocal_metrics: list[dict], diar_segments: list[dict],
) -> list[dict]:
    """Build per-speaker aggregated vocal profiles."""
    import numpy as np

    speaker_metrics: dict[str, list[dict]] = {}
    for seg, metrics in zip(segments, vocal_metrics):
        if not metrics:
            continue
        seg_mid = (seg["start_time"] + seg["end_time"]) / 2
        best_speaker = None
        best_overlap = 0.0
        for diar in diar_segments:
            overlap_start = max(seg["start_time"], diar["start_time"])
            overlap_end = min(seg["end_time"], diar["end_time"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar["speaker_id"]
        if best_speaker is None and diar_segments:
            best_speaker = min(
                diar_segments,
                key=lambda d: abs((d["start_time"] + d["end_time"]) / 2 - seg_mid),
            )["speaker_id"]
        if best_speaker:
            speaker_metrics.setdefault(best_speaker, []).append(metrics)

    profiles = []
    for speaker_id, metrics_list in speaker_metrics.items():
        if not metrics_list:
            continue

        def avg(key):
            vals = [m.get(key, 0) for m in metrics_list if not math.isnan(m.get(key, 0))]
            return float(np.mean(vals)) if vals else 0.0

        jitter = avg("jitter_local")
        shimmer = avg("shimmer_local")
        hnr = avg("hnr")
        f0_std = avg("f0_std")

        stress = min(10, (jitter * 2 + shimmer * 1.5 + f0_std / 20) * 2)
        confidence = min(10, max(0, hnr / 5 + (1 - jitter) * 3 + (1 - shimmer) * 2))

        profiles.append({
            "speaker_id": speaker_id,
            "speaker_name": None,
            "segment_count": len(metrics_list),
            "avg_f0": round(avg("f0_mean"), 2),
            "avg_rms_energy": round(avg("rms_energy_mean"), 6),
            "avg_speech_rate": round(avg("speech_rate"), 2),
            "avg_jitter": round(jitter, 4),
            "avg_shimmer": round(shimmer, 4),
            "avg_hnr": round(hnr, 2),
            "stress_score": round(stress, 2),
            "confidence_score": round(confidence, 2),
        })

    return profiles
