"""Modal App: GPU-only transcription + diarization pipeline.

GPU does: Whisper + pyannote diarization + speaker embeddings.
Vocal analysis (librosa/parselmouth) runs locally — no reason to pay GPU rates for CPU work.

Deploy:  modal deploy src/modal_app/app.py
"""

from __future__ import annotations

import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

app = modal.App("earnings-gpu")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "faster-whisper>=1.0.0",
        "pyannote.audio>=3.3.0",
        "huggingface-hub>=0.20.0,<1.0.0",
        "numpy<2",
        "soundfile>=0.12.0",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

volume = modal.Volume.from_name("earnings-model-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Helper functions (run inside Modal container on GPU)
# ---------------------------------------------------------------------------

def _download_from_hf(hf_path: str, hf_token: str, work_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(
        repo_id="jchilling/taiwan-earnings-calls",
        filename=hf_path,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(work_dir),
    )
    return Path(local_path)


def _preprocess_audio(input_path: Path, work_dir: Path) -> Path:
    wav_path = work_dir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
        capture_output=True, check=True,
    )
    return wav_path


def _run_whisper(wav_path: Path, language: str | None) -> dict:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "large-v3", device="cuda", compute_type="float16",
        download_root="/cache/huggingface/whisper",
    )
    segments_iter, info = model.transcribe(
        str(wav_path), language=language, beam_size=5,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
    )

    segments = []
    for seg in segments_iter:
        confidence = getattr(seg, "avg_logprob", None) or getattr(seg, "avg_log_prob", None) or 0.0
        segments.append({
            "text": seg.text.strip(),
            "start_time": round(seg.start, 3),
            "end_time": round(seg.end, 3),
            "language": info.language,
            "confidence": round(confidence, 4),
        })

    return {
        "segments": segments,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 2),
    }


def _run_diarization(wav_path: Path, hf_token: str) -> dict:
    import torch
    from pyannote.audio import Pipeline

    _original_load = torch.load
    torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
            cache_dir="/cache/huggingface/pyannote",
        )
    finally:
        torch.load = _original_load

    pipeline.to(torch.device("cuda"))
    diarization = pipeline(str(wav_path))

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start_time": round(turn.start, 3),
            "end_time": round(turn.end, 3),
            "speaker_id": speaker,
        })

    embeddings = _extract_embeddings(wav_path, diarization, hf_token)
    return {"segments": segments, "embeddings": embeddings}


def _extract_embeddings(wav_path: Path, diarization: object, hf_token: str) -> dict[str, list[float]]:
    import torch
    import torchaudio
    from pyannote.audio import Inference, Model

    _original_load = torch.load
    torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})
    try:
        model = Model.from_pretrained(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            use_auth_token=hf_token,
            cache_dir="/cache/huggingface/pyannote",
        )
    finally:
        torch.load = _original_load

    inference = Inference(model, window="whole")
    inference.to(torch.device("cuda"))

    waveform, sample_rate = torchaudio.load(str(wav_path))
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    speaker_segments: dict[str, list[tuple[float, float]]] = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.setdefault(speaker, []).append((turn.start, turn.end))

    embeddings: dict[str, list[float]] = {}
    for speaker, segs in speaker_segments.items():
        import numpy as np
        segs_sorted = sorted(segs, key=lambda s: s[1] - s[0], reverse=True)[:10]
        seg_embeddings = []
        for start, end in segs_sorted:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            if end_sample - start_sample < sample_rate:
                continue
            chunk = waveform[:, start_sample:end_sample]
            emb = inference({"waveform": chunk, "sample_rate": sample_rate})
            seg_embeddings.append(emb)

        if seg_embeddings:
            avg_emb = np.mean(seg_embeddings, axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
            embeddings[speaker] = avg_emb.tolist()

    return embeddings


def _run_gpu_pipeline(audio_path: Path, hf_token: str, language: str | None) -> dict:
    """GPU-only pipeline: preprocess → whisper + pyannote in parallel → return."""
    import time
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        print("Preprocessing audio...")
        wav_path = _preprocess_audio(audio_path, work_dir)
        t_preprocess = time.time() - t0

        print("Running Whisper + Pyannote in parallel...")
        with ThreadPoolExecutor(max_workers=2) as pool:
            whisper_future = pool.submit(_run_whisper, wav_path, language)
            diar_future = pool.submit(_run_diarization, wav_path, hf_token)
            whisper_result = whisper_future.result()
            diar_result = diar_future.result()

        t_gpu = time.time() - t0
        print(f"  GPU done: {t_gpu - t_preprocess:.1f}s "
              f"({len(whisper_result['segments'])} segments, "
              f"{len(diar_result['segments'])} diarization turns, "
              f"{len(diar_result['embeddings'])} embeddings)")

    t_total = time.time() - t0
    volume.commit()

    return {
        "whisper": whisper_result,
        "diarization": diar_result["segments"],
        "speaker_embeddings": diar_result["embeddings"],
        "timing": {
            "gpu_s": round(t_gpu - t_preprocess, 1),
            "total_s": round(t_total, 1),
        },
    }


# ---------------------------------------------------------------------------
# Modal entry points
# ---------------------------------------------------------------------------

@app.function(gpu="A10G", timeout=1800, image=image, volumes={"/cache": volume}, retries=1)
def process_earnings_call(hf_path: str, hf_token: str, language: str | None = None) -> dict:
    """GPU pipeline: download from HF → whisper + pyannote → return."""
    import time
    t0 = time.time()
    work_dir = Path(tempfile.mkdtemp())
    try:
        print(f"Downloading {hf_path} from HuggingFace...")
        mp3_path = _download_from_hf(hf_path, hf_token, work_dir)
        t_download = time.time() - t0
        print(f"  Download: {t_download:.1f}s")
        result = _run_gpu_pipeline(mp3_path, hf_token, language)
        result["timing"]["download_s"] = round(t_download, 1)
        return result
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


@app.function(gpu="A10G", timeout=1800, image=image, volumes={"/cache": volume}, retries=1)
def process_earnings_call_bytes(audio_bytes: bytes, filename: str, hf_token: str, language: str | None = None) -> dict:
    """GPU pipeline: receive bytes → whisper + pyannote → return."""
    import time
    t0 = time.time()
    work_dir = Path(tempfile.mkdtemp())
    try:
        audio_path = work_dir / filename
        audio_path.write_bytes(audio_bytes)
        print(f"  Received {len(audio_bytes) / 1024 / 1024:.1f}MB: {filename}")
        result = _run_gpu_pipeline(audio_path, hf_token, language)
        result["timing"]["upload_s"] = round(time.time() - t0 - result["timing"]["total_s"], 1)
        return result
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
