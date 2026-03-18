"""Data models for GPU pipeline results.

All dataclasses serialize to/from plain dicts for Modal transport.
"""

from dataclasses import asdict, dataclass


@dataclass
class VocalMetrics:
    """Per-segment vocal/prosodic features."""

    f0_mean: float = 0.0           # Hz, pitch
    f0_std: float = 0.0            # pitch variability
    rms_energy_mean: float = 0.0   # dB, loudness
    rms_energy_std: float = 0.0    # loudness variability
    speech_rate: float = 0.0       # words/sec
    pause_count: int = 0           # pauses > 200ms
    pause_total_sec: float = 0.0   # total pause time
    jitter_local: float = 0.0      # % pitch perturbation (stress)
    shimmer_local: float = 0.0     # % amplitude perturbation (stress)
    hnr: float = 0.0               # dB harmonics-to-noise (voice quality)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VocalMetrics":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SpeakerVocalProfile:
    """Aggregated vocal profile for one speaker across all their segments."""

    speaker_id: str = ""
    speaker_name: str | None = None
    segment_count: int = 0
    avg_f0: float = 0.0
    avg_rms_energy: float = 0.0
    avg_speech_rate: float = 0.0
    avg_jitter: float = 0.0
    avg_shimmer: float = 0.0
    avg_hnr: float = 0.0
    stress_score: float = 0.0      # 0-10 composite
    confidence_score: float = 0.0  # 0-10 composite

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpeakerVocalProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AudioQuality:
    """Audio quality metrics for the full recording."""

    snr_db: float = 0.0            # signal-to-noise ratio
    clipping_ratio: float = 0.0    # fraction of clipped samples
    sample_rate: int = 16000
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AudioQuality":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
