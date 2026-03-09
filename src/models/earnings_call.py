"""EarningsCall, Transcript, and AudioFile models."""

import enum

from sqlalchemy import Date, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class CallStatus(str, enum.Enum):
    """Processing status of an earnings call."""

    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class EarningsCall(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A single earnings call event."""

    __tablename__ = "earnings_calls"

    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id"), nullable=False)
    call_date: Mapped[str] = mapped_column(Date, nullable=False)
    fiscal_year: Mapped[int | None] = mapped_column(Integer)
    fiscal_quarter: Mapped[int | None] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(10), default="en")
    status: Mapped[CallStatus] = mapped_column(
        Enum(CallStatus), default=CallStatus.DISCOVERED, nullable=False
    )
    source_url: Mapped[str | None] = mapped_column(String(1000))

    company: Mapped["Company"] = relationship(back_populates="earnings_calls")
    audio_file: Mapped["AudioFile | None"] = relationship(
        back_populates="earnings_call", uselist=False
    )
    transcript: Mapped["Transcript | None"] = relationship(
        back_populates="earnings_call", uselist=False
    )

    def __repr__(self) -> str:
        return f"<EarningsCall {self.company_id} {self.call_date}>"


class AudioFile(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Downloaded audio file metadata."""

    __tablename__ = "audio_files"

    earnings_call_id: Mapped[str] = mapped_column(
        ForeignKey("earnings_calls.id"), unique=True, nullable=False
    )
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    format: Mapped[str] = mapped_column(String(10), nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)
    download_url: Mapped[str | None] = mapped_column(String(1000))

    earnings_call: Mapped["EarningsCall"] = relationship(back_populates="audio_file")

    def __repr__(self) -> str:
        return f"<AudioFile {self.file_path}>"


class Transcript(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Transcription output for an earnings call."""

    __tablename__ = "transcripts"

    earnings_call_id: Mapped[str] = mapped_column(
        ForeignKey("earnings_calls.id"), unique=True, nullable=False
    )
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    language_detected: Mapped[str | None] = mapped_column(String(10))
    whisper_model_used: Mapped[str | None] = mapped_column(String(50))
    is_diarized: Mapped[bool] = mapped_column(default=False)
    segments_json: Mapped[str | None] = mapped_column(Text)

    earnings_call: Mapped["EarningsCall"] = relationship(back_populates="transcript")

    def __repr__(self) -> str:
        return f"<Transcript call={self.earnings_call_id}>"


from src.models.company import Company  # noqa: E402, F401
