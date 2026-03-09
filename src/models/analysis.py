"""Analysis models: CallSummary, KeyMetric, SentimentScore."""

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class CallSummary(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """LLM-generated summary of an earnings call."""

    __tablename__ = "call_summaries"

    earnings_call_id: Mapped[str] = mapped_column(
        ForeignKey("earnings_calls.id"), unique=True, nullable=False
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    key_points: Mapped[str | None] = mapped_column(Text)
    management_outlook: Mapped[str | None] = mapped_column(Text)
    model_used: Mapped[str | None] = mapped_column(String(100))

    earnings_call: Mapped["EarningsCall"] = relationship()

    def __repr__(self) -> str:
        return f"<CallSummary call={self.earnings_call_id}>"


class KeyMetric(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Extracted financial metric from an earnings call."""

    __tablename__ = "key_metrics"

    earnings_call_id: Mapped[str] = mapped_column(
        ForeignKey("earnings_calls.id"), nullable=False, index=True
    )
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[str] = mapped_column(String(200), nullable=False)
    unit: Mapped[str | None] = mapped_column(String(50))
    period: Mapped[str | None] = mapped_column(String(50))
    context: Mapped[str | None] = mapped_column(Text)

    earnings_call: Mapped["EarningsCall"] = relationship()

    def __repr__(self) -> str:
        return f"<KeyMetric {self.metric_name}={self.metric_value}>"


class SentimentScore(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Sentiment analysis of an earnings call."""

    __tablename__ = "sentiment_scores"

    earnings_call_id: Mapped[str] = mapped_column(
        ForeignKey("earnings_calls.id"), unique=True, nullable=False
    )
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    tone_label: Mapped[str | None] = mapped_column(String(50))
    forward_looking_score: Mapped[float | None] = mapped_column(Float)
    risk_score: Mapped[float | None] = mapped_column(Float)
    details_json: Mapped[str | None] = mapped_column(Text)

    earnings_call: Mapped["EarningsCall"] = relationship()

    def __repr__(self) -> str:
        return f"<SentimentScore call={self.earnings_call_id} score={self.overall_score}>"


from src.models.earnings_call import EarningsCall  # noqa: E402, F401
