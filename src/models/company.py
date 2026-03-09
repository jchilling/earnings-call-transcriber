"""Company, Exchange, and Sector models."""

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Exchange(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Stock exchange (e.g. TWSE, HKEX, TSE, KRX)."""

    __tablename__ = "exchanges"

    code: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    country: Mapped[str] = mapped_column(String(50), nullable=False)

    companies: Mapped[list["Company"]] = relationship(back_populates="exchange")

    def __repr__(self) -> str:
        return f"<Exchange {self.code}>"


class Sector(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Industry sector classification."""

    __tablename__ = "sectors"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(500))

    companies: Mapped[list["Company"]] = relationship(back_populates="sector")

    def __repr__(self) -> str:
        return f"<Sector {self.name}>"


class Company(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Publicly listed company."""

    __tablename__ = "companies"
    __table_args__ = (
        UniqueConstraint("ticker", "exchange_id", name="uq_company_ticker_exchange"),
    )

    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    name_local: Mapped[str | None] = mapped_column(String(200))
    exchange_id: Mapped[str] = mapped_column(ForeignKey("exchanges.id"), nullable=False)
    sector_id: Mapped[str | None] = mapped_column(ForeignKey("sectors.id"))
    ir_url: Mapped[str | None] = mapped_column(String(500))

    exchange: Mapped["Exchange"] = relationship(back_populates="companies")
    sector: Mapped["Sector | None"] = relationship(back_populates="companies")
    earnings_calls: Mapped[list["EarningsCall"]] = relationship(back_populates="company")

    def __repr__(self) -> str:
        return f"<Company {self.ticker} ({self.name})>"


# Avoid circular import at type-check time
from src.models.earnings_call import EarningsCall  # noqa: E402
