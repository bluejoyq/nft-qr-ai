from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase


class QrHistoryBase(
    DeclarativeBase,
):
    pass


class QrHistory(QrHistoryBase):
    __tablename__ = "qr_histories"
    id: Mapped[Optional[int]] = mapped_column(
        primary_key=True, index=True, autoincrement=True
    )
    image_src: Mapped[str]
    qr_data: Mapped[str]
