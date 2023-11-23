from typing import Optional
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, MappedAsDataclass


class QrHistoryBase(
    DeclarativeBase,
):
    pass


class QrHistory(QrHistoryBase):
    __tablename__ = "qr_histories"
    id: Mapped[Optional[int]] = mapped_column(
        primary_key=True, index=True, autoincrement=True
    )
    address: Mapped[str]
    contract_address: Mapped[str]
    token_id: Mapped[str]
    image_name: Mapped[str]
    qr_data: Mapped[str]
