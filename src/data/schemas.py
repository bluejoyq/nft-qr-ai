from typing import List
from pydantic import BaseModel


class QrHistory(BaseModel):
    id: int
    address: str
    contract_address: str
    token_id: str
    image_src: str
    qr_data: str

    class Config:
        orm_mode = True
        from_attributes = True


class QrHistoriesRes(BaseModel):
    data: List[QrHistory]
    next: int | None

    class Config:
        orm_mode = True
        from_attributes = True
