from pydantic import BaseModel


class QrDto(BaseModel):
    qr_data: str
    additional_prompt: str = ""
