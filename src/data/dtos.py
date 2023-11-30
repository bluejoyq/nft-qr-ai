from pydantic import BaseModel


class QrDto(BaseModel):
    image_url: str
    qr_data: str
    additional_prompt: str = ""
