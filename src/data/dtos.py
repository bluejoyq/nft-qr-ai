from pydantic import BaseModel


class QrDto(BaseModel):
    address: str
    contract_address: str
    token_id: str
    image_url: str
    qr_data: str
    additional_prompt: str = ""
