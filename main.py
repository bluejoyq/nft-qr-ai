from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from model import gen_qr_image
from dtos import QrDto
from PIL import Image
import io
import requests
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = ["http://localhost:5173", "https://nft-qr.web.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to load image: {response.status_code}")
    image_bytes = io.BytesIO(response.content)
    return Image.open(image_bytes)


def from_image_to_bytes(image: Image.Image) -> io.BytesIO:
    """
    pillow image 객체를 bytes로 변환
    """
    # Pillow 이미지 객체를 Bytes로 변환
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes


@app.get("/health")
def health_check():
    return "good"


@app.post("/qr")
async def generate_qr(dto: QrDto):
    try:
        image = load_image_from_url(dto.image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail="image load fail")
    try:
        result_image = gen_qr_image(image, dto.qr_data, dto.prompt)
        image_bytes = from_image_to_bytes(result_image)
        return StreamingResponse(image_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)
