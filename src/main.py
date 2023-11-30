from fastapi import Depends, FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from src.domain.inference import gen_qr_image
from src.data.dtos import QrDto
from PIL import Image
import io
import requests
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from .data import database, models, crud, firebase, schemas
import time
import os
from firebase_admin import storage

app = FastAPI()
origins = ["http://localhost:5173", "https://nft-qr.web.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


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


@app.get("/public/{image_name}")
def read_image(image_name: str):
    return FileResponse(f"{os.getcwd()}/public/{image_name}")


@app.post("/qr")
def generate_qr(dto: QrDto, db: Session = Depends(get_db)):
    try:
        image = load_image_from_url(dto.image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail="image load fail")
    try:
        result_image = gen_qr_image(image, dto.qr_data, dto.additional_prompt)
        image_src = firebase.save_image(result_image)
        qr_history = models.QrHistory(
            image_src=image_src,
            qr_data=dto.qr_data,
        )
        crud.create_qr_history(db=db, qr_history=qr_history)
        return schemas.QrHistory.model_validate(qr_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)


@app.get("/qr")
def get_qr_histories(offset: int = 0, db: Session = Depends(get_db)):
    data = crud.get_qr_histories(db=db, offset=offset)

    next = offset + 10
    if len(data) != 10:
        next = None
    qr_histories = list(map(schemas.QrHistory.model_validate, data))
    return schemas.QrHistoriesRes(data=qr_histories, next=next)
