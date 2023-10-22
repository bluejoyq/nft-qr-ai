from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from model import gen_qr_image
from PIL import Image
import io
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def generate_qr(file: UploadFile = File()):
    try:
        if "image" not in file.content_type:
            raise HTTPException(status_code=400, detail="File is not an image.")
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        result_image = gen_qr_image(image, "data")
        image_bytes = from_image_to_bytes(result_image)
        return StreamingResponse(image_bytes, media_type="image/png")
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
