import io
import firebase_admin
from firebase_admin import credentials, storage
from PIL import Image
import time

cred = credentials.Certificate("nft-qr-firebase-adminsdk-f9461-e119ec4c73.json")
firebase_admin.initialize_app(cred)
bucket = storage.bucket("nft-qr.appspot.com")


def save_image(image: Image.Image) -> str:
    image_name = f"{time.time()}.png"

    bs = io.BytesIO()
    image.save(bs, "png")
    blob = bucket.blob(image_name)
    blob.upload_from_string(bs.getvalue(), content_type="image/png")
    return blob.public_url
