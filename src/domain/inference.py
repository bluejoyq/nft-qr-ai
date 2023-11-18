import torch
from PIL import Image, ImageEnhance
import qrcode
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
)

from typing import Tuple

controlnet = ControlNetModel.from_pretrained(
    "DionTimmer/controlnet_qrcode-control_v11p_sd21", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

IMAGE_SIZE = 768


def white_bg_to_transparent(qr_image: Image.Image) -> Image.Image:
    datas = qr_image.convert("RGBA").getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img = Image.new(mode="RGBA", size=qr_image.size)
    img.putdata(newData)
    return img


def transparent_to_white(image: Image.Image) -> Image.Image:
    datas = image.convert("RGBA").getdata()
    newData = []
    for item in datas:
        if item[3] < 128:
            newData.append((255, 255, 255, 255))
        else:
            newData.append(item)

    new_image = Image.new(mode="RGBA", size=image.size)
    new_image.putdata(newData)
    return new_image


def crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    size = min(width, height)

    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    return image.crop((left, top, right, bottom))


def inference(
    blended_image: Image.Image,
    qr_image: Image.Image,
    additional_prompt: str,
    base_prompt="high quality, cinematic, render:2.0, HD, 4k, 8k, epic, master piece,",
    negative_prompt="ugly, disfigured, low quality, blurry, nsfw, typography:2.0, text:2.0, letter:2.0",
    guidance_scale=30,
    controlnet_conditioning_scale=0.5,
    strength=0.5,
    num_inference_steps=50,
) -> Image.Image:
    image = pipe(
        prompt=base_prompt + additional_prompt,
        negative_prompt=negative_prompt,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        image=blended_image,
        control_image=qr_image,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        strength=strength,
        num_inference_steps=num_inference_steps,
    ).images[0]

    return image


def prepare(image: Image.Image, qr_data: str) -> Tuple[Image.Image, Image.Image]:
    main_image = crop_square(image).resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGBA")
    main_image = transparent_to_white(main_image)
    qr_image: Image.Image = (
        qrcode.make(
            data=qr_data,
            version=1,
            error_correction=qrcode.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        .resize((IMAGE_SIZE, IMAGE_SIZE))
        .convert("RGBA")
    )

    alpha_qr_image = qr_image.copy()
    alpha_qr_image.putalpha(128)
    blended_image = Image.alpha_composite(main_image, alpha_qr_image)
    color_enhancer = ImageEnhance.Color(blended_image)
    blended_image = color_enhancer.enhance(1.8)
    contrast_enhancer = ImageEnhance.Contrast(blended_image)
    blended_image = contrast_enhancer.enhance(1.35)
    return blended_image, qr_image


def gen_qr_image(
    image: Image.Image, qr_data: str, additional_prompt: str
) -> Image.Image:
    base_image = image

    blendend_image, qr_image = prepare(base_image, qr_data)
    result_image = inference(blendend_image, qr_image, additional_prompt)
    color_enhancer = ImageEnhance.Color(result_image)
    result_image = color_enhancer.enhance(1.5)
    contrast_enhancer = ImageEnhance.Contrast(result_image)
    result_image = contrast_enhancer.enhance(1.2)
    return result_image
