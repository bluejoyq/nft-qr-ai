import torch
from PIL import Image
import qrcode
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
)


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

image_size = 768
prompt = "nft:2.0, high quality, cinematic, render:2.0, HD, 4k, 8k, randscape"
negative_prompt = "ugly, disfigured, low quality, blurry, nsfw, typography"
data = "naver.com"
# main_image_path = "test/data.png"


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


def inference(image: Image.Image, data: str) -> Image.Image:
    main_image = crop_square(image).resize((image_size, image_size)).convert("RGBA")
    main_image = transparent_to_white(main_image)
    main_image.save("main.png")
    qr_image: Image.Image = (
        qrcode.make(
            data=data,
            version=1,
            error_correction=qrcode.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        .resize((image_size, image_size))
        .convert("RGBA")
    )

    alpha_qr_image = qr_image.copy()
    alpha_qr_image.putalpha(128)
    alpha_qr_image.save("alpha.png")
    blended_image = Image.alpha_composite(main_image, alpha_qr_image)
    blended_image.save("blended.png")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=blended_image,
        control_image=qr_image,
        width=image_size,
        height=image_size,
        guidance_scale=30,
        controlnet_conditioning_scale=0.5,
        strength=0.6,
        num_inference_steps=50,
    ).images[0]

    return image
