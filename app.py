import torch
from PIL import Image, ImageOps
import qrcode
import torchvision.transforms as T
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
        if item[3] == 0:
            newData.append((255, 255, 255, 1))
        else:
            newData.append(item)

    mew_image = Image.new(mode="RGBA", size=image.size)
    mew_image.putdata(newData)
    return mew_image


prompt = "high quality, pixel perfect, nft"
negative_prompt = "ugly, disfigured, low quality, blurry, nsfw"
main_image_path = "test/data.png"
data = "naver.com"
main_image = (
    Image.open(main_image_path).resize((image_size, image_size)).convert("RGBA")
)
main_image = transparent_to_white(main_image)
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

alpha_qr_image = white_bg_to_transparent(qr_image)
alpha_qr_image.putalpha(100)
alpha_qr_image.save("alpha.png")
blended_image = Image.alpha_composite(main_image, alpha_qr_image)
blended_image.save("blended.png")
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=blended_image,
    control_image=alpha_qr_image,
    width=image_size,
    height=image_size,
    guidance_scale=5,
    controlnet_conditioning_scale=0.3,
    strength=0.6,
    num_inference_steps=50,
).images[0]


image.save(f'{main_image_path.split(".")[0]}.result.png')
