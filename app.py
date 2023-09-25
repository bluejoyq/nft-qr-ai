import torch
from PIL import Image, ImageOps
import qrcode
import torchvision.transforms as T
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
)


controlnet = ControlNetModel.from_pretrained(
    "DionTimmer/controlnet_qrcode-control_v1p_sd15", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
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


prompt = "pixel perfect, contrast, nft"
negative_prompt = "ugly, disfigured, low quality, blurry, nsfw"
main_image_path = "test/test.jpg"
data = "naver.com"
main_image = (
    Image.open(main_image_path).resize((image_size, image_size)).convert("RGBA")
)
qr_image = qrcode.make(
    data=data,
    version=1,
    error_correction=qrcode.ERROR_CORRECT_H,
    box_size=10,
    border=4,
).resize((image_size, image_size))
qr_image = white_bg_to_transparent(qr_image)
blended_image = Image.blend(main_image, qr_image, 0.4).convert("RGB")


image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=blended_image,
    control_image=qr_image,
    width=image_size,
    height=image_size,
    guidance_scale=10,
    controlnet_conditioning_scale=1.2,
    strength=0.6,
    num_inference_steps=35,
).images[0]


image.save(f'{main_image_path.split(".")[0]}.result.png')