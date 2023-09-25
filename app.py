import torch
from PIL import Image
import qrcode

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

main_image = (
    Image.open("test/image.png").convert("RGB").resize((image_size, image_size))
)
prompt = "pixel perfect, detailed"
negative_prompt = "ugly, disfigured, low quality, blurry, nsfw"
data = "naver.com"
qr_image = (
    qrcode.make(
        data=data,
        version=1,
        error_correction=qrcode.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    .convert("RGB")
    .resize((image_size, image_size))
)
blended_image = Image.blend(main_image, qr_image, 0.4)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=blended_image,
    control_image=qr_image,
    width=768,
    height=768,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.1,
    strength=0.9,
    num_inference_steps=10,
).images[0]

image.save("yellow_cat.png")
