# 生成一張「模擬使用者手機隨手拍」的商品照當 POC 測試輸入
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

OUT = Path(__file__).parent
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
img = pipe(
    prompt="amateur smartphone photo of a brown leather wallet on a cluttered desk, "
           "harsh overhead fluorescent light, slightly blurry, casual snapshot",
    negative_prompt="professional photography, studio lighting, bokeh",
    width=1024, height=1024, num_inference_steps=30,
    generator=torch.Generator("cuda").manual_seed(123),
).images[0]
img.save(OUT / "test_product_input.png")
print("saved", flush=True)
