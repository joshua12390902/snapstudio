# 補跑：只載 ControlNet 管線（正式版的記憶體配置），生成第二角色 + 並排圖
import time, torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL

OUT = Path(__file__).parent
torch.cuda.reset_peak_memory_stats()

t0 = time.time()
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(OUT / "cn-openpose", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet, vae=vae, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
print(f"[load] {time.time()-t0:.1f}s | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

pose = Image.open(OUT / "02_pose_skeleton.png")
t0 = time.time()
img = pipe(
    prompt="futuristic cyborg warrior with glowing blue circuits, "
           "neon cyberpunk city street at night, rain reflections, "
           "cinematic sci-fi movie still, octane render",
    negative_prompt="low quality, bad anatomy, extra limbs, deformed, watermark, text",
    image=pose, controlnet_conditioning_scale=0.8,
    width=832, height=1216, num_inference_steps=30,
    generator=torch.Generator("cuda").manual_seed(7),
).images[0]
img.save(OUT / "04_cyborg.png")
print(f"[gen 04_cyborg] {time.time()-t0:.1f}s | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

imgs = [Image.open(OUT / f) for f in
        ["01_source_photo.png", "02_pose_skeleton.png", "03_knight.png", "04_cyborg.png"]]
w, h = imgs[0].size
grid = Image.new("RGB", (w * 4 + 30, h), "white")
for i, im in enumerate(imgs):
    grid.paste(im.resize((w, h)), (i * (w + 10), 0))
grid.save(OUT / "05_grid.png")
print("DONE", flush=True)
