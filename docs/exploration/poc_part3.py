# 調參驗證：conditioning scale 0.8 → 1.0，重生成騎士與賽博格
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
print(f"[load] {time.time()-t0:.1f}s", flush=True)

pose = Image.open(OUT / "02_pose_skeleton.png")
jobs = {
    "03_knight_s10": "fantasy knight in ornate silver armor with flowing red cape, "
                     "epic mountain battlefield, dramatic sunset lighting, "
                     "fantasy concept art, highly detailed, artstation quality",
    "04_cyborg_s10": "futuristic cyborg warrior with glowing blue circuits, "
                     "neon cyberpunk city street at night, rain reflections, "
                     "cinematic sci-fi movie still, octane render",
}
for name, p in jobs.items():
    t0 = time.time()
    img = pipe(
        prompt=p,
        negative_prompt="low quality, bad anatomy, extra limbs, deformed, watermark, text",
        image=pose, controlnet_conditioning_scale=1.0,
        width=832, height=1216, num_inference_steps=30,
        generator=torch.Generator("cuda").manual_seed(7),
    ).images[0]
    img.save(OUT / f"{name}.png")
    print(f"[gen {name}] {time.time()-t0:.1f}s | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

imgs = [Image.open(OUT / f) for f in
        ["01_source_photo.png", "02_pose_skeleton.png", "03_knight_s10.png", "04_cyborg_s10.png"]]
w, h = imgs[0].size
grid = Image.new("RGB", (w * 4 + 30, h), "white")
for i, im in enumerate(imgs):
    grid.paste(im.resize((w, h)), (i * (w + 10), 0))
grid.save(OUT / "06_grid_s10.png")
print("DONE", flush=True)
