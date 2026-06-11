# PosePainter POC — 驗證姿態鏈端到端：照片 → OpenPose 骨架 → ControlNet 同姿勢角色生成
# 環境：/workspace/.venv-1（torch 2.5.1+cu121, diffusers 0.39.0.dev0, RTX 3090）
import time, torch
from pathlib import Path
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from controlnet_aux import OpenposeDetector

OUT = Path(__file__).parent
torch.cuda.reset_peak_memory_stats()

def stamp(tag, t0):
    print(f"[{tag}] {time.time()-t0:.1f}s | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

# ---------- Stage 1: 生成一張「來源照片」（模擬使用者上傳的照片） ----------
t0 = time.time()
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
stamp("load base", t0)

t0 = time.time()
src = base(
    prompt="photograph of a young man performing a high martial arts kick, "
           "full body visible, gym interior, natural lighting, sharp focus",
    negative_prompt="blurry, cropped, out of frame, bad anatomy, extra limbs",
    width=832, height=1216, num_inference_steps=30,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
src.save(OUT / "01_source_photo.png")
stamp("gen source", t0)

# ---------- Stage 2: OpenPose 姿態估計 → 火柴人骨架圖 ----------
t0 = time.time()
openpose = OpenposeDetector.from_pretrained("/workspace/Deep_Generative_Model/HW7_poc/annotators")
pose = openpose(src, detect_resolution=1024, image_resolution=1024,
                include_body=True, include_hand=False, include_face=False)
pose = pose.resize(src.size)
pose.save(OUT / "02_pose_skeleton.png")
stamp("pose detect", t0)

# ---------- Stage 3: ControlNet-OpenPose 同姿勢生成兩種角色 ----------
t0 = time.time()
controlnet = ControlNetModel.from_pretrained(
    "/workspace/Deep_Generative_Model/HW7_poc/cn-openpose", torch_dtype=torch.float16
)
pipe = StableDiffusionXLControlNetPipeline.from_pipe(base, controlnet=controlnet).to("cuda")
stamp("load controlnet", t0)

# 正式 App 中這兩段 prompt 由 LLM 從使用者一句話展開
characters = {
    "03_knight": "fantasy knight in ornate silver armor with flowing red cape, "
                 "epic mountain battlefield, dramatic sunset lighting, "
                 "fantasy concept art, highly detailed, artstation quality",
    "04_cyborg": "futuristic cyborg warrior with glowing blue circuits, "
                 "neon cyberpunk city street at night, rain reflections, "
                 "cinematic sci-fi movie still, octane render",
}
negative = "low quality, bad anatomy, extra limbs, deformed, watermark, text"

for name, prompt in characters.items():
    t0 = time.time()
    img = pipe(
        prompt=prompt, negative_prompt=negative,
        image=pose, controlnet_conditioning_scale=0.8,
        width=832, height=1216, num_inference_steps=30,
        generator=torch.Generator("cuda").manual_seed(7),
    ).images[0]
    img.save(OUT / f"{name}.png")
    stamp(f"gen {name}", t0)

# ---------- Stage 4: 並排對照圖 ----------
from PIL import Image
imgs = [Image.open(OUT / f) for f in
        ["01_source_photo.png", "02_pose_skeleton.png", "03_knight.png", "04_cyborg.png"]]
w, h = imgs[0].size
grid = Image.new("RGB", (w * 4 + 30, h), "white")
for i, im in enumerate(imgs):
    grid.paste(im.resize((w, h)), (i * (w + 10), 0))
grid.save(OUT / "05_grid.png")
print("DONE — all outputs in", OUT, flush=True)
