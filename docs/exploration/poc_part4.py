# 換姿態偵測器：YOLO11-pose（COCO-17）→ OpenPose-18 格式 → 標準骨架渲染 → 重生成
import time, torch
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from controlnet_aux.dwpose.util import draw_bodypose

OUT = Path(__file__).parent

# ---------- YOLO11 姿態偵測 ----------
t0 = time.time()
model = YOLO("yolo11x-pose.pt")
src = Image.open(OUT / "01_source_photo.png")
W, H = src.size
res = model(np.array(src), verbose=False)[0]
xy = res.keypoints.xy[0].cpu().numpy()      # (17,2) 像素座標
conf = res.keypoints.conf[0].cpu().numpy()  # (17,)
print(f"[yolo pose] {time.time()-t0:.1f}s, kpts>0.3: {(conf>0.3).sum()}/17", flush=True)

# COCO-17 → OpenPose-18：op 索引 → coco 索引（neck=雙肩中點另算）
C2O = {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9,
       8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15,
       14: 2, 15: 1, 16: 4, 17: 3}
THR = 0.3
cand = -np.ones((18, 2))
vis = np.zeros(18, dtype=bool)
for op_i, co_i in C2O.items():
    if conf[co_i] > THR:
        cand[op_i] = xy[co_i] / [W, H]
        vis[op_i] = True
if conf[5] > THR and conf[6] > THR:  # neck = 雙肩中點
    cand[1] = (xy[5] + xy[6]) / 2 / [W, H]
    vis[1] = True

subset = np.array([[j if vis[j] else -1 for j in range(18)]], dtype=float)
canvas = np.zeros((H, W, 3), dtype=np.uint8)
canvas = draw_bodypose(canvas, cand, subset)
pose = Image.fromarray(canvas)
pose.save(OUT / "02b_pose_yolo.png")
print("[pose rendered]", flush=True)

# ---------- ControlNet 重生成（與 part3 相同 prompt/seed，只換骨架圖） ----------
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(OUT / "cn-openpose", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet, vae=vae, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
print(f"[load] {time.time()-t0:.1f}s", flush=True)

jobs = {
    "03_knight_v2": "fantasy knight in ornate silver armor with flowing red cape, "
                    "epic mountain battlefield, dramatic sunset lighting, "
                    "fantasy concept art, highly detailed, artstation quality",
    "04_cyborg_v2": "futuristic cyborg warrior with glowing blue circuits, "
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
        ["01_source_photo.png", "02b_pose_yolo.png", "03_knight_v2.png", "04_cyborg_v2.png"]]
w, h = imgs[0].size
grid = Image.new("RGB", (w * 4 + 30, h), "white")
for i, im in enumerate(imgs):
    grid.paste(im.resize((w, h)), (i * (w + 10), 0))
grid.save(OUT / "07_grid_v2.png")
print("DONE", flush=True)
