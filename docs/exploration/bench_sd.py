# 即時魔鏡單幀成本基準：SD1.5 + ControlNet-OpenPose + 少步 LoRA（LCM-LoRA / Hyper-SD）
# 用法（每個配置開獨立行程，VRAM 量測才乾淨）：
#   HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   /workspace/.venv-1/bin/python bench_sd.py --lora lcm --steps 4 --width 512 --height 512 --guidance 1.0 --tag a_lcm4_512
import argparse
import json
import os
import time

import torch
from diffusers import (
    ControlNetModel,
    LCMScheduler,
    StableDiffusionControlNetPipeline,
    TCDScheduler,
)
from PIL import Image

BASE = "/workspace/Deep_Generative_Model/HW7_poc/realtime_bench"
MODELS = f"{BASE}/models"
PROMPT = "futuristic cyborg warrior, neon city, cinematic"

parser = argparse.ArgumentParser()
parser.add_argument("--lora", choices=["lcm", "hyper1", "hyper4"], required=True)
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--guidance", type=float, default=1.0)
parser.add_argument("--tag", required=True)
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--runs", type=int, default=10)
args = parser.parse_args()

cond = Image.open(f"{BASE}/cond_{args.width}x{args.height}.png").convert("RGB")

controlnet = ControlNetModel.from_pretrained(
    f"{MODELS}/controlnet_openpose", torch_dtype=torch.float16, variant="fp16"
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    f"{MODELS}/sd15",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

if args.lora == "lcm":
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(
        f"{MODELS}/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors"
    )
else:  # Hyper-SD：模型卡建議 TCDScheduler、guidance 0、eta 1.0
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    fname = "Hyper-SD15-1step-lora.safetensors" if args.lora == "hyper1" else "Hyper-SD15-4steps-lora.safetensors"
    pipe.load_lora_weights(f"{MODELS}/hyper-sd", weight_name=fname)
pipe.fuse_lora()  # 融合 LoRA，貼近實際部署的推理速度

extra = {"eta": 1.0} if args.lora.startswith("hyper") else {}


def run(seed: int) -> Image.Image:
    gen = torch.Generator("cuda").manual_seed(seed)
    return pipe(
        PROMPT,
        image=cond,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        generator=gen,
        **extra,
    ).images[0]


# 暖機
for i in range(args.warmup):
    run(100 + i)

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
times = []
for i in range(args.runs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = run(42)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

img.save(f"{BASE}/out_{args.tag}.png")
ms = sum(times) / len(times)
result = {
    "tag": args.tag,
    "lora": args.lora,
    "steps": args.steps,
    "size": f"{args.width}x{args.height}",
    "guidance": args.guidance,
    "ms_per_frame": round(ms, 1),
    "fps": round(1000 / ms, 2),
    "vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "times_ms": [round(t, 1) for t in times],
}
with open(f"{BASE}/results.jsonl", "a") as f:
    f.write(json.dumps(result, ensure_ascii=False) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
