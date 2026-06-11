# IC-Light v1 (fc) 在 diffusers 0.39.0.dev0 上的改寫驗證（已跑通：3.1s/張、VRAM 3.6GB）
# 機制：SD1.5 UNet conv_in 4→8 通道（串接前景 latent），權重 = 原版 + IC-Light offset
import time
import torch
import numpy as np
import safetensors.torch as sf
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from rembg import remove, new_session

ROOT = Path(__file__).parent.parent          # HW7_snapstudio/
EXAMPLES = ROOT / "examples"
SD15 = ROOT / "weights" / "sd15"
ICLIGHT_FC = ROOT / "weights" / "iclight" / "iclight_sd15_fc.safetensors"

# ---------- 1. 去背 → 灰底合成（IC-Light 的前景條件格式） ----------
src = Image.open(EXAMPLES / "test_product_input.png").convert("RGB")
fg_rgba = remove(src, session=new_session("u2net"))
fg_rgba.save(EXAMPLES / "test_cutout.png")
arr = np.array(fg_rgba).astype(np.float32)
alpha = arr[..., 3:4] / 255.0
comp = (arr[..., :3] * alpha + 127.0 * (1 - alpha)).astype(np.uint8)
fg = Image.fromarray(comp).resize((768, 768), Image.LANCZOS)
fg.save(EXAMPLES / "test_graybg.png")
print("[matting done]", flush=True)

# ---------- 2. 載入 SD1.5 並合併 IC-Light 權重 ----------
pipe = StableDiffusionPipeline.from_pretrained(
    str(SD15), torch_dtype=torch.float16, variant="fp16",
    safety_checker=None, requires_safety_checker=False,
)
unet = pipe.unet
with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(
        8, unet.conv_in.out_channels,
        unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding,
    )
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4].copy_(unet.conv_in.weight)
    new_conv_in.bias.copy_(unet.conv_in.bias)
    unet.conv_in = new_conv_in.to(unet.dtype)

sd_offset = sf.load_file(str(ICLIGHT_FC))
sd_origin = unet.state_dict()
unet.load_state_dict(
    {k: sd_origin[k] + sd_offset[k].to(sd_origin[k]) for k in sd_origin}, strict=True
)
del sd_offset, sd_origin
print("[ic-light weights merged]", flush=True)

# 攔截 forward：把前景 latent 串接進每一步的輸入
original_forward = unet.forward
def hooked_forward(sample, timestep, encoder_hidden_states, **kwargs):
    cak = kwargs.get("cross_attention_kwargs") or {}
    c = cak["concat_conds"].to(sample)
    c = torch.cat([c] * (sample.shape[0] // c.shape[0]), dim=0)
    kwargs["cross_attention_kwargs"] = {}
    return original_forward(
        torch.cat([sample, c], dim=1), timestep, encoder_hidden_states, **kwargs
    )
unet.forward = hooked_forward

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True
)
pipe.to("cuda")

# ---------- 3. 前景 latent ----------
with torch.no_grad():
    t = torch.from_numpy(np.array(fg)).float() / 127.5 - 1.0
    t = t.permute(2, 0, 1).unsqueeze(0).to("cuda", torch.float16)
    fg_latent = pipe.vae.encode(t).latent_dist.mode() * pipe.vae.config.scaling_factor

# ---------- 4. 兩種光線方案重打光（正式版由 LLM 場景企劃輸出這段 prompt） ----------
scenes = {
    "out_warm_window": "brown leather wallet on dark walnut table, warm morning window "
                       "light from left side, cozy cafe atmosphere, soft shadows, "
                       "professional product photography, best quality",
    "out_studio_rim": "brown leather wallet on glossy black acrylic surface, dramatic "
                      "rim lighting from behind, dark moody background, luxury product "
                      "photography, best quality",
}
neg = "lowres, bad quality, watermark, text, cluttered"
for name, prompt in scenes.items():
    t0 = time.time()
    img = pipe(
        prompt=prompt, negative_prompt=neg,
        width=768, height=768, num_inference_steps=25, guidance_scale=2.0,
        cross_attention_kwargs={"concat_conds": fg_latent},
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    img.save(EXAMPLES / f"{name}.png")
    print(f"[{name}] {time.time()-t0:.1f}s | peak VRAM "
          f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
print("DONE", flush=True)
