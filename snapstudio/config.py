"""SnapStudio 全域設定：路徑、模型位置、LLM 端點。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
SD15_DIR = WEIGHTS / "sd15"
ICLIGHT_FC = WEIGHTS / "iclight" / "iclight_sd15_fc.safetensors"
ICLIGHT_FBC = WEIGHTS / "iclight" / "iclight_sd15_fbc.safetensors"
LCM_LORA_DIR = WEIGHTS / "lcm-lora-sdv1-5"
REALVISXL_PATH = WEIGHTS / "realvisxl" / "RealVisXL_V5.0_fp16.safetensors"
# inpaint-grounded 主流程：SDXL 專用 9 通道 inpaint 權重 + LCM-LoRA-SDXL 少步加速
SDXL_INPAINT_DIR = WEIGHTS / "sdxl-inpaint"            # 官方 SDXL 1.0 inpaint（底）
REALVISXL_INPAINT_DIR = WEIGHTS / "realvisxl-inpaint"  # RealVisXL V4 9 通道 inpaint（美感更佳，主用）
LCM_LORA_SDXL_DIR = WEIGHTS / "lcm-lora-sdxl"
# 主流程預設用的 inpaint 權重（存在才用 RealVisXL，否則退官方）
INPAINT_DIR = REALVISXL_INPAINT_DIR if REALVISXL_INPAINT_DIR.exists() else SDXL_INPAINT_DIR
EXAMPLES = ROOT / "examples"

# 場景模型：RealVisXL 單檔權重（上方路徑）；SDXL_ID 供 from_single_file 借
# 設定檔/tokenizer（HF 快取），RealVisXL 缺檔時亦作為退路模型
SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

# 環境鐵則：本機到 HF CDN 的長連線會停滯 → 權重先用 scripts/download_weights.sh
# 抓齊後一律離線載入
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# LLM 端點（皆為 OpenAI 相容介面），環境變數可覆寫
LLM_BASE_URL = os.getenv("SNAPSTUDIO_LLM_BASE_URL", "https://opencode.ai/zen/v1")
LLM_MODEL = os.getenv("SNAPSTUDIO_LLM_MODEL", "opencode/big-pickle")
LLM_API_KEY = os.getenv("SNAPSTUDIO_LLM_API_KEY", os.getenv("OPENCODE_API_KEY", ""))
OLLAMA_BASE_URL = os.getenv("SNAPSTUDIO_OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_TEXT_MODEL = os.getenv("SNAPSTUDIO_OLLAMA_TEXT_MODEL", "qwen3:14b")
# 用「限制 context 的 32b」兼顧強度與速度：原生 32b 預設 context 太大(模型+ctx≈51GB)→只 24GB
# 進 GPU、其餘 CPU offload→單次 263s 不能用。實測縮 num_ctx 後 32b 大致進 GPU：8192→29s、
# 4096→9s，且輸出比 7b 強(會填對 worn_framing)。用 8192 平衡(可吃 1024 圖、輸出完整)。
#   建模型：ollama create qwen2.5vl:32b-ctx8k -f Modelfile（內容 FROM qwen2.5vl:32b / PARAMETER
#   num_ctx 8192）。若該卡更小或要更快，可改 SNAPSTUDIO_OLLAMA_VISION_MODEL=qwen2.5vl:7b。
OLLAMA_VISION_MODEL = os.getenv("SNAPSTUDIO_OLLAMA_VISION_MODEL", "qwen2.5vl:32b-ctx8k")
OLLAMA_JUDGE_MODEL = os.getenv("SNAPSTUDIO_OLLAMA_JUDGE_MODEL", "qwen2.5vl:32b-ctx8k")
