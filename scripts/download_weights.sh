#!/usr/bin/env bash
# SnapStudio 權重下載腳本（總量約 12 GB）
#
# 為什麼不用 huggingface_hub 直接下載？
#   實測本機到 HF CDN 的長連線會「停滯」：連線不斷但流量歸零，hub 的下載器會永遠掛住。
#   解法：wget 的 --read-timeout=20 會在 20 秒無資料時主動斷線，配合 -c 斷點續傳與
#   --tries=50 自動重試，停滯多少次都能續完。執行期則一律 HF_HUB_OFFLINE=1 離線載入
#   （見 snapstudio/config.py）。
#
# 用法：
#   bash scripts/download_weights.sh              # 全部
#   bash scripts/download_weights.sh sd15 iclight # 只抓指定群組（sd15/iclight/lcm/realvisxl）
#
# 每個檔案附「期望位元組數」：已完整則跳過，下載後驗證大小，不符即報錯退出。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$ROOT/weights"

fetch() { # fetch <weights/ 下相對路徑> <期望 bytes> <resolve URL>
  local rel="$1" size="$2" url="$3"
  local dest="$W/$rel"
  if [[ -f "$dest" && "$(stat -c%s "$dest")" == "$size" ]]; then
    echo "[skip] $rel（已完整）"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  # curl -C - 斷點續傳 + --retry 自動重連（本機部分環境 wget 不穩，curl 較可靠）
  curl -sL -C - --retry 50 --retry-delay 2 "$url" -o "$dest"
  local got
  got="$(stat -c%s "$dest")"
  if [[ "$got" != "$size" ]]; then
    echo "[error] $rel 大小不符：得到 $got，期望 $size" >&2
    exit 1
  fi
  echo "[ok] $rel"
}

# ---- SD1.5 base（diffusers 格式、fp16 variant，約 2.1 GB）----
# 原 runwayml/stable-diffusion-v1-5 已自 HF 下架，此為官方接手的鏡像 repo（檔案逐位元組相同）。
SD15="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main"
dl_sd15() {
  fetch sd15/model_index.json                                  541        "$SD15/model_index.json"
  fetch sd15/feature_extractor/preprocessor_config.json        342        "$SD15/feature_extractor/preprocessor_config.json"
  fetch sd15/scheduler/scheduler_config.json                   308        "$SD15/scheduler/scheduler_config.json"
  fetch sd15/text_encoder/config.json                          617        "$SD15/text_encoder/config.json"
  fetch sd15/text_encoder/model.fp16.safetensors               246144864  "$SD15/text_encoder/model.fp16.safetensors"
  fetch sd15/tokenizer/merges.txt                              524619     "$SD15/tokenizer/merges.txt"
  fetch sd15/tokenizer/special_tokens_map.json                 472        "$SD15/tokenizer/special_tokens_map.json"
  fetch sd15/tokenizer/tokenizer_config.json                   806        "$SD15/tokenizer/tokenizer_config.json"
  fetch sd15/tokenizer/vocab.json                              1059962    "$SD15/tokenizer/vocab.json"
  fetch sd15/unet/config.json                                  743        "$SD15/unet/config.json"
  fetch sd15/unet/diffusion_pytorch_model.fp16.safetensors     1719125304 "$SD15/unet/diffusion_pytorch_model.fp16.safetensors"
  fetch sd15/vae/config.json                                   547        "$SD15/vae/config.json"
  fetch sd15/vae/diffusion_pytorch_model.fp16.safetensors      167335342  "$SD15/vae/diffusion_pytorch_model.fp16.safetensors"
}

# ---- IC-Light v1 權重 offset（fc：純文字條件 / fbc：前景+背景條件，各約 1.7 GB）----
ICL="https://huggingface.co/lllyasviel/ic-light/resolve/main"
dl_iclight() {
  fetch iclight/iclight_sd15_fc.safetensors  1719148312 "$ICL/iclight_sd15_fc.safetensors"
  fetch iclight/iclight_sd15_fbc.safetensors 1719171352 "$ICL/iclight_sd15_fbc.safetensors"
}

# ---- LCM-LoRA（SD1.5 用，快速預覽檔位，約 135 MB）----
dl_lcm() {
  fetch lcm-lora-sdv1-5/pytorch_lora_weights.safetensors 134621556 \
    "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors"
}

# ---- RealVisXL V5.0（場景背景生成，單檔 fp16 checkpoint，約 6.9 GB）----
# 注意：from_single_file 載入時仍需 SDXL 的設定檔（本機 HF 快取的
# stabilityai/stable-diffusion-xl-base-1.0 與 madebyollin/sdxl-vae-fp16-fix）。
dl_realvisxl() {
  fetch realvisxl/RealVisXL_V5.0_fp16.safetensors 6938065488 \
    "https://huggingface.co/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors"
}

# ---- SDXL inpaint 0.1（v2 主流程，專用 9 通道 inpaint 權重，fp16 約 6.5 GB）----
# 必須 from_pretrained（資料夾格式）；RealVisXL 等 4 通道 base 無法用於真正的 inpaint。
SDIN="https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main"
dl_sdxl_inpaint() {
  fetch sdxl-inpaint/model_index.json                              690        "$SDIN/model_index.json"
  fetch sdxl-inpaint/scheduler/scheduler_config.json               479        "$SDIN/scheduler/scheduler_config.json"
  fetch sdxl-inpaint/text_encoder/config.json                      746        "$SDIN/text_encoder/config.json"
  fetch sdxl-inpaint/text_encoder/model.fp16.safetensors           246144867  "$SDIN/text_encoder/model.fp16.safetensors"
  fetch sdxl-inpaint/text_encoder_2/config.json                    758        "$SDIN/text_encoder_2/config.json"
  fetch sdxl-inpaint/text_encoder_2/model.fp16.safetensors         1389382884 "$SDIN/text_encoder_2/model.fp16.safetensors"
  fetch sdxl-inpaint/tokenizer/merges.txt                          524619     "$SDIN/tokenizer/merges.txt"
  fetch sdxl-inpaint/tokenizer/special_tokens_map.json             472        "$SDIN/tokenizer/special_tokens_map.json"
  fetch sdxl-inpaint/tokenizer/tokenizer_config.json               737        "$SDIN/tokenizer/tokenizer_config.json"
  fetch sdxl-inpaint/tokenizer/vocab.json                          1059962    "$SDIN/tokenizer/vocab.json"
  fetch sdxl-inpaint/tokenizer_2/merges.txt                        524619     "$SDIN/tokenizer_2/merges.txt"
  fetch sdxl-inpaint/tokenizer_2/special_tokens_map.json           460        "$SDIN/tokenizer_2/special_tokens_map.json"
  fetch sdxl-inpaint/tokenizer_2/tokenizer_config.json             725        "$SDIN/tokenizer_2/tokenizer_config.json"
  fetch sdxl-inpaint/tokenizer_2/vocab.json                        1059962    "$SDIN/tokenizer_2/vocab.json"
  fetch sdxl-inpaint/unet/config.json                              1932       "$SDIN/unet/config.json"
  fetch sdxl-inpaint/unet/diffusion_pytorch_model.fp16.safetensors 5135178560 "$SDIN/unet/diffusion_pytorch_model.fp16.safetensors"
  fetch sdxl-inpaint/vae/config.json                               659        "$SDIN/vae/config.json"
  fetch sdxl-inpaint/vae/diffusion_pytorch_model.fp16.safetensors  167335338  "$SDIN/vae/diffusion_pytorch_model.fp16.safetensors"
}

# ---- LCM-LoRA SDXL（inpaint 少步加速，約 394 MB）----
dl_lcm_sdxl() {
  fetch lcm-lora-sdxl/pytorch_lora_weights.safetensors 393855224 \
    "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors"
}

# ---- RealVisXL V4 inpaint（9 通道 inpaint，美感優於官方 0.1，主流程預設，約 6.9 GB）----
# 經審查員 A/B 確認真實感/接地/海景全面優於官方 SDXL inpaint，設為預設權重。
RVIN="https://huggingface.co/OzzyGT/RealVisXL_V4.0_inpainting/resolve/main"
dl_realvisxl_inpaint() {
  fetch realvisxl-inpaint/model_index.json                              721        "$RVIN/model_index.json"
  fetch realvisxl-inpaint/scheduler/scheduler_config.json               563        "$RVIN/scheduler/scheduler_config.json"
  fetch realvisxl-inpaint/text_encoder/config.json                      560        "$RVIN/text_encoder/config.json"
  fetch realvisxl-inpaint/text_encoder/model.fp16.safetensors           246144152  "$RVIN/text_encoder/model.fp16.safetensors"
  fetch realvisxl-inpaint/text_encoder_2/config.json                    570        "$RVIN/text_encoder_2/config.json"
  fetch realvisxl-inpaint/text_encoder_2/model.fp16.safetensors         1389382176 "$RVIN/text_encoder_2/model.fp16.safetensors"
  fetch realvisxl-inpaint/tokenizer/merges.txt                          524619     "$RVIN/tokenizer/merges.txt"
  fetch realvisxl-inpaint/tokenizer/special_tokens_map.json             588        "$RVIN/tokenizer/special_tokens_map.json"
  fetch realvisxl-inpaint/tokenizer/tokenizer_config.json               705        "$RVIN/tokenizer/tokenizer_config.json"
  fetch realvisxl-inpaint/tokenizer/vocab.json                          1059962    "$RVIN/tokenizer/vocab.json"
  fetch realvisxl-inpaint/tokenizer_2/merges.txt                        524619     "$RVIN/tokenizer_2/merges.txt"
  fetch realvisxl-inpaint/tokenizer_2/special_tokens_map.json           462        "$RVIN/tokenizer_2/special_tokens_map.json"
  fetch realvisxl-inpaint/tokenizer_2/tokenizer_config.json             856        "$RVIN/tokenizer_2/tokenizer_config.json"
  fetch realvisxl-inpaint/tokenizer_2/vocab.json                        1059962    "$RVIN/tokenizer_2/vocab.json"
  fetch realvisxl-inpaint/unet/config.json                              1778       "$RVIN/unet/config.json"
  fetch realvisxl-inpaint/unet/diffusion_pytorch_model.fp16.safetensors 5135178560 "$RVIN/unet/diffusion_pytorch_model.fp16.safetensors"
  fetch realvisxl-inpaint/vae/config.json                               739        "$RVIN/vae/config.json"
  fetch realvisxl-inpaint/vae/diffusion_pytorch_model.fp16.safetensors  167335342  "$RVIN/vae/diffusion_pytorch_model.fp16.safetensors"
}

# 預設群組：主流程所需（RealVisXL inpaint 為預設生成權重；sdxl_inpaint 為退路）
if [[ $# -eq 0 ]]; then
  groups=(sd15 iclight lcm sdxl_inpaint lcm_sdxl realvisxl_inpaint)
else
  groups=("$@")
fi
for g in "${groups[@]}"; do
  case "$g" in
    sd15)             dl_sd15 ;;
    iclight)          dl_iclight ;;
    lcm)              dl_lcm ;;
    sdxl_inpaint)     dl_sdxl_inpaint ;;
    lcm_sdxl)         dl_lcm_sdxl ;;
    realvisxl_inpaint) dl_realvisxl_inpaint ;;
    realvisxl)        dl_realvisxl ;;
    *) echo "[error] 未知群組：$g（可用：sd15 iclight lcm sdxl_inpaint lcm_sdxl realvisxl_inpaint realvisxl）" >&2; exit 1 ;;
  esac
done
echo "全部權重就緒：$W"
