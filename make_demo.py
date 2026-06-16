#!/usr/bin/env python3
"""一鍵產出 HW7 demo 影片：錄真實生成全流程 → 上傳段保持原速、生成段加速 → 加字卡 → 314831024_HW7.mp4

前提：app 已在 http://127.0.0.1:7860 跑著（python app.py）、GPU 有空、playwright+ffmpeg 已裝。
用法：cd 到 repo，執行  /workspace/.venv-1/bin/python make_demo.py
"""
import os, sys, glob, time, subprocess, shutil
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT); sys.path.insert(0, ".")
SHOW = os.path.join(ROOT, "examples/showcase")
WORK = os.path.join(ROOT, "examples/outputs/debug/demo_build")
VIDDIR = os.path.join(WORK, "rec")
RAW = os.path.join(ROOT, "examples/readme_pic")
PHOTOS = [os.path.join(RAW, f) for f in ("IMG_0352.jpg", "IMG_0353.jpg", "IMG_0354.jpg")]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
OUT = os.path.join(ROOT, "314831024_HW7.mp4")
shutil.rmtree(WORK, ignore_errors=True); os.makedirs(VIDDIR, exist_ok=True)

def sh(cmd):
    subprocess.run(cmd, check=True)

# ---------- 1. 錄真實生成（回傳 webm 路徑 + 按下生成的影片時間點 t_click 秒） ----------
def record():
    from playwright.sync_api import sync_playwright
    print("[rec] 開始錄影…", flush=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1280, "height": 800},
                            record_video_dir=VIDDIR,
                            record_video_size={"width": 1280, "height": 800})
        pg = ctx.new_page(); t_ref = time.time()      # ≈ 影片 t=0
        pg.set_default_timeout(60000)
        pg.goto("http://127.0.0.1:7860", wait_until="domcontentloaded"); time.sleep(5)
        # 一次上傳 3 張（第一張主圖，其餘當不同角度）；上傳後停久一點讓人看清 3 張縮圖
        pg.locator('input[type="file"]').first.set_input_files(PHOTOS)
        print("[rec] 已上傳 3 張", flush=True); time.sleep(7)
        try:
            num = pg.locator('input[type="number"]').first; num.fill("1"); num.press("Enter")
        except Exception:
            pass
        time.sleep(1.5)
        try:
            tb = pg.get_by_placeholder("質感文青風、目標客群上班族"); tb.click()
            for ch in "高質感 陽光 花園":      # 逐字打，看得出在輸入
                tb.type(ch); time.sleep(0.12)
        except Exception:
            pg.locator('textarea').first.fill("高質感 陽光 花園")
        time.sleep(2.5)
        t_click = time.time() - t_ref
        pg.get_by_role("button", name="開始生成").click(); print(f"[rec] 按生成 @ {t_click:.1f}s，等出圖…", flush=True)
        t0 = time.time()
        while time.time() - t0 < 480:
            if pg.locator('#snap-gallery img').count() > 0:
                print(f"[rec] 出圖 @ {int(time.time()-t0)}s", flush=True); break
            time.sleep(3)
        time.sleep(5); pg.mouse.wheel(0, 400); time.sleep(3)   # 停留展示成品
        ctx.close(); b.close()
    webms = sorted(glob.glob(os.path.join(VIDDIR, "*.webm")), key=os.path.getmtime)
    if not webms:
        raise RuntimeError("沒錄到 webm")
    return webms[-1], t_click

# ---------- 2. 字卡 ----------
def slides():
    from PIL import Image, ImageDraw, ImageFont
    W, H = 1280, 800; BG = (10, 10, 11); AM = (244, 184, 96); INK = (237, 237, 237); MU = (138, 138, 143)
    def F(s): return ImageFont.truetype(FONT, s)
    def ct(d, y, t, f, c):
        bb = d.textbbox((0, 0), t, font=f); d.text((W//2-(bb[2]-bb[0])//2, y), t, font=f, fill=c)
    def newp():
        im = Image.new("RGB", (W, H), BG); return im, ImageDraw.Draw(im)
    im, d = newp()
    ct(d, 250, "SnapStudio", F(96), AM); ct(d, 380, "AI 商品攝影棚", F(56), INK)
    ct(d, 480, "一張手機隨手拍 → 整套電商素材 · 單張 RTX 3090 本機推論", F(28), MU)
    im.save(os.path.join(WORK, "s_title.png"))
    im, d = newp(); ct(d, 60, "IC-Light A 護字：產品吃場景光、文字/logo 保持銳利", F(34), AM)
    ab = Image.open(os.path.join(SHOW, "iclight_ab.png")).convert("RGB"); ab.thumbnail((1040, 1040))
    im.paste(ab, ((W-ab.width)//2, 230)); im.save(os.path.join(WORK, "s_icl.png"))
    im, d = newp()
    ct(d, 260, "LLM ↔ Diffusion 參數層咬合 · 雙模式自動路由", F(40), INK)
    ct(d, 370, "去背 → VLM 識別 → LLM 企劃 → Diffusion 生成 → IC-Light 打光", F(28), MU)
    ct(d, 480, "github.com/joshua12390902/snapstudio", F(34), AM)
    im.save(os.path.join(WORK, "s_end.png"))

# ---------- 3. 組裝（上傳段原速、生成段加速） ----------
def assemble(webm, t_click):
    def seg_img(png, t, out):
        sh(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", png, "-t", str(t),
            "-r", "30", "-vf", "scale=1280:800,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", out])
    def ffprobe_dur(f):
        return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "default=nokey=1:noprint_wrappers=1", f],
                                    capture_output=True, text=True).stdout.strip() or "0")
    total = ffprobe_dur(webm)
    print(f"[asm] 影片總長 {total:.1f}s，按生成 @ {t_click:.1f}s", flush=True)
    # Part A：上傳+輸入段（0→t_click），原速（看得清楚 3 張照片）
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", webm, "-t", f"{t_click:.2f}",
        "-vf", "scale=1280:800,fps=30,format=yuv420p", "-an", "-c:v", "libx264", "-preset", "veryfast",
        os.path.join(WORK, "gA.mp4")])
    # Part B：生成等待段（t_click→結尾），加速到約 28s
    genlen = max(1.0, total - t_click)
    speed = max(1.0, genlen / 28.0)
    sh(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t_click:.2f}", "-i", webm,
        "-vf", f"setpts=PTS/{speed:.3f},scale=1280:800,fps=30,format=yuv420p", "-an",
        "-c:v", "libx264", "-preset", "veryfast", os.path.join(WORK, "gB.mp4")])
    seg_img(os.path.join(WORK, "s_title.png"), 3.0, os.path.join(WORK, "g0.mp4"))
    seg_img(os.path.join(WORK, "s_icl.png"), 3.5, os.path.join(WORK, "gC.mp4"))
    seg_img(os.path.join(WORK, "s_end.png"), 3.5, os.path.join(WORK, "gD.mp4"))
    lst = os.path.join(WORK, "list.txt")
    with open(lst, "w") as f:
        for g in ["g0.mp4", "gA.mp4", "gB.mp4", "gC.mp4", "gD.mp4"]:
            f.write(f"file '{os.path.join(WORK, g)}'\n")
    sh(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", OUT])
    shutil.copy(OUT, os.path.join(SHOW, "demo.mp4"))

if __name__ == "__main__":
    webm, t_click = record()
    print(f"[ok] 錄影完成：{webm}", flush=True)
    slides(); print("[ok] 字卡完成", flush=True)
    assemble(webm, t_click)
    print(f"[DONE] 產出 {OUT}（已同步 examples/showcase/demo.mp4）", flush=True)
