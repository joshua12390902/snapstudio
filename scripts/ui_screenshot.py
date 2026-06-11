"""用 Playwright 自動操作 Gradio UI 並截圖（demo 素材用）。

用法：先確保 GPU 空閒，然後
    /workspace/.venv-1/bin/python scripts/ui_screenshot.py
產出：examples/demo_run/ui_shots/{tab1_empty,tab1_done,tab2_refined}.png
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "demo_run" / "ui_shots"
OUT.mkdir(parents=True, exist_ok=True)
PORT = 7861
URL = f"http://127.0.0.1:{PORT}"
TEST_IMG = str(ROOT / "examples" / "test_product_input.png")


def wait_server(timeout=120):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(URL, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    env = dict(os.environ, GRADIO_SERVER_PORT=str(PORT))
    server = subprocess.Popen(
        [sys.executable, str(ROOT / "app.py")], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        assert wait_server(), "Gradio server 沒起來"
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(URL, wait_until="domcontentloaded")
            page.wait_for_selector("button", timeout=30_000)
            time.sleep(3)
            page.screenshot(path=str(OUT / "tab1_empty.png"), full_page=True)

            # 上傳圖片 + 填需求 + 快速檔 → 生成
            page.set_input_files("input[type=file]", TEST_IMG)
            time.sleep(2)
            brief = page.locator("textarea").first
            brief.fill("質感文青風，目標客群是注重質感的上班族")
            page.get_by_text("快速", exact=False).first.click()
            n0 = page.locator("img").count()
            page.locator("button", has_text="開始").first.click()
            # 完成訊號：gallery 縮圖出現（頁面 img 數量超過點擊前），範例縮圖不會誤判
            page.wait_for_function(
                f"() => document.querySelectorAll('img').length > {n0}",
                timeout=420_000,
            )
            time.sleep(3)
            page.screenshot(path=str(OUT / "tab1_done.png"), full_page=True)

            # Tab2 修改流程（生成後下拉已自動帶入第一方案）：下指令 → 重生 → 等對比圖
            page.get_by_role("tab").nth(1).click()
            time.sleep(1)
            instr = page.locator("textarea").last
            instr.fill("光再暖一點，背景換成大理石")
            # refine 是「替換」既有圖的 src（on_generate 已預載修改前/後圖），
            # 所以等待條件是 src 集合改變而非 img 數量增加
            srcs0 = page.evaluate(
                "Array.from(document.querySelectorAll('img')).map(i => i.src).sort().join('|')"
            )
            page.locator("button", has_text="重生").first.click()
            page.wait_for_function(
                """(prev) => Array.from(document.querySelectorAll('img'))
                    .map(i => i.src).sort().join('|') !== prev""",
                arg=srcs0,
                timeout=420_000,
            )
            time.sleep(3)
            page.screenshot(path=str(OUT / "tab2_refined.png"), full_page=True)
            browser.close()
        print("截圖完成 →", OUT, flush=True)
    finally:
        server.terminate()
        server.wait(timeout=15)


if __name__ == "__main__":
    main()
