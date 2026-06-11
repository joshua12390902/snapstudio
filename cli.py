"""SnapStudio CLI：一鍵 照片 → 素材包（成品圖 + cutout + plans.json + copy.json）。

用法：
    python cli.py --image examples/test_product_input.png \
                  --brief "質感路線、文青風" --n 3 --quality fine \
                  --out examples/demo_run/
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # 允許從任意 cwd 執行

from PIL import Image  # noqa: E402


def _to_jsonable(obj):
    """pydantic v2 / dataclass / dict 通吃的序列化。"""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return dict(obj)


def _safe_name(name: str, fallback: str) -> str:
    """方案名轉檔名：只留中英數字。"""
    kept = "".join(c for c in name if c.isalnum() or "一" <= c <= "鿿")
    return kept[:24] or fallback


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SnapStudio：AI 商品攝影棚（CLI）")
    parser.add_argument("--image", required=True, help="商品照片路徑")
    parser.add_argument("--brief", default="", help="口語需求，如「質感路線、文青風」")
    parser.add_argument("--n", type=int, default=3, help="場景方案數")
    parser.add_argument("--quality", choices=["fast", "fine"], default="fine",
                        help="fast=LCM 快速預覽 / fine=fbc+兩段式高清")
    parser.add_argument("--out", default="examples/demo_run/", help="輸出資料夾")
    args = parser.parse_args(argv)

    from snapstudio.pipeline import SnapStudio  # 延遲匯入：--help 不需載模型

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image)

    def progress(stage: str, frac: float) -> None:
        print(f"[{frac * 100:5.1f}%] {stage}", flush=True)

    t0 = time.time()
    studio = SnapStudio(quality=args.quality)
    result = studio.process(
        image, user_brief=args.brief, n_plans=args.n, progress_cb=progress
    )

    # 成品圖（每方案一張）＋主角擺放預覽中間產物＋去背前景
    shot_paths = []
    for i, (plan, shot) in enumerate(zip(result.plans, result.shots), start=1):
        path = out_dir / f"shot_{i:02d}_{_safe_name(plan.plan_name, f'plan{i}')}.png"
        shot.save(path)
        shot_paths.append(str(path))
    for i, prev in enumerate(result.previews, start=1):
        prev.save(out_dir / f"placement_{i:02d}.png")
    if result.fg is not None:
        result.fg.save(out_dir / "cutout.png")

    # plans.json：商品卡 + 全部方案 + 計時
    plans_payload = {
        "product_card": _to_jsonable(result.card),
        "plans": [_to_jsonable(p) for p in result.plans],
        "timings": result.timings,
    }
    (out_dir / "plans.json").write_text(
        json.dumps(plans_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # copy.json：文案包（LLM 失敗時為 null）
    (out_dir / "copy.json").write_text(
        json.dumps(_to_jsonable(result.copy), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if result.copy is None:
        print("警告：文案生成失敗，copy.json 內容為 null", flush=True)

    print(f"\n完成（{time.time() - t0:.1f}s）→ {out_dir.resolve()}", flush=True)
    for p in shot_paths:
        print(f"  成品圖  {p}", flush=True)
    print(f"  企劃    {out_dir / 'plans.json'}", flush=True)
    print(f"  文案    {out_dir / 'copy.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
