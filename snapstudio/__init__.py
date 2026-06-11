"""SnapStudio：AI 商品攝影棚（去背 → AI 場景 → 重打光 → 文案）。

重類別採 PEP 562 延遲匯出：``import snapstudio`` 不會拖入 torch/diffusers，
存取對應屬性時才載入子模組（CLI/UI 的 --help 與啟動因此不需等模型依賴）。
"""
import importlib

from . import config  # noqa: F401  # 輕量；匯入即設定 HF_HUB_OFFLINE 等環境變數
from .schemas import CopyPack, ProductCard, ScenePlan  # noqa: F401

__version__ = "1.0.0"

# 屬性名 → (子模組, 類別名)
_LAZY = {
    "Matting": ("matting", "Matting"),
    "SceneGenerator": ("scene", "SceneGenerator"),
    "Relighter": ("relight", "Relighter"),
    "LLMClient": ("llm", "LLMClient"),
    "SnapStudio": ("pipeline", "SnapStudio"),
    "StudioResult": ("pipeline", "StudioResult"),
}

__all__ = [
    "__version__", "config",
    "ProductCard", "ScenePlan", "CopyPack",
    *_LAZY,
]


def __getattr__(name: str):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(f".{module_name}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
