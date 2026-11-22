from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

# 既製のモデルの一式の入力の項目の種類と, 出力の形
READY_INPUTS = ("image",)
READY_FORMS = ("class",)

# ready の欄
READY_KEYS = (
    "library", "model", "weights", "input", "preprocess", "url", "file",
    "license"
)

# 前処理の欄と, 大きさを変えるときの補い方
PREPROCESS_KEYS = ("resize", "crop", "interpolation", "mean", "std")
INTERPOLATIONS = ("bilinear", "bicubic")

# ライセンスの記録に写す pretrained_info の項目 (画面が警告に使う)
LICENSE_KEYS = (
    "license", "status", "redistributable", "gated", "alert", "message",
    "source", "url"
)


def torchvision_weights(model: str, weights: str) -> dict:
    """
    torchvision の画像の分類のモデルの重みの情報を読む関数

    重みのファイルは読まない (名前の表と前処理は重みの説明から)

    Parameters
    ----------
    model: str
        モデルの名前 (resnet18 など)
    weights: str
        重みの名前 (IMAGENET1K_V1 など. DEFAULT はその時の名前に直す)

    Returns
    ----------
    found: dict
        unit, params, classes, input, preprocess, url, file

    Raises
    ----------
    ImportError
        torchvision が入っていない場合
    ValueError
        モデルか重みの名前が無い場合と, 画像の分類の重みでない場合
    """
    # 重みの名前の一覧 (DEFAULT は版の決まった名前に)
    try:
        from torchvision import models
    except ImportError as error:
        raise ImportError(
            "torchvision の既製のモデルには torchvision が要ります "
            "(ML-ModelControl の追加の依存 torchvision を入れる)"
        ) from error
    try:
        choices = models.get_model_weights(model)
    except Exception as error:
        raise ValueError(
            f"torchvision に重みのあるモデル {model} がありません"
        ) from error
    if weights not in choices.__members__:
        raise ValueError(
            f"{model} の重みに {weights} がありません (重み: "
            f"{', '.join(choices.__members__)})"
        )
    chosen = choices[weights]

    # 画像の分類の前処理 (大きさ, 切り出し, 正規化) と種類の名前
    transform = chosen.transforms()
    needed = ("resize_size", "crop_size", "mean", "std", "interpolation")
    if not all(hasattr(transform, k) for k in needed) or not (
        chosen.meta.get("categories")
    ):
        raise ValueError(
            f"{model} の {chosen.name} は画像の分類の重みではありません "
            "(既製のモデルの一式は今は画像の分類だけ)"
        )
    found = {
        "unit": "tv.Model",
        "params": {"name": model, "weights": chosen.name},
        "classes": [str(c) for c in chosen.meta["categories"]],
        "input": "image",
        "preprocess": {
            "resize": int(transform.resize_size[0]),
            "crop": int(transform.crop_size[0]),
            "interpolation": str(transform.interpolation.value),
            "mean": [float(v) for v in transform.mean],
            "std": [float(v) for v in transform.std],
        },
        "url": str(chosen.url),
        "file": Path(str(chosen.url)).name,
    }

    return found


# 既製のモデルの出どころ (ライブラリ) ごとの重みの情報の読み方
READY_LIBRARIES: dict[str, Callable[[str, str], dict]] = {
    "torchvision": torchvision_weights,
}


def check_ready(ready: Any) -> list[str]:
    """
    既製のモデルの一式の ready の形を確かめる関数

    Parameters
    ----------
    ready: Any
        inference.yaml の ready

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 出どころの名前, 入力の項目の種類, 前処理
    if not isinstance(ready, dict):
        return ["ready が辞書ではありません"]
    errors = []
    unknown = [str(k) for k in ready if k not in READY_KEYS]
    if unknown:
        errors.append(f"ready に知らない欄があります: {', '.join(unknown)}")
    for key in ("library", "model", "weights"):
        if not isinstance(ready.get(key), str) or not ready.get(key):
            errors.append(f"ready.{key} に名前の文字が要ります")
    if ready.get("input") not in READY_INPUTS:
        errors.append(
            f"ready.input は {', '.join(READY_INPUTS)} にしてください"
        )
    errors.extend(check_preprocess(ready.get("preprocess")))
    if ready.get("license") is not None and not isinstance(
        ready["license"], dict
    ):
        errors.append("ready.license が辞書ではありません")

    return errors


def check_preprocess(preprocess: Any) -> list[str]:
    """
    既製のモデルの前処理の形を確かめる関数

    Parameters
    ----------
    preprocess: Any
        ready.preprocess

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 大きさは正の整数, 補い方は決まった名前, 正規化は 3 つの数
    if not isinstance(preprocess, dict):
        return ["ready.preprocess が辞書ではありません"]
    errors = []
    unknown = [str(k) for k in preprocess if k not in PREPROCESS_KEYS]
    if unknown:
        errors.append(
            f"ready.preprocess に知らない欄があります: {', '.join(unknown)}"
        )
    for key in ("resize", "crop"):
        value = preprocess.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or (
            value < 1
        ):
            errors.append(f"ready.preprocess.{key} は正の整数にしてください")
    if preprocess.get("interpolation") not in INTERPOLATIONS:
        errors.append(
            "ready.preprocess.interpolation は "
            f"{', '.join(INTERPOLATIONS)} のどれかにしてください"
        )
    for key in ("mean", "std"):
        values = preprocess.get(key)
        if not isinstance(values, list) or len(values) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in values
        ):
            errors.append(
                f"ready.preprocess.{key} は 3 つの数 (チャンネルごと) に"
                "してください"
            )
        elif key == "std" and not all(v > 0 for v in values):
            errors.append("ready.preprocess.std は正の数にしてください")
    if not errors and preprocess["crop"] > preprocess["resize"]:
        errors.append(
            "ready.preprocess.crop は resize 以下にしてください"
        )

    return errors


def rgb_of(image: np.ndarray) -> np.ndarray:
    """
    読んだ画像を (高さ, 幅, 3) の RGB にする関数

    Parameters
    ----------
    image: np.ndarray
        (高さ, 幅) か (高さ, 幅, チャンネル) の uint8 の画像

    Returns
    ----------
    rgb: np.ndarray
        (高さ, 幅, 3) の uint8

    Raises
    ----------
    ValueError
        チャンネルの数が 1, 3, 4 でない場合
    """
    # 灰色は 3 つに重ね, 透明度の付いたものは透明度を除く (Pillow の
    # RGB への変換と同じ)
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"画像の形 {tuple(np.shape(image))} を RGB にできません"
        )
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    rgb = np.ascontiguousarray(array[:, :, :3].astype(np.uint8))

    return rgb


def prepare_ready(image: np.ndarray, preprocess: dict) -> np.ndarray:
    """
    1 枚の画像に既製のモデルの前処理をかける関数

    短い辺を resize にそろえ (縦横の比はそのまま. 長い辺は比で切り捨て.
    Pillow で補う), 真ん中の crop の正方形を切り出し, 255 で割って
    チャンネルごとに (値 - mean) / std にする (torchvision の画像の分類の
    前処理に Pillow の画像を渡したときと同じ順と求め方)

    Parameters
    ----------
    image: np.ndarray
        (高さ, 幅) か (高さ, 幅, チャンネル) の uint8 の画像
    preprocess: dict
        ready.preprocess

    Returns
    ----------
    array: np.ndarray
        (3, crop, crop) の float32
    """
    # 短い辺をそろえる (画像を読めるなら Pillow は入っている)
    from PIL import Image

    rgb = rgb_of(image)
    height, width = rgb.shape[:2]
    size = int(preprocess["resize"])
    if height <= width:
        shape = (size, int(size * width / height))
    else:
        shape = (int(size * height / width), size)
    if shape != (height, width):
        method = Image.Resampling[str(preprocess["interpolation"]).upper()]
        rgb = np.asarray(
            Image.fromarray(rgb).resize((shape[1], shape[0]), method)
        )

    # 真ん中の切り出しと正規化
    crop = int(preprocess["crop"])
    top = int(round((shape[0] - crop) / 2.0))
    left = int(round((shape[1] - crop) / 2.0))
    part = rgb[top:top + crop, left:left + crop].astype(np.float32) / 255.0
    mean = np.asarray(preprocess["mean"], dtype=np.float32)
    std = np.asarray(preprocess["std"], dtype=np.float32)
    array = ((part - mean) / std).transpose(2, 0, 1).astype(np.float32)

    return array


def weights_file(ready: dict) -> Path:
    """
    出どころから読む重みのファイルを置く場所を返す関数

    Parameters
    ----------
    ready: dict
        inference.yaml の ready

    Returns
    ----------
    path: Path
        PyTorch の置き場 (torch.hub.get_dir() の checkpoints) の, 出どころ
        のファイルの名前の場所
    """
    # torchvision は初回に取りに行き, ここに置いて次から読む
    path = Path(torch.hub.get_dir()) / "checkpoints" / str(
        ready.get("file") or ""
    )

    return path
