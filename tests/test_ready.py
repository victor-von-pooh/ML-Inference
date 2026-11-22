from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from ml_modelcontrol import register_unit
from torch import nn

from ml_inference import (
    READY_LIBRARIES, decide, describe_bundle, infer, load_bundle,
    make_ready_bundle, predict
)
from ml_inference.ready import prepare_ready

# 出どころの重みの倍率 (入れ替わった出どころを試すため変える)
SOURCE = {"scale": 1.0}

# 作り物の種類の名前の表と前処理
CLASSES = ["cat", "dog", "bird"]
PREPROCESS = {
    "resize": 10, "crop": 8, "interpolation": "bilinear",
    "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
}

# 作り物のライセンスの情報 (ネットワークを使わない)
LICENSE = {
    "unit": "test.ReadyNet", "model": "tiny", "hub_id": None,
    "revision": None, "pinned_revision": None, "license": None,
    "status": "unknown", "redistributable": False, "gated": False,
    "alert": True, "message": "ライセンスを確かめてください",
    "source": "テストの情報", "url": None,
}


class ReadyNet(nn.Module):
    def __init__(self, name: str = "tiny", weights: str = "V1"):
        # 親クラスのコンストラクタを呼び出す
        super(ReadyNet, self).__init__()

        # 出どころの重み (チャンネルの平均をそのまま種類の logit に)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3, 3)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(3) * SOURCE["scale"])
            self.fc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        チャンネルの平均から種類の logit を求める関数

        Parameters
        ----------
        x: torch.Tensor
            (件数, 3, 高さ, 幅) の画像

        Returns
        ----------
        out: torch.Tensor
            (件数, 3) の logit
        """
        # 平均して全結合
        out = self.fc(self.flatten(self.pool(x)))

        return out


register_unit(ReadyNet, name="test.ReadyNet", overwrite=True)


def tiny_weights(model: str, weights: str) -> dict:
    """
    作り物の出どころの重みの説明を返す関数

    Parameters
    ----------
    model: str
        モデルの名前
    weights: str
        重みの名前

    Returns
    ----------
    found: dict
        READY_LIBRARIES の関数の結果の形
    """
    # 名前は 1 つだけ
    if model != "tiny" or weights != "V1":
        raise ValueError(f"{model} の {weights} はありません")
    found = {
        "unit": "test.ReadyNet",
        "params": {"name": model, "weights": weights},
        "classes": list(CLASSES), "input": "image",
        "preprocess": dict(PREPROCESS),
        "url": "https://example.com/tiny-0123abcd.pth",
        "file": "tiny-0123abcd.pth",
    }

    return found


@pytest.fixture
def ready_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    作り物の出どころの既製のモデルの一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        出どころの差し替え

    Returns
    ----------
    path: Path
        一式
    """
    # 作り物の出どころを足して作る
    monkeypatch.setitem(READY_LIBRARIES, "test", tiny_weights)
    path = tmp_path / "tiny"
    make_ready_bundle(path, "tiny", "V1", library="test", license=LICENSE,
                      conclusion={"min_confidence": 0.5})

    return path


def write_images(folder: Path) -> Path:
    """
    赤, 緑, 青の画像と読めないファイルを書く関数

    Parameters
    ----------
    folder: Path
        書くディレクトリ

    Returns
    ----------
    folder: Path
        書いたディレクトリ
    """
    # 大きさの違う 3 枚と, 壊れた 1 枚
    image = pytest.importorskip("PIL.Image")
    folder.mkdir()
    for index, (name, size) in enumerate(
        [("a_red", (20, 12)), ("b_green", (9, 30)), ("c_blue", (10, 10))]
    ):
        array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        array[:, :, index] = 255
        image.fromarray(array).save(folder / f"{name}.png")
    (folder / "d_broken.png").write_bytes(b"not an image")

    return folder


def test_ready_bundle_pins_weights(ready_bundle: Path) -> None:
    """
    重みを写さずに版と hash を固定し, 説明が種類の名前を返すことを
    確かめる関数

    Parameters
    ----------
    ready_bundle: Path
        既製のモデルの一式

    Returns
    ----------
    None
    """
    # 版 6. 重みは空で, 出どころから読む記録
    info = yaml.safe_load((ready_bundle / "inference.yaml").read_text("utf-8"))
    assert info["format_version"] == 6 and info["dataset"] is None
    assert info["output"] == {"form": "class", "classes": CLASSES}
    record = info["model"]["pretrained"][0]
    assert record["place"] == "source" and record["pinned_revision"] == "V1"
    assert len(record["sha256"]) == 64 and record["keys"] == 2
    assert torch.load(ready_bundle / "final_model.pt",
                      weights_only=True) == {}
    assert info["ready"]["preprocess"] == PREPROCESS
    assert info["ready"]["license"]["alert"] is True
    assert any("商用で使う前に" in n for n in info["notes"])

    # 説明は種類の名前の表と, 出どころと置き場
    description = describe_bundle(ready_bundle)
    assert description["kind"] == "ready"
    assert description["task"] == "classification"
    assert description["classes"] == CLASSES
    assert description["ready"]["weights"] == "V1"
    assert description["ready"]["weights_file"].endswith("tiny-0123abcd.pth")
    assert description["ready"]["weights_cached"] is False
    assert description["pretrained"][0]["sha256"] == record["sha256"]


def test_ready_bundle_predicts_class_names(
    ready_bundle: Path, tmp_path: Path
) -> None:
    """
    画像を一式の前処理で読み, 種類の名前を予測することを確かめる関数

    Parameters
    ----------
    ready_bundle: Path
        既製のモデルの一式
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 色ごとの種類と, 読めない画像の理由
    folder = write_images(tmp_path / "images")
    result = predict(ready_bundle, folder)
    rows = result.set_index("path")
    assert list(rows.loc[["a_red.png", "b_green.png", "c_blue.png"],
                         "prediction"]) == CLASSES
    assert rows.loc["d_broken.png", "status"] == "skipped"
    assert "画像を読めません" in rows.loc["d_broken.png", "reason"]
    assert {f"p:{c}" for c in CLASSES} <= set(result.columns)

    # 結論の既定値も使え, 意思決定は使えない
    concluded = infer(ready_bundle, folder)
    assert concluded["summary"]["rows"] == 4
    assert concluded["rules"] == {"min_confidence": 0.5}
    with pytest.raises(ValueError, match="既製のモデルの一式"):
        decide(load_bundle(ready_bundle), result,
               {"type": "triage", "positive": "cat"})
    with pytest.raises(ValueError, match="options"):
        predict(ready_bundle, folder, options={"x": 1})


def test_ready_bundle_stops_on_other_weights(
    ready_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    出どころの重みの hash が違えば, 推論と一式づくりを止めることを
    確かめる関数

    Parameters
    ----------
    ready_bundle: Path
        既製のモデルの一式
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        出どころの差し替え

    Returns
    ----------
    None
    """
    # 入れ替わった出どころは誤り
    folder = write_images(tmp_path / "images")
    info = yaml.safe_load((ready_bundle / "inference.yaml").read_text("utf-8"))
    pinned = info["model"]["pretrained"][0]["sha256"]
    monkeypatch.setitem(SOURCE, "scale", 2.0)
    with pytest.raises(ValueError, match="hash が一式と違います"):
        predict(ready_bundle, folder)

    # 決めておいた hash と違えば作らない
    monkeypatch.setitem(READY_LIBRARIES, "test", tiny_weights)
    with pytest.raises(ValueError, match="決めておいたもの"):
        make_ready_bundle(tmp_path / "other", "tiny", "V1", library="test",
                          sha256=pinned, license=LICENSE)
    assert not (tmp_path / "other").exists()


def test_ready_bundle_checks(
    ready_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    出どころの名前の誤りと, 版 6 の形の誤りを確かめる関数

    Parameters
    ----------
    ready_bundle: Path
        既製のモデルの一式
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        出どころの差し替え

    Returns
    ----------
    None
    """
    # 無い出どころと重みの名前
    with pytest.raises(ValueError, match="出どころ nothing"):
        make_ready_bundle(tmp_path / "x", "tiny", "V1", library="nothing")
    monkeypatch.setitem(READY_LIBRARIES, "test", tiny_weights)
    with pytest.raises(ValueError, match="V2"):
        make_ready_bundle(tmp_path / "x", "tiny", "V2", library="test",
                          license=LICENSE)

    # ready の無い版 6, 前処理の誤り, 版 5 の ready
    file = ready_bundle / "inference.yaml"
    info = yaml.safe_load(file.read_text("utf-8"))
    cases: list[tuple[dict, str]] = [
        ({**info, "ready": None}, "ready"),
        ({**info, "ready": {**info["ready"], "preprocess": {
            **PREPROCESS, "crop": 12}}}, "crop は resize 以下"),
        ({**info, "format_version": 5}, "版 6 にしてください"),
        ({**info, "output": {"form": "class", "classes": ["a", "a"]}},
         "重ならない"),
        ({**info, "ready": {**info["ready"], "file": "../evil.pth"}},
         "ready.file"),
        ({**info, "ready": {**info["ready"], "url": 1}}, "ready.url"),
        ({**info, "ready": {**info["ready"], "library": "nothing"}},
         "ready.library"),
        ({**info, "ready": {**info["ready"], "weights": "V2"}},
         "ready.weights の V2"),
        ({**info, "ready": {**info["ready"], "model": "other"}},
         "ready.model の other"),
    ]
    for changed, message in cases:
        file.write_text(yaml.safe_dump(changed, allow_unicode=True),
                        encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_bundle(ready_bundle)


def test_prepare_ready_matches_definition() -> None:
    """
    前処理が短い辺をそろえ, 真ん中を切り出して正規化することを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 横長の画像の左右の端の色は切り出しで落ちる
    pytest.importorskip("PIL.Image")
    image = np.full((20, 60, 3), 51, dtype=np.uint8)
    image[:, :10] = 255
    image[:, -10:] = 255
    array = prepare_ready(image, PREPROCESS)
    assert array.shape == (3, 8, 8) and array.dtype == np.float32
    np.testing.assert_allclose(array, (51 / 255 - 0.5) / 0.5, atol=1e-6)

    # 灰色と透明度の付いた画像も RGB に
    gray = prepare_ready(np.full((8, 8), 255, dtype=np.uint8), PREPROCESS)
    np.testing.assert_allclose(gray, 1.0, atol=1e-6)
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    assert prepare_ready(rgba, PREPROCESS).shape == (3, 8, 8)


def test_torchvision_weights_reads_categories() -> None:
    """
    torchvision の ResNet18 の重みの説明から, 1000 の種類の名前と前処理を
    読むことを確かめる関数 (重みのファイルは取りに行かない)

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # torchvision が入っている場合だけ
    pytest.importorskip("torchvision")
    found: Any = READY_LIBRARIES["torchvision"]("resnet18", "DEFAULT")
    assert found["params"] == {"name": "resnet18", "weights": "IMAGENET1K_V1"}
    assert len(found["classes"]) == 1000 and found["classes"][0] == "tench"
    assert found["preprocess"]["resize"] == 256
    assert found["preprocess"]["crop"] == 224
    assert found["file"] == "resnet18-f37072fd.pth"
