from pathlib import Path

import pytest
import yaml
from ml_eda import export_capture_clouds, export_image_masks, load_items
from ml_eda.synthetic import write_image_masks, write_shape_captures

from ml_inference import POINT_KINDS, describe_items
from ml_inference.items import resolve_items

# 画像と深度の読み書き (任意の依存. 無い環境では飛ばす)
pytest.importorskip("PIL.Image")


def test_mask_description(tmp_path: Path) -> None:
    """
    大きさをそろえた画像と領域の版 3 のデータセットの説明と items の
    既定を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 12 組を 12 x 16 にそろえて書き出す
    raw = write_image_masks(tmp_path / "raw", count=12, seed=0)
    dataset = tmp_path / "masks"
    steps = [{"op": "resize", "width": 16, "height": 12}]
    export_image_masks(raw["images"], raw["masks"], dataset,
                       classes=raw["classes"], origin={"type": "synthetic"},
                       item_steps={"image": steps})
    description = describe_items(str(dataset))
    assert description["format_version"] == 3
    assert description["role"] == "predict"
    assert description["target"]["kind"] == "mask"
    assert description["target"]["entries"] == ["mask"]
    assert description["target"]["origin"]["type"] == "synthetic"
    rows = {row["name"]: row for row in description["entries"]}
    assert rows["image"]["kind"] == "image"
    assert rows["image"]["role"] == "input"
    assert rows["image"]["steps"] == steps
    assert not rows["image"]["points"]
    assert sum(description["splits"].values()) == 12
    assert [c["value"] for c in description["classes"]] == [0, 1, 2]
    assert not description["normalized"]

    # items の既定は make_bundle が省いた欄に使うものと同じ
    resolved, _ = resolve_items(load_items(dataset), "mask", None)
    assert description["items"] == resolved
    assert description["problem"] is None


def test_cloud_description(tmp_path: Path) -> None:
    """
    撮影と点群の版 2 のデータセットで, 点を選ぶ項目と点の数を空にした
    items の既定を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 乱数の形 8 つ
    raw = write_shape_captures(tmp_path / "raw", count=8, seed=0,
                               points=400, views=6)
    dataset = tmp_path / "clouds"
    export_capture_clouds(raw["captures"], raw["clouds"], dataset,
                          split={"method": "random", "valid": 0.25,
                                 "test": 0.0, "seed": 0})
    assert POINT_KINDS == ("capture", "point_cloud")
    description = describe_items(dataset)
    rows = {row["name"]: row for row in description["entries"]}
    assert rows["capture"]["points"] and rows["cloud"]["points"]
    assert description["normalized"]
    assert description["classes"] == []
    assert description["items"] == {
        "input": "capture", "target": "cloud", "prepare": "capture_points",
        "options": {}, "values": None, "restore": "shift",
    }
    assert description["problem"] is None

    # 点の数を足すと make_bundle の items と同じ
    options = {"points": 64}
    resolved, _ = resolve_items(load_items(dataset), "point_cloud",
                                {"options": options})
    assert resolved == {**description["items"], "options": options}


def test_description_problems(tmp_path: Path) -> None:
    """
    一式を作れないデータセットは理由を problem に返し, 読めないデータ
    セットは誤りにすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 予測するでないデータセットは説明だけ
    raw = write_image_masks(tmp_path / "raw", count=12, seed=0)
    dataset = tmp_path / "masks"
    export_image_masks(raw["images"], raw["masks"], dataset,
                       classes=raw["classes"])
    file = dataset / "dataset.yaml"
    info = yaml.safe_load(file.read_text(encoding="utf-8"))
    file.write_text(yaml.safe_dump({**info, "role": "generate"},
                                   allow_unicode=True), encoding="utf-8")
    description = describe_items(dataset)
    assert description["items"] is None
    assert "予測する" in description["problem"]
    assert description["entries"]

    # dataset.yaml の無い場所
    with pytest.raises(ValueError, match="dataset.yaml"):
        describe_items(tmp_path / "raw")
