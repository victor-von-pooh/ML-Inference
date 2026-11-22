from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import flat_spec, save_weights
from ml_eda import export_dataset, load_data, load_dataset
from ml_modelcontrol import build_model

from ml_inference import make_bundle, predict

# 画像の読み書き (任意の依存. 無い環境では画像のテストを飛ばす)
Image = pytest.importorskip("PIL.Image")


def test_images(tmp_path: Path) -> None:
    """
    画像の一式で, クラスのフォルダの無い新しい画像のディレクトリを推論
    できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 明るい画像と暗い画像 (大きさは 2 通り)
    rng = np.random.default_rng(0)
    for label, base in (("light", 200), ("dark", 50)):
        (tmp_path / "pets" / label).mkdir(parents=True)
        for index in range(6):
            size = (12, 10) if index == 0 else (8, 8)
            pixels = np.clip(
                base + rng.normal(0, 10, (*size[::-1], 3)), 0, 255
            ).astype(np.uint8)
            Image.fromarray(pixels).save(
                tmp_path / "pets" / label / f"{index}.png"
            )
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "item_steps": [{"op": "resize", "width": 8, "height": 8},
                       {"op": "normalize"}],
        "split": {"method": "stratified", "valid": 0.34, "seed": 0},
    }
    export_dataset(load_data(tmp_path / "pets"), recipe, tmp_path / "ds")
    spec = flat_spec([3, 8, 8], 2)
    weights = save_weights(spec, tmp_path / "w.pt")
    make_bundle(tmp_path / "bundle", spec, tmp_path / "ds", weights)

    # クラスのフォルダの無い新しい画像 (大きさの違うものと読めないもの)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    Image.fromarray(np.full((8, 8, 3), 210, np.uint8)).save(fresh / "a.png")
    Image.fromarray(np.full((20, 16, 3), 40, np.uint8)).save(fresh / "b.png")
    (fresh / "c.png").write_bytes(b"not an image")
    result = predict(tmp_path / "bundle", fresh)
    names = result["path"].map(lambda value: Path(value).name)
    assert list(result["status"]) == ["ok", "ok", "skipped"]
    assert names.iloc[2] == "c.png" and "読めません" in result["reason"][2]
    ok = result[result["status"] == "ok"]
    assert np.allclose(ok[["p:dark", "p:light"]].sum(axis=1), 1.0)

    # 手で前処理してモデルに通した確率と同じ
    dataset = load_dataset(tmp_path / "ds")
    image = np.asarray(Image.open(fresh / "a.png").convert("RGB"))
    x = torch.from_numpy(dataset.transform_item(image)[None])
    model = build_model(spec)
    model.load_state_dict(torch.load(weights))
    with torch.no_grad():
        expected = model(x).softmax(dim=1)[0].numpy()
    got = ok.loc[names == "a.png", ["p:dark", "p:light"]].to_numpy()[0]
    assert np.allclose(got, expected, atol=1e-5)


def test_pointclouds(tmp_path: Path) -> None:
    """
    点群の一式で, 点の数の違う新しい点群を推論できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 平らな点群と丸い点群
    rng = np.random.default_rng(1)

    def cloud(kind: str, count: int) -> np.ndarray:
        """
        平らか丸い点群を作る関数

        Parameters
        ----------
        kind: str
            flat か ball
        count: int
            点の数

        Returns
        ----------
        points: np.ndarray
            (点の数, 3) の座標
        """
        # 平らは z が 0, 丸いは球の上
        points = rng.normal(size=(count, 3))
        if kind == "flat":
            points[:, 2] = 0
        else:
            points /= np.linalg.norm(points, axis=1, keepdims=True)

        return points

    for kind in ("flat", "ball"):
        (tmp_path / "shapes" / kind).mkdir(parents=True)
        for index in range(6):
            np.savetxt(tmp_path / "shapes" / kind / f"{index}.xyz",
                       cloud(kind, 40 + index))
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "item_steps": [{"op": "sample_points", "points": 32},
                       {"op": "normalize_cloud"}],
        "split": {"method": "stratified", "valid": 0.34, "seed": 0},
    }
    export_dataset(load_data(tmp_path / "shapes"), recipe, tmp_path / "ds")
    spec = flat_spec([32, 3], 2)
    weights = save_weights(spec, tmp_path / "w.pt")
    make_bundle(tmp_path / "bundle", spec, tmp_path / "ds", weights)

    # 点の数の違う新しい点群のディレクトリ
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    np.savetxt(fresh / "a.xyz", cloud("flat", 100))
    np.savetxt(fresh / "b.xyz", cloud("ball", 10))
    result = predict(tmp_path / "bundle", fresh)
    assert list(result["status"]) == ["ok", "ok"]
    assert set(result["prediction"]) <= {"flat", "ball"}
