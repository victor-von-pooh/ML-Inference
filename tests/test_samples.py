from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_inference import infer, load_bundle, make_sample_bundle
from ml_inference.samples import sample_spec


def sample_frames(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """
    表の分類, 表の回帰, 文, 時系列の見本のデータを作る関数

    Parameters
    ----------
    rng: np.random.Generator
        乱数

    Returns
    ----------
    frames: dict[str, pd.DataFrame]
        見本の種類から表
    """
    # 分類は x の符号, 回帰は 50 x + 300, 文は言葉, 時系列は店ごとの週の波
    x = rng.normal(size=300)
    words = rng.choice(["良い", "悪い"], size=300)
    days = pd.date_range("2025-01-01", periods=150).strftime("%Y-%m-%d")
    daily = pd.concat([
        pd.DataFrame({
            "date": days, "shop": shop,
            "sales": level + 10 * np.sin(np.arange(150) * 2 * np.pi / 7)
            + rng.normal(0, 1, 150),
        }) for shop, level in (("a", 100.0), ("b", 60.0))
    ], ignore_index=True)
    frames = {
        "class": pd.DataFrame({
            "x": x, "y": np.where(x > 0, "上", "下"),
        }),
        "value": pd.DataFrame({
            "x": x, "y": 50 * x + 300 + rng.normal(0, 1, 300),
        }),
        "text": pd.DataFrame({
            "review": [f"{word}です" for word in words], "label": words,
        }),
        "series": daily,
    }

    return frames


def test_make_sample_bundle(tmp_path: Path) -> None:
    """
    表の分類, 表の回帰, 文, 時系列の見本の一式を作り, 推論できることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 見本の種類ごとのレシピと読み込み
    frames = sample_frames(np.random.default_rng(0))
    split = {"method": "random", "valid": 0.2, "seed": 0}
    settings = {
        "class": ({"target": "y", "task": "classification", "steps": [],
                   "split": split}, {}),
        "value": ({"target": "y", "task": "regression",
                   "steps": [{"op": "standardize", "columns": "auto"}],
                   "split": split}, {}),
        "text": ({"target": "label", "task": "classification", "steps": [],
                  "item_steps": [{"op": "vocabulary", "unit": "char",
                                  "length": 8}],
                  "split": split}, {"text_column": "review"}),
        "series": ({"target": "sales", "task": "forecasting", "steps": [
            {"op": "lag", "columns": ["sales"], "lags": [1, 7],
             "by": ["shop"]},
            {"op": "drop_missing"},
            {"op": "one_hot", "columns": ["shop"]},
            {"op": "standardize", "columns": "auto"},
        ], "split": {"method": "time", "valid": 0.2}}, {}),
    }
    for kind, (recipe, options) in settings.items():
        source = tmp_path / f"{kind}.csv"
        frames[kind].to_csv(source, index=False)
        info = make_sample_bundle(
            tmp_path / kind, source, recipe, options=options,
            name=f"見本 {kind}"
        )
        assert info["name"] == f"見本 {kind}"
        bundle = load_bundle(tmp_path / kind)
        assert bundle.dataset.info["source"]["path"] == str(source)
        result = infer(bundle, frames[kind].head(20))
        assert (result["frame"]["status"] == "ok").sum() > 0

        # 学習した小さなモデルの当たり具合 (回帰と予測は元の単位に戻した
        # 平均絶対誤差)
        check = info["check"]
        if recipe["task"] == "classification":
            assert check["value"] > 0.9
            assert info["model"]["target_scale"] is None
        else:
            assert info["model"]["target_scale"]["scale"] > 0
            assert check["value"] < 0.3 * frames[kind][
                recipe["target"]
            ].std()

    # 一式を置き換えるのは overwrite のときだけ
    with pytest.raises(FileExistsError):
        make_sample_bundle(tmp_path / "class", tmp_path / "class.csv",
                           settings["class"][0])
    make_sample_bundle(tmp_path / "class", tmp_path / "class.csv",
                       settings["class"][0], overwrite=True, steps=5)

    # 目的の値が一定なら学習できないので誤り (NaN の一式を作らない)
    flat = tmp_path / "flat.csv"
    frames["value"].assign(y=5.0).to_csv(flat, index=False)
    with pytest.raises(ValueError, match="一定"):
        make_sample_bundle(tmp_path / "flat", flat, settings["value"][0])
    assert not (tmp_path / "flat").exists()


def test_sample_needs_table_or_text() -> None:
    """
    表と文でないデータセットでは見本のモデルを作らないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 画像のデータセット (説明だけ)
    dataset = SimpleNamespace(info={"kind": "image"})
    with pytest.raises(ValueError, match="表, 時系列, 文"):
        sample_spec(dataset, 2)
