from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ml_eda import export_dataset, load_data


def linear_spec(width: int, outputs: int, name: str = "x") -> dict:
    """
    全結合の 1 層のモデルの定義を返す関数

    Parameters
    ----------
    width: int
        入力の列の数
    outputs: int
        出力の幅
    name: str = "x"
        入力の名前

    Returns
    ----------
    spec: dict
        ML-ModelControl の定義
    """
    # 入力から全結合の 1 層
    spec = {
        "format_version": 1,
        "name": "linear",
        "inputs": {name: {"shape": [4, width]}},
        "nodes": {
            "head": {
                "unit": "nn.Linear",
                "params": {"in_features": width, "out_features": outputs},
                "inputs": name,
            }
        },
        "outputs": {"logits": "head"},
    }

    return spec


def flat_spec(shape: list[int], classes: int) -> dict:
    """
    中身を平らにして全結合の 1 層で分類するモデルの定義を返す関数

    Parameters
    ----------
    shape: list[int]
        1 件の中身の形
    classes: int
        クラスの数

    Returns
    ----------
    spec: dict
        ML-ModelControl の定義
    """
    # 平らにしてから全結合
    width = int(np.prod(shape))
    spec = {
        "format_version": 1,
        "inputs": {"x": {"shape": [2, *shape]}},
        "nodes": {
            "flat": {"unit": "nn.Flatten", "inputs": "x"},
            "head": {"unit": "nn.Linear",
                     "params": {"in_features": width,
                                "out_features": classes},
                     "inputs": "flat"},
        },
        "outputs": {"logits": "head"},
    }

    return spec


def save_weights(spec: dict, path: Path, seed: int = 0) -> Path:
    """
    定義から組み立てたモデルの重みを保存する関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    path: Path
        保存先
    seed: int = 0
        重みの初期値の乱数の種

    Returns
    ----------
    path: Path
        保存先
    """
    from ml_modelcontrol import build_model

    # 乱数の初期値のままの重み
    model = build_model(spec, seed=seed)
    torch.save(model.state_dict(), path)

    return path


@pytest.fixture
def table_csv(tmp_path: Path) -> Path:
    """
    3 クラスの分類の表を CSV に書く関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    path: Path
        CSV のファイル
    """
    # 数の列 2 つ, カテゴリの列 1 つ, クラス
    rng = np.random.default_rng(0)
    count = 60
    frame = pd.DataFrame({
        "id": np.arange(count),
        "a": rng.normal(size=count),
        "b": rng.normal(10, 3, size=count),
        "color": rng.choice(["赤", "青"], size=count),
        "label": rng.choice(["犬", "猫", "鳥"], size=count),
    })
    path = tmp_path / "table.csv"
    frame.to_csv(path, index=False)

    return path


@pytest.fixture
def table_dataset(tmp_path: Path, table_csv: Path) -> Path:
    """
    表の分類のデータセットを書き出す関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_csv: Path
        元の CSV

    Returns
    ----------
    path: Path
        データセットのディレクトリ
    """
    # id を落とし, ワンホットと標準化
    recipe = {
        "format_version": 1,
        "target": "label",
        "task": "classification",
        "steps": [
            {"op": "drop_columns", "columns": ["id"]},
            {"op": "one_hot", "columns": ["color"]},
            {"op": "standardize", "columns": "auto"},
        ],
        "split": {"method": "random", "valid": 0.2, "test": 0.1, "seed": 0},
    }
    path = tmp_path / "dataset"
    export_dataset(load_data(table_csv), recipe, path)

    return path


@pytest.fixture
def regression_dataset(tmp_path: Path) -> Path:
    """
    目的の列も標準化した回帰のデータセットを書き出す関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    path: Path
        データセットのディレクトリ
    """
    # 目的の列 y を対数と標準化で変換
    rng = np.random.default_rng(1)
    count = 50
    frame = pd.DataFrame({
        "shop": rng.choice(["東", "西"], size=count),
        "x": rng.normal(size=count),
        "y": rng.uniform(100, 1000, size=count),
    })
    source = tmp_path / "sales.csv"
    frame.to_csv(source, index=False)
    recipe = {
        "format_version": 1,
        "target": "y",
        "task": "regression",
        "steps": [
            {"op": "drop_columns", "columns": ["shop"]},
            {"op": "log", "columns": ["y"]},
            {"op": "standardize", "columns": ["x", "y"]},
        ],
        "split": {"method": "random", "valid": 0.2, "test": 0.1, "seed": 0},
    }
    path = tmp_path / "regression"
    export_dataset(load_data(source), recipe, path)

    return path
