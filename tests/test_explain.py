from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import linear_spec, save_weights
from ml_eda import export_dataset, load_data, load_dataset
from ml_modelcontrol import build_model

from ml_inference import explain, make_bundle


def test_effects_of_columns(tmp_path: Path) -> None:
    """
    効く列は効きが大きく, 前処理で落とす列は 0 になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # y は a だけで決まる. id は前処理で落とす
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "id": np.arange(200), "a": rng.normal(size=200),
        "b": rng.normal(size=200),
    })
    frame["y"] = 3 * frame["a"]
    source = tmp_path / "d.csv"
    frame.to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "y", "task": "regression",
        "steps": [{"op": "drop_columns", "columns": ["id"]}],
        "split": {"method": "random", "valid": 0.2, "seed": 0},
    }
    export_dataset(load_data(source), recipe, tmp_path / "ds")

    # 重みを a に 3, b に 0 にしたモデル
    spec = linear_spec(2, 1)
    model = build_model(spec)
    with torch.no_grad():
        model.nodes["head"].weight.copy_(torch.tensor([[3.0, 0.0]]))
        model.nodes["head"].bias.zero_()
    torch.save(model.state_dict(), tmp_path / "w.pt")
    make_bundle(tmp_path / "bundle", spec, tmp_path / "ds", tmp_path / "w.pt")

    # a が最も効き, b と id は 0
    result = explain(tmp_path / "bundle", frame.drop(columns=["y"]))
    effects = {c["column"]: c["effect"] for c in result["columns"]}
    assert result["measure"] == "value" and result["rows"] == 200
    assert result["columns"][0]["column"] == "a" and effects["a"] > 1
    assert effects["b"] == 0 and effects["id"] == 0


def test_effects_with_lagged_target(tmp_path: Path) -> None:
    """
    過去の値の列を作る一式で, 目的の列を過去の値として求めることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 前の行の y から次の y を当てる系列
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"y": np.cumsum(rng.normal(size=120)),
                          "noise": rng.normal(size=120)})
    source = tmp_path / "s.csv"
    frame.to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "y", "task": "regression",
        "steps": [{"op": "lag", "columns": ["y"], "lags": [1]},
                  {"op": "drop_missing"}],
        "split": {"method": "random", "valid": 0.2, "seed": 0},
    }
    export_dataset(load_data(source), recipe, tmp_path / "ds")
    width = len(load_dataset(tmp_path / "ds").features)
    spec = linear_spec(width, 1)
    weights = save_weights(spec, tmp_path / "w.pt")
    make_bundle(tmp_path / "bundle", spec, tmp_path / "ds", weights)

    # 目的の列は「過去の値」の名前で並ぶ. 行は選ばずに全て
    result = explain(tmp_path / "bundle", frame, max_rows=50)
    names = [c["column"] for c in result["columns"]]
    assert "y (過去の値)" in names
    assert result["rows"] == 119


def test_text_bundle_is_rejected(tmp_path: Path) -> None:
    """
    文の一式では列の効きを求めないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 文字の語彙の文の一式
    source = tmp_path / "t.csv"
    pd.DataFrame({"text": ["良い", "悪い"] * 6,
                  "label": ["良", "悪"] * 6}).to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [],
        "item_steps": [{"op": "vocabulary", "unit": "char", "length": 2}],
        "split": {"method": "random", "valid": 0.25, "seed": 0},
    }
    export_dataset(load_data(source, text_column="text"), recipe,
                   tmp_path / "ds")
    vocabulary = len(next(
        s["state"]["vocabulary"]
        for s in load_dataset(tmp_path / "ds").recipe["item_steps"]
    ))
    spec = {
        "format_version": 1,
        "inputs": {"ids": {"shape": [2, 2], "dtype": "int64"}},
        "nodes": {
            "bag": {"unit": "nn.EmbeddingBag",
                    "params": {"num_embeddings": vocabulary,
                               "embedding_dim": 2},
                    "inputs": {"input": "ids"}},
            "head": {"unit": "nn.Linear",
                     "params": {"in_features": 2, "out_features": 2},
                     "inputs": "bag"},
        },
        "outputs": {"logits": "head"},
    }
    weights = save_weights(spec, tmp_path / "w.pt")
    make_bundle(tmp_path / "bundle", spec, tmp_path / "ds", weights)
    with pytest.raises(ValueError, match="列の効き"):
        explain(tmp_path / "bundle", source)
