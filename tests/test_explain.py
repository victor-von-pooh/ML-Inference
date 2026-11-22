from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import linear_spec, save_weights
from ml_eda import export_dataset, load_data, load_dataset
from ml_modelcontrol import build_model

from ml_inference import explain, explain_rows, make_bundle


def linear_bundle(tmp_path: Path) -> pd.DataFrame:
    """
    y = 3 a (b は効かず, id は前処理で落とす) の回帰の一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ (bundle に一式を作る)

    Returns
    ----------
    frame: pd.DataFrame
        学習に使ったデータ (目的の列 y を含む)
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

    return frame


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
    # y = 3 a の一式
    frame = linear_bundle(tmp_path)

    # a が最も効き, b と id は 0
    result = explain(tmp_path / "bundle", frame.drop(columns=["y"]))
    effects = {c["column"]: c["effect"] for c in result["columns"]}
    assert result["measure"] == "value" and result["rows"] == 200
    assert result["columns"][0]["column"] == "a" and effects["a"] > 1
    assert effects["b"] == 0 and effects["id"] == 0


def test_row_effects(tmp_path: Path) -> None:
    """
    行ごとの理由が, 効く列ではその行の値の向きと大きさになり, 効かない列と
    落とす列では 0 になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # y = 3 a なので, a の効きは 3 (a - 平均) に近い
    frame = linear_bundle(tmp_path).drop(columns=["y"])
    result = explain_rows(tmp_path / "bundle", frame, repeats=40)
    table = result["frame"]
    assert result["measure"] == "value" and len(table) == 200
    assert result["columns"] == ["effect:id", "effect:a", "effect:b"]
    assert result["errors"] == {}
    # (並べ替えの揺れは 1 行あたり 3 / √40 ほど)
    expected = 3 * (frame["a"] - frame["a"].mean())
    gap = (table["effect:a"] - expected).abs()
    assert gap.mean() < 0.6 and np.corrcoef(
        table["effect:a"], expected
    )[0, 1] > 0.95
    far = expected.abs() > 2
    assert (np.sign(table["effect:a"][far]) == np.sign(expected[far])).all()
    assert (table["effect:b"] == 0).all() and (table["effect:id"] == 0).all()

    # 理由の文は効く列だけで, 向きの印が付く
    top = int(np.argmax(frame["a"].to_numpy()))
    assert table["reason"][top].startswith("a (+")
    assert table["reason"][int(np.argmin(frame["a"]))].startswith("a (-")

    # 回数は 1 以上の整数
    with pytest.raises(ValueError, match="repeats"):
        explain_rows(tmp_path / "bundle", frame, repeats=0)

    # 同じシードなら同じ値
    again = explain_rows(tmp_path / "bundle", frame, repeats=40)["frame"]
    assert again["effect:a"].equals(table["effect:a"])


def test_row_effects_with_lagged_target(tmp_path: Path) -> None:
    """
    過去の値の列を作る一式で, 目的の列を過去の値として行ごとに求め,
    予測できない行は空にすることを確かめる関数

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
    frame = pd.DataFrame({"y": np.cumsum(rng.normal(size=60)),
                          "noise": rng.normal(size=60)})
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

    # 目的の列は「過去の値」の名前. 最初の行は予測しないので空
    result = explain_rows(tmp_path / "bundle", frame, repeats=3)
    table = result["frame"]
    assert "effect:y (過去の値)" in result["columns"]
    assert table["status"].iloc[0] != "ok"
    assert table["effect:y (過去の値)"].iloc[1:].notna().all()
    assert np.isnan(table["effect:y (過去の値)"].iloc[0])
    assert pd.isna(table["reason"].iloc[0])


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

    # 目的の列は「過去の値」の名前で並ぶ. 行は最後の続いた 50 行 (最初の
    # 行は過去の値が無い)
    result = explain(tmp_path / "bundle", frame, max_rows=50)
    names = [c["column"] for c in result["columns"]]
    assert "y (過去の値)" in names
    assert result["rows"] == 49


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
    with pytest.raises(ValueError, match="行ごとの理由"):
        explain_rows(tmp_path / "bundle", source)
