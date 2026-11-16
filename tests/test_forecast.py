from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import linear_spec
from ml_eda import export_dataset, load_data
from ml_modelcontrol import build_model

from ml_inference import forecast, infer, make_bundle, predict


def test_forecast_steps_per_series(tmp_path: Path) -> None:
    """
    系列ごとに, 予測した値を次の入力に使って先の行を順に埋めることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2 つの系列で y が 1 ずつ増える (a は 0 から, b は 100 から)
    frame = pd.DataFrame({
        "s": ["a"] * 30 + ["b"] * 30,
        "y": np.r_[np.arange(30), 100 + np.arange(30)].astype(float),
    })
    source = tmp_path / "s.csv"
    frame.to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "y", "task": "forecasting",
        "steps": [
            {"op": "lag", "columns": ["y"], "lags": [1], "by": ["s"]},
            {"op": "drop_missing"},
            {"op": "drop_columns", "columns": ["s"]},
        ],
        "split": {"method": "random", "valid": 0.2, "seed": 0},
    }
    export_dataset(load_data(source), recipe, tmp_path / "ds")

    # 次の値は前の値 + 1 のモデル
    spec = linear_spec(1, 1)
    model = build_model(spec)
    with torch.no_grad():
        model.nodes["head"].weight.fill_(1.0)
        model.nodes["head"].bias.fill_(1.0)
    torch.save(model.state_dict(), tmp_path / "w.pt")
    make_bundle(tmp_path / "bundle", spec, tmp_path / "ds", tmp_path / "w.pt")

    # 各系列の後ろに 3 つと 2 つの先の行 (系列が交互に並んでも系列ごと)
    future = pd.DataFrame({"s": ["a", "b", "a", "b", "a"], "y": np.nan})
    given = pd.concat([frame, future], ignore_index=True)
    result = forecast(tmp_path / "bundle", given)
    tail = result.iloc[60:]
    assert list(tail["step"]) == [1, 1, 2, 2, 3]
    assert np.allclose(tail["prediction"], [30, 130, 31, 131, 32])
    assert np.allclose(tail["y"], [30, 130, 31, 131, 32])
    assert result["step"].iloc[:60].isna().all()
    assert (tail["status"] == "ok").all()

    # infer では先の行だけの結論
    concluded = infer(tmp_path / "bundle", given, forecast=True)
    assert len(concluded["frame"]) == 5
    assert concluded["sentences"][0] == "5 件のうち 5 件を予測"


def test_forecast_edge_cases(tmp_path: Path) -> None:
    """
    過去の途中の欠損, 数でない予測, 文字の目的の列を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 前の値 + x の系列 (x が無限大なら予測も無限大)
    frame = pd.DataFrame({"x": np.ones(30), "y": np.arange(30.0)})
    source = tmp_path / "s.csv"
    frame.to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "y", "task": "forecasting",
        "steps": [{"op": "lag", "columns": ["y"], "lags": [1]},
                  {"op": "drop_missing"}],
        "split": {"method": "random", "valid": 0.2, "seed": 0},
    }
    export_dataset(load_data(source), recipe, tmp_path / "ds")
    spec = linear_spec(2, 1)
    model = build_model(spec)
    with torch.no_grad():
        model.nodes["head"].weight.copy_(torch.tensor([[1.0, 1.0]]))
        model.nodes["head"].bias.zero_()
    torch.save(model.state_dict(), tmp_path / "w.pt")
    make_bundle(tmp_path / "bundle", spec, tmp_path / "ds", tmp_path / "w.pt")

    # 過去の途中の空の行は先の行にしない
    given = pd.concat([frame, pd.DataFrame({"x": [1.0, 1.0], "y": np.nan})],
                      ignore_index=True)
    given.loc[10, "y"] = np.nan
    result = forecast(tmp_path / "bundle", given)
    assert np.isnan(result.loc[10, "step"])
    assert list(result["step"].iloc[30:]) == [1, 2]
    assert np.allclose(result["prediction"].iloc[30:], [30, 31])

    # 無限大の入力で止まり, 後ろの先の行は理由付き
    given = pd.concat(
        [frame, pd.DataFrame({"x": [np.inf, 1.0], "y": np.nan})],
        ignore_index=True
    )
    result = forecast(tmp_path / "bundle", given)
    assert list(result["status"].iloc[30:]) == ["skipped", "skipped"]
    assert "数ではありません" in result.loc[30, "reason"]
    assert "揃いません" in result.loc[31, "reason"]
    plain = predict(tmp_path / "bundle", given)
    assert plain.loc[30, "status"] == "skipped"

    # 数にできない目的の列
    given = frame.astype({"y": str})
    given.loc[3, "y"] = "12円"
    with pytest.raises(ValueError, match="数にできない"):
        forecast(tmp_path / "bundle", given)
