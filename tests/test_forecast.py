from pathlib import Path

import numpy as np
import pandas as pd
import torch
from conftest import linear_spec
from ml_eda import export_dataset, load_data
from ml_modelcontrol import build_model

from ml_inference import forecast, make_bundle


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
