from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import linear_spec
from ml_eda import export_dataset, load_data
from ml_modelcontrol import build_model
from test_decision import classifier_bundle, regression_bundle

from ml_inference import (
    decide, describe_bundle, forecast, load_bundle, make_bundle, predict,
    remeasure, restore_uncertainty
)


def test_remeasure_regression(tmp_path: Path) -> None:
    """
    外れ方の広がった実績で測り直すと, 前の記録の当たり具合を返し, 一式の
    外れ方と量の判断が変わることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 誤りが比で ±40 % に広がった実績
    path = regression_bundle(tmp_path)
    rng = np.random.default_rng(5)
    x = rng.uniform(1, 10, 300)
    recent = pd.DataFrame({"x": x, "y": 10 * x * rng.uniform(0.6, 1.4, 300)})
    bundle = load_bundle(path)
    decision = {"type": "quantity", "under_cost": 9, "over_cost": 1}
    before = decide(bundle, predict(bundle, recent), decision)

    # 書き戻さない場合の一式 (変わらない)
    checked = remeasure(path, recent, write=False)
    assert checked["rows"] == 300 and checked["written"] is False
    assert checked["before"]["coverage"]["80"] < 0.75
    assert "外れ方を小さく見積もっていた" in checked["sentences"][0]
    assert load_bundle(path).info["uncertainty"]["split"] == "valid"

    # 書き戻した後の実績の分割の記録と, 9 : 1 の量の増え
    result = remeasure(path, recent)
    saved = load_bundle(path).info["uncertainty"]
    assert saved["split"] == "recent" and saved["rows"] == 300
    assert saved["previous"]["split"] == "valid"
    assert "test_below" not in saved and "coverage" not in saved
    assert result["uncertainty"]["kind"] == saved["kind"]
    bundle = load_bundle(path)
    after = decide(bundle, predict(bundle, recent), decision)
    assert after["frame"]["level"].sum() > before["frame"]["level"].sum()
    assert any("300 行で測り直した" in note for note in after["notes"])
    summary = describe_bundle(path)["uncertainty"]
    assert summary["split"] == "recent" and summary["rows"] == 300
    assert summary["measured_at"] and "absolute" not in summary
    assert summary["restorable"] is True

    # 2 回測り直しても, 戻すのは一式を作ったときの記録
    first = load_bundle(path).info["uncertainty"]["original"]
    remeasure(path, recent.head(100))
    assert load_bundle(path).info["uncertainty"]["original"] == first
    restored = restore_uncertainty(path)
    assert restored["split"] == "valid" and "test_below" in restored
    assert load_bundle(path).info["uncertainty"] == first
    with pytest.raises(ValueError, match="戻せる記録"):
        restore_uncertainty(path)


def test_remeasure_count_and_errors(tmp_path: Path) -> None:
    """
    数える行の印と, 測れない場合の誤りを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 印のある行と目的の値の分かる行だけの測り直し
    path = regression_bundle(tmp_path)
    x = np.linspace(1, 10, 50)
    recent = pd.DataFrame({"x": x, "y": 10 * x})
    recent.loc[:4, "y"] = np.nan
    count = np.arange(50) >= 10
    result = remeasure(path, recent, count=count, write=False)
    assert result["rows"] == 40

    # 少なすぎる行と, 目的の列の無いデータ
    with pytest.raises(ValueError, match="20 行以上"):
        remeasure(path, recent.head(15), write=False)
    with pytest.raises(ValueError, match="目的の列"):
        remeasure(path, recent.drop(columns="y"), write=False)
    with pytest.raises(ValueError, match="印の数"):
        remeasure(path, recent, count=count[:10], write=False)

    # 比の形の記録では, 予測が 0 以下の行を確かめから除く
    negative = pd.DataFrame({"x": np.linspace(-5, 5, 60),
                             "y": np.linspace(-50, 50, 60)})
    checked = remeasure(path, negative, write=False)
    assert checked["before"]["excluded"] == 30
    assert any("0 以下の 30 行" in s for s in checked["sentences"])


def test_remeasure_classification(tmp_path: Path) -> None:
    """
    分類で, 確率の当たり具合を測り直し, 前の記録のずれと比べることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 当たりの確率が逆向きになった実績 (確率のずれが大きい)
    path = classifier_bundle(tmp_path)
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    label = np.where(rng.uniform(size=400) < 1 / (1 + np.exp(2 * x)),
                     "yes", "no")
    result = remeasure(path, pd.DataFrame({"x": x, "label": label}))
    saved = load_bundle(path).info["uncertainty"]
    assert saved["split"] == "recent" and saved["calibration_of"] == "yes"
    assert saved["calibration"]["ece"] > 0.2
    assert result["before"]["ece"] < saved["calibration"]["ece"]
    assert "記録したときは" in result["sentences"][0]
    assert result["before"]["dropped"] == 0
    bundle = load_bundle(path)
    triage = {"type": "triage", "positive": "yes", "miss_cost": 3,
              "false_alarm_cost": 1}
    notes = decide(bundle, predict(bundle, pd.DataFrame({"x": x})),
                   triage)["notes"]
    assert any("400 行で測り直した値" in note for note in notes)


def walk_bundle(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """
    2 つの店の売上が日ごとにランダムウォークする時系列と, 前の日の値を
    そのまま予測する一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    result: tuple[Path, pd.DataFrame]
        一式のディレクトリと, 元のデータ
    """
    # k 段先の誤りの広がりは 1 段先の √k 倍
    rng = np.random.default_rng(0)
    days = pd.date_range("2025-01-01", periods=200).strftime("%Y-%m-%d")
    walks = [500 + np.cumsum(rng.normal(0, 5, 200)) for _ in range(2)]
    frame = pd.DataFrame({
        "date": np.r_[days, days], "s": ["a"] * 200 + ["b"] * 200,
        "y": np.r_[walks[0], walks[1]],
    })
    source = tmp_path / "w.csv"
    frame.to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "y", "task": "forecasting",
        "steps": [
            {"op": "lag", "columns": ["y"], "lags": [1], "by": ["s"],
             "time": "date"},
            {"op": "drop_missing"},
            {"op": "drop_columns", "columns": ["s", "date"]},
        ],
        "split": {"method": "random", "valid": 0.3, "seed": 0},
    }
    export_dataset(load_data(source), recipe, tmp_path / "ds")
    spec = linear_spec(1, 1)
    model = build_model(spec)
    with torch.no_grad():
        model.nodes["head"].weight.fill_(1.0)
        model.nodes["head"].bias.zero_()
    torch.save(model.state_dict(), tmp_path / "w.pt")
    make_bundle(tmp_path / "bundle", spec, tmp_path / "ds", tmp_path / "w.pt")
    result = (tmp_path / "bundle", frame)

    return result


def test_remeasure_steps(tmp_path: Path) -> None:
    """
    何段も先の外れ方を段ごとに測り, 先の段ほど広く, 量の判断がその段の
    外れ方を使うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 後ろ 100 日で, 5 段先まで
    path, frame = walk_bundle(tmp_path)
    count = (pd.to_datetime(frame["date"]) >= "2025-04-11").to_numpy()
    result = remeasure(path, frame, count=count, horizon=5)
    steps = result["uncertainty"]["steps"]
    assert result["uncertainty"]["horizon"] == 5
    assert set(steps) == {"rows", "1", "2", "3", "4", "5"}
    width = {k: steps[k][89] - steps[k][9] for k in ("1", "5")}
    assert width["5"] > 1.5 * width["1"]
    assert any("何段も先の外れ方" in s for s in result["sentences"])

    # 先の 5 日を予測して 9 : 1 で置くと, 5 段目は段ごとの外れ方で多め
    future = pd.DataFrame({
        "date": np.repeat(pd.date_range("2025-07-20", periods=5).strftime(
            "%Y-%m-%d"
        ), 2),
        "s": ["a", "b"] * 5, "y": np.nan,
    })
    ahead = forecast(path, pd.concat([frame, future], ignore_index=True))
    rows = ahead[ahead["step"].notna()].reset_index(drop=True)
    decision = {"type": "quantity", "under_cost": 9, "over_cost": 1}
    bundle = load_bundle(path)
    decided = decide(bundle, rows, decision)
    assert decided["summary"]["by_step"] is True
    gap = decided["frame"]["level"] - rows["prediction"]
    assert gap[rows["step"] == 5].mean() > gap[rows["step"] == 1].mean()
    assert any("段ごとに測った外れ方" in n for n in decided["notes"])
    empty = decide(bundle, rows.head(0), decision)
    assert "決められる行がありません" in empty["sentences"][0]
    skipped = rows.head(3).assign(status="skipped")
    assert "決められる行" in decide(bundle, skipped, decision)["sentences"][0]

    # 使えない場合
    with pytest.raises(ValueError, match="2 以上"):
        remeasure(path, frame, horizon=1, write=False)
    (tmp_path / "r").mkdir()
    reg = regression_bundle(tmp_path / "r")
    with pytest.raises(ValueError, match="過去の値を使う"):
        remeasure(reg, frame.assign(x=1.0), horizon=3, write=False)
