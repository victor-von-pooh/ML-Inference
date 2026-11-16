from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import linear_spec
from ml_eda import export_dataset, load_data
from ml_modelcontrol import build_model

from ml_inference import backtest, check_decision, decide, infer, make_bundle
from ml_inference import load_bundle, predict


def regression_bundle(tmp_path: Path) -> Path:
    """
    y = 10 x (誤りは比で ±10 % ほど) の回帰の一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    path: Path
        一式のディレクトリ
    """
    # 正の値で, 外れ方が水準に比例するデータ
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 10, 400)
    frame = pd.DataFrame({"x": x, "y": 10 * x * rng.uniform(0.9, 1.1, 400)})
    source = tmp_path / "d.csv"
    frame.to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "y", "task": "regression",
        "steps": [],
        "split": {"method": "random", "valid": 0.25, "test": 0.25,
                  "seed": 0},
    }
    export_dataset(load_data(source), recipe, tmp_path / "ds")
    spec = linear_spec(1, 1)
    model = build_model(spec)
    with torch.no_grad():
        model.nodes["head"].weight.fill_(10.0)
        model.nodes["head"].bias.zero_()
    torch.save(model.state_dict(), tmp_path / "w.pt")
    path = tmp_path / "bundle"
    make_bundle(path, spec, tmp_path / "ds", tmp_path / "w.pt")

    return path


def test_uncertainty_and_quantity(tmp_path: Path) -> None:
    """
    外れ方を比で測り, 損の比の分位点まで置くことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 比で ±10 % ほどの外れ方, テスト用で幅が覆う
    path = regression_bundle(tmp_path)
    uncertainty = load_bundle(path).info["uncertainty"]
    assert uncertainty["kind"] == "relative"
    assert 0.8 <= uncertainty["coverage"]["90"] <= 1.0
    assert np.exp(uncertainty["relative"][94]) == pytest.approx(1.09,
                                                               abs=0.02)

    # 足りない損が 9 倍なら 9 割の点まで, 今ある量を引き, 刻みで切り上げ
    frame = pd.DataFrame({"x": [5.0, 5.0], "stock": [0.0, 30.0]})
    table = predict(path, frame)
    decision = {"type": "quantity", "under_cost": 9, "over_cost": 1,
                "stock": "stock", "step": 5}
    result = decide(path, table, decision)
    level = 50 * np.exp(uncertainty["relative"][89])
    expected = [np.ceil(level / 5) * 5, np.ceil((level - 30) / 5) * 5]
    assert list(result["frame"]["action"]) == expected
    assert result["summary"]["expected_cost"] < result["summary"][
        "baseline_cost"
    ]
    assert "90 %" in result["sentences"][0]


def test_backtest_quantity(tmp_path: Path) -> None:
    """
    実際の値で, 非対称な損では方針が予測どおりより損が小さいことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 新しいデータでたどり直す (数える行を選べる)
    path = regression_bundle(tmp_path)
    rng = np.random.default_rng(5)
    x = rng.uniform(1, 10, 300)
    frame = pd.DataFrame({"x": x, "y": 10 * x * rng.uniform(0.9, 1.1, 300),
                          "now": 10 * x})
    decision = {"type": "quantity", "under_cost": 9, "over_cost": 1,
                "current": "now"}
    result = backtest(path, frame, decision)
    policies = result["policies"]
    assert policies["この方針"] > policies["予測の値どおりに置く"]
    assert "今のやり方" in policies
    half = backtest(path, frame, decision, count=np.arange(300) < 150)
    assert half["rows"] == 150


def classifier_bundle(tmp_path: Path) -> Path:
    """
    x が大きいほど当たりの 2 クラスの分類の一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    path: Path
        一式のディレクトリ
    """
    # 当たりの確率 = シグモイド (x) の一式
    rng = np.random.default_rng(1)
    x = rng.normal(size=600)
    label = np.where(rng.uniform(size=600) < 1 / (1 + np.exp(-2 * x)),
                     "yes", "no")
    source = tmp_path / "c.csv"
    pd.DataFrame({"x": x, "label": label}).to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [],
        "split": {"method": "random", "valid": 0.3, "test": 0.2, "seed": 0},
    }
    export_dataset(load_data(source), recipe, tmp_path / "ds")
    spec = linear_spec(1, 2)
    model = build_model(spec)
    with torch.no_grad():
        model.nodes["head"].weight.copy_(torch.tensor([[-1.0], [1.0]]))
        model.nodes["head"].bias.zero_()
    torch.save(model.state_dict(), tmp_path / "w.pt")
    path = tmp_path / "bundle"
    make_bundle(path, spec, tmp_path / "ds", tmp_path / "w.pt")

    return path


def test_select_and_triage(tmp_path: Path) -> None:
    """
    見込みの得で上限まで選ぶことと, 損の比のしきい値で回すことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 確率の当たり具合を測っている
    path = classifier_bundle(tmp_path)
    calibration = load_bundle(path).info["uncertainty"]["calibration"]
    assert calibration["ece"] < 0.1

    # 価値の大きい行を優先して 2 件まで
    frame = pd.DataFrame({"x": [2.0, 1.0, 0.5, -3.0],
                          "value": [1000.0, 10000.0, 100.0, 100.0]})
    table = predict(path, frame)
    select = {"type": "select", "positive": "yes", "value": "value",
              "cost": 50, "effect": 0.5, "budget": 2}
    result = decide(path, table, select)
    assert list(result["frame"]["action"]) == [True, True, False, False]
    assert "全てに行動" not in result["summary"]["baselines"]
    assert any("仮定" in note for note in result["notes"])

    # 見逃しが空振りの 9 倍なら確率 0.1 以上を回す
    triage = {"type": "triage", "positive": "yes", "miss_cost": 9,
              "false_alarm_cost": 1}
    result = decide(path, table, triage)
    assert result["summary"]["threshold_min"] == pytest.approx(0.1)
    p = table["p:yes"].to_numpy()
    assert list(result["frame"]["action"]) == list(p >= 0.1)

    # infer で結論と一緒に, たどり直しで比べる
    inferred = infer(path, frame, decision=triage)
    assert "action" in inferred["frame"].columns
    assert inferred["decision"]["summary"]["acting"] >= 1
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    past = pd.DataFrame({"x": x, "label": np.where(
        rng.uniform(size=300) < 1 / (1 + np.exp(-2 * x)), "yes", "no"
    )})
    policies = backtest(path, past, triage)["policies"]
    assert policies["この方針"] >= policies["確率 0.5 で分ける"]


def test_decision_errors(tmp_path: Path) -> None:
    """
    意思決定の決め方の誤りを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 型と一式の組み合わせ, 仮定の書き忘れ, 無い列
    bundle = load_bundle(classifier_bundle(tmp_path))
    assert check_decision({"type": "price"}, bundle)
    assert check_decision({"type": "quantity", "under_cost": 1,
                           "over_cost": 1}, bundle)
    assert any("effect" in e for e in check_decision(
        {"type": "select", "positive": "yes", "value": 1, "cost": 1}, bundle
    ))
    assert check_decision(
        {"type": "triage", "positive": "yes", "miss_cost": "m",
         "false_alarm_cost": 1}, bundle, ["x"]
    )
    assert check_decision(
        {"type": "triage", "positive": "yes", "miss_cost": 3,
         "false_alarm_cost": 1}, bundle, ["x"]
    ) == []


def test_review_fixes(tmp_path: Path) -> None:
    """
    レビューで直した点 (損の端, 欠けた値, 費用の上限, 小数のラベル) を
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 足りない損が 0 なら足さない, 欠けた損の行は決めない
    (tmp_path / "r").mkdir()
    (tmp_path / "c").mkdir()
    path = regression_bundle(tmp_path / "r")
    frame = pd.DataFrame({"x": [5.0, 5.0], "under": [0.0, np.nan]})
    table = predict(path, frame)
    result = decide(path, table, {"type": "quantity", "under_cost": "under",
                                  "over_cost": 1})
    assert result["frame"]["action"].iloc[0] == 0
    assert np.isnan(result["frame"]["action"].iloc[1])
    assert result["summary"]["missing"] == 1
    assert any("欠けた 1 行" in note for note in result["notes"])

    # 費用の上限では, 費用あたりの得の順の方が良ければそれを使い, 比べる
    # やり方も上限を守る
    path = classifier_bundle(tmp_path / "c")
    frame = pd.DataFrame({
        "x": [3.0] * 11,
        "value": [200.0] + [70.0] * 10,
        "cost": [100.0] + [10.0] * 10,
    })
    table = predict(path, frame)
    select = {"type": "select", "positive": "yes", "value": "value",
              "cost": "cost", "effect": 1.0, "budget_cost": 100}
    result = decide(path, table, select)
    chosen = result["frame"]["action"].astype(bool)
    assert not chosen.iloc[0] and chosen.iloc[1:].all()
    spent = frame["cost"][chosen].sum()
    assert spent <= 100

    # 空欄で小数に読まれたラベルの列 (1.0) も, クラスの名前 (1) と比べる
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    source = tmp_path / "n.csv"
    pd.DataFrame({"x": x, "label": (x > 0).astype(int)}).to_csv(
        source, index=False
    )
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [],
        "split": {"method": "random", "valid": 0.3, "seed": 0},
    }
    export_dataset(load_data(source), recipe, tmp_path / "nds")
    spec = linear_spec(1, 2)
    model = build_model(spec)
    with torch.no_grad():
        model.nodes["head"].weight.copy_(torch.tensor([[-3.0], [3.0]]))
        model.nodes["head"].bias.zero_()
    torch.save(model.state_dict(), tmp_path / "nw.pt")
    make_bundle(tmp_path / "nb", spec, tmp_path / "nds", tmp_path / "nw.pt")
    past = pd.DataFrame({"x": x, "label": (x > 0).astype(float)})
    triage = {"type": "triage", "positive": "1", "miss_cost": 5,
              "false_alarm_cost": 1}
    policies = backtest(tmp_path / "nb", past, triage)["policies"]
    assert policies["どれも回さない"] < 0
