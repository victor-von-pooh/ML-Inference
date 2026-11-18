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

    # 時刻でずらさない一式の自動は行ごと. まとまりで選び直すと幅が広がる
    # (10 行ずつ同じ値の並び: 似た行のまとまり)
    assert result["group"] is None
    block = frame.iloc[np.repeat(np.arange(30), 10)].reset_index(drop=True)
    rowwise = backtest(path, block, decision, group=None)
    grouped = backtest(path, block, decision,
                       group=np.repeat(np.arange(30), 10))
    assert grouped["group"] == {"name": "渡したまとまり", "count": 30}
    low, high = rowwise["gaps"]["予測の値どおりに置く"]["range"]
    wide_low, wide_high = grouped["gaps"]["予測の値どおりに置く"]["range"]
    assert wide_high - wide_low > 2 * (high - low)
    assert "ごとに選び直した" in grouped["sentences"][1]
    named = backtest(path, block.assign(day=np.repeat(np.arange(30), 10)),
                     decision, group="day")
    assert named["group"]["count"] == 30
    with pytest.raises(ValueError, match="まとまり"):
        backtest(path, block, decision, group="no_such")
    with pytest.raises(ValueError, match="まとまり"):
        backtest(path, block, decision, group=np.arange(5))

    # まとまりが少なければ行ごとに戻し, 欠けた値は 1 行ずつ別のまとまり
    few = backtest(path, block, decision, group=np.zeros(300))
    assert few["group"] is None
    assert "1 しか無いので, 行ごとに" in few["sentences"][1]
    holes = np.repeat(np.arange(30), 10).astype(float)
    holes[:10] = np.nan
    missing = backtest(path, block, decision, group=holes)
    assert missing["group"]["count"] == 39


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


def test_current_flags(tmp_path: Path) -> None:
    """
    今のやり方で回したかの列を, 文字や数でも行動したかとして読むことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 全て no なら, どれも回さないのと同じ損 (文字のまま真偽値にしない)
    path = classifier_bundle(tmp_path)
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    past = pd.DataFrame({"x": x, "label": np.where(x > 0, "yes", "no")})
    triage = {"type": "triage", "positive": "yes", "miss_cost": 9,
              "false_alarm_cost": 1, "current": "now"}
    nothing = backtest(path, past.assign(now="no"), triage)["policies"]
    assert nothing["今のやり方"] == nothing["どれも回さない"]
    everything = backtest(path, past.assign(now="はい"), triage)["policies"]
    assert everything["今のやり方"] == everything["全てを回す"]
    numbers = backtest(path, past.assign(now=(x > 0).astype(int)),
                       triage)["policies"]
    words = backtest(path, past.assign(now=np.where(x > 0, "Yes", "No")),
                     triage)["policies"]
    assert numbers["今のやり方"] == words["今のやり方"] == 0

    # 決めるだけなら列が無くてもよい. たどり直しでは要る
    assert "action" in decide(path, predict(path, past), triage)["frame"]
    with pytest.raises(ValueError, match="current の列"):
        backtest(path, past, triage)

    # 読めない値と欠けた値は誤り
    with pytest.raises(ValueError, match="読めない値"):
        backtest(path, past.assign(now="たぶん"), triage)
    with pytest.raises(ValueError, match="欠けた値"):
        backtest(path, past.assign(now=np.where(x > 0, "yes", None)),
                 triage)


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
    assert any("範囲の外の) 1 行" in note for note in result["notes"])

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


def test_default_decision(tmp_path: Path) -> None:
    """
    一式の既定の意思決定を infer が使い, 空の辞書で外せることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 既定値を持つ一式
    path = classifier_bundle(tmp_path)
    triage = {"type": "triage", "positive": "yes", "miss_cost": 9,
              "false_alarm_cost": 1}
    make_bundle(tmp_path / "with", tmp_path / "bundle" / "model_spec.yaml",
                tmp_path / "ds", tmp_path / "w.pt", decision=triage)
    frame = pd.DataFrame({"x": [1.0, -1.0]})
    assert "decision" in infer(tmp_path / "with", frame)
    assert "decision" not in infer(tmp_path / "with", frame, decision={})
    with pytest.raises(ValueError, match="意思決定の決め方の誤り"):
        make_bundle(tmp_path / "bad", tmp_path / "bundle" / "model_spec.yaml",
                    tmp_path / "ds", tmp_path / "w.pt",
                    decision={"type": "quantity", "under_cost": 1,
                              "over_cost": 1})
    assert path.exists()


def test_bool_labels_and_row_values(tmp_path: Path) -> None:
    """
    真偽の目的の列のたどり直しと, 列で書いた範囲の外の値を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 真偽の目的の列 (クラスは False, True)
    rng = np.random.default_rng(4)
    x = rng.normal(size=200)
    source = tmp_path / "b.csv"
    pd.DataFrame({"x": x, "label": x > 0}).to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [],
        "split": {"method": "random", "valid": 0.3, "seed": 0},
    }
    export_dataset(load_data(source), recipe, tmp_path / "ds")
    spec = linear_spec(1, 2)
    model = build_model(spec)
    with torch.no_grad():
        model.nodes["head"].weight.copy_(torch.tensor([[-3.0], [3.0]]))
        model.nodes["head"].bias.zero_()
    torch.save(model.state_dict(), tmp_path / "w.pt")
    make_bundle(tmp_path / "bundle", spec, tmp_path / "ds", tmp_path / "w.pt")
    positive = load_bundle(tmp_path / "bundle").classes[1]
    triage = {"type": "triage", "positive": positive, "miss_cost": 5,
              "false_alarm_cost": 1}
    past = pd.DataFrame({"x": x, "label": x > 0})
    policies = backtest(tmp_path / "bundle", past, triage)["policies"]
    assert policies["どれも回さない"] < 0

    # 範囲の外の効果の行と, 整数でない件数
    frame = pd.DataFrame({"x": [1.0, 1.0], "effect": [0.5, 5.0]})
    table = predict(tmp_path / "bundle", frame)
    select = {"type": "select", "positive": positive, "value": 100,
              "cost": 1, "effect": "effect"}
    result = decide(tmp_path / "bundle", table, select)
    assert result["summary"]["missing"] == 1
    bundle = load_bundle(tmp_path / "bundle")
    assert check_decision({**select, "budget": 1.5}, bundle)
