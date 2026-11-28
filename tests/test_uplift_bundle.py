import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from ml_eda import export_dataset, load_data, load_dataset
from ml_modelcontrol import build_model, load_example

import ml_inference.bundle as bundle_module
from ml_inference.uplift import (
    qini_coefficient, qini_points, read_estimator, uplift_at_k
)
from ml_inference import (
    backtest, check_decision, decide, describe_bundle, infer, load_bundle,
    make_bundle, make_uplift_bundle, predict, read_result, record_review,
    write_result
)

# 顧客の数と, 配る数の上限と評価の上位の割合
CUSTOMERS = 80
BUDGET = 5
K = 0.3

# 見積もりに使う列
FEATURES = ["age", "visits"]

# ML-Found の docs/DESIGN.md の照合の見本 (見積もり, 処置, 結果, 曲線の
# (rows, qini) の点, 係数, K ごとの上位 K 割の効果の差)
SAMPLES = [
    ([0.9, 0.8, 0.8, 0.6, 0.5, 0.5, 0.3, 0.2], [1, 1, 0, 1, 0, 1, 0, 0],
     [1, 0, 1, 1, 0, 0, 0, 1],
     [(0, 0), (1, 1), (3, -1), (4, -1), (6, 0), (7, 2 / 3), (8, 0)],
     -5 / 72, {0.25: -1 / 3, 0.3: -1 / 3, 0.75: 0.0, 1: 0.0}),
    ([0.9, 0.7, 0.7, 0.4, 0.4, 0.1], [1, 0, 1, 1, 0, 0],
     [1, 0, 1, 0, 0, 1],
     [(0, 0), (1, 1), (3, 2), (5, 2), (6, 1)],
     12 / 13, {0.25: math.nan, 0.5: 1.0, 1: 1 / 3}),
    ([3.0, 2.0, 2.0, 1.0, 0.5, 0.5], [1, 1, 0, 1, 0, 0],
     [500, 0, 200, 300, 0, 100],
     [(0, 0), (1, 500), (3, 100), (4, 200), (6, 500)],
     2 / 23, {0.25: math.nan, 0.5: 50.0, 1: 500 / 3}),
    ([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0], [1, 0, 0, 1],
     [(0, 0), (4, 0)], 0.0, {0.25: 0.0, 0.5: 0.0}),
    ([0.3, 0.2, 0.1], [1, 0, 1], [0, 0, 0],
     [(0, 0), (1, 0), (2, 0), (3, 0)], 0.0, {0.5: math.nan, 1: 0.0}),
]


def coupon_rows(outcome: str) -> pd.DataFrame:
    """
    来店の多い顧客ほど配ると買う, 無作為に配った合成の記録を作る関数

    Parameters
    ----------
    outcome: str
        binary (買ったか) か amount (買った額)

    Returns
    ----------
    frame: pd.DataFrame
        customer_id, age, visits, coupon, purchased の表
    """
    # 処置は無作為, 結果は来店の回数と処置で決める
    rng = np.random.default_rng(0)
    rows = []
    for number in range(CUSTOMERS):
        visits = int(rng.integers(0, 10))
        coupon = int(number % 2)
        chance = 0.2 + (0.06 * visits if coupon else 0.0)
        bought = int(rng.random() < chance)
        rows.append({
            "customer_id": f"c{number:03d}",
            "age": float(rng.integers(20, 70)),
            "visits": float(visits),
            "coupon": coupon,
            "purchased": bought if outcome == "binary"
            else float(bought * rng.integers(1000, 5000)),
        })
    frame = pd.DataFrame(rows)

    return frame


def uplift_spec(learner: str = "t_learner") -> dict:
    """
    列の数に合わせた効果のモデルの定義を返す関数

    Parameters
    ----------
    learner: str = "t_learner"
        効果の決め方 (t_learner か s_learner)

    Returns
    ----------
    spec: dict
        ML-ModelControl の同梱の例 tabular_uplift (s_learner は
        tabular_uplift_s) の入力の幅を変えたもの
    """
    # s_learner は列の数と処置の列の 1 の和の幅の 1 つの枝
    if learner == "s_learner":
        spec = load_example("tabular_uplift_s")
        width = len(FEATURES) + 1
        spec["inputs"]["x"]["shape"] = [8, width]
        spec["nodes"]["outcome1"]["params"]["in_features"] = width
        return spec

    # 入力の幅と, 2 つの枝の最初の全結合の入力の幅
    spec = load_example("tabular_uplift")
    width = len(FEATURES)
    spec["inputs"]["x"]["shape"] = [8, width]
    spec["nodes"]["treated1"]["params"]["in_features"] = width
    spec["nodes"]["control1"]["params"]["in_features"] = width

    return spec


def skeleton_estimate(
    model: torch.nn.Module, x: np.ndarray, scale: dict,
    learner: str = "t_learner"
) -> dict:
    """
    骨組みと同じ式で, 行ごとの 2 つの群の予測と効果を求める関数

    Parameters
    ----------
    model: torch.nn.Module
        効果のモデル
    x: np.ndarray
        (行の数, 列の数) の float32
    scale: dict
        量の標準化の mean と std (0 と 1 は None)
    learner: str = "t_learner"
        効果の決め方 (s_learner は処置の列を 1 と 0 にして入力の最後に
        足した 2 回の予測)

    Returns
    ----------
    found: dict
        treated, control, effect の float64 の配列
    """
    # 2 つの予測と, 0 と 1 はシグモイド, 量は元の単位に戻してからの差
    with torch.no_grad():
        if learner == "s_learner":
            pair = [
                model(x=torch.tensor(np.column_stack([
                    x, np.full(len(x), value, dtype=np.float32)
                ]))).reshape(-1).double()
                for value in (1.0, 0.0)
            ]
            treated, control = pair
        else:
            outputs = model(x=torch.tensor(x))
            treated = outputs["treated"].reshape(-1).double()
            control = outputs["control"].reshape(-1).double()
    if scale is None:
        treated, control = torch.sigmoid(treated), torch.sigmoid(control)
    else:
        treated = treated * scale["std"] + scale["mean"]
        control = control * scale["std"] + scale["mean"]
    found = {"treated": treated.numpy(), "control": control.numpy(),
             "effect": (treated - control).numpy()}

    return found


def write_uplift_run(
    root: Path, raw: Path, outcome: str, spec: bool = True,
    budget: object = BUDGET, learner: str = "t_learner"
) -> Path:
    """
    効果の見積もりの骨組みと同じ成果物を持つ実験のディレクトリを作る関数

    Parameters
    ----------
    root: Path
        作業のディレクトリ (データセットと実験を置く)
    raw: Path
        元の表の CSV
    outcome: str
        binary か amount
    spec: bool = True
        モデルの定義を実験に置くか (置かなければ仮のモデルの実験)
    budget: object = BUDGET
        成果物の配る数の上限
    learner: str = "t_learner"
        効果の決め方 (t_learner か s_learner)

    Returns
    ----------
    run: Path
        実験のディレクトリ
    """
    # 介入の記録の表を ML-EDA で書き出し, 分割ごとの行の特徴
    data_dir = root / "dataset"
    recipe = {"task": "uplift", "target": "purchased", "treatment": "coupon",
              "outcome": outcome, "id": "customer_id",
              "steps": [{"op": "standardize", "columns": "auto"}],
              "split": {"method": "stratified", "valid": 0.2, "test": 0.2,
                        "seed": 0, "unlabeled": "drop"}}
    export_dataset(load_data(raw), recipe, data_dir)
    dataset = load_dataset(data_dir)
    run = root / "run"
    run.mkdir()
    splits = {}
    for split in dataset.frames:
        x, y = dataset.arrays(split)
        splits[split] = {
            "x": np.array(x, dtype=np.float32),
            "t": dataset.treatment(split),
            "y": np.asarray(y, dtype=np.float64),
            "ids": dataset.frames[split]["customer_id"].to_numpy(),
        }

    # 学習しない重みのモデルと, 量の標準化の値
    model = build_model(uplift_spec(learner), seed=0, load_weights=False)
    model.eval()
    torch.save(model.state_dict(), run / "final_model.pt")
    if spec:
        (run / "model_spec.yaml").write_text(
            yaml.safe_dump(uplift_spec(learner), allow_unicode=True),
            encoding="utf-8"
        )
    train_y = splits["train"]["y"]
    scale = None if outcome == "binary" else {
        "mean": float(train_y.mean()), "std": float(train_y.std())
    }

    # 行ごとの効果, 効果の決め方, レシピの写し, 評価
    tables = []
    for split, part in splits.items():
        part.update(skeleton_estimate(model, part["x"], scale, learner))
        tables.append(pd.DataFrame({
            "customer_id": part["ids"], "split": split,
            "effect": part["effect"], "treated": part["treated"],
            "control": part["control"], "treatment": part["t"],
            "outcome": part["y"],
        }))
    pd.concat(tables).to_csv(run / "uplift_effects.csv", index=False)
    effects = splits["train"]["effect"]
    train = {"rows": len(effects), "mean": float(effects.mean()),
             "min": float(effects.min())}
    for point in (10, 50, 90):
        train[f"p{point}"] = float(np.percentile(effects, point))
    train["max"] = float(effects.max())
    train["positive"] = float((effects > 0).mean())
    names = ({"input": "x", "outcome": "outcome"} if learner == "s_learner"
             else {"input": "x", "treated": "treated", "control": "control"})
    estimator = {
        "format_version": 1, "learner": learner, "outcome": outcome,
        "recipe": "uplift_recipe.yaml", "effects": "uplift_effects.csv",
        "features": FEATURES, "id": "customer_id", "treatment": "coupon",
        "budget": budget, "k": K,
        "model": {**names,
                  "link": "sigmoid" if outcome == "binary" else "identity"},
        "scale": scale, "train": train,
    }
    (run / "uplift_estimator.json").write_text(
        json.dumps(estimator, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    (run / "uplift_recipe.yaml").write_text(
        (data_dir / "recipe.yaml").read_text(encoding="utf-8"),
        encoding="utf-8"
    )
    (run / "evaluations").mkdir()
    (run / "evaluations" / "uplift_valid.json").write_text(json.dumps({
        "name": "uplift_valid", "split": "valid", "kind": "table",
        "rows": [{"rows": 16, "treated": 8, "control": 8, "k": K,
                  "qini": 0.12, "uplift_at_k": 0.25, "effect": 0.1,
                  "estimated": 0.02}],
    }), encoding="utf-8")
    (run / "evaluations" / "uplift_test.json").write_text(json.dumps({
        "name": "uplift_test", "split": "test", "kind": "table",
        "rows": [{"rows": 16, "treated": 8, "control": 8, "k": K,
                  "qini": -0.05, "uplift_at_k": None, "effect": 0.0,
                  "estimated": 0.02}],
    }), encoding="utf-8")
    (run / "evaluations" / "qini_curve_valid.json").write_text(json.dumps({
        "name": "qini_curve_valid", "split": "valid", "kind": "curve",
        "x": "fraction",
        "rows": [{"rows": 0, "fraction": 0.0, "qini": 0.0, "random": 0.0},
                 {"rows": 8, "fraction": 0.5, "qini": 2.0, "random": 0.5},
                 {"rows": 16, "fraction": 1.0, "qini": 1.0,
                  "random": 1.0}],
    }), encoding="utf-8")

    return run


@pytest.fixture(params=["binary", "amount"])
def outcome(request: pytest.FixtureRequest) -> str:
    """
    結果の種類を返す関数

    Parameters
    ----------
    request: pytest.FixtureRequest
        引数の値

    Returns
    ----------
    outcome: str
        binary か amount
    """
    # 2 つの結果の種類
    outcome = str(request.param)

    return outcome


def raw_table(tmp_path: Path, outcome: str) -> Path:
    """
    合成の配った記録を CSV に書く関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    outcome: str
        binary か amount

    Returns
    ----------
    path: Path
        CSV のファイル
    """
    # 結果の種類ごとの表
    path = tmp_path / f"coupons_{outcome}.csv"
    coupon_rows(outcome).to_csv(path, index=False)

    return path


def test_uplift_bundle(tmp_path: Path, outcome: str) -> None:
    """
    効果の一式の効果と, 効果の上位から配る判断を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    outcome: str
        結果の種類

    Returns
    ----------
    None
    """
    # 一式を作り, 元の表の効果を実験の効果と比べる
    raw = raw_table(tmp_path, outcome)
    run = write_uplift_run(tmp_path / "work", raw, outcome)
    info = make_uplift_bundle(tmp_path / "bundle", run, data=raw)
    assert info["format_version"] == 12
    assert (info["role"], info["output"]) == ("effect", {"form": "effect"})
    assert info["decision"] == {"type": "treat", "budget": BUDGET}
    uplift = info["uplift"]
    assert uplift["outcome"] == outcome and uplift["features"] == FEATURES
    assert (uplift["treatment"], uplift["k"]) == ("coupon", K)
    assert uplift["check"]["rows"] == CUSTOMERS
    assert uplift["check"]["agreement"] == 1.0
    # (額は元の単位の差. float32 の前処理の丸めの分)
    assert uplift["check"]["max_diff"] < (1e-6 if outcome == "binary"
                                          else 1e-3)
    assert uplift["evaluations"]["test"]["uplift_at_k"] is None
    assert uplift["curves"]["valid"]["qini"] == [0.0, 2.0, 1.0]
    assert uplift["curves"]["valid"]["fraction"] == [0.0, 0.5, 1.0]
    assert "test" not in uplift["curves"]
    assert uplift["rows"] == CUSTOMERS
    for name in ("uplift_estimator.json", "uplift_recipe.yaml",
                 "model_spec.yaml", "final_model.pt"):
        assert (tmp_path / "bundle" / name).is_file()

    # 説明
    bundle = load_bundle(tmp_path / "bundle")
    found = describe_bundle(bundle)
    assert (found["kind"], found["task"]) == ("uplift", "")
    assert found["features"] == FEATURES and found["steps"] == ["standardize"]
    assert found["uplift"]["budget"] == BUDGET
    assert found["uplift"]["curves"]["valid"]["rows"] == [0, 8, 16]
    assert found["anomaly"] is None and not found["column_output"]

    # 新しい行の効果 (処置と結果の列は要らない. 前処理で欠けた行は飛ばす)
    rows = coupon_rows(outcome).drop(columns=["coupon", "purchased"])
    rows.loc[1, "visits"] = None
    frame = predict(bundle, rows)
    assert list(frame.columns[-6:]) == ["status", "reason", "effect",
                                        "treated", "control", "rank"]
    assert frame.loc[1, "status"] == "skipped"
    assert "前処理の後に欠損: visits" in frame.loc[1, "reason"]
    ok = frame[frame["status"] == "ok"]
    np.testing.assert_allclose(ok["effect"], ok["treated"] - ok["control"])
    assert sorted(ok["rank"].tolist()) == list(range(1, len(ok) + 1))
    assert ok.sort_values("rank")["effect"].is_monotonic_decreasing
    if outcome == "binary":
        assert ok["treated"].between(0, 1).all()

    # 入力の列の並びに依らない効果と順位
    swapped = predict(bundle, rows[["visits", "customer_id", "age"]])
    np.testing.assert_allclose(swapped["effect"], frame["effect"])
    assert swapped["rank"].equals(frame["rank"])

    # 結論と既定の配る判断 (上位から上限まで, 効果が正の行だけ)
    result = infer(bundle, rows)
    summary = result["summary"]
    assert summary["predicted"] == CUSTOMERS - 1
    assert summary["skipped"] == 1 and summary["id"] == "customer_id"
    assert summary["outcome"] == outcome and summary["measured"] is None
    assert summary["top_k"]["rows"] == int(0.3 * (CUSTOMERS - 1))
    assert len(summary["top"]) == 10
    assert summary["top"][0]["row"] == int(ok["rank"].idxmin())
    assert summary["distribution"]["max"] == pytest.approx(
        ok["effect"].max()
    )
    assert "効果 (処置したときの結果 − しないときの結果" in result[
        "sentences"
    ][0]
    assert any("検証用の Qini の係数 0.120" in s
               for s in result["sentences"])
    assert any("上位 30 % の効果の差は無し" in s
               for s in result["sentences"])
    acted = result["frame"]["action"]
    decided = result["decision"]
    positive = int((ok["effect"] > 0).sum())
    assert int(acted.sum()) == min(BUDGET, positive)
    assert pd.isna(acted.iloc[1])
    chosen = result["frame"][acted.fillna(False).astype(bool)]
    assert sorted(chosen["rank"].tolist()) == list(
        range(1, len(chosen) + 1)
    )
    assert (chosen["effect"] > 0).all()
    assert decided["summary"]["gain"] == pytest.approx(
        chosen["effect"].sum()
    )
    assert decided["summary"]["budget"] == BUDGET
    assert decided["sentences"][:3] == decided["notes"]
    assert "効果の高い順に" in decided["sentences"][3]
    assert any("無作為" in note for note in decided["notes"])

    # 空の辞書は配る判断無し
    plain = infer(bundle, rows, decision={})
    assert "decision" not in plain and "action" not in plain["frame"]

    # 書き出しと読み直し (人の確かめの記録は異常の度合いの結果だけ)
    write_result(result, tmp_path / "result")
    back = read_result(tmp_path / "result")
    assert back["summary"]["id"] == "customer_id"
    assert back["frame"]["effect"].tolist() == pytest.approx(
        result["frame"]["effect"].tolist(), nan_ok=True
    )
    with pytest.raises(ValueError, match="異常の度合いの結果ではありません"):
        record_review(tmp_path / "result", "c000", "normal", "山田")


def test_treat_decision(tmp_path: Path) -> None:
    """
    配る判断の形と, 同じ効果と 0 以下の効果の扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # ML-Quest の action.decision の配る判断と同じ名前と形だけを受ける
    raw = raw_table(tmp_path, "binary")
    run = write_uplift_run(tmp_path / "work", raw, "binary")
    make_uplift_bundle(tmp_path / "bundle", run)
    bundle = load_bundle(tmp_path / "bundle")
    assert check_decision({"type": "treat", "budget": 500}, bundle) == []
    errors = check_decision({"type": "treat", "budget": 0, "cost": 10},
                            bundle)
    assert any("cost" in e for e in errors)
    assert any("budget" in e for e in errors)
    assert check_decision({"type": "treat", "budget": True}, bundle)
    assert check_decision({"type": "treat"}, bundle)
    assert check_decision({"type": "select", "budget": 2}, bundle)
    assert check_decision({"type": "triage", "capacity": 2}, bundle)

    # 同じ効果は前の行が先, 効果が 0 以下の行は上限の中でも配らない
    frame = pd.DataFrame({
        "status": ["ok", "ok", "ok", "ok", "ok", "skipped"],
        "effect": [0.2, 0.5, 0.5, -0.1, 0.0, math.nan],
    })
    found = decide(bundle, frame, {"type": "treat", "budget": 4})
    assert found["frame"]["action"].tolist()[:5] == [True, True, True,
                                                     False, False]
    assert pd.isna(found["frame"]["action"].iloc[5])
    assert found["summary"]["nonpositive"] == 1
    assert found["summary"]["lowest"] == 0.2
    assert any("0 以下の 1 行" in s for s in found["sentences"])
    found = decide(bundle, frame, {"type": "treat", "budget": 1})
    assert found["frame"]["action"].tolist()[:3] == [False, True, False]
    assert found["summary"]["left_positive"] == 2
    assert any("上限で配らない行が 2 行" in s for s in found["sentences"])
    nothing = decide(bundle, frame.iloc[3:5], {"type": "treat",
                                              "budget": 3})
    assert nothing["summary"]["acted"] == 0
    assert nothing["summary"]["lowest"] is None
    with pytest.raises(ValueError, match="書けない項目"):
        decide(bundle, frame, {"type": "treat", "budget": 2, "value": 1})

    # 結論の決め方, たどり直し, 列の予測の形の一式づくりは誤り
    with pytest.raises(ValueError, match="効果の一式には"):
        infer(bundle, raw, rules={"min_confidence": 0.5})
    with pytest.raises(ValueError, match="効果の一式には"):
        backtest(bundle, raw, {"type": "treat", "budget": 2})
    with pytest.raises(ValueError, match="make_uplift_bundle"):
        make_bundle(tmp_path / "other", uplift_spec(),
                    tmp_path / "work" / "dataset", run / "final_model.pt",
                    form="effect")


def test_uplift_metrics() -> None:
    """
    Qini の曲線と係数, 上位 K 割の効果の差が ML-Found の見本と同じ値に
    なることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 見本ごとの曲線の点, 係数, K ごとの効果の差
    for scores, treatment, outcome, curve, coefficient, lifts in SAMPLES:
        values = [np.asarray(v, dtype=np.float64)
                  for v in (scores, treatment, outcome)]
        rows, qini = qini_points(*values)
        np.testing.assert_allclose(np.column_stack([rows, qini]),
                                   np.asarray(curve, dtype=np.float64),
                                   atol=1e-12)
        assert qini_coefficient(*values) == pytest.approx(coefficient,
                                                          abs=1e-12)
        for k, lift in lifts.items():
            assert uplift_at_k(*values, k) == pytest.approx(
                lift, abs=1e-12, nan_ok=True
            )

    # 同点の行の並べる順に依らない, 無限大の見積もりも同点
    scores = np.array([0.8, 0.9, 0.8, 0.6, 0.5, 0.3, 0.5, 0.2])
    treatment = np.array([1, 1, 0, 1, 1, 0, 0, 0], dtype=np.float64)
    outcome = np.array([0, 1, 1, 1, 0, 0, 0, 1], dtype=np.float64)
    assert qini_coefficient(scores, treatment, outcome) == pytest.approx(
        -5 / 72
    )
    assert uplift_at_k(scores, treatment, outcome, 0.25) == pytest.approx(
        -1 / 3
    )
    infinite = np.array([math.inf, math.inf, 0.0, -math.inf])
    assert uplift_at_k(infinite, np.array([1.0, 0.0, 1.0, 0.0]),
                       np.array([1.0, 0.0, 0.0, 1.0]), 0.5) == 1.0


def test_measured(tmp_path: Path) -> None:
    """
    処置と結果のある表で, 効果の見積もりの評価を数え直すことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 処置と結果のある元の表 (結果の欠けた行と読めない処置の行は除く)
    raw = raw_table(tmp_path, "binary")
    run = write_uplift_run(tmp_path / "work", raw, "binary")
    make_uplift_bundle(tmp_path / "bundle", run)
    rows = coupon_rows("binary").astype({"coupon": object})
    rows.loc[0, "purchased"] = None
    rows.loc[2, "coupon"] = "配布"
    result = infer(tmp_path / "bundle", rows)
    measured = result["summary"]["measured"]
    assert (measured["rows"], measured["excluded"]) == (CUSTOMERS - 2, 2)
    assert measured["treated"] + measured["control"] == CUSTOMERS - 2
    used = result["frame"].drop(index=[0, 2])
    t = used["coupon"].astype(float).to_numpy()
    y = used["purchased"].astype(float).to_numpy()
    effects = used["effect"].to_numpy()
    assert measured["qini"] == pytest.approx(
        qini_coefficient(effects, t, y)
    )
    assert measured["uplift_at_k"] == pytest.approx(
        uplift_at_k(effects, t, y, K)
    )
    assert measured["effect"] == pytest.approx(y[t == 1].mean()
                                               - y[t == 0].mean())
    assert any("渡した表の処置と結果で数えた Qini の係数" in s
               for s in result["sentences"])

    # 処置の群だけの表は数えず, 文で知らせる
    only = coupon_rows("binary").query("coupon == 1")
    result = infer(tmp_path / "bundle", only)
    assert result["summary"]["measured"] is None
    assert any("数え直さない" in s for s in result["sentences"])


def test_uplift_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    一式にできない実験と成果物, 古い読む側, 食い違う一式を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        読める版を古くする

    Returns
    ----------
    None
    """
    # 仮のモデルで学習した実験 (定義無し)
    raw = raw_table(tmp_path, "amount")
    placeholder = write_uplift_run(tmp_path / "a", raw, "amount",
                                   spec=False)
    with pytest.raises(ValueError, match="PlaceholderModel"):
        make_uplift_bundle(tmp_path / "x", placeholder)

    # 成果物の版と形の誤り (結果の種類が文字でない成果物も)
    run = write_uplift_run(tmp_path / "b", raw, "amount")
    path = run / "uplift_estimator.json"
    estimator = json.loads(path.read_text("utf-8"))
    model = estimator["model"]
    for change, words in (({"format_version": 2}, "版 2"),
                          ({"outcome": ["binary"]}, "outcome"),
                          ({"learner": "x_learner"}, "learner"),
                          ({"learner": "s_learner"}, "input, outcome の"),
                          ({"budget": 0}, "budget"),
                          ({"k": 1.5}, "k は"),
                          ({"k": True}, "k は"),
                          ({"scale": None}, "scale"),
                          ({"model": {**model, "link": "sigmoid"}}, "link"),
                          ({"outcome": "binary"}, "link")):
        path.write_text(json.dumps({**estimator, **change}),
                        encoding="utf-8")
        with pytest.raises(ValueError, match=words):
            make_uplift_bundle(tmp_path / "x", run)
    path.write_text(json.dumps({**estimator, "budget": None}),
                    encoding="utf-8")
    assert not (tmp_path / "x").exists()

    # 配る数の上限の無い成果物は既定の判断無し
    info = make_uplift_bundle(tmp_path / "bundle", run)
    assert info["decision"] is None and info["uplift"]["budget"] is None
    assert info["uplift"]["check"] is None
    assert read_estimator(tmp_path / "bundle"
                          / "uplift_estimator.json")["budget"] is None
    result = infer(tmp_path / "bundle", raw)
    assert "decision" not in result

    # JSON の外の NaN を持つ評価
    evaluation = run / "evaluations" / "uplift_valid.json"
    original = evaluation.read_text("utf-8")
    evaluation.write_text(original.replace("0.12", "NaN"), encoding="utf-8")
    with pytest.raises(ValueError, match="NaN"):
        make_uplift_bundle(tmp_path / "y", run)
    evaluation.write_text(original, encoding="utf-8")

    # 入力の幅が列の数と違う定義と, 出力の 1 つ欠けた定義
    spec = uplift_spec()
    spec["inputs"]["x"]["shape"] = [8, len(FEATURES) + 1]
    (run / "model_spec.yaml").write_text(yaml.safe_dump(spec),
                                         encoding="utf-8")
    with pytest.raises(ValueError, match="幅 3 が, 見積もりに使う列の数 2"):
        make_uplift_bundle(tmp_path / "y", run)
    spec = uplift_spec()
    spec["outputs"].pop("control")
    (run / "model_spec.yaml").write_text(yaml.safe_dump(spec),
                                         encoding="utf-8")
    with pytest.raises(ValueError, match="control"):
        make_uplift_bundle(tmp_path / "y", run)

    # 行の id の列の無い元の表は比べない
    rows = pd.read_csv(raw).drop(columns=["customer_id"])
    run = write_uplift_run(tmp_path / "c", raw, "amount")
    info = make_uplift_bundle(tmp_path / "bundle", run, data=rows,
                              overwrite=True)
    assert info["uplift"]["check"] is None
    assert any("比べません" in note for note in info["notes"])

    # 版 11 までしか知らない読む側と, 食い違う一式
    monkeypatch.setattr(bundle_module, "BUNDLE_FORMAT_VERSION", 11)
    with pytest.raises(ValueError, match="format_version 12 の一式"):
        load_bundle(tmp_path / "bundle")
    monkeypatch.undo()
    path = tmp_path / "bundle" / "inference.yaml"
    info = yaml.safe_load(path.read_text("utf-8"))
    for change, words in (({"format_version": 11}, "uplift を持つ一式"),
                          ({"output": {"form": "value"}}, "effect"),
                          ({"role": "predict"}, "受け持ち"),
                          ({"uplift": {**info["uplift"],
                                       "outcome": ["binary"]}}, "outcome"),
                          ({"uplift": {**info["uplift"],
                                       "treatment": "other"}},
                           "treatment")):
        path.write_text(yaml.safe_dump({**info, **change},
                                       allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match=words):
            load_bundle(tmp_path / "bundle")


def test_s_learner_bundle(
    tmp_path: Path, outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    処置の列を入力に入れる 1 つの予測 (s_learner) の効果の一式を確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    outcome: str
        結果の種類
    monkeypatch: pytest.MonkeyPatch
        読める版を古くする

    Returns
    ----------
    None
    """
    # 版 18 の一式 (出力の名前は output) と, 実験の効果との一致
    raw = raw_table(tmp_path, outcome)
    run = write_uplift_run(tmp_path / "work", raw, outcome,
                           learner="s_learner")
    info = make_uplift_bundle(tmp_path / "bundle", run, data=raw,
                              batch_size=7)
    assert info["format_version"] == 18
    assert (info["role"], info["output"]) == ("effect", {"form": "effect"})
    assert info["decision"] == {"type": "treat", "budget": BUDGET}
    uplift = info["uplift"]
    assert uplift["learner"] == "s_learner"
    assert (uplift["input"], uplift["output"]) == ("x", "outcome")
    assert "treated" not in uplift and "control" not in uplift
    assert uplift["check"]["rows"] == CUSTOMERS
    assert uplift["check"]["agreement"] == 1.0
    assert uplift["check"]["max_diff"] < (1e-6 if outcome == "binary"
                                          else 1e-3)

    # 新しい行の 2 つの予測は処置の列を 1 と 0 にした予測 (実験と同じ)
    bundle = load_bundle(tmp_path / "bundle")
    assert describe_bundle(bundle)["uplift"]["output"] == "outcome"
    rows = coupon_rows(outcome).drop(columns=["coupon", "purchased"])
    frame = predict(bundle, rows)
    assert (frame["status"] == "ok").all()
    np.testing.assert_allclose(frame["effect"],
                               frame["treated"] - frame["control"])
    recorded = pd.read_csv(run / "uplift_effects.csv").set_index(
        "customer_id"
    ).loc[frame["customer_id"]]
    for key in ("treated", "control"):
        np.testing.assert_allclose(frame[key], recorded[key],
                                   rtol=1e-4, atol=1e-6)
    if outcome == "binary":
        assert frame["treated"].between(0, 1).all()
    result = infer(bundle, rows)
    acted = result["frame"]["action"]
    assert int(acted.sum()) == min(BUDGET, int((frame["effect"] > 0).sum()))

    # 版 17 までしか知らない読む側と, 版と食い違う一式
    monkeypatch.setattr(bundle_module, "BUNDLE_FORMAT_VERSION", 17)
    with pytest.raises(ValueError, match="format_version 18 の一式"):
        load_bundle(tmp_path / "bundle")
    monkeypatch.undo()
    path = tmp_path / "bundle" / "inference.yaml"
    info = yaml.safe_load(path.read_text("utf-8"))
    for change, words in (({"uplift": {**info["uplift"],
                                       "learner": "t_learner"}},
                           "uplift.learner は s_learner"),
                          ({"uplift": {**info["uplift"],
                                       "treated": "outcome"}},
                           "知らない項目"),
                          ({"uplift": {**info["uplift"], "output": "y"}},
                           "output"),
                          ({"format_version": 12}, "treated")):
        path.write_text(yaml.safe_dump({**info, **change},
                                       allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match=words):
            load_bundle(tmp_path / "bundle")


def test_s_learner_refusals(tmp_path: Path) -> None:
    """
    s_learner の成果物のモデルの名前の欠けと, 入力の幅の食い違いが, 一式を
    作る前に誤りになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 結果の予測の名前の欠けた成果物 (t_learner の名前だけも)
    raw = raw_table(tmp_path, "binary")
    run = write_uplift_run(tmp_path / "work", raw, "binary",
                           learner="s_learner")
    path = run / "uplift_estimator.json"
    estimator = json.loads(path.read_text("utf-8"))
    for model in ({"input": "x", "link": "sigmoid"},
                  {"input": "x", "treated": "treated",
                   "control": "control", "link": "sigmoid"}):
        path.write_text(json.dumps({**estimator, "model": model}),
                        encoding="utf-8")
        with pytest.raises(ValueError, match="input, outcome の名前"):
            make_uplift_bundle(tmp_path / "x", run)
    path.write_text(json.dumps(estimator), encoding="utf-8")

    # 定義に無い出力の名前と, 処置の列の 1 を足していない入力の幅
    path.write_text(json.dumps({**estimator, "model": {
        **estimator["model"], "outcome": "y"
    }}), encoding="utf-8")
    with pytest.raises(ValueError, match="出力に y がありません"):
        make_uplift_bundle(tmp_path / "x", run)
    path.write_text(json.dumps(estimator), encoding="utf-8")
    spec = uplift_spec("s_learner")
    spec["inputs"]["x"]["shape"] = [8, len(FEATURES)]
    (run / "model_spec.yaml").write_text(yaml.safe_dump(spec),
                                         encoding="utf-8")
    with pytest.raises(ValueError,
                       match="幅 2 が, 見積もりに使う列の数と処置の列の 1"):
        make_uplift_bundle(tmp_path / "x", run)
    assert not (tmp_path / "x").exists()
