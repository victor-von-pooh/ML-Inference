import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from ml_eda import export_dataset, load_data, load_dataset
from ml_modelcontrol import build_model, load_example

import ml_inference.bundle as bundle_module
from ml_inference.demand import (
    best_orders, expected_costs, order_cost, pinball_loss, tau_of
)
from ml_inference import (
    backtest, check_decision, decide, describe_bundle, infer, load_bundle,
    make_demand_bundle, predict, read_result, write_result
)

# 商品の数, 日の数, 先の期間の日数, 学ぶ水準, 損と上限 (ML_tester の
# TASK_TYPES.md の 40 の発注の判断の例)
PRODUCTS = 4
DAYS = 70
HORIZON = 7
LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
SHORTAGE = 3
HOLDING = 1
MAX_ORDER = 60

# ML-Quest の時間をまたぐ量の判断の骨組みの train.py を書き換えずに短く
# 動かした実験の成果物と元の表 (ML-Quest の tests/test_demand_skeleton.py
# の合成の表を 4 商品, 60 日にし, ML-ModelControl の同梱の例
# tabular_demand の定義で CPU で 8 エポック)
SKELETON_OUTPUT = Path(__file__).parent / "demand_skeleton"

# 予測に使う列 (前処理の後)
FEATURES = ["category", "units_lag1", "units_lag7", "units_mean7",
            "date_weekday"]

# ML-Found の docs/DESIGN.md の分位点の損失の照合の見本 (実際の値, 予測,
# 水準, 水準ごとの損失, 平均)
PINBALL_SAMPLES = [
    ([10, 4, 7], [[8], [6], [7]], [0.5], [2 / 3], 2 / 3),
    ([10, 4, 0], [[6, 9, 13], [2, 5, 4], [-1, 0, 2]], [0.1, 0.5, 0.9],
     [7 / 30, 1 / 3, 1 / 6], 11 / 45),
]

# 同じく見込みの損の見本 (需要, 発注の数, 品切れの損, 売れ残りの損, 比べる
# 発注の数, 足りない数, 余った数, 損, 比べる発注の数での損, total,
# baseline_total, saving, ratio)
COST_SAMPLES = [
    ([10, 5, 8, 0], [12, 3, 8, 2], 3, 1, [7, 9, 8, 0],
     [0, 2, 0, 0], [2, 0, 0, 2], [2, 6, 0, 2], [9, 4, 0, 0],
     10, 13, 3, 10 / 13),
    ([4, 6], [6, 2.5], 2, 0.5, None, [0, 3.5], [2, 0], [1, 7], None,
     8, math.nan, math.nan, math.nan),
    ([3, 0], [3, 1], 4, 1, [3, 0], [0, 0], [0, 1], [0, 1], [0, 0],
     1, 0, -1, math.nan),
]


def daily_rows(stock: bool) -> pd.DataFrame:
    """
    商品ごとの日ごとの売れた数の合成の記録を作る関数

    Parameters
    ----------
    stock: bool
        在庫の数の列を持たせるか

    Returns
    ----------
    frame: pd.DataFrame
        record_id, product_id, date, category, units, stock (stock のとき
        だけ) の表
    """
    # 商品と曜日で水準の違う売れた数
    rng = np.random.default_rng(0)
    rows = []
    days = pd.date_range("2026-01-01", periods=DAYS)
    for product in range(PRODUCTS):
        for number, day in enumerate(days):
            rows.append({
                "record_id": f"r{product}_{number:02d}",
                "product_id": f"p{product:02d}",
                "date": day.strftime("%Y-%m-%d"),
                "category": float(product % 2),
                "units": float(rng.poisson(2 + product
                                           + 2 * (day.weekday() >= 5))),
                "stock": float(rng.integers(0, 6)),
            })
    frame = pd.DataFrame(rows)
    if not stock:
        frame = frame.drop(columns=["stock"])

    return frame


def demand_spec() -> dict:
    """
    列の数と水準の数に合わせた需要のモデルの定義を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    spec: dict
        ML-ModelControl の同梱の例 tabular_demand の入力の幅と水準の数を
        変えたもの
    """
    # 入力の幅と, 最初と最後の全結合の幅
    spec = load_example("tabular_demand")
    spec["inputs"]["x"]["shape"] = [8, len(FEATURES)]
    spec["nodes"]["demand1"]["params"]["in_features"] = len(FEATURES)
    spec["nodes"]["demand2"]["params"]["out_features"] = len(LEVELS)

    return spec


def skeleton_quantiles(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    """
    学習のスクリプトと同じ式で, 行ごとの分位点を予測する関数

    Parameters
    ----------
    model: torch.nn.Module
        需要のモデル
    x: np.ndarray
        (行の数, 列の数) の float32

    Returns
    ----------
    values: np.ndarray
        (行の数, 水準の数) の float64 の予測
    """
    # 出力が 1 つの定義のテンソル
    with torch.no_grad():
        outputs = model(x=torch.tensor(np.asarray(x, dtype=np.float32)))
    values = np.sort(np.clip(outputs.double().numpy(), 0.0, None), axis=1)

    return values


def write_demand_run(
    root: Path, raw: Path, stock: bool, spec: bool = True
) -> Path:
    """
    時間をまたぐ量の判断の学習のスクリプトと同じ成果物を持つ実験の
    ディレクトリを作る関数

    Parameters
    ----------
    root: Path
        作業のディレクトリ (データセットと実験を置く)
    raw: Path
        元の日ごとの記録の表の CSV
    stock: bool
        在庫の数の列を使うか
    spec: bool = True
        モデルの定義を実験に置くか (置かなければ仮のモデルの実験)

    Returns
    ----------
    run: Path
        実験のディレクトリ
    """
    # 需要の記録の表を ML-EDA で書き出す
    data_dir = root / "dataset"
    recipe = {
        "task": "demand", "target": "units", "product": "product_id",
        "date": "date", "horizon": HORIZON, "id": "record_id",
        "steps": [
            {"op": "lag", "columns": ["units"], "lags": [1, 7],
             "windows": [7], "by": ["product_id"], "time": "date",
             "freq": "1D"},
            {"op": "datetime_parts", "columns": ["date"],
             "parts": ["weekday"]},
            {"op": "standardize", "columns": "auto"},
        ],
        "split": {"method": "time", "valid": 0.2, "test": 0.2},
    }
    if stock:
        recipe["stock"] = "stock"
    export_dataset(load_data(raw), recipe, data_dir)
    dataset = load_dataset(data_dir)
    assert dataset.features == FEATURES
    run = root / "run"
    run.mkdir()

    # 学習しない重みのモデル
    model = build_model(demand_spec(), seed=0, load_weights=False)
    model.eval()
    torch.save(model.state_dict(), run / "final_model.pt")
    if spec:
        (run / "model_spec.yaml").write_text(
            yaml.safe_dump(demand_spec(), allow_unicode=True),
            encoding="utf-8"
        )

    # 分割ごとの発注を決める日の行の発注の数
    tau, _ = tau_of(SHORTAGE, HOLDING)
    tables = []
    for split, frame in dataset.frames.items():
        values = skeleton_quantiles(model, dataset.arrays(split)[0])
        held = dataset.stock(split)
        held = np.zeros(len(frame)) if held is None else held
        best = best_orders(LEVELS, values, held, tau, SHORTAGE, HOLDING,
                           MAX_ORDER)
        tables.append(pd.DataFrame({
            "product_id": frame["product_id"].astype(str),
            "date": pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d"),
            "split": split,
            "stock": dataset.stock(split) if stock else math.nan,
            "quantile": best["quantile"], "order": best["order"],
            "previous": dataset.previous(split),
            "demand": frame["demand"].to_numpy(),
        }))
    pd.concat(tables).to_csv(run / "demand_orders.csv", index=False)

    # 需要の決め方, レシピの写し, 評価
    train = dataset.info["demand"]["splits"]["train"]
    estimator = {
        "format_version": 1, "recipe": "demand_recipe.yaml",
        "orders": "demand_orders.csv", "features": FEATURES,
        "product": "product_id", "date": "date", "units": "units",
        "stock": "stock" if stock else None, "id": "record_id",
        "horizon": HORIZON, "interval": HORIZON, "levels": LEVELS,
        "shortage_cost": SHORTAGE, "holding_cost": HOLDING,
        "max_order": MAX_ORDER,
        "model": {"input": "x", "quantiles": "quantiles"},
        "train": {"rows": train["rows"], "products": train["products"],
                  "demand_mean": train["demand_mean"]},
    }
    (run / "demand_estimator.json").write_text(
        json.dumps(estimator, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    (run / "demand_recipe.yaml").write_text(
        (data_dir / "recipe.yaml").read_text(encoding="utf-8"),
        encoding="utf-8"
    )
    (run / "evaluations").mkdir()
    row = {"rows": 8, "products": PRODUCTS, "mae": 4.5,
           "pinball_loss": 1.25, "order_cost": 40.0, "baseline_cost": 50.0,
           "order_cost_saving": 10.0, "order_cost_ratio": 0.8, "capped": 0}
    (run / "evaluations" / "demand_valid.json").write_text(json.dumps({
        "name": "demand_valid", "split": "valid", "kind": "table",
        "rows": [row],
    }), encoding="utf-8")
    (run / "evaluations" / "demand_test.json").write_text(json.dumps({
        "name": "demand_test", "split": "test", "kind": "table",
        "rows": [{**row, "order_cost_ratio": None, "baseline_cost": 0.0,
                  "order_cost_saving": -40.0}],
    }), encoding="utf-8")

    return run


def raw_table(tmp_path: Path, stock: bool) -> Path:
    """
    合成の日ごとの記録を CSV に書く関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    stock: bool
        在庫の数の列を持たせるか

    Returns
    ----------
    path: Path
        CSV のファイル
    """
    # 在庫の数の列の有無ごとの表
    path = tmp_path / f"daily_{int(stock)}.csv"
    daily_rows(stock).to_csv(path, index=False)

    return path


def made_bundle(tmp_path: Path, stock: bool) -> Path:
    """
    実験と需要の一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    stock: bool
        在庫の数の列を使うか

    Returns
    ----------
    path: Path
        需要の一式のディレクトリ
    """
    # 元の表を渡さない一式
    raw = raw_table(tmp_path, stock)
    run = write_demand_run(tmp_path / "work", raw, stock)
    make_demand_bundle(tmp_path / "bundle", run)
    path = tmp_path / "bundle"

    return path


@pytest.fixture(params=[False, True])
def stock(request: pytest.FixtureRequest) -> bool:
    """
    在庫の数の列を使うかを返す関数

    Parameters
    ----------
    request: pytest.FixtureRequest
        引数の値

    Returns
    ----------
    stock: bool
        在庫の数の列を使うか
    """
    # 在庫の数の列の有無
    stock = bool(request.param)

    return stock


def test_demand_bundle(tmp_path: Path, stock: bool) -> None:
    """
    需要の一式の予測と, 水準 τ の分位点から発注の数を決める判断を確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    stock: bool
        在庫の数の列を使うか

    Returns
    ----------
    None
    """
    # 一式づくりと, 元の表で決めた発注の数と実験の発注の数の比べ
    raw = raw_table(tmp_path, stock)
    run = write_demand_run(tmp_path / "work", raw, stock)
    info = make_demand_bundle(tmp_path / "bundle", run, data=raw)
    assert info["format_version"] == 14
    assert info["role"] == "predict"
    assert info["output"] == {"form": "distribution", "quantiles": LEVELS}
    assert info["decision"] == {"type": "order", "shortage_cost": SHORTAGE,
                                "holding_cost": HOLDING,
                                "max_order": MAX_ORDER}
    section = info["demand"]
    assert section["features"] == FEATURES and section["units"] == "units"
    assert section["stock"] == ("stock" if stock else None)
    check = section["check"]
    orders = pd.read_csv(run / "demand_orders.csv")
    assert check["rows"] == len(orders) == section["rows"]
    assert check["agreement"] == 1.0 and check["max_diff"] < 1e-6
    assert check["pinball_loss"] > 0
    assert section["evaluations"]["test"]["order_cost_ratio"] is None
    for name in ("demand_estimator.json", "demand_recipe.yaml",
                 "model_spec.yaml", "final_model.pt"):
        assert (tmp_path / "bundle" / name).is_file()

    # 説明
    bundle = load_bundle(tmp_path / "bundle")
    found = describe_bundle(bundle)
    assert (found["kind"], found["task"]) == ("demand", "")
    assert found["features"] == FEATURES
    assert found["steps"] == ["lag", "datetime_parts", "standardize"]
    assert found["demand"]["levels"] == LEVELS
    assert found["price"] is None and not found["column_output"]

    # 商品ごとの最後の日の行 (その日の売れた数は要らない)
    rows = daily_rows(stock)
    last = rows.groupby("product_id").tail(1).index
    rows.loc[last, "units"] = None
    frame = predict(bundle, rows)
    names = [f"q:{level:g}" for level in LEVELS]
    assert list(frame.columns[-11:]) == [
        "status", "reason", "order_date", "on_hand", "previous_demand",
        "prediction", *names
    ]
    assert len(frame) == PRODUCTS and (frame["status"] == "ok").all()
    assert frame["record_id"].tolist() == rows.loc[last, "record_id"].tolist()
    assert (frame["order_date"] == "2026-03-11").all()
    np.testing.assert_allclose(frame["prediction"], frame["q:0.5"])
    values = frame[names].to_numpy()
    assert (np.diff(values, axis=1) >= 0).all()
    previous = rows[(rows["date"] >= "2026-03-04")
                    & (rows["date"] < "2026-03-11")]
    np.testing.assert_allclose(
        frame["previous_demand"],
        previous.groupby("product_id")["units"].sum().to_numpy()
    )
    np.testing.assert_allclose(
        frame["on_hand"],
        rows.loc[last, "stock"].to_numpy() if stock else np.zeros(PRODUCTS)
    )

    # 結論と既定の発注の判断 (τ = 0.75 は学んだ水準)
    result = infer(bundle, rows)
    summary = result["summary"]
    assert (summary["predicted"], summary["skipped"]) == (PRODUCTS, 0)
    assert summary["product"] == "product_id"
    assert summary["stock"] == ("stock" if stock else None)
    assert summary["dates"] == {"min": "2026-03-11", "max": "2026-03-11"}
    assert summary["prediction"]["total"] == pytest.approx(
        frame["prediction"].sum()
    )
    assert "先の 7 日の需要の分位点を予測した" in result["sentences"][0]
    assert any("検証用の需要の 0.5 の分位点の平均絶対誤差 4.5" in s
               for s in result["sentences"])
    assert any("在庫の数の列が無い" in s for s in result["sentences"]) == (
        not stock
    )
    decided = result["frame"]
    np.testing.assert_allclose(decided["quantile"], decided["q:0.75"])
    wanted = decided["quantile"] - decided["on_hand"]
    assert (decided["order"] >= 0).all()
    assert (decided["order"] <= MAX_ORDER).all()
    assert (decided["order"] == decided["order"].round()).all()
    assert ((decided["order"] - wanted.clip(lower=0)).abs() <= 1).all()
    assert not decided["capped"].any()
    assert decided["action"].tolist() == (decided["order"] > 0).tolist()
    assert (decided["expected_cost"]
            <= decided["baseline_cost"] + 1e-9).all()
    decision = result["decision"]
    assert decision["summary"]["rows"] == PRODUCTS
    assert decision["summary"]["level"] == 0.75
    assert not decision["summary"]["outside"]
    assert decision["summary"]["total"] == decided["order"].sum()
    assert decision["summary"]["compared"] == PRODUCTS
    assert decision["summary"]["saving"] >= 0
    count = len(decision["notes"])
    assert decision["sentences"][:count] == decision["notes"]
    assert "商品で発注する" in decision["sentences"][count]
    assert any("前の期間と同じ数を発注すると" in s
               for s in decision["sentences"])

    # 空の辞書は発注の判断無し, 書き出しと読み直し
    plain = infer(bundle, rows, decision={})
    assert "decision" not in plain and "order" not in plain["frame"]
    write_result(result, tmp_path / "result")
    back = read_result(tmp_path / "result")
    assert back["frame"]["order"].tolist() == decided["order"].tolist()


def test_order_decision(tmp_path: Path) -> None:
    """
    発注の判断の形と, 水準の補い, 上限, 在庫の扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # ML-Quest の action.decision の発注の判断と同じ名前と形だけの受け入れ
    # (損と上限は省けない)
    bundle = load_bundle(made_bundle(tmp_path, True))
    quest = {"type": "order", "shortage_cost": 3, "holding_cost": 1,
             "max_order": 60}
    assert check_decision(quest, bundle) == []
    for change, words in (({"max_order": None}, "省けません"),
                          ({"max_order": -1}, "max_order"),
                          ({"max_order": True}, "max_order"),
                          ({"shortage_cost": 0}, "shortage_cost"),
                          ({"holding_cost": -1}, "holding_cost"),
                          ({"holding_cost": "1"}, "holding_cost"),
                          ({"stock": 3}, "書けない項目")):
        errors = check_decision({**quest, **change}, bundle)
        assert any(words in e for e in errors), (change, errors)
    lost = {key: value for key, value in quest.items()
            if key != "shortage_cost"}
    assert any("省けません" in e for e in check_decision(lost, bundle))
    assert check_decision({"type": "price", "candidates": [0],
                           "max_change": 0.1}, bundle)

    # τ = 2/3 は隣の 2 つの水準 (0.5 と 0.75) を直線で補う
    rows = daily_rows(True)
    prediction = predict(bundle, rows)
    between = decide(bundle, prediction, {**quest, "shortage_cost": 2})
    weight = (2 / 3 - 0.5) / 0.25
    np.testing.assert_allclose(
        between["frame"]["quantile"],
        (1 - weight) * prediction["q:0.5"] + weight * prediction["q:0.75"]
    )
    assert not any("範囲" in note for note in between["notes"])

    # 学んだ水準の外は端の水準 (前提の文で知らせる)
    above = decide(bundle, prediction, {**quest, "shortage_cost": 19})
    np.testing.assert_allclose(above["frame"]["quantile"],
                               prediction["q:0.9"])
    assert above["summary"]["outside"]
    assert above["summary"]["used_level"] == 0.9
    assert any("範囲 [0.1, 0.9] の外" in note for note in above["notes"])
    below = decide(bundle, prediction, {**quest, "holding_cost": 57})
    np.testing.assert_allclose(below["frame"]["quantile"],
                               prediction["q:0.1"])
    assert any("端の水準 0.1" in note for note in below["notes"])

    # 上限を超えない発注と, 上限で切った商品の印
    # (上限 1.5 の整数の上限は 1. 上限が無ければ 2 以上を発注する商品に印)
    free = decide(bundle, prediction, {**quest, "max_order": 1000})
    held = decide(bundle, prediction, {**quest, "max_order": 1.5})
    frame = held["frame"]
    assert (frame["order"] <= 1).all()
    assert frame["capped"].tolist() == (free["frame"]["order"] > 1).tolist()
    assert held["summary"]["capped"] == int(frame["capped"].sum()) > 0
    np.testing.assert_allclose(frame["order"],
                               free["frame"]["order"].clip(upper=1))
    assert any("上限で切った商品" in s for s in held["sentences"])
    zero = decide(bundle, prediction, {**quest, "max_order": 0})
    assert (zero["frame"]["order"] == 0).all()
    assert not zero["frame"]["action"].any()

    # 在庫の数の列の無い表は在庫 0, 読めない在庫の数の行は飛ばす
    plain = infer(bundle, rows.drop(columns=["stock"]))
    assert (plain["frame"]["on_hand"] == 0).all()
    assert plain["summary"]["stock"] is None
    assert any("表に在庫の数の列 stock が無い" in s
               for s in plain["sentences"])
    assert any("在庫は 0 として発注の数を決めた" in note
               for note in plain["decision"]["notes"])
    broken = rows.astype({"stock": object})
    broken.loc[broken.groupby("product_id").tail(1).index[0], "stock"] = -1
    found = infer(bundle, broken)
    assert found["frame"].loc[0, "reason"] == (
        "在庫の数が 0 以上の数ではありません"
    )
    assert found["frame"].loc[0, "decision"] == "予測なし"
    assert found["decision"]["summary"]["rows"] == PRODUCTS - 1
    assert pd.isna(found["frame"].loc[0, "order"])

    # 結論の決め方とたどり直しは誤り
    with pytest.raises(ValueError, match="需要の一式には"):
        infer(bundle, rows, rules={"min_confidence": 0.5})
    with pytest.raises(ValueError, match="需要の一式には"):
        backtest(bundle, rows, quest)


def test_daily_table(tmp_path: Path) -> None:
    """
    日ごとの記録の表の誤りと, 前の期間の欠けを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同じ商品の同じ日, 商品の id の欠け, 日付の誤り, 列の無い表
    bundle = load_bundle(made_bundle(tmp_path, False))
    rows = daily_rows(False)
    for change, words in (
        (lambda f: pd.concat([f, f.iloc[:1]]), "同じ商品の同じ日"),
        (lambda f: f.assign(product_id=f["product_id"].where(
            f.index != 3)), "欠けた値"),
        (lambda f: f.assign(date=f["date"].where(f.index != 3, "不明")),
         "読めない"),
        (lambda f: f.assign(date=f["date"].where(
            f.index != 3, "2026-01-04 10:00")), "時刻"),
        (lambda f: f.drop(columns=["date"]), "日付の列 date"),
    ):
        with pytest.raises(ValueError, match=words):
            predict(bundle, change(rows))

    # 前の期間に抜けの日のある商品は前の期間の合計が無く, 比べに数えない
    gap = rows.drop(index=rows[(rows["product_id"] == "p01")
                               & (rows["date"] == "2026-03-07")].index)
    result = infer(bundle, gap)
    frame = result["frame"]
    assert frame["previous_demand"].isna().tolist() == [
        False, True, False, False
    ]
    assert frame["baseline_order"].isna().tolist() == [
        False, True, False, False
    ]
    assert result["decision"]["summary"]["compared"] == PRODUCTS - 1
    assert result["summary"]["previous"]["counted"] == PRODUCTS - 1


def test_best_orders() -> None:
    """
    発注の数と見込みの損が, 総当たりと細かい数え方と同じになることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 分位点の関数 (水準の間は直線, 外は端の値) で決まる分布の細かい積み
    rng = np.random.default_rng(1)
    values = np.sort(rng.uniform(0, 30, (6, len(LEVELS))), axis=1)
    values[0] = 5.0
    stock = rng.integers(0, 8, 6).astype(float)
    grid = (np.arange(200000) + 0.5) / 200000
    demand = np.stack([np.interp(grid, LEVELS, row) for row in values])
    for supply in (stock + 3.0, stock + 12.4):
        found = expected_costs(LEVELS, values, supply, 3.0, 1.0)
        fine = (3.0 * np.clip(demand - supply[:, None], 0, None)
                + np.clip(supply[:, None] - demand, 0, None)).mean(axis=1)
        np.testing.assert_allclose(found, fine, rtol=1e-4, atol=1e-6)

    # 上限の中の整数の発注の数の総当たり (τ = 2/3 と, 学んだ水準の外)
    for shortage, holding, limit in ((2.0, 1.0, 40.0), (19.0, 1.0, 9.5),
                                     (1.0, 30.0, 40.0)):
        tau, _ = tau_of(shortage, holding)
        best = best_orders(LEVELS, values, stock, tau, shortage, holding,
                           limit)
        choices = np.arange(0, math.floor(limit) + 1, dtype=float)
        costs = np.stack([
            expected_costs(LEVELS, values, stock + c, shortage, holding)
            for c in choices
        ], axis=1)
        np.testing.assert_allclose(best["order"],
                                   choices[np.argmin(costs, axis=1)])
        np.testing.assert_allclose(best["expected_cost"], costs.min(axis=1))
        assert (best["order"] <= limit).all()


def test_demand_metrics() -> None:
    """
    分位点の損失と見込みの損が ML-Found の見本と同じ値になることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 分位点の損失の見本
    for actual, quantiles, levels, loss, mean in PINBALL_SAMPLES:
        found = pinball_loss(actual, quantiles, levels)
        assert found["levels"]["level"] == levels
        np.testing.assert_allclose(found["levels"]["loss"], loss)
        assert found["mean"] == pytest.approx(mean)

    # 見込みの損の見本
    for (demand, order, shortage, holding, baseline, short, excess, cost,
         baseline_cost, total, baseline_total, saving, ratio) in (
        COST_SAMPLES
    ):
        found = order_cost(demand, order, shortage, holding,
                           baseline=baseline)
        items = found["items"]
        np.testing.assert_allclose(items["shortage"], short)
        np.testing.assert_allclose(items["excess"], excess)
        np.testing.assert_allclose(items["cost"], cost)
        if baseline is None:
            assert "baseline_cost" not in items
        else:
            np.testing.assert_allclose(items["baseline_cost"], baseline_cost)
        assert found["total"] == pytest.approx(total)
        assert found["baseline_total"] == pytest.approx(baseline_total,
                                                        nan_ok=True)
        assert found["saving"] == pytest.approx(saving, nan_ok=True)
        assert found["ratio"] == pytest.approx(ratio, nan_ok=True)

    # 誤りの入力
    with pytest.raises(ValueError, match="0 以上"):
        order_cost([1, 2], [1, -1], 1, 1)
    with pytest.raises(ValueError, match="holding_cost"):
        order_cost([1], [1], 1, 0)
    with pytest.raises(ValueError, match="水準"):
        pinball_loss([1], [[1, 2]], [0.5, 0.5])


def test_demand_sources(
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
    raw = raw_table(tmp_path, True)
    placeholder = write_demand_run(tmp_path / "a", raw, True, spec=False)
    with pytest.raises(ValueError, match="PlaceholderModel"):
        make_demand_bundle(tmp_path / "x", placeholder)

    # 成果物の版と形の誤り
    run = write_demand_run(tmp_path / "b", raw, True)
    path = run / "demand_estimator.json"
    estimator = json.loads(path.read_text("utf-8"))
    for change, words in (
        ({"format_version": 2}, "版 2"),
        ({"features": ["category", "stock"]}, "予測に使う列に stock"),
        ({"levels": [0.5, 0.1]}, "小さい順"),
        ({"levels": [0, 0.5]}, "levels"),
        ({"shortage_cost": 0}, "shortage_cost"),
        ({"max_order": None}, "省けません"),
        ({"horizon": 0}, "horizon"),
        ({"horizon": 14}, "horizon"),
        ({"stock": None}, "stock"),
        ({"model": {"input": "x"}}, "quantiles"),
        ({"train": {"rows": -1, "products": 0, "demand_mean": None}},
         "train"),
    ):
        path.write_text(json.dumps({**estimator, **change}),
                        encoding="utf-8")
        with pytest.raises(ValueError, match=words):
            make_demand_bundle(tmp_path / "x", run)
    path.write_text(json.dumps(estimator), encoding="utf-8")
    assert not (tmp_path / "x").exists()

    # JSON の外の NaN を持つ評価と, 列の違う発注の数の表
    evaluation = run / "evaluations" / "demand_valid.json"
    original = evaluation.read_text("utf-8")
    evaluation.write_text(original.replace("4.5", "NaN"), encoding="utf-8")
    with pytest.raises(ValueError, match="NaN"):
        make_demand_bundle(tmp_path / "y", run)
    evaluation.write_text(original, encoding="utf-8")
    orders = run / "demand_orders.csv"
    table = pd.read_csv(orders)
    table.drop(columns=["previous"]).to_csv(orders, index=False)
    with pytest.raises(ValueError, match="の列は"):
        make_demand_bundle(tmp_path / "y", run)
    table.to_csv(orders, index=False)

    # 分位点の出力の無い定義と, 水準の数と合わない出力
    spec = demand_spec()
    spec["outputs"] = {"levels": "demand_cumsum"}
    (run / "model_spec.yaml").write_text(yaml.safe_dump(spec),
                                         encoding="utf-8")
    with pytest.raises(ValueError, match="quantiles"):
        make_demand_bundle(tmp_path / "y", run)
    spec = demand_spec()
    spec["nodes"]["demand2"]["params"]["out_features"] = 3
    (run / "model_spec.yaml").write_text(yaml.safe_dump(spec),
                                         encoding="utf-8")
    weights = (run / "final_model.pt").read_bytes()
    torch.save(build_model(spec, seed=0, load_weights=False).state_dict(),
               run / "final_model.pt")
    with pytest.raises(ValueError, match=r"\(2, 5\) ではありません"):
        make_demand_bundle(tmp_path / "y", run)
    (run / "final_model.pt").write_bytes(weights)
    (run / "model_spec.yaml").write_text(yaml.safe_dump(demand_spec()),
                                         encoding="utf-8")

    # 商品の id の列の無い元の表は比べない
    rows = pd.read_csv(raw).drop(columns=["product_id"])
    info = make_demand_bundle(tmp_path / "bundle", run, data=rows)
    assert info["demand"]["check"] is None
    assert any("比べません" in note for note in info["notes"])

    # 版 13 までしか知らない読む側と, 食い違う一式
    monkeypatch.setattr(bundle_module, "BUNDLE_FORMAT_VERSION", 13)
    with pytest.raises(ValueError, match="format_version 14 の一式"):
        load_bundle(tmp_path / "bundle")
    monkeypatch.undo()
    path = tmp_path / "bundle" / "inference.yaml"
    info = yaml.safe_load(path.read_text("utf-8"))
    for change, words in (({"format_version": 13}, "demand を持つ一式"),
                          ({"output": {"form": "value"}}, "distribution"),
                          ({"role": "effect"}, "受け持ち"),
                          ({"output": {"form": "distribution",
                                       "quantiles": [0.5]}},
                           "output.quantiles"),
                          ({"demand": {**info["demand"],
                                       "max_order": None}}, "省けません"),
                          ({"demand": {**info["demand"],
                                       "holding_cost": 2}},
                           "holding_cost")):
        path.write_text(yaml.safe_dump({**info, **change},
                                       allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match=words):
            load_bundle(tmp_path / "bundle")


def test_skeleton_output(tmp_path: Path) -> None:
    """
    骨組みが書いた成果物から一式を作り, 元の表で決めた発注の数が実験の
    発注の数と一致することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 実験の決め方 (在庫の数の列を使う発注の判断)
    run = tmp_path / "run"
    shutil.copytree(SKELETON_OUTPUT, run)
    estimator = json.loads((run / "demand_estimator.json")
                           .read_text("utf-8"))
    assert (estimator["stock"], estimator["levels"]) == ("stock", LEVELS)
    assert (estimator["shortage_cost"], estimator["holding_cost"],
            estimator["max_order"]) == (SHORTAGE, HOLDING, MAX_ORDER)

    # 一式づくりと, 元の表で決めた発注の数と実験の発注の数の比べ
    info = make_demand_bundle(tmp_path / "bundle", run,
                              data=run / "daily.csv")
    orders = pd.read_csv(run / "demand_orders.csv")
    assert info["format_version"] == 14
    assert info["decision"] == {"type": "order", "shortage_cost": SHORTAGE,
                                "holding_cost": HOLDING,
                                "max_order": MAX_ORDER}
    check = info["demand"]["check"]
    assert check["rows"] == len(orders)
    assert check["agreement"] == 1.0
    # (骨組みの float32 のまとまりごとの予測との丸めの分)
    assert check["max_diff"] < 1e-4
    assert set(info["demand"]["evaluations"]) == {"valid", "test"}
    assert not info["notes"]

    # 発注を決める日までの表での推論の, 商品ごとの発注の数と在庫と前の期間
    rows = pd.read_csv(run / "daily.csv")
    for day, recorded in orders.groupby("date"):
        result = infer(tmp_path / "bundle", rows[rows["date"] <= day])
        decided = result["frame"].set_index("product_id")
        decided = decided.loc[recorded["product_id"]]
        assert (decided["order_date"] == day).all()
        np.testing.assert_array_equal(decided["order"], recorded["order"])
        np.testing.assert_allclose(decided["quantile"],
                                   recorded["quantile"], atol=1e-4)
        np.testing.assert_allclose(decided["on_hand"], recorded["stock"])
        np.testing.assert_allclose(decided["previous_demand"],
                                   recorded["previous"])
