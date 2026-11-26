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
from ml_inference.price import price_elasticity, profit_lift
from ml_inference import (
    backtest, check_decision, decide, describe_bundle, infer, load_bundle,
    make_bundle, make_price_bundle, predict, read_result, write_result
)

# 行の数, 値段の候補と変える幅の上限 (ML-Quest の docs/DESIGN.md の 5 節
# の値段の判断の例)
ROWS = 60
CANDIDATES = [-0.1, -0.05, 0, 0.05, 0.1]
MAX_CHANGE = 0.1

# ML-Quest の値段を変える判断の骨組みの train.py を書き換えずに短く動かした
# 実験の成果物と元の表 (商品ごとに選ぶ実験と, 商品の id と原価の無い行ごと
# に選ぶ実験)
SKELETON_OUTPUT = Path(__file__).parent / "price_skeleton"
SKELETON_ROWS = Path(__file__).parent / "price_skeleton_rows"

# 予測に使う列 (最後が値段の列)
FEATURES = ["category", "weekday", "price"]

# ML-Found の docs/DESIGN.md の弾力性の照合の見本 (今の値段と 1 % 上げた
# 値段での予測した数, 弾力性, 平均, 平均に数えた数)
ELASTICITY_SAMPLES = [
    ([100, 50, 0, 20, 0], [98, 50.5, 3, 0, 0],
     [-2, 1, math.nan, -100, math.nan], -101 / 3, 3),
    ([0, 0], [1, 0], [math.nan, math.nan], math.nan, 0),
]

# 同じく利益の比べの見本 (今の値段, 原価, 候補, 上限, 予測した数, 選んだ
# 割合, 選んだ値段, 利益, 今の利益, total, current_total, gain, ratio,
# changed)
LIFT_SAMPLES = [
    ([10, 20, 5], [6, 12, 5], [-0.1, 0, 0.1, 0.2], 0.1,
     [[120, 100, 85, 80], [60, 50, 40, 10], [30, 20, 10, 0]],
     [0.1, 0, 0.1], [11, 20, 5.5], [425, 400, 5], [400, 400, 0],
     830, 800, 30, 83 / 80, 2),
    ([10, 8, 5], None, [-0.2, 0, 0.2], None,
     [[130, 100, 70], [10, 10, 9], [20, 18, 10]],
     [-0.2, 0.2, 0], [8, 9.6, 5], [1040, 86.4, 90], [1000, 80, 90],
     1216.4, 1170, 46.4, 3041 / 2925, 2),
    ([10, 10], [12, 10], [0, 0.5], None, [[5, 2], [0, 0]],
     [0.5, 0], [15, 10], [6, 0], [-10, 0], 6, -10, 16, math.nan, 1),
]


def sales_rows(product: bool) -> pd.DataFrame:
    """
    値段を無作為に変えた合成の売れ行きの記録を作る関数

    Parameters
    ----------
    product: bool
        商品の id の列を持たせるか (2 行ずつ同じ商品)

    Returns
    ----------
    frame: pd.DataFrame
        record_id, product_id (product のときだけ), category, weekday,
        price, unit_cost, units の表
    """
    # 無作為の値段と, 種類と値段で決まる売れた数
    rng = np.random.default_rng(0)
    rows = []
    for number in range(ROWS):
        category = float(number % 3)
        price = float(rng.choice([8.0, 10.0, 12.0]))
        rows.append({
            "record_id": f"r{number:03d}",
            "product_id": f"p{number // 2:02d}",
            "category": category,
            "weekday": float(rng.integers(0, 7)),
            "price": price,
            "unit_cost": 4.0 + category,
            "units": float(max(0, 40 - (1 + category) * price
                               + rng.integers(0, 5))),
        })
    frame = pd.DataFrame(rows)
    if not product:
        frame = frame.drop(columns=["product_id"])

    return frame


def price_spec() -> dict:
    """
    列の数に合わせた売れ行きのモデルの定義を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    spec: dict
        ML-ModelControl の同梱の例 tabular_price の入力の幅を変えたもの
    """
    # 入力の幅と最初の全結合の入力の幅
    spec = load_example("tabular_price")
    spec["inputs"]["x"]["shape"] = [8, len(FEATURES)]
    spec["nodes"]["units1"]["params"]["in_features"] = len(FEATURES)

    return spec


def skeleton_units(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    """
    骨組みと同じ式で, 行ごとの売れた数を予測する関数

    Parameters
    ----------
    model: torch.nn.Module
        売れ行きのモデル
    x: np.ndarray
        (行の数, 列の数) の float32

    Returns
    ----------
    units: np.ndarray
        float64 の予測 (負の値は 0 に切る)
    """
    # 出力が 1 つの定義のテンソル
    with torch.no_grad():
        outputs = model(x=torch.tensor(np.asarray(x, dtype=np.float32)))
    units = np.clip(outputs.reshape(-1).double().numpy(), 0.0, None)

    return units


def write_price_run(
    root: Path, raw: Path, product: bool, spec: bool = True
) -> Path:
    """
    値段を変える判断の骨組みと同じ成果物を持つ実験のディレクトリを作る関数

    Parameters
    ----------
    root: Path
        作業のディレクトリ (データセットと実験を置く)
    raw: Path
        元の表の CSV
    product: bool
        商品の id の列で選ぶか
    spec: bool = True
        モデルの定義を実験に置くか (置かなければ仮のモデルの実験)

    Returns
    ----------
    run: Path
        実験のディレクトリ
    """
    # 値段を変えた記録の表を ML-EDA で書き出す
    data_dir = root / "dataset"
    recipe = {"task": "price", "target": "units", "price": "price",
              "cost": "unit_cost", "id": "record_id",
              "steps": [{"op": "standardize", "columns": "auto"}],
              "split": {"method": "stratified", "valid": 0.2, "test": 0.2,
                        "seed": 0, "unlabeled": "drop"}}
    if product:
        recipe["product"] = "product_id"
    export_dataset(load_data(raw), recipe, data_dir)
    dataset = load_dataset(data_dir)
    run = root / "run"
    run.mkdir()

    # 学習しない重みのモデル
    model = build_model(price_spec(), seed=0, load_weights=False)
    model.eval()
    torch.save(model.state_dict(), run / "final_model.pt")
    if spec:
        (run / "model_spec.yaml").write_text(
            yaml.safe_dump(price_spec(), allow_unicode=True),
            encoding="utf-8"
        )

    # 今の値段 (商品ごとの全ての分割の中央値か, 行の値段)
    key = "product_id" if product else "record_id"
    frames = dataset.frames
    records = pd.concat([pd.DataFrame({
        "key": frames[split][key].astype(str),
        "price": dataset.price(split),
    }) for split in frames], ignore_index=True)
    medians = records.groupby("key", sort=False)["price"].median()

    # 分割ごと商品ごとの選んだ値段 (候補ごとの予測の商品ごとの平均)
    tables = []
    for split, frame in frames.items():
        keys = frame[key].astype(str).to_numpy()
        current = medians.loc[keys].to_numpy(np.float64)
        units = np.stack([
            skeleton_units(model, dataset.price_inputs(split,
                                                       current * (1 + c)))
            for c in CANDIDATES
        ], axis=1)
        raised = skeleton_units(model, dataset.price_inputs(
            split, current * 1.01
        ))
        order = pd.unique(keys)
        table = pd.DataFrame({"key": keys, "current": current,
                              "cost": dataset.cost(split),
                              "raised": raised})
        for j in range(len(CANDIDATES)):
            table[f"u{j}"] = units[:, j]
        means = table.groupby("key", sort=False).mean().loc[order]
        rows = table.groupby("key", sort=False).size().loc[order]
        found = means[[f"u{j}" for j in range(len(CANDIDATES))]].to_numpy()
        lift = profit_lift(means["current"], CANDIDATES, found,
                           means["cost"], MAX_CHANGE)
        elasticity = price_elasticity(found[:, CANDIDATES.index(0)],
                                      means["raised"])
        tables.append(pd.DataFrame({
            key: order, "split": split, "rows": rows.to_numpy(),
            "current_price": means["current"].to_numpy(),
            "cost": means["cost"].to_numpy(),
            "change": lift["items"]["change"],
            "price": lift["items"]["price"],
            "profit": lift["items"]["profit"],
            "current_profit": lift["items"]["current_profit"],
            "elasticity": elasticity["items"]["elasticity"],
        }))
    pd.concat(tables).to_csv(run / "price_choices.csv", index=False)

    # 値段の決め方, レシピの写し, 評価
    estimator = {
        "format_version": 1, "recipe": "price_recipe.yaml",
        "choices": "price_choices.csv", "features": FEATURES,
        "price": "price", "cost": "unit_cost",
        "product": "product_id" if product else None, "id": "record_id",
        "current": "median" if product else "row",
        "candidates": CANDIDATES, "max_change": MAX_CHANGE,
        "elasticity_step": 0.01,
        "model": {"input": "x", "units": "units", "link": "softplus"},
        "scale": 12.5,
        "train": {"products": 20, "changed": 5, "elasticity": -1.5},
    }
    (run / "price_estimator.json").write_text(
        json.dumps(estimator, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    (run / "price_recipe.yaml").write_text(
        (data_dir / "recipe.yaml").read_text(encoding="utf-8"),
        encoding="utf-8"
    )
    (run / "evaluations").mkdir()
    row = {"rows": 12, "products": 12, "mae": 3.5, "rmse": 4.25,
           "elasticity": -1.2, "elasticity_counted": 12, "profit": 830.0,
           "current_profit": 800.0, "profit_gain": 30.0,
           "profit_ratio": 1.0375, "changed": 7, "max_change": MAX_CHANGE}
    (run / "evaluations" / "price_valid.json").write_text(json.dumps({
        "name": "price_valid", "split": "valid", "kind": "table",
        "rows": [row],
    }), encoding="utf-8")
    (run / "evaluations" / "price_test.json").write_text(json.dumps({
        "name": "price_test", "split": "test", "kind": "table",
        "rows": [{**row, "elasticity": None, "profit_ratio": None,
                  "current_profit": -5.0, "profit": 0.0,
                  "profit_gain": 5.0}],
    }), encoding="utf-8")

    return run


def raw_table(tmp_path: Path, product: bool) -> Path:
    """
    合成の売れ行きの記録を CSV に書く関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    product: bool
        商品の id の列を持たせるか

    Returns
    ----------
    path: Path
        CSV のファイル
    """
    # 商品の id の有無ごとの表
    path = tmp_path / f"sales_{int(product)}.csv"
    sales_rows(product).to_csv(path, index=False)

    return path


@pytest.fixture(params=[False, True])
def product(request: pytest.FixtureRequest) -> bool:
    """
    商品の id の列で選ぶかを返す関数

    Parameters
    ----------
    request: pytest.FixtureRequest
        引数の値

    Returns
    ----------
    product: bool
        商品の id の列を持つか
    """
    # 行ごとと商品ごと
    product = bool(request.param)

    return product


def test_price_bundle(tmp_path: Path, product: bool) -> None:
    """
    値段の一式の予測と, 上限の中で利益が最も大きい値段の判断を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    product: bool
        商品の id の列で選ぶか

    Returns
    ----------
    None
    """
    # 一式づくりと, 元の表で選んだ値段と実験の選んだ値段の比べ
    raw = raw_table(tmp_path, product)
    run = write_price_run(tmp_path / "work", raw, product)
    info = make_price_bundle(tmp_path / "bundle", run, data=raw)
    assert info["format_version"] == 13
    assert (info["role"], info["output"]) == ("effect", {"form": "effect"})
    assert info["decision"] == {"type": "price", "candidates": CANDIDATES,
                                "max_change": MAX_CHANGE}
    section = info["price"]
    assert section["features"] == FEATURES and section["units"] == "units"
    assert section["current"] == ("median" if product else "row")
    check = section["check"]
    assert check["agreement"] == 1.0 and check["max_diff"] < 1e-6
    if product:
        # (分割をまたぐ商品は比べない)
        assert 0 < check["rows"] < section["rows"]
        assert any("比べたのは" in note for note in info["notes"])
    else:
        assert check["rows"] == ROWS == section["rows"]
    assert section["evaluations"]["test"]["profit_ratio"] is None
    for name in ("price_estimator.json", "price_recipe.yaml",
                 "model_spec.yaml", "final_model.pt"):
        assert (tmp_path / "bundle" / name).is_file()

    # 説明
    bundle = load_bundle(tmp_path / "bundle")
    found = describe_bundle(bundle)
    assert (found["kind"], found["task"]) == ("price", "")
    assert found["features"] == FEATURES and found["steps"] == ["standardize"]
    assert found["price"]["candidates"] == CANDIDATES
    assert found["uplift"] is None and not found["column_output"]

    # 新しい行 (売れた数の列は要らない. 前処理で欠けた行と, 値段と原価の
    # 読めない行は飛ばす)
    rows = sales_rows(product).drop(columns=["units"]).astype(
        {"price": object, "unit_cost": object}
    )
    rows.loc[1, "weekday"] = None
    rows.loc[2, "price"] = 0
    rows.loc[3, "unit_cost"] = "不明"
    frame = predict(bundle, rows)
    assert list(frame.columns[-7:]) == [
        "status", "reason", "current_price", "cost", "units",
        "current_profit", "elasticity"
    ]
    assert "price (入力)" in frame.columns and "price" not in frame.columns
    assert "前処理の後に欠損: weekday" in frame.loc[1, "reason"]
    assert frame.loc[2, "reason"] == "値段が 0 より大きい数ではありません"
    assert frame.loc[3, "reason"] == "原価が 0 以上の数ではありません"
    ok = frame[frame["status"] == "ok"]
    assert len(ok) == ROWS - 3
    np.testing.assert_allclose(ok["current_price"], ok["price (入力)"]
                               .astype(float))
    np.testing.assert_allclose(
        ok["current_profit"], (ok["current_price"] - ok["cost"])
        * ok["units"]
    )

    # 弾力性は 1 % 上げた値段との予測した数の差
    raised = rows.loc[ok.index].copy()
    raised["price"] = raised["price"].astype(float) * 1.01
    up = predict(bundle, raised)
    np.testing.assert_allclose(
        ok["elasticity"],
        100 * (up["units"].to_numpy() - ok["units"].to_numpy())
        / ok["units"].to_numpy()
    )

    # 結論と既定の値段の判断 (上限の中で利益が最も大きい値段)
    result = infer(bundle, rows)
    summary = result["summary"]
    assert (summary["predicted"], summary["skipped"]) == (ROWS - 3, 3)
    assert summary["id"] == "record_id" and summary["cost"] == "unit_cost"
    assert summary["product"] == ("product_id" if product else None)
    assert summary["units"]["total"] == pytest.approx(ok["units"].sum())
    assert summary["elasticity"]["counted"] == int(
        ok["elasticity"].notna().sum()
    )
    assert summary["measured"] is None
    assert "売れた数を今の値段 (行の値段の列) で予測した" in result[
        "sentences"
    ][0]
    assert any("検証用の売れた数の平均絶対誤差 3.5" in s
               for s in result["sentences"])
    assert any("比は無し" in s for s in result["sentences"])
    decided = result["frame"][result["frame"]["status"] == "ok"]
    assert decided["change"].abs().max() <= MAX_CHANGE
    assert (decided["profit"] >= decided["current_profit"] - 1e-9).all()
    np.testing.assert_allclose(
        decided["price"], decided["current_price"] * (1 + decided["change"])
    )
    assert result["frame"]["action"].iloc[:4].isna().tolist() == [
        False, True, True, True
    ]
    assert decided["action"].tolist() == (decided["change"] != 0).tolist()
    decision = result["decision"]
    assert decision["summary"]["rows"] == ROWS - 3
    assert decision["summary"]["total"] == pytest.approx(
        decided["profit"].sum()
    )
    assert decision["summary"]["gain"] >= 0
    assert decision["sentences"][:3] == decision["notes"]
    assert any("試しに値段を変えた記録" in note for note in decision["notes"])
    assert "行の値段を変える" in decision["sentences"][3]

    # 空の辞書は値段の判断無し, 書き出しと読み直し
    plain = infer(bundle, rows, decision={})
    assert "decision" not in plain and "change" not in plain["frame"]
    write_result(result, tmp_path / "result")
    back = read_result(tmp_path / "result")
    assert back["frame"]["price"].tolist() == pytest.approx(
        result["frame"]["price"].tolist(), nan_ok=True
    )


def test_price_decision(tmp_path: Path) -> None:
    """
    値段の判断の形と, 上限, 候補, 原価の扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # ML-Quest の action.decision の値段の判断と同じ名前と形だけの受け入れ
    # (max_change は省けない)
    raw = raw_table(tmp_path, False)
    run = write_price_run(tmp_path / "work", raw, False)
    make_price_bundle(tmp_path / "bundle", run)
    bundle = load_bundle(tmp_path / "bundle")
    quest = {"type": "price", "candidates": [-0.1, -0.05, 0, 0.05, 0.1],
             "max_change": 0.1}
    assert check_decision(quest, bundle) == []
    for change, words in (({"max_change": None}, "省けません"),
                          ({"max_change": -0.1}, "max_change"),
                          ({"max_change": True}, "max_change"),
                          ({"candidates": [0.1, 0.2]}, "今の値段 (0)"),
                          ({"candidates": [0, 0.1, 0.10]}, "同じ割合"),
                          ({"candidates": [0, -1]}, "-1 より大きい"),
                          ({"candidates": [0, True]}, "-1 より大きい"),
                          ({"candidates": "0, 0.1"}, "-1 より大きい"),
                          ({"cost": 3}, "書けない項目")):
        errors = check_decision({**quest, **change}, bundle)
        assert any(words in e for e in errors), (change, errors)
    lost = {key: value for key, value in quest.items()
            if key != "max_change"}
    assert any("省けません" in e for e in check_decision(lost, bundle))
    assert check_decision({"type": "treat", "budget": 2}, bundle)

    # 上限 0 は今の値段のまま, 上限の外の候補の方が利益の大きい行の数
    rows = sales_rows(False).drop(columns=["units"])
    prediction = predict(bundle, rows)
    same = decide(bundle, prediction, {**quest, "max_change": 0})
    assert same["summary"]["changed"] == 0
    assert same["summary"]["gain"] == 0
    assert not same["frame"]["action"].any()
    wide = {"type": "price", "candidates": [-0.5, 0, 0.5, 2.0],
            "max_change": 0.5}
    held = decide(bundle, prediction, wide)
    free = decide(bundle, prediction, {**wide, "max_change": 2.0})
    assert held["frame"]["change"].abs().max() <= 0.5
    better = free["frame"]["profit"] > held["frame"]["profit"] + 1e-6
    assert held["summary"]["limited"] == int(better.sum())
    assert free["summary"]["limited"] == 0
    assert any("上限を超える候補" in s for s in held["sentences"]) == (
        held["summary"]["limited"] > 0
    )

    # 原価の列の無い表は売上で選ぶ (原価 0)
    plain = rows.drop(columns=["unit_cost"])
    result = infer(bundle, plain)
    assert (result["frame"]["cost"] == 0).all()
    assert result["summary"]["cost"] is None
    assert any("表に原価の列 unit_cost が無い" in s
               for s in result["sentences"])
    frame = result["frame"]
    np.testing.assert_allclose(frame["profit"],
                               frame["price"] * frame["price_units"])
    assert any("売上 (原価 0) で比べた" in note
               for note in result["decision"]["notes"])

    # 結論の決め方, たどり直し, 列の予測の形の一式づくりは誤り
    with pytest.raises(ValueError, match="値段の一式には"):
        infer(bundle, raw, rules={"min_confidence": 0.5})
    with pytest.raises(ValueError, match="値段の一式には"):
        backtest(bundle, raw, quest)
    with pytest.raises(ValueError, match="make_price_bundle"):
        make_bundle(tmp_path / "other", price_spec(),
                    tmp_path / "work" / "dataset", run / "final_model.pt",
                    form="effect")


def test_price_metrics() -> None:
    """
    弾力性と利益の比べが ML-Found の見本と同じ値になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 弾力性の見本
    for now, raised, items, mean, counted in ELASTICITY_SAMPLES:
        found = price_elasticity(now, raised)
        np.testing.assert_allclose(found["items"]["elasticity"], items)
        assert found["mean"] == pytest.approx(mean, nan_ok=True)
        assert found["counted"] == counted

    # 利益の比べの見本
    for (price, cost, candidates, limit, units, change, chosen, profit,
         current, total, current_total, gain, ratio, changed) in (
        LIFT_SAMPLES
    ):
        found = profit_lift(price, candidates, units, cost=cost,
                            max_change=limit)
        items = found["items"]
        np.testing.assert_allclose(items["change"], change)
        np.testing.assert_allclose(items["price"], chosen)
        np.testing.assert_allclose(items["profit"], profit)
        np.testing.assert_allclose(items["current_profit"], current)
        assert found["total"] == pytest.approx(total)
        assert found["current_total"] == pytest.approx(current_total)
        assert found["gain"] == pytest.approx(gain)
        assert found["ratio"] == pytest.approx(ratio, nan_ok=True)
        assert found["changed"] == changed

    # 上限と割合は 10 進の表記どおりに比べる, 誤りの入力
    found = profit_lift([10.0], [-0.3, 0, 0.3], [[200, 100, 90]],
                        max_change=0.3)
    assert found["items"]["change"] == [-0.3]
    with pytest.raises(ValueError, match="0 以上"):
        profit_lift([10.0], [0, 0.1], [[1, -1]])
    with pytest.raises(ValueError, match="0 以上"):
        price_elasticity([1, -1], [1, 1])


def test_measured(tmp_path: Path) -> None:
    """
    売れた数のある表で, 今の値段での予測の誤差を数え直すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 売れた数のある元の表 (売れた数の欠けた行は除く)
    raw = raw_table(tmp_path, False)
    run = write_price_run(tmp_path / "work", raw, False)
    make_price_bundle(tmp_path / "bundle", run)
    rows = sales_rows(False)
    rows.loc[0, "units"] = None
    result = infer(tmp_path / "bundle", rows)
    measured = result["summary"]["measured"]
    assert (measured["rows"], measured["excluded"]) == (ROWS - 1, 1)
    used = result["frame"].drop(index=[0])
    errors = used["units"] - used["units (入力)"]
    assert measured["mae"] == pytest.approx(errors.abs().mean())
    assert measured["rmse"] == pytest.approx(
        math.sqrt((errors ** 2).mean())
    )
    assert any("渡した表の売れた数で数えた平均絶対誤差" in s
               for s in result["sentences"])


def test_price_sources(
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
    raw = raw_table(tmp_path, False)
    placeholder = write_price_run(tmp_path / "a", raw, False, spec=False)
    with pytest.raises(ValueError, match="PlaceholderModel"):
        make_price_bundle(tmp_path / "x", placeholder)

    # 成果物の版と形の誤り
    run = write_price_run(tmp_path / "b", raw, False)
    path = run / "price_estimator.json"
    estimator = json.loads(path.read_text("utf-8"))
    model = estimator["model"]
    for change, words in (
        ({"format_version": 2}, "版 2"),
        ({"current": "median"}, "current"),
        ({"features": ["price", "category", "weekday"]}, "最後は値段"),
        ({"features": ["category", "unit_cost", "price"]}, "unit_cost"),
        ({"candidates": [0.1, 0.2]}, r"今の値段 \(0\)"),
        ({"max_change": None}, "省けません"),
        ({"elasticity_step": 0.02}, "elasticity_step"),
        ({"model": {**model, "link": "identity"}}, "softplus"),
        ({"scale": 0}, "scale"),
        ({"train": {"products": -1, "changed": 0, "elasticity": None}},
         "train"),
    ):
        path.write_text(json.dumps({**estimator, **change}),
                        encoding="utf-8")
        with pytest.raises(ValueError, match=words):
            make_price_bundle(tmp_path / "x", run)
    path.write_text(json.dumps(estimator), encoding="utf-8")
    assert not (tmp_path / "x").exists()

    # JSON の外の NaN を持つ評価と, 列の違う選んだ値段の表
    evaluation = run / "evaluations" / "price_valid.json"
    original = evaluation.read_text("utf-8")
    evaluation.write_text(original.replace("3.5", "NaN"), encoding="utf-8")
    with pytest.raises(ValueError, match="NaN"):
        make_price_bundle(tmp_path / "y", run)
    evaluation.write_text(original, encoding="utf-8")
    choices = run / "price_choices.csv"
    table = pd.read_csv(choices)
    table.drop(columns=["elasticity"]).to_csv(choices, index=False)
    with pytest.raises(ValueError, match="の列は"):
        make_price_bundle(tmp_path / "y", run)
    table.to_csv(choices, index=False)

    # 売れた数の出力の無い定義
    spec = price_spec()
    spec["outputs"] = {"count": "units_softplus"}
    (run / "model_spec.yaml").write_text(yaml.safe_dump(spec),
                                         encoding="utf-8")
    with pytest.raises(ValueError, match="units"):
        make_price_bundle(tmp_path / "y", run)
    (run / "model_spec.yaml").write_text(yaml.safe_dump(price_spec()),
                                         encoding="utf-8")

    # 鍵の列の無い元の表は比べない
    rows = pd.read_csv(raw).drop(columns=["record_id"])
    info = make_price_bundle(tmp_path / "bundle", run, data=rows)
    assert info["price"]["check"] is None
    assert any("比べません" in note for note in info["notes"])

    # 版 12 までしか知らない読む側と, 食い違う一式
    monkeypatch.setattr(bundle_module, "BUNDLE_FORMAT_VERSION", 12)
    with pytest.raises(ValueError, match="format_version 13 の一式"):
        load_bundle(tmp_path / "bundle")
    monkeypatch.undo()
    path = tmp_path / "bundle" / "inference.yaml"
    info = yaml.safe_load(path.read_text("utf-8"))
    for change, words in (({"format_version": 12}, "price を持つ一式"),
                          ({"output": {"form": "value"}}, "effect"),
                          ({"role": "predict"}, "受け持ち"),
                          ({"price": {**info["price"],
                                      "max_change": None}}, "省けません"),
                          ({"price": {**info["price"],
                                      "candidates": [0, 0.2]}},
                           "candidates")):
        path.write_text(yaml.safe_dump({**info, **change},
                                       allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match=words):
            load_bundle(tmp_path / "bundle")


def test_skeleton_output(tmp_path: Path) -> None:
    """
    骨組みが書いた成果物から一式を作り, 値段を選べることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 行ごとに選ぶ実験の, 元の表で選んだ値段と実験の選んだ値段の比べ
    run = tmp_path / "rows"
    shutil.copytree(SKELETON_ROWS, run)
    estimator = json.loads((run / "price_estimator.json").read_text("utf-8"))
    assert (estimator["current"], estimator["cost"],
            estimator["product"]) == ("row", None, None)
    info = make_price_bundle(tmp_path / "rows_bundle", run,
                             data=run / "sales.csv")
    choices = pd.read_csv(run / "price_choices.csv")
    assert info["price"]["check"]["rows"] == len(choices)
    assert info["price"]["check"]["agreement"] == 1.0
    # (骨組みの float32 のまとまりごとの予測との丸めの分)
    assert info["price"]["check"]["max_diff"] < 1e-3
    assert info["decision"] == {"type": "price",
                                "candidates": estimator["candidates"],
                                "max_change": estimator["max_change"]}
    assert set(info["price"]["evaluations"]) == {"valid", "test"}
    result = infer(tmp_path / "rows_bundle", run / "sales.csv")
    assert (result["frame"]["cost"] == 0).all()
    assert result["frame"]["change"].abs().max() <= estimator["max_change"]

    # 商品ごとに選ぶ実験 (分割をまたぐ商品は比べない) と, 骨組みの今の値段
    # (商品ごとの元の値段の中央値)
    run = tmp_path / "products"
    shutil.copytree(SKELETON_OUTPUT, run)
    info = make_price_bundle(tmp_path / "bundle", run,
                             data=run / "sales.csv")
    assert info["price"]["current"] == "median"
    assert info["price"]["cost"] == "unit_cost"
    assert info["price"]["check"] is None
    assert any("比べません" in note for note in info["notes"])
    sales = pd.read_csv(run / "sales.csv")
    choices = pd.read_csv(run / "price_choices.csv")
    medians = sales.groupby("product_id")["price"].median()
    np.testing.assert_allclose(
        choices["current_price"], medians.loc[choices["product_id"]]
    )
    result = infer(tmp_path / "bundle", sales)
    decided = result["frame"]
    assert (decided["status"] == "ok").all()
    assert (decided["profit"] >= decided["current_profit"] - 1e-9).all()
    assert decided["change"].abs().max() <= estimator["max_change"]
    assert result["summary"]["evaluations"]["test"]["max_change"] == (
        estimator["max_change"]
    )
