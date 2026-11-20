from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import linear_spec, save_weights
from ml_eda import export_dataset, load_data
from test_remeasure import walk_bundle

from ml_inference import backtest, check_decision, current_from_baseline
from ml_inference import load_bundle, make_bundle


def shop_days() -> pd.DataFrame:
    """
    2 つの店の日ごとの売上の表を作る関数

    駅前店は 2025-01-05 が休み

    Parameters
    ----------
    None

    Returns
    ----------
    frame: pd.DataFrame
        date, shop, sales の表 (店の順でなく日付の順に混ぜて並べる)
    """
    # 店ごとに 10 日. 駅前店は 5 日目の行が無い
    days = pd.date_range("2025-01-01", periods=10).strftime("%Y-%m-%d")
    rows = [
        {"date": day, "shop": shop, "sales": float(base + number)}
        for number, day in enumerate(days)
        for shop, base in (("駅前店", 100), ("郊外店", 200))
        if not (shop == "駅前店" and day == "2025-01-05")
    ]
    frame = pd.DataFrame(rows)

    return frame


def test_lag_and_mean() -> None:
    """
    k 期前は同じ系列の暦の k 期前の値, 平均は直前の k 期の値の平均に
    なることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 期前: 同じ店のちょうど 2 日前. 休みの日の 2 日後は欠損 (近い行で
    # 代えない). 間隔は時刻の差の最頻値 (1 日)
    frame = shop_days()
    lag = current_from_baseline(
        frame, {"type": "lag", "periods": 2, "time": "date",
                "by": ["shop"]}, target="sales"
    )
    assert lag.name == "今のやり方"
    assert list(lag.index) == list(frame.index)
    station = frame["shop"] == "駅前店"
    expected = {"2025-01-03": 100.0, "2025-01-04": 101.0,
                "2025-01-07": np.nan, "2025-01-10": 107.0}
    for day, value in expected.items():
        got = lag[station & (frame["date"] == day)].iloc[0]
        assert (np.isnan(got) and np.isnan(value)) or got == value
    suburb = lag[(frame["shop"] == "郊外店") & (frame["date"] == "2025-01-03")]
    assert suburb.iloc[0] == 200.0
    assert lag[frame["date"] <= "2025-01-02"].isna().all()

    # 直前の 3 日の平均 (その日は含めない. 期間にある値だけの平均)
    mean = current_from_baseline(
        frame, {"type": "mean", "periods": 3, "time": "date",
                "by": "shop", "freq": "1D", "target": "sales"}
    )
    day6 = mean[station & (frame["date"] == "2025-01-06")].iloc[0]
    assert day6 == pytest.approx((102.0 + 103.0) / 2)
    first = mean[station & (frame["date"] == "2025-01-02")].iloc[0]
    assert first == 100.0

    # 列はそのまま (文字も読み替えない)
    marks = frame.assign(done=["はい", "いいえ"] * 9 + ["はい"])
    column = current_from_baseline(marks, {"type": "column",
                                           "column": "done"})
    assert list(column) == list(marks["done"])


def test_baseline_errors() -> None:
    """
    決まりの誤りと, 求められない場合に止まることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 種類, 期の数, 列, 時刻, 間隔
    frame = shop_days()
    cases = [
        ({"type": "median", "periods": 2}, "type は"),
        ({"type": "lag", "periods": 0, "time": "date"}, "periods"),
        ({"type": "lag", "periods": True, "time": "date"}, "periods"),
        ({"type": "column", "column": "no_such"}, "no_such"),
        ({"type": "lag", "periods": 1}, "時刻の列"),
        ({"type": "lag", "periods": 1, "time": "date", "freq": "x"}, "freq"),
        ({"type": "lag", "periods": 1, "time": "date", "by": ["x"]},
         "列 x が"),
        ({"type": "lag", "periods": 1.5, "time": "date"}, "periods"),
        ({"type": "lag", "periods": 1, "time": "date", "by": [1]}, "by"),
        ({"type": "lag", "periods": 1, "time": "date", "by": "sales"},
         "系列を分ける列にできません"),
        ({"type": "lag", "periods": 1, "time": "date", "extra": 1}, "extra"),
    ]
    for baseline, message in cases:
        with pytest.raises(ValueError, match=message):
            current_from_baseline(frame, baseline, target="sales")
    with pytest.raises(ValueError, match="目的の列"):
        current_from_baseline(frame, {"type": "lag", "periods": 1,
                                      "time": "date"})
    single = frame.head(1)
    with pytest.raises(ValueError, match="間隔"):
        current_from_baseline(single, {"type": "lag", "periods": 1,
                                       "time": "date"}, target="sales")
    short = pd.DataFrame({"t": ["2025-01-01 00:00:00.5",
                                "2025-01-01 00:00:01"], "sales": [1, 2]})
    with pytest.raises(ValueError, match="間隔を求められません"):
        current_from_baseline(short, {"type": "lag", "periods": 1,
                                      "time": "t"}, target="sales")


def test_backtest_with_baseline(tmp_path: Path) -> None:
    """
    たどり直しの今のやり方を決まりで渡すと, 同じ値を列で渡した場合と
    同じ損になり, 値の無い行は全ての方針で数えないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 時刻と系列を分ける列は一式の過去の値の手順から補う
    path, frame = walk_bundle(tmp_path)
    rule = {"type": "lag", "periods": 7}
    decision = {"type": "quantity", "under_cost": 3, "over_cost": 1,
                "current": rule}
    result = backtest(path, frame, decision)
    level = current_from_baseline(
        frame, {**rule, "time": "date", "by": ["s"]}, target="y"
    )
    with_column = backtest(
        path, frame.assign(now=level),
        {**decision, "current": "now"}, count=level.notna().to_numpy()
    )
    assert result["rows"] == with_column["rows"] == 400 - 2 * 7
    assert result["policies"] == pytest.approx(with_column["policies"])
    assert result["decision"]["current"] == rule
    assert any("7 期前の値と同じ量" in s and "12 行" in s
               for s in result["sentences"])

    # 今ある量が決まりの量より多い行の, 足さない今のやり方 (決まりは
    # 置いた後の量)
    stock = np.where(np.arange(400) % 2 == 0, 0.0, 1000.0)
    stocked = frame.assign(stock=stock)
    placed = backtest(path, stocked, {**decision, "stock": "stock"})
    added = np.maximum(level.to_numpy() - stock, 0)
    by_column = backtest(
        path, stocked.assign(now=added),
        {**decision, "stock": "stock", "current": "now"},
        count=level.notna().to_numpy()
    )
    assert placed["policies"]["今のやり方"] == pytest.approx(
        by_column["policies"]["今のやり方"]
    )

    # 量の決まりの列も置いた後の量で, k 期前の決まりと同じ損. 空欄の行は
    # 数えない (問題の設定と同じ)
    by_rule_column = backtest(
        path, stocked.assign(now=level),
        {**decision, "stock": "stock",
         "current": {"type": "column", "column": "now"}}
    )
    assert by_rule_column["rows"] == placed["rows"]
    assert by_rule_column["policies"] == pytest.approx(placed["policies"])

    # 平均の決まり (始めの行は前の期の値が無いので, 数えない行は 0)
    mean = backtest(path, frame, {**decision, "current": {
        "type": "mean", "periods": 3
    }})
    assert mean["rows"] == 398
    assert not any("数えない" in s for s in mean["sentences"])


def test_baseline_in_decision(tmp_path: Path) -> None:
    """
    k 期前と平均は量を決める型だけで, 一式の既定値には書けないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 振り分けの k 期前は誤り, 列の決まりは誤りでない
    source = tmp_path / "two.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"x": rng.normal(size=100),
                  "y": rng.choice(["a", "b"], 100)}).to_csv(source,
                                                           index=False)
    recipe = {"format_version": 1, "target": "y",
              "task": "classification", "steps": [],
              "split": {"method": "stratified", "valid": 0.3, "seed": 0}}
    export_dataset(load_data(source), recipe, tmp_path / "two")
    spec = linear_spec(1, 2)
    weights = save_weights(spec, tmp_path / "w.pt")
    make_bundle(tmp_path / "b", spec, tmp_path / "two", weights)
    bundle = load_bundle(tmp_path / "b")
    triage = {"type": "triage", "positive": "a", "miss_cost": 3,
              "false_alarm_cost": 1}
    errors = check_decision({**triage, "current": {"type": "lag",
                                                   "periods": 1}}, bundle)
    assert any("量を決める型" in e for e in errors)
    assert check_decision(
        {**triage, "current": {"type": "column", "column": "done"}}, bundle
    ) == []
    assert any("current は今のやり方" in e for e in check_decision(
        {**triage, "current": 3}, bundle
    ))

    with pytest.raises(ValueError, match="量を決める型"):
        backtest(tmp_path / "b", tmp_path / "two.csv",
                 {**triage, "current": {"type": "lag", "periods": 1}})

    # 既定値の current は列の名前だけ (決まりは誤り)
    with pytest.raises(ValueError, match="既定値の current"):
        make_bundle(tmp_path / "c", spec, tmp_path / "two", weights,
                    decision={**triage, "current": {"type": "column",
                                                    "column": "done"}})
