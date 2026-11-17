from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_decision import classifier_bundle, regression_bundle
from test_remeasure import walk_bundle

from ml_inference import backtest, decide, load_bundle, predict, remeasure
from ml_inference.decision import bootstrap_gap, choose_rows

# 答えの分かっているデータでの, 意思決定の計算と理論の最適な答えの比べ.
# y = 10 x に比で一様に ±10 % の誤りが乗る一式 (regression_bundle) と,
# 当たりの確率がちょうど sigmoid (2 x) の一式 (classifier_bundle) は,
# 最適な量としきい値を式で書けるデータ


def uniform_sales(
    rows: int, seed: int, low: float = 0.9, high: float = 1.1
) -> pd.DataFrame:
    """
    y = 10 x に比で一様な誤りが乗った新しい実績を作る関数

    Parameters
    ----------
    rows: int
        行の数
    seed: int
        乱数のシード
    low: float = 0.9
        比の下限
    high: float = 1.1
        比の上限

    Returns
    ----------
    frame: pd.DataFrame
        x と y の表
    """
    # 一式の作り方と同じ x の範囲
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 10, rows)
    frame = pd.DataFrame({"x": x, "y": 10 * x * rng.uniform(low, high, rows)})

    return frame


def newsvendor_loss(
    level: np.ndarray, actual: np.ndarray, under: float, over: float
) -> float:
    """
    置いた量と実際の値から, 足りない損と余る損の合計を求める関数

    Parameters
    ----------
    level: np.ndarray
        置いた量
    actual: np.ndarray
        実際の値
    under: float
        1 単位足りないときの損
    over: float
        1 単位余ったときの損

    Returns
    ----------
    loss: float
        損の合計
    """
    # 足りない分と余った分
    loss = float((under * np.maximum(actual - level, 0)
                  + over * np.maximum(level - actual, 0)).sum())

    return loss


@pytest.mark.parametrize("tau", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_quantity_level_is_true_quantile(tmp_path: Path, tau: float) -> None:
    """
    量の型が置く量が, 本当の分布の τ の分位点 (予測 × (0.9 + 0.2 τ)) に
    合うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    tau: float
        足りない損 / (足りない損 + 余る損)

    Returns
    ----------
    None
    """
    # 予測 500 で刻み 1 (刻みの誤りは 0.2 % 以下). 外れ方は 100 行で
    # 測るので, 分位点の推定の揺れ (比で 0.006 ほど) の 2.5 倍の許容
    path = regression_bundle(tmp_path)
    table = predict(path, pd.DataFrame({"x": [50.0]}))
    decision = {"type": "quantity", "under_cost": tau,
                "over_cost": 1 - tau}
    level = decide(path, table, decision)["frame"]["level"].iloc[0]
    assert level / 500 == pytest.approx(0.9 + 0.2 * tau, abs=0.015)


def test_quantity_loss_near_oracle(tmp_path: Path) -> None:
    """
    9 : 1 の損で, 方針の実際の損が本当の分位点まで置いた場合 (最適) に
    近く, たどり直しの損が手で数えた損と一致することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 一式を作ったデータと別の 5,000 行
    path = regression_bundle(tmp_path)
    fresh = uniform_sales(5000, seed=11)
    decision = {"type": "quantity", "under_cost": 9, "over_cost": 1}
    decided = decide(path, predict(path, fresh), decision)["frame"]
    actual = fresh["y"].to_numpy()
    point = decided["prediction"].to_numpy(float)
    policy = newsvendor_loss(decided["level"].to_numpy(), actual, 9, 1)
    oracle = newsvendor_loss(point * 1.08, actual, 9, 1)
    naive = newsvendor_loss(np.round(point), actual, 9, 1)
    assert policy <= 1.03 * oracle
    assert naive > 1.5 * oracle

    # たどり直しの実際の損 (得の符号を反転) は手で数えた損と同じ
    policies = backtest(path, fresh, decision)["policies"]
    assert -policies["この方針"] == pytest.approx(policy, rel=1e-9)
    assert -policies["予測の値どおりに置く"] == pytest.approx(naive,
                                                              rel=1e-9)


def test_triage_threshold_minimizes_realized_loss(tmp_path: Path) -> None:
    """
    確率が当たっている場合, 損の比で決まるしきい値の振り分けの実際の損が,
    どの一定のしきい値より (揺れの範囲で) 小さいことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 当たりの確率がちょうど sigmoid (2 x) の 20,000 行. 見逃し 4, 空振り 1
    # なら最適なしきい値は 0.2
    path = classifier_bundle(tmp_path)
    rng = np.random.default_rng(3)
    x = rng.normal(size=20000)
    hit = rng.uniform(size=20000) < 1 / (1 + np.exp(-2 * x))
    table = predict(path, pd.DataFrame({"x": x}))
    triage = {"type": "triage", "positive": "yes", "miss_cost": 4,
              "false_alarm_cost": 1}
    action = decide(path, table, triage)["frame"]["action"]
    flag = action.astype(bool).to_numpy()
    probability = table["p:yes"].to_numpy(float)
    assert probability == pytest.approx(1 / (1 + np.exp(-2 * x)), abs=1e-5)

    # しきい値を 0.05 から 0.95 まで動かした実際の損の最小との比べ
    policy_loss = float((4 * (~flag & hit) + 1 * (flag & ~hit)).sum())
    grid = np.arange(0.05, 0.96, 0.05)
    losses = [float((4 * ((probability < t) & hit)
                     + 1 * ((probability >= t) & ~hit)).sum())
              for t in grid]
    assert policy_loss <= 1.01 * min(losses)

    # データの確かめ (実際の損の最も小さいしきい値が理論の 0.2 の近く)
    assert abs(grid[int(np.argmin(losses))] - 0.2) <= 0.1


def test_select_without_limit_is_optimal() -> None:
    """
    上限が無い場合, 対象を選ぶ型が得の正の行をちょうど選ぶ (最適) ことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 正と負の得が混ざった行
    rng = np.random.default_rng(4)
    gain = rng.normal(size=200)
    cost = rng.uniform(1, 5, 200)
    chosen = choose_rows(gain, cost, np.ones(200, dtype=bool), None, None)
    assert (chosen == (gain > 0)).all()


def test_select_cost_limit_near_optimum() -> None:
    """
    費用の上限がある場合 (詰め込みの問題), 選んだ行の得の合計が, 全ての
    組み合わせを調べた最適の半分以上 (得の順と費用あたりの得の順の良い方
    の保証. 件数の上限の無い場合) で, 多くは最適に近いことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 費用あたりの得の順だけでは解けない例 (得の 1 を先に取ると 10 が
    # 入らない). 得の順の答えとの良い方の選び
    chosen = choose_rows(np.array([1.0, 10.0]), np.array([0.1, 10.0]),
                         np.ones(2, dtype=bool), None, 10.0)
    assert list(chosen) == [False, True]

    # 12 行の詰め込みの問題 200 個と, 全ての組み合わせ (4,096) の比べ
    rng = np.random.default_rng(6)
    subsets = np.array(list(product([False, True], repeat=12)))
    ratios = []
    for _ in range(200):
        gain = rng.uniform(-1, 10, 12)
        cost = rng.uniform(1, 10, 12)
        budget_cost = float(rng.uniform(5, 30))
        chosen = choose_rows(gain, cost, np.ones(12, dtype=bool), None,
                             budget_cost)
        assert cost[chosen].sum() <= budget_cost
        fits = subsets[(subsets * cost).sum(axis=1) <= budget_cost]
        best = float((fits * np.maximum(gain, 0)).sum(axis=1).max())
        ratios.append(gain[chosen].sum() / best if best > 0 else 1.0)
    assert min(ratios) >= 0.5
    assert np.mean(ratios) >= 0.95


def test_remeasure_recovers_known_spread(tmp_path: Path) -> None:
    """
    比で一様に ±40 % の誤りの実績で測り直すと, 10 % と 90 % の点が本当の
    値 (0.68 倍と 1.32 倍) に合うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2,000 行の実績 (分位点の推定の揺れは比で 0.01 ほど)
    path = regression_bundle(tmp_path)
    remeasure(path, uniform_sales(2000, seed=12, low=0.6, high=1.4))
    uncertainty = load_bundle(path).info["uncertainty"]
    assert uncertainty["kind"] == "relative"
    points = np.exp(np.asarray(uncertainty["relative"]))
    assert points[9] == pytest.approx(0.68, abs=0.02)
    assert points[89] == pytest.approx(1.32, abs=0.02)


def test_steps_widen_with_random_walk(tmp_path: Path) -> None:
    """
    ランダムウォークでは k 段先の誤りの広がりが 1 段先の √k 倍になるので,
    段ごとの外れ方の 80 % の幅がその比に (誤りの数の少なさの揺れの範囲
    で) 合うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 後ろ 100 日で 5 段先まで (始める時刻 20 個 × 2 店 = 1 段 40 個の誤り)
    path, frame = walk_bundle(tmp_path)
    count = (pd.to_datetime(frame["date"]) >= "2025-04-11").to_numpy()
    steps = remeasure(path, frame, count=count, horizon=5,
                      write=False)["uncertainty"]["steps"]
    width = {k: steps[str(k)][89] - steps[str(k)][9] for k in range(1, 6)}
    for k in range(2, 6):
        assert width[k] / width[1] == pytest.approx(np.sqrt(k), rel=0.25)


def test_bootstrap_range_covers_true_gap() -> None:
    """
    差の 90 % の幅が本当の差を覆う割合が, 行が独立なら行ごとの選び直しで,
    同じ日の行が似ているなら日ごとの選び直しで 90 % 近くになり, 似ている
    行を行ごとに選び直すと幅が狭すぎることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 本当の差の平均は 1 行 0.1. 30 日 × 10 店で, 日ごとに共通の揺れ
    rng = np.random.default_rng(7)
    days = np.repeat(np.arange(30), 10)
    covered = {"独立": 0, "日ごと": 0, "似た行を行ごと": 0}
    trials = 500
    for _ in range(trials):
        noise = rng.normal(size=300)
        shared = rng.normal(size=30)[days] * 2
        cases = {
            "独立": (0.1 + noise, None),
            "日ごと": (0.1 + noise + shared, days),
            "似た行を行ごと": (0.1 + noise + shared, None),
        }
        for name, (gap, groups) in cases.items():
            low, high = bootstrap_gap(gap, np.zeros(300), groups)
            covered[name] += low <= 30 <= high
    rate = {name: hits / trials for name, hits in covered.items()}
    assert 0.84 <= rate["独立"] <= 0.96
    assert 0.80 <= rate["日ごと"] <= 0.96
    assert rate["似た行を行ごと"] < 0.6
