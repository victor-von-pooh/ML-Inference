from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from ml_eda import Data

from ml_inference.bundle import Bundle, load_bundle
from ml_inference.conclude import is_number, plain
from ml_inference.predict import predict

# 意思決定の型と, 型ごとに書ける項目
DECISION_KEYS = {
    "quantity": ("type", "under_cost", "over_cost", "stock", "step",
                 "current"),
    "select": ("type", "positive", "value", "cost", "effect", "budget",
               "budget_cost", "current"),
    "triage": ("type", "positive", "miss_cost", "false_alarm_cost",
               "capacity", "current"),
}

# 型ごとの, 行ごとに要る損得の値 (欠けた行は決めない)
AMOUNTS = {
    "quantity": ("under_cost", "over_cost", "stock"),
    "select": ("value", "cost", "effect"),
    "triage": ("miss_cost", "false_alarm_cost"),
}

# 意思決定が予測の表に足す列
DECISION_COLUMNS = (
    "action", "level", "expected_cost", "point_expected_cost",
    "gain_if_acted", "expected_gain"
)

# 分割の日本語の名前
SPLIT_NAMES = {"train": "学習用", "valid": "検証用", "test": "テスト用"}

# たどり直しの差の幅を求める選び直しの回数と乱数のシード
BOOTSTRAP = 500
BOOTSTRAP_SEED = 0


def check_amount(
    decision: dict, name: str, columns: Optional[list[str]],
    errors: list[str], required: bool = True, upper: Optional[float] = None
) -> None:
    """
    損得の値が 0 以上の数か, 予測の表の列の名前かを確かめる関数

    Parameters
    ----------
    decision: dict
        意思決定の決め方
    name: str
        項目の名前
    columns: Optional[list[str]]
        予測の表の列 (None なら列の有無は確かめない)
    errors: list[str]
        誤りの一覧 (書き足す)
    required: bool = True
        省けないか
    upper: Optional[float] = None
        数の上限 (割合なら 1)

    Returns
    ----------
    None
    """
    # 列の名前なら列があるか, 数なら 0 以上 (と上限以下) か
    value = decision.get(name)
    if value is None:
        if required:
            errors.append(f"{name} を書いてください")
        return
    if isinstance(value, str):
        if columns is not None and value not in columns:
            errors.append(f"{name} の列 {value} が予測の表にありません")
        return
    if not is_number(value) or value < 0 or (
        upper is not None and value > upper
    ):
        limit = f"から {upper} の" if upper is not None else "以上の"
        errors.append(f"{name} は 0 {limit}数か列の名前にしてください")


def check_decision(
    decision: Any, bundle: Bundle, columns: Optional[list[str]] = None
) -> list[str]:
    """
    意思決定の決め方の誤りを一覧で返す関数

    Parameters
    ----------
    decision: Any
        意思決定の決め方
    bundle: Bundle
        推論の一式
    columns: Optional[list[str]] = None
        予測の表の列 (渡すと, 列の名前で書いた値があるかも確かめる)

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 型と知らない項目
    if not isinstance(decision, dict):
        return ["意思決定の決め方は辞書にしてください"]
    kind = decision.get("type")
    if kind not in DECISION_KEYS:
        return [f"type は {', '.join(DECISION_KEYS)} のどれかにしてください"]
    errors = [
        f"{kind} に知らない項目です: {key}"
        for key in decision if key not in DECISION_KEYS[kind]
    ]

    # 型と一式の組み合わせ
    if kind == "quantity":
        if bundle.task == "classification":
            errors.append("quantity は回帰と予測の一式にだけ使えます")
    elif bundle.task != "classification" or len(bundle.classes) != 2:
        errors.append(f"{kind} は 2 クラスの分類の一式にだけ使えます")
    elif str(decision.get("positive")) not in bundle.classes:
        errors.append(
            f"positive は {', '.join(bundle.classes)} のどれかにして"
            "ください"
        )

    # 型ごとの損得と制約
    for name in AMOUNTS[kind]:
        check_amount(
            decision, name, columns, errors, required=name != "stock",
            upper=1.0 if name == "effect" else None
        )
    for name in ("step", "budget", "budget_cost", "capacity"):
        value = decision.get(name)
        if value is not None and (not is_number(value) or value < 0 or (
            name == "step" and value == 0
        )):
            errors.append(f"{name} は正の数にしてください")
        elif name in ("budget", "capacity") and value is not None and (
            value != int(value)
        ):
            errors.append(f"{name} は件数なので整数にしてください")
    current = decision.get("current")
    if current is not None and columns is not None and current not in columns:
        errors.append(f"current の列 {current} が予測の表にありません")

    return errors


def values_of(frame: pd.DataFrame, value: Any) -> np.ndarray:
    """
    数か列の名前で書いた値を, 行ごとの数にする関数

    Parameters
    ----------
    frame: pd.DataFrame
        予測の表
    value: Any
        数か列の名前 (None は 0)

    Returns
    ----------
    values: np.ndarray
        行ごとの数 (列の欠けた値と数にできない値は欠損)
    """
    # 列は数に (欠けた値は欠損のまま), 数はそろえる
    if value is None:
        return np.zeros(len(frame))
    if isinstance(value, str):
        values = pd.to_numeric(frame[value], errors="coerce")
        return values.to_numpy(dtype=float)
    values = np.full(len(frame), float(value))

    return values


def amounts_of(
    frame: pd.DataFrame, decision: dict
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    型ごとの損得の値を行ごとに取り出し, 欠けた行の印を返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        予測の表
    decision: dict
        意思決定の決め方

    Returns
    ----------
    result: tuple[dict[str, np.ndarray], np.ndarray]
        項目ごとの行ごとの値と, 値の欠けた行の印
    """
    # 欠けた値と範囲の外の値 (負, 1 を超える効果) の行は決めない (0 と
    # みなすと, ただの行動や 1 % の点になる)
    values = {
        name: values_of(frame, decision.get(name))
        for name in AMOUNTS[decision["type"]]
    }
    missing = np.zeros(len(frame), dtype=bool)
    for name, array in values.items():
        missing |= np.isnan(array) | (array < 0)
        if name == "effect":
            missing |= array > 1
    result = (values, missing)

    return result


def residual_samples(bundle: Bundle) -> tuple[str, np.ndarray]:
    """
    一式の外れ方 (誤りの分位点) を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    samples: tuple[str, np.ndarray]
        形 (relative か absolute) と, 1 % から 99 % の分位点

    Raises
    ----------
    ValueError
        外れ方を残していない一式 (版 1) の場合
    """
    # 版 2 の一式が作るときに残す
    uncertainty = bundle.info.get("uncertainty") or {}
    kind = uncertainty.get("kind")
    if not kind or not uncertainty.get(kind):
        raise ValueError(
            "この一式は予測の外れ方を残していません. 量を決めるには一式を"
            "作り直してください (版 2 から外れ方を測る)"
        )
    samples = (kind, np.asarray(uncertainty[kind], dtype=float))

    return samples


def outcomes(
    prediction: np.ndarray, kind: str, quantiles: np.ndarray
) -> np.ndarray:
    """
    予測と誤りの分位点から, ありうる実際の値の並びを作る関数

    Parameters
    ----------
    prediction: np.ndarray
        行ごとの予測
    kind: str
        relative か absolute
    quantiles: np.ndarray
        誤りの 1 % から 99 % の分位点

    Returns
    ----------
    values: np.ndarray
        (行の数, 99) のありうる実際の値 (同じ重み. 1 % より外と 99 % より
        外の裾は含まないので, τ が 0.99 に近いと見込みの損は甘い)
    """
    # 比は掛け, 差は足す
    if kind == "relative":
        values = prediction[:, None] * np.exp(quantiles[None, :])
    else:
        values = prediction[:, None] + quantiles[None, :]

    return values


def quantity_cost(
    level: np.ndarray, actual: np.ndarray, under: np.ndarray,
    over: np.ndarray
) -> np.ndarray:
    """
    置いた量と実際の値から, 足りない損と余る損を求める関数

    Parameters
    ----------
    level: np.ndarray
        置いた量 (今ある量 + 足す量). 行ごと
    actual: np.ndarray
        実際の値 (行ごと, または (行, 見込み) の並び)
    under: np.ndarray
        1 単位足りないときの損 (行ごと)
    over: np.ndarray
        1 単位余ったときの損 (行ごと)

    Returns
    ----------
    cost: np.ndarray
        行ごと (または見込みごと) の損
    """
    # 実際の値が 2 次元なら, 行ごとの値を見込みの向きに広げる
    if actual.ndim == 2:
        level, under, over = level[:, None], under[:, None], over[:, None]
    cost = under * np.maximum(actual - level, 0) + over * np.maximum(
        level - actual, 0
    )

    return cost


def best_amount(
    goal: np.ndarray, stock: np.ndarray, step: float,
    possible: np.ndarray, under: np.ndarray, over: np.ndarray
) -> np.ndarray:
    """
    目指す量を刻みにそろえる関数 (切り下げと切り上げの見込みの損の小さい
    方)

    Parameters
    ----------
    goal: np.ndarray
        目指す置いた後の量
    stock: np.ndarray
        今ある量
    step: float
        刻み
    possible: np.ndarray
        ありうる実際の値
    under: np.ndarray
        足りない損
    over: np.ndarray
        余る損

    Returns
    ----------
    action: np.ndarray
        足す量 (0 以上で刻みの倍数)
    """
    # 刻みの上下の 2 つの量を比べる
    raw = (goal - stock) / step
    low = np.maximum(np.floor(raw + 1e-9), 0) * step
    high = np.maximum(np.ceil(raw - 1e-9), 0) * step
    low_cost = quantity_cost(stock + low, possible, under, over).mean(axis=1)
    high_cost = quantity_cost(stock + high, possible, under, over).mean(
        axis=1
    )
    action = np.where(high_cost < low_cost, high, low)

    return action


def decide_quantity(
    bundle: Bundle, frame: pd.DataFrame, decision: dict, ok: np.ndarray
) -> tuple[dict, dict]:
    """
    足りない損と余る損の比で, 次の 1 期間に置く量を決める関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    frame: pd.DataFrame
        予測の表
    decision: dict
        意思決定の決め方
    ok: np.ndarray
        決める行 (予測し, 損得の値が揃った行) の印

    Returns
    ----------
    result: tuple[dict, dict]
        足す列 (action, level, expected_cost, point_expected_cost) と
        まとめ
    """
    # 損の比 τ で決まる分位点まで置く (新聞売りの問題の答え). 足りない損
    # が 0 なら足さない, 余る損が 0 なら 99 % の点
    kind, quantiles = residual_samples(bundle)
    amounts, _ = amounts_of(frame, decision)
    under = np.nan_to_num(amounts["under_cost"])
    over = np.nan_to_num(amounts["over_cost"])
    stock = np.nan_to_num(amounts["stock"])
    prediction = np.nan_to_num(pd.to_numeric(
        frame["prediction"], errors="coerce"
    ).to_numpy(dtype=float))
    step = float(decision.get("step", 1))
    tau = under / np.maximum(under + over, 1e-12)
    index = np.clip(np.round(tau * 100).astype(int), 1, 99) - 1
    possible = outcomes(prediction, kind, quantiles)
    goal = possible[np.arange(len(frame)), index]
    action = best_amount(goal, stock, step, possible, under, over)
    action = np.where(tau <= 0, 0.0, action)
    level = stock + action

    # 外れ方の記録が正しいとしたときの見込みの損 (分位点をありうる値に)
    expected = quantity_cost(level, possible, under, over).mean(axis=1)
    point = stock + np.maximum(
        np.round((prediction - stock) / step) * step, 0
    )
    baseline = quantity_cost(point, possible, under, over).mean(axis=1)
    columns = {
        "action": np.where(ok, action, np.nan),
        "level": np.where(ok, level, np.nan),
        "expected_cost": np.where(ok, expected, np.nan),
        "point_expected_cost": np.where(ok, baseline, np.nan),
    }
    summary = {
        "acting": int((action[ok] > 0).sum()),
        "amount": float(action[ok].sum()),
        "expected_cost": float(expected[ok].sum()),
        "baseline_cost": float(baseline[ok].sum()),
        "tau_min": float(tau[ok].min()) if ok.any() else None,
        "tau_max": float(tau[ok].max()) if ok.any() else None,
        "kind": kind,
    }
    result = (columns, summary)

    return result


def greedy(
    order: np.ndarray, usable: np.ndarray, cost: np.ndarray,
    budget: Optional[float], budget_cost: Optional[float],
    count: Optional[int] = None
) -> np.ndarray:
    """
    並びの順に, 件数と費用の上限まで行を選ぶ関数

    Parameters
    ----------
    order: np.ndarray
        選ぶ順の行の位置
    usable: np.ndarray
        選んでよい行の印
    cost: np.ndarray
        行ごとの費用
    budget: Optional[float]
        件数の上限
    budget_cost: Optional[float]
        費用の上限 (越える行は飛ばして次を見る)
    count: Optional[int] = None
        選ぶ件数 (比べるやり方を同じ件数にそろえる場合)

    Returns
    ----------
    chosen: np.ndarray
        選んだ行の印
    """
    # 上限に届くまで, 選べる行を順に
    chosen = np.zeros(len(usable), dtype=bool)
    spent, picked = 0.0, 0
    for row in order:
        if not usable[row]:
            continue
        if (budget is not None and picked >= budget) or (
            count is not None and picked >= count
        ):
            break
        if budget_cost is not None and spent + cost[row] > budget_cost:
            continue
        chosen[row] = True
        spent += cost[row]
        picked += 1

    return chosen


def choose_rows(
    gain: np.ndarray, cost: np.ndarray, ok: np.ndarray,
    budget: Optional[float], budget_cost: Optional[float]
) -> np.ndarray:
    """
    得が正の行を, 上限の中で得の合計が大きくなるように選ぶ関数

    Parameters
    ----------
    gain: np.ndarray
        行動した場合の見込みの得
    cost: np.ndarray
        行ごとの費用
    ok: np.ndarray
        選んでよい行の印
    budget: Optional[float]
        件数の上限
    budget_cost: Optional[float]
        費用の上限

    Returns
    ----------
    chosen: np.ndarray
        選んだ行の印
    """
    # 得の順と, (費用の上限があれば) 費用あたりの得の順の良い方 (詰め込み
    # の問題の近い答え)
    usable = ok & (gain > 0)
    by_gain = greedy(
        np.argsort(-np.where(ok, gain, -np.inf), kind="stable"), usable,
        cost, budget, budget_cost
    )
    if budget_cost is None:
        return by_gain
    ratio = gain / np.maximum(cost, 1e-12)
    by_ratio = greedy(
        np.argsort(-np.where(ok, ratio, -np.inf), kind="stable"), usable,
        cost, budget, budget_cost
    )
    chosen = by_ratio if gain[by_ratio].sum() > gain[by_gain].sum() else (
        by_gain
    )

    return chosen


def decide_select(
    bundle: Bundle, frame: pd.DataFrame, decision: dict, ok: np.ndarray
) -> tuple[dict, dict]:
    """
    見込みの得の大きい行を, 上限まで行動する行に選ぶ関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    frame: pd.DataFrame
        予測の表
    decision: dict
        意思決定の決め方
    ok: np.ndarray
        決める行の印

    Returns
    ----------
    result: tuple[dict, dict]
        足す列 (action, gain_if_acted, expected_gain) と, まとめ
    """
    # 行動した場合の見込みの得 = 確率 × 価値 × 効果 − 費用 (効果は確率に
    # 比例するという仮定)
    probability = np.nan_to_num(pd.to_numeric(
        frame[f"p:{decision['positive']}"], errors="coerce"
    ).to_numpy(dtype=float))
    amounts, _ = amounts_of(frame, decision)
    cost = np.nan_to_num(amounts["cost"])
    gain = np.nan_to_num(
        probability * amounts["value"] * amounts["effect"] - cost
    )
    budget, budget_cost = decision.get("budget"), decision.get("budget_cost")
    chosen = choose_rows(gain, cost, ok, budget, budget_cost)

    # 比べ: 確率の高い順に同じ件数 (同じ費用の上限の中で). 全てに行動は
    # 上限の無い場合だけ (上限を破る比べは公平でない)
    ranked = np.argsort(-np.where(ok, probability, -np.inf), kind="stable")
    top = greedy(ranked, ok, cost, None, budget_cost, int(chosen.sum()))
    baselines = {"確率の高い順に同じ件数": float(gain[top].sum())}
    if budget is None and budget_cost is None:
        baselines["全てに行動"] = float(gain[ok].sum())
    columns = {
        "action": np.where(ok, chosen, None),
        "gain_if_acted": np.where(ok, gain, np.nan),
        "expected_gain": np.where(ok, np.where(chosen, gain, 0.0), np.nan),
    }
    summary = {
        "acting": int(chosen.sum()),
        "amount": float(cost[chosen].sum()),
        "expected_gain": float(gain[chosen].sum()),
        "baselines": baselines,
    }
    result = (columns, summary)

    return result


def triage_loss(
    flag: np.ndarray, probability: np.ndarray, miss: np.ndarray,
    alarm: np.ndarray
) -> np.ndarray:
    """
    回すかの印から, 行ごとの見込みの損を求める関数

    Parameters
    ----------
    flag: np.ndarray
        回すかの印
    probability: np.ndarray
        当たりの確率
    miss: np.ndarray
        見逃しの損
    alarm: np.ndarray
        空振りの損

    Returns
    ----------
    values: np.ndarray
        行ごとの見込みの損
    """
    # 回すなら外れの分, 回さないなら当たりの分
    values = np.where(flag, (1 - probability) * alarm, probability * miss)

    return values


def decide_triage(
    bundle: Bundle, frame: pd.DataFrame, decision: dict, ok: np.ndarray
) -> tuple[dict, dict]:
    """
    見逃しの損と空振りの損の比のしきい値で, 回すかを決める関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    frame: pd.DataFrame
        予測の表
    decision: dict
        意思決定の決め方
    ok: np.ndarray
        決める行の印

    Returns
    ----------
    result: tuple[dict, dict]
        足す列 (action, expected_cost) と, まとめ
    """
    # 回す損の見込みが, 回さない損の見込みより小さい行 (しきい値以上)
    probability = np.nan_to_num(pd.to_numeric(
        frame[f"p:{decision['positive']}"], errors="coerce"
    ).to_numpy(dtype=float))
    amounts, _ = amounts_of(frame, decision)
    miss = np.nan_to_num(amounts["miss_cost"])
    alarm = np.nan_to_num(amounts["false_alarm_cost"])
    threshold = alarm / np.maximum(miss + alarm, 1e-12)
    benefit = probability * miss - (1 - probability) * alarm

    # 人手の上限があれば, 回して減る損の大きい順に上限まで
    capacity = decision.get("capacity")
    order = np.argsort(-np.where(ok, benefit, -np.inf), kind="stable")
    act = greedy(order, ok & (probability >= threshold), alarm, capacity,
                 None)
    by_half = greedy(
        np.argsort(-np.where(ok, probability, -np.inf), kind="stable"),
        ok & (probability >= 0.5), alarm, capacity, None
    )
    baselines = {"確率 0.5 で分ける": float(
        triage_loss(by_half, probability, miss, alarm)[ok].sum()
    )}
    if capacity is None:
        baselines["全てを回す"] = float(
            triage_loss(ok, probability, miss, alarm)[ok].sum()
        )
    baselines["どれも回さない"] = float(triage_loss(
        np.zeros(len(frame), dtype=bool), probability, miss, alarm
    )[ok].sum())
    expected = triage_loss(act, probability, miss, alarm)
    columns = {
        "action": np.where(ok, act, None),
        "expected_cost": np.where(ok, expected, np.nan),
    }
    summary = {
        "acting": int(act.sum()),
        "threshold_min": float(threshold[ok].min()) if ok.any() else None,
        "threshold_max": float(threshold[ok].max()) if ok.any() else None,
        "expected_cost": float(expected[ok].sum()),
        "baselines": baselines,
    }
    result = (columns, summary)

    return result


def amount_text(value: float) -> str:
    """
    量や金額を桁に合わせた短い文字にする関数

    Parameters
    ----------
    value: float
        値

    Returns
    ----------
    text: str
        大きな値はカンマ付きの整数, 小さな値は有効数字 3 桁
    """
    # 1,000 以上は整数
    text = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"

    return text


def range_text(low: float, high: float, percent: bool = False) -> str:
    """
    行ごとに違いうる値を, 1 つの値か範囲の文字にする関数

    Parameters
    ----------
    low: float
        最小
    high: float
        最大
    percent: bool = False
        割合を % で書くか

    Returns
    ----------
    text: str
        同じなら 1 つ, 違えば「行ごとに x から y」
    """
    # 割合は % に, 他は有効数字 2 桁
    if percent:
        low_text, high_text = f"{low * 100:.0f} %", f"{high * 100:.0f} %"
    else:
        low_text, high_text = f"{low:.2g}", f"{high:.2g}"
    text = low_text if abs(high - low) < 1e-9 else (
        f"行ごとに {low_text} から {high_text}"
    )

    return text


def caveats(
    bundle: Bundle, prediction: pd.DataFrame, decision: dict,
    summary: dict
) -> list[str]:
    """
    意思決定の前提 (外れ方の記録, 確率のずれ, 仮定, 抜けた行) を述べる文
    を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    prediction: pd.DataFrame
        予測の表
    decision: dict
        意思決定の決め方
    summary: dict
        まとめ

    Returns
    ----------
    notes: list[str]
        前提の文
    """
    # 値の欠けた行と, 外れ方を測った行の少なさ
    kind = decision["type"]
    uncertainty = bundle.info.get("uncertainty") or {}
    notes = []
    if summary["missing"]:
        notes.append(
            f"損得の値が欠けた (か範囲の外の) {summary['missing']:,} 行は"
            "決めていない"
        )
    rows = uncertainty.get("rows")
    if rows is not None and rows < 100:
        notes.append(
            f"外れ方と確率の当たり具合は {rows} 行だけで測ったので粗い"
        )

    # 量: 1 期間, 外れ方の記録, 打ち切られた需要, 何段も先の予測
    if kind == "quantity":
        notes.append(
            "量は次の 1 期間の需要に対するもの (届くまでの日数, 発注の周期,"
            " 余りの持ち越しは考えていない. 持ち越す場合の余る損は保管の"
            "費用などで, 1 期間の答えは目安)"
        )
        below = uncertainty.get("test_below")
        if below and summary["tau_min"] is not None:
            level = int(np.clip(round(summary["tau_min"] * 100), 1, 99))
            notes.append(
                f"外れ方は"
                f"{SPLIT_NAMES.get(uncertainty.get('split'), '検証用')}"
                f"の分割で測った. {level}"
                f" % の点を, テスト用の分割の実際の値が下回ったのは "
                f"{below[level - 1] * 100:.0f} %"
                + (" (外れ方を小さく見積もっている. 量は少なめ)"
                   if below[level - 1] < level / 100 - 0.05 else "")
            )
        if uncertainty.get("kind") == "absolute":
            notes.append(
                "外れ方は水準に依らない差で持っている (大きな行では外れ方"
                "を小さく, 小さな行では大きく見積もりうる)"
            )
        notes.append(
            "過去の値が欠品などで抑えられていたなら, 学習した需要と外れ方"
            "は小さめで, 量も少なめ"
        )
        if "step" in prediction.columns and (
            pd.to_numeric(prediction["step"], errors="coerce") > 1
        ).any():
            notes.append(
                "2 段目より先の予測にも 1 つ先の外れ方を当てている (先の段"
                "ほど外れ方は大きく, 量は少なめ)"
            )
        return notes

    # 選ぶと振り分け: 確率のずれ, 効果の仮定, 人の判断
    calibration = uncertainty.get("calibration") or {}
    ece = uncertainty.get("test_ece", calibration.get("ece"))
    if ece is not None:
        notes.append(
            f"確率と実際に当たった割合のずれは平均 {ece * 100:.1f} ポイント"
            + (" (大きいので, 確率を額面どおりに使う判断は粗い)"
               if ece > 0.05 else "")
        )
    if kind == "select":
        notes.append(
            f"行動で当たりの損を防げる割合 ({decision['effect']}) と, その"
            "効果が当たりの確率に比例することは仮定 (確率の高い行ほど防げ"
            "るとは限らない. 確かめるには, 行動した組としない組を比べる"
            "試験が要る)"
        )
    notes.append(
        "人の判断を補う材料で, 自動で決めるものではない (重い判断では, "
        "グループごとの見逃しの割合なども確かめる)"
    )

    return notes


def sentences_of(
    bundle: Bundle, kind: str, summary: dict, decision: dict
) -> list[str]:
    """
    意思決定のまとめを日本語の文にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    kind: str
        意思決定の型
    summary: dict
        まとめ
    decision: dict
        意思決定の決め方

    Returns
    ----------
    sentences: list[str]
        文 (前提のもとの最も良い行動と, 同じ前提での比べ)
    """
    # 前提のもとの比べは前提から必ずそうなるので, 実際の差はたどり直しで
    rows = summary["rows"]
    target = bundle.dataset.info.get("target") or "値"
    tail = (
        " (同じ前提から計算した比べなので, この方針は必ず同じか良く出る. "
        "実際の差は過去のデータでたどり直して確かめる)"
    )
    if kind == "quantity":
        tau = range_text(summary["tau_min"], summary["tau_max"], True)
        sentences = [
            f"足りない損 {decision['under_cost']} と余る損 "
            f"{decision['over_cost']}, 予測の外れ方の記録が正しいとすれば, "
            f"損が最も小さいのは {target} の外れ方の {tau} の点まで置くこと",
            f"{rows:,} 行のうち {summary['acting']:,} 行に, 合計 "
            f"{amount_text(summary['amount'])} を足す",
            f"その前提での見込みの損は {amount_text(summary['expected_cost'])}"
            f", 予測の値どおりに置くと {amount_text(summary['baseline_cost'])}"
            + tail,
        ]
    elif kind == "select":
        sentences = [
            "確率と, 書いた価値, 費用, 効果 (仮定) のもとで, 見込みの得が正の"
            f"行を{'上限まで' if summary['limited'] else '全て'}選ぶと, "
            f"{rows:,} 件のうち {summary['acting']:,} 件 (費用の合計 "
            f"{amount_text(summary['amount'])})",
            f"その前提での見込みの得は {amount_text(summary['expected_gain'])}"
            ", " + ", ".join(
                f"{name}なら {amount_text(value)}"
                for name, value in summary["baselines"].items()
            ) + tail,
        ]
    else:
        threshold = range_text(
            summary["threshold_min"], summary["threshold_max"]
        )
        sentences = [
            f"見逃しの損 {decision['miss_cost']} と空振りの損 "
            f"{decision['false_alarm_cost']}, 確率が当たっているとすれば, "
            f"損が最も小さいのは確率 {threshold} 以上を回すこと"
            + (f" (回せるのは {decision['capacity']} 件まで)"
               if decision.get("capacity") is not None else ""),
            f"{rows:,} 件のうち {summary['acting']:,} 件を回す",
            f"その前提での見込みの損は {amount_text(summary['expected_cost'])}"
            ", " + ", ".join(
                f"{name}なら {amount_text(value)}"
                for name, value in summary["baselines"].items()
            ) + tail,
        ]

    return sentences


def decide(
    bundle: Union[str, Path, Bundle], prediction: pd.DataFrame,
    decision: dict
) -> dict:
    """
    予測の表から, 行ごとの行動と見込みの得と損を決める関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ
    prediction: pd.DataFrame
        predict (か forecast) の表
    decision: dict
        意思決定の決め方 (type と型ごとの損得と制約)

    Returns
    ----------
    result: dict
        frame (行動の列を足した表), summary, sentences, notes (前提),
        decision (使った決め方)

    Raises
    ----------
    ValueError
        決め方に誤りがある場合と, 外れ方を残していない一式の場合
    """
    # 決め方の確認
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    decision = plain(dict(decision))
    errors = check_decision(decision, bundle, list(prediction.columns))
    if errors:
        raise ValueError("意思決定の決め方の誤り: " + " / ".join(errors))

    # 予測し, 損得の値が揃った行だけを決める
    kind = decision["type"]
    _, missing = amounts_of(prediction, decision)
    predicted = (prediction["status"] == "ok").to_numpy()
    ok = predicted & ~missing
    maker = {"quantity": decide_quantity, "select": decide_select,
             "triage": decide_triage}[kind]
    columns, summary = maker(bundle, prediction, decision, ok)
    summary.update(
        rows=int(ok.sum()), missing=int((predicted & missing).sum()),
        limited=decision.get("budget") is not None
        or decision.get("budget_cost") is not None
    )
    frame = prediction.copy()
    for name in columns:
        if name in frame.columns:
            frame = frame.rename(columns={name: f"{name} (入力)"})
    for name, values in columns.items():
        frame[name] = values
    result = {
        "frame": frame,
        "summary": summary,
        "sentences": sentences_of(bundle, kind, summary, decision),
        "notes": caveats(bundle, prediction, decision, summary),
        "decision": decision,
    }

    return result


def realized(
    kind: str, flag: np.ndarray, part: pd.DataFrame, decision: dict,
    actual: np.ndarray
) -> np.ndarray:
    """
    方針の行動から, 行ごとの実際の得 (正) と損 (負) を求める関数

    Parameters
    ----------
    kind: str
        意思決定の型
    flag: np.ndarray
        量の型は置いた量, 他は行動したかの印
    part: pd.DataFrame
        数える行の予測の表
    decision: dict
        意思決定の決め方
    actual: np.ndarray
        量の型は実際の値, 他は実際に当たりだったかの印

    Returns
    ----------
    values: np.ndarray
        行ごとの得 (損は負)
    """
    # 量は足りない損と余る損, 選ぶは当たりの価値 × 効果 − 費用 (効果は
    # 仮定のまま), 振り分けは空振りと見逃しの損
    amounts, _ = amounts_of(part, decision)
    if kind == "quantity":
        values = -quantity_cost(
            flag, actual, amounts["under_cost"], amounts["over_cost"]
        )
        return values
    if kind == "select":
        gains = actual * amounts["value"] * amounts["effect"] - amounts[
            "cost"
        ]
        values = np.where(flag, gains, 0.0)
        return values
    values = -(
        amounts["false_alarm_cost"] * (flag & ~actual)
        + amounts["miss_cost"] * (~flag & actual)
    )

    return values


def policies_of(
    kind: str, part: pd.DataFrame, decision: dict
) -> dict[str, np.ndarray]:
    """
    たどり直す方針 (この方針と素朴なやり方) ごとの行動を返す関数

    Parameters
    ----------
    kind: str
        意思決定の型
    part: pd.DataFrame
        数える行の, 行動の列を足した予測の表
    decision: dict
        意思決定の決め方

    Returns
    ----------
    flags: dict[str, np.ndarray]
        方針の名前から, 量の型は置いた量, 他は行動したかの印
    """
    # 量: この方針, 予測の値どおり, (書けば) 今のやり方で足す量
    amounts, _ = amounts_of(part, decision)
    if kind == "quantity":
        stock = np.nan_to_num(amounts["stock"])
        step = float(decision.get("step", 1))
        point = pd.to_numeric(part["prediction"]).to_numpy(float)
        flags = {
            "この方針": part["level"].to_numpy(float),
            "予測の値どおりに置く": stock + np.maximum(
                np.round((point - stock) / step) * step, 0
            ),
        }
        if decision.get("current"):
            flags["今のやり方"] = stock + values_of(part, decision["current"])
        return flags

    # 選ぶと振り分け: 同じ上限の中での確率の高い順と, 全てと何もしない
    probability = pd.to_numeric(
        part[f"p:{decision['positive']}"]
    ).to_numpy(float)
    acted = part["action"].astype(bool).to_numpy()
    usable = np.ones(len(part), dtype=bool)
    ranked = np.argsort(-probability, kind="stable")
    if kind == "select":
        cost = np.nan_to_num(amounts["cost"])
        flags = {"この方針": acted, "確率の高い順に同じ件数": greedy(
            ranked, usable, cost, None, decision.get("budget_cost"),
            int(acted.sum())
        )}
        if decision.get("budget") is None and decision.get(
            "budget_cost"
        ) is None:
            flags["全てに行動"] = usable
    else:
        flags = {"この方針": acted, "確率 0.5 で分ける": greedy(
            ranked, probability >= 0.5, np.zeros(len(part)),
            decision.get("capacity"), None
        )}
        if decision.get("capacity") is None:
            flags["全てを回す"] = usable
        flags["どれも回さない"] = np.zeros(len(part), dtype=bool)
    if decision.get("current"):
        flags["今のやり方"] = part[decision["current"]].astype(
            bool
        ).to_numpy()

    return flags


def bootstrap_gap(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    """
    2 つの方針の行ごとの得の合計の差の, 90 % の幅を求める関数

    Parameters
    ----------
    first: np.ndarray
        1 つ目の方針の行ごとの得
    second: np.ndarray
        2 つ目の方針の行ごとの得

    Returns
    ----------
    bounds: tuple[float, float]
        行を選び直した差の合計の 5 % と 95 % の点
    """
    # 行を重複を許して選び直す (乱数は固定)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gap = first - second
    picks = rng.integers(0, len(gap), size=(BOOTSTRAP, len(gap)))
    totals = gap[picks].sum(axis=1)
    bounds = (float(np.percentile(totals, 5)),
              float(np.percentile(totals, 95)))

    return bounds


def label_text(values: pd.Series) -> pd.Series:
    """
    目的の列の値を, 一式のクラスの名前と比べられる文字にする関数

    Parameters
    ----------
    values: pd.Series
        目的の列

    Returns
    ----------
    texts: pd.Series
        文字 (1.0 のような整数の小数は 1 に)
    """
    # 真偽の列はそのまま文字に (True, False). 数の列は, 整数の値だけ
    # 整数の文字に (空欄があると小数で読まれるため)
    if pd.api.types.is_bool_dtype(values):
        texts = values.astype(str)
        return texts
    numbers = pd.to_numeric(values, errors="coerce")
    whole = numbers.notna() & np.isfinite(numbers) & (
        numbers == numbers.round()
    ) & (numbers.abs() < 2 ** 53)
    texts = values.astype(str)
    texts[whole] = numbers[whole].astype("int64").astype(str)

    return texts


def backtest(
    bundle: Union[str, Path, Bundle],
    source: Union[str, Path, pd.DataFrame, Data], decision: dict,
    options: Optional[dict] = None, count: Optional[Any] = None
) -> dict:
    """
    目的の列が分かる過去のデータで, 方針を取っていた場合の実際の得と損を
    素朴なやり方と比べる関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ
    source: Union[str, Path, pd.DataFrame, Data]
        目的の列を持つ過去のデータ
    decision: dict
        意思決定の決め方
    options: Optional[dict] = None
        読み込みの設定
    count: Optional[Any] = None
        数える行の印 (行ごとの真偽値. 過去の値の列を作るために前の行も
        渡し, 比べる期間だけを数える場合など. 省略は全ての行)

    Returns
    ----------
    result: dict
        rows (数えた行), policies (方針ごとの実際の得 (正) と損 (負) の
        合計), gaps (この方針と他の方針の差の合計と 90 % の幅),
        sentences, decision

    Raises
    ----------
    ValueError
        目的の列が無いデータの場合と, 数える行が無い場合
    """
    # 全ての行で予測し (過去の値の列のため), 数える行だけで方針を決める
    # (上限のある選び方を, 数えない行に使わないように)
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    table = predict(bundle, source, options)
    target = bundle.dataset.info.get("target")
    if target not in table.columns:
        raise ValueError(f"たどり直すには目的の列 {target} が要ります")
    keep = np.array(
        (table["status"] == "ok") & table[target].notna(), dtype=bool
    )
    if count is not None:
        keep &= np.asarray(count, dtype=bool)
    if not keep.any():
        raise ValueError("たどり直せる行 (予測し, 目的の値が分かる行) が"
                         "ありません")
    decided = decide(bundle, table[keep].reset_index(drop=True), decision)
    frame, decision = decided["frame"], decided["decision"]
    part = frame[frame["action"].notna().to_numpy()]

    # 方針ごとの行ごとの実際の得と, この方針との差の幅
    kind = decision["type"]
    actual = pd.to_numeric(part[target], errors="coerce").to_numpy(
        float
    ) if kind == "quantity" else (
        label_text(part[target]) == str(decision["positive"])
    ).to_numpy()
    rows = {
        name: realized(kind, flag, part, decision, actual)
        for name, flag in policies_of(kind, part, decision).items()
    }
    policies = {name: float(values.sum()) for name, values in rows.items()}
    gaps = {
        name: {"gap": policies["この方針"] - value,
               "range": bootstrap_gap(rows["この方針"], rows[name])}
        for name, value in policies.items() if name != "この方針"
    }

    # 文 (量と振り分けは損, 選ぶは得. 符号は残す)
    word = "得" if kind == "select" else "損"
    sign = 1 if kind == "select" else -1
    sentences = [
        f"過去の {len(part):,} 行でたどり直すと, 実際の{word}はこの方針で "
        f"{amount_text(sign * policies['この方針'])}, " + ", ".join(
            f"{name}なら {amount_text(sign * value)}"
            for name, value in policies.items() if name != "この方針"
        ),
    ]
    for name, item in gaps.items():
        # 改善 (得の増え方, 損の減り方) と, 行を選び直した 90 % の幅
        low, high = item["range"]
        better = "多い" if kind == "select" else "少ない"
        worse = "少ない" if kind == "select" else "多い"
        verdict = "この方針が良い" if low > 0 else (
            "この方針が悪い" if high < 0 else "差は行の選び方の揺れの範囲"
        )
        sentences.append(
            f"{name}と比べて, この方針は{word}が "
            f"{amount_text(abs(item['gap']))} "
            f"{better if item['gap'] >= 0 else worse} (改善の 90 % の幅 "
            f"{amount_text(low)} から {amount_text(high)}. {verdict})"
        )
    if kind == "select":
        sentences.append(
            "得は行動の効果の仮定のもとでの値 (行動しなかった過去の結果"
            "しか分からないため)"
        )
    result = {
        "rows": int(len(part)),
        "policies": policies,
        "gaps": gaps,
        "sentences": sentences,
        "decision": decision,
    }

    return result
