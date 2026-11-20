from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from ml_eda import Data

from ml_inference.baseline import (
    BASELINE_NAME, baseline_text, check_baseline, current_from_baseline
)
from ml_inference.bundle import Bundle, load_bundle
from ml_inference.conclude import is_number, plain
from ml_inference.predict import lag_settings, predict

# 意思決定の型と, 型ごとに書ける項目
DECISION_KEYS = {
    "quantity": ("type", "under_cost", "over_cost", "stock", "step",
                 "current"),
    "select": ("type", "positive", "value", "cost", "effect",
               "effect_kind", "effect_goal", "budget", "budget_cost",
               "current"),
    "triage": ("type", "positive", "miss_cost", "false_alarm_cost",
               "capacity", "current"),
}

# 型ごとの, 行ごとに要る損得の値 (欠けた行は決めない)
AMOUNTS = {
    "quantity": ("under_cost", "over_cost", "stock"),
    "select": ("value", "cost", "effect"),
    "triage": ("miss_cost", "false_alarm_cost"),
}

# 対象を選ぶ型の効果の渡し方. share は行動で当たりの損を防げる割合 (確率
# に掛ける. 仮定), uplift は行動したときの当たりの確率 − しないときの当たり
# の確率 (効果のモデルの見積もり)
EFFECT_KINDS = ("share", "uplift")

# uplift の向き. reduce は当たりを減らしたい (解約など. 差が負なら良い),
# raise は当たりを増やしたい (成約など. 差が正なら良い)
EFFECT_GOALS = ("reduce", "raise")

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

# まとまりごとに選び直すのに要るまとまりの数の下限 (少ないと幅が狭く出る)
MIN_GROUPS = 20


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


def is_uplift(decision: dict) -> bool:
    """
    対象を選ぶ型の効果を, 確率の差 (uplift) で渡したかを返す関数

    Parameters
    ----------
    decision: dict
        意思決定の決め方

    Returns
    ----------
    uplift: bool
        確率の差で渡したか (effect_kind の無い場合と null は share)
    """
    uplift = (
        decision.get("type") == "select"
        and (decision.get("effect_kind") or "share") == "uplift"
    )

    return uplift


def uplift_sign(decision: dict) -> float:
    """
    確率の差を得の向きに直す符号を返す関数

    Parameters
    ----------
    decision: dict
        意思決定の決め方

    Returns
    ----------
    sign: float
        当たりを減らしたいなら -1, 増やしたいなら 1
    """
    sign = -1.0 if decision.get("effect_goal") == "reduce" else 1.0

    return sign


def check_uplift(
    decision: dict, columns: Optional[list[str]], errors: list[str]
) -> None:
    """
    確率の差で渡した効果が, -1 から 1 の数か, 予測の表の列の名前かを
    確かめる関数

    Parameters
    ----------
    decision: dict
        意思決定の決め方
    columns: Optional[list[str]]
        予測の表の列 (None なら列の有無は確かめない)
    errors: list[str]
        誤りの一覧 (書き足す)

    Returns
    ----------
    None
    """
    # 向き (差の符号の読み方) は必ず書く
    if decision.get("effect_goal") not in EFFECT_GOALS:
        errors.append(
            "effect_goal (当たりを減らしたいか増やしたいか) は "
            f"{', '.join(EFFECT_GOALS)} のどれかにしてください"
        )

    # 行ごとに違う見積もりはふつう列 (効果のモデルの出力)
    value = decision.get("effect")
    if value is None:
        errors.append("effect を書いてください")
        return
    if isinstance(value, str):
        if columns is not None and value not in columns:
            errors.append(f"effect の列 {value} が予測の表にありません")
        return
    if not is_number(value) or not -1 <= value <= 1:
        errors.append(
            "effect (確率の差) は -1 から 1 の数か列の名前にしてください"
        )


def check_decision(
    decision: Any, bundle: Bundle, columns: Optional[list[str]] = None,
    with_current: bool = True
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
    with_current: bool = True
        今のやり方の列 (current) があるかも確かめるか (使うのはたどり
        直しだけなので, 決めるだけなら確かめない)

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

    # 型ごとの損得と制約 (uplift の効果は -1 から 1 と向き)
    uplift = is_uplift(decision)
    if kind == "select" and (
        decision.get("effect_kind") or "share"
    ) not in EFFECT_KINDS:
        errors.append(
            f"effect_kind は {', '.join(EFFECT_KINDS)} のどれかにしてください"
        )
    if not uplift and decision.get("effect_goal") is not None:
        errors.append(
            "effect_goal は effect_kind が uplift のときだけ書いてください"
            " (防げる割合では使わない)"
        )
    for name in AMOUNTS[kind]:
        if name == "effect" and uplift:
            check_uplift(decision, columns, errors)
            continue
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
    errors += check_current(
        decision.get("current"), kind, columns if with_current else None
    )

    return errors


def check_current(
    current: Any, kind: str, columns: Optional[list[str]]
) -> list[str]:
    """
    今のやり方 (current) の誤りを一覧で返す関数

    Parameters
    ----------
    current: Any
        列の名前か, 今のやり方の決まり (辞書)
    kind: str
        意思決定の型
    columns: Optional[list[str]]
        予測の表の列 (渡すと, 列があるかも確かめる)

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 列の名前か決まり. k 期前と平均は量を決める型だけ (行動したかは
    # 前の期の値から作れない)
    errors: list[str] = []
    if current is None:
        return errors
    if isinstance(current, dict):
        errors = [f"current: {message}"
                  for message in check_baseline(current, columns)]
        if current.get("type") in ("lag", "mean") and kind != "quantity":
            errors.append(
                "current の k 期前と平均 (lag, mean) は量を決める型"
                " (quantity) だけに使えます. 行動したかは列で書いて"
                "ください"
            )
        return errors
    if not isinstance(current, str):
        errors.append("current は今のやり方の列の名前か決まり (辞書) に"
                      "してください")
    elif columns is not None and current not in columns:
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
    # 欠けた値と範囲の外の値 (負, 1 を超える効果. uplift の効果は絶対値が
    # 1 を超えるもの) の行は決めない (0 とみなすと, ただの行動や 1 % の点に
    # なる)
    values = {
        name: values_of(frame, decision.get(name))
        for name in AMOUNTS[decision["type"]]
    }
    missing = np.zeros(len(frame), dtype=bool)
    uplift = is_uplift(decision)
    for name, array in values.items():
        if name == "effect" and uplift:
            missing |= np.isnan(array) | (np.abs(array) > 1)
            continue
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


def row_quantiles(
    bundle: Bundle, frame: pd.DataFrame, quantiles: np.ndarray
) -> tuple[np.ndarray, bool]:
    """
    行ごとの誤りの分位点を返す関数 (何段も先の予測の行は, 測ってあれば
    その段の外れ方)

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    frame: pd.DataFrame
        予測の表 (何段も先の予測は step の列を持つ)
    quantiles: np.ndarray
        1 段目 (全ての行) の誤りの 1 % から 99 % の分位点

    Returns
    ----------
    result: tuple[np.ndarray, bool]
        (行の数, 99) の分位点と, 段ごとの外れ方を使った行があるか
    """
    # 1 段目は全ての行で測った記録のまま (段ごとの 1 段目は予測を始めた
    # 時刻の行だけで少ない). 測った段より先は, 測った中で最も先の段
    rows = np.tile(quantiles, (len(frame), 1))
    steps = (bundle.info.get("uncertainty") or {}).get("steps") or {}
    measured = sorted(int(k) for k in steps if k != "rows")
    if not measured or "step" not in frame.columns:
        return rows, False
    step = pd.to_numeric(frame["step"], errors="coerce").to_numpy()
    used = False
    for number in np.unique(step[np.isfinite(step)]).astype(int):
        if number <= 1:
            continue
        near = [m for m in measured if 1 < m <= number]
        if not near:
            continue
        rows[step == number] = np.asarray(steps[str(near[-1])], dtype=float)
        used = True
    result = (rows, used)

    return result


def max_step(uncertainty: dict) -> int:
    """
    段ごとの外れ方を測れた最も先の段を返す関数

    Parameters
    ----------
    uncertainty: dict
        一式の外れ方の記録

    Returns
    ----------
    step: int
        段 (測れていなければ 0)
    """
    # rows 以外の段の名前
    steps = uncertainty.get("steps") or {}
    step = max((int(k) for k in steps if k != "rows"), default=0)

    return step


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
        誤りの 1 % から 99 % の分位点 (全ての行で 1 つか, (行の数, 99)
        の行ごと)

    Returns
    ----------
    values: np.ndarray
        (行の数, 99) のありうる実際の値 (同じ重み. 1 % より外と 99 % より
        外の裾は含まないので, τ が 0.99 に近いと見込みの損は甘い)
    """
    # 比は掛け, 差は足す
    grid = quantiles if quantiles.ndim == 2 else quantiles[None, :]
    if kind == "relative":
        values = prediction[:, None] * np.exp(grid)
    else:
        values = prediction[:, None] + grid

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
    grid, by_step = row_quantiles(bundle, frame, quantiles)
    possible = outcomes(prediction, kind, grid)
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
        "by_step": by_step,
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
    # 行動した場合の見込みの得. share は 確率 × 価値 × 効果 − 費用 (効果は
    # 確率に比例するという仮定), uplift は 価値 × 向きを揃えた確率の差 −
    # 費用 (確率を二重に掛けない. 減らしたい当たりは差の符号を反転)
    probability = np.nan_to_num(pd.to_numeric(
        frame[f"p:{decision['positive']}"], errors="coerce"
    ).to_numpy(dtype=float))
    amounts, _ = amounts_of(frame, decision)
    cost = np.nan_to_num(amounts["cost"])
    if is_uplift(decision):
        effect = uplift_sign(decision) * amounts["effect"]
    else:
        effect = probability * amounts["effect"]
    gain = np.nan_to_num(amounts["value"] * effect - cost)
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
        elif uncertainty.get("split") == "recent":
            notes.append(
                f"外れ方は渡した実績の {uncertainty.get('rows', 0):,} 行で"
                "測り直した (その後に傾向が変われば, また測り直す)"
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
                "2 段目より先の予測には, 段ごとに測った外れ方を当てた ("
                f"{max_step(uncertainty)} 段まで. それより先は最も先の段の"
                "外れ方)"
                if summary.get("by_step") else
                "2 段目より先の予測にも 1 つ先の外れ方を当てている (先の段"
                "ほど外れ方は大きく, 量は少なめ. 実績で段ごとの外れ方を"
                "測り直せる)"
            )
        return notes

    # 選ぶと振り分け: 確率のずれ, 効果の仮定, 人の判断
    calibration = uncertainty.get("calibration") or {}
    ece = uncertainty.get("test_ece", calibration.get("ece"))
    if ece is not None:
        notes.append(
            f"確率と実際に当たった割合のずれは平均 {ece * 100:.1f} ポイント"
            + ((" (大きい. uplift の得は確率を使わず, 比べの順にだけ"
                "使う)" if is_uplift(decision)
                else " (大きいので, 確率を額面どおりに使う判断は粗い)")
               if ece > 0.05 else "")
            + (f". 渡した実績の {uncertainty.get('rows', 0):,} 行で測り"
               "直した値"
               if uncertainty.get("split") == "recent" else "")
        )
    if is_uplift(decision):
        goal = (
            "当たりを減らしたいので, 差が負の行ほど得"
            if decision.get("effect_goal") == "reduce"
            else "当たりを増やしたいので, 差が正の行ほど得"
        )
        notes.append(
            f"効果 ({decision['effect']}) は, 行動したときの当たりの確率 −"
            f" しないときの当たりの確率の見積もり ({goal}). 見積もりは, 行動"
            "を無作為に割り当てた記録でしか確かめられない (行動した人を選ん"
            "だ記録だけでは偏る). 行動すると悪くなる行は選ばない"
        )
    elif kind == "select":
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
    if rows == 0:
        sentences = ["決められる行がありません (予測でき, 損得の値が揃った"
                     "行が要る)"]
        return sentences
    tail = (
        " (同じ前提から計算した比べなので, この方針は必ず同じか良く出る. "
        "実際の差は過去のデータでたどり直して確かめる)"
    )
    if is_uplift(decision):
        tail = (
            " (同じ見積もりから計算した比べなので, この方針は必ず同じか良く"
            "出る. 実際の差は, 行動を無作為に割り当てた試験で確かめる)"
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
        basis = (
            "確率の差の見積もり (効果のモデル) と, 書いた価値, 費用のもとで"
            if is_uplift(decision)
            else "確率と, 書いた価値, 費用, 効果 (仮定) のもとで"
        )
        sentences = [
            f"{basis}, 見込みの得が正の"
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
    bundle.require_column("意思決定")
    decision = plain(dict(decision))
    errors = check_decision(
        decision, bundle, list(prediction.columns), with_current=False
    )
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


# 行動した印と, 行動しなかった印として読む言葉 (小文字にして比べる)
YES_WORDS = ("yes", "y", "true", "1", "はい", "済")
NO_WORDS = ("no", "n", "false", "0", "いいえ", "未")


def acted_flags(column: pd.Series) -> np.ndarray:
    """
    今のやり方で行動したかの列を, 行ごとの真偽値にする関数

    文字の列をそのまま真偽値にすると no も行動したことになるので, 決まった
    言葉で読む

    Parameters
    ----------
    column: pd.Series
        列 (真偽値, 数, yes や no などの文字)

    Returns
    ----------
    flags: np.ndarray
        行動したか

    Raises
    ----------
    ValueError
        読めない値か欠けた値がある場合
    """
    # 真偽値はそのまま, 数は 0 より大きいか, 文字は決まった言葉か数
    # (全角は半角にそろえる)
    name = column.name
    if column.isna().any():
        raise ValueError(
            f"今のやり方の列 {name} に欠けた値があります"
            " (行動したかを全ての行に書いてください)"
        )
    if pd.api.types.is_bool_dtype(column):
        flags = column.to_numpy(bool)
        return flags
    if pd.api.types.is_numeric_dtype(column):
        flags = (pd.to_numeric(column) > 0).to_numpy(bool)
        return flags
    text = column.astype(str).str.normalize("NFKC").str.strip().str.lower()
    number = pd.to_numeric(text, errors="coerce")
    yes = text.isin(YES_WORDS) | (number > 0)
    known = yes | text.isin(NO_WORDS) | number.notna()
    unknown = sorted(set(text[~known]))
    if unknown:
        raise ValueError(
            f"今のやり方の列 {name} に, 行動したかとして読めない値が"
            f"あります: {', '.join(unknown[:5])} (yes と no, はい と"
            " いいえ, 1 と 0 などで書いてください)"
        )
    flags = yes.to_numpy(bool)

    return flags


def policies_of(
    kind: str, part: pd.DataFrame, decision: dict,
    current: Optional[tuple[str, bool]] = None
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
    current: Optional[tuple[str, bool]] = None
        今のやり方の列の名前と, 置いた後の量の列か (決まりから作った列.
        偽なら量は足した量の列). 無ければ比べない

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
        if current is not None:
            values = values_of(part, current[0])
            if np.isnan(values).any():
                raise ValueError(
                    f"今のやり方の列 {current[0]} に欠けた値か"
                    "数でない値があります (足した量を全ての行に書いて"
                    "ください)"
                )
            # 決まりの値は置いた後の量 (今ある量が多ければ足さない)
            flags["今のやり方"] = np.maximum(values, stock) if current[1] else (
                stock + values
            )
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
    if current is not None:
        flags["今のやり方"] = acted_flags(part[current[0]])

    return flags


def bootstrap_gap(
    first: np.ndarray, second: np.ndarray,
    groups: Optional[np.ndarray] = None
) -> tuple[float, float]:
    """
    2 つの方針の行ごとの得の合計の差の, 90 % の幅を求める関数

    Parameters
    ----------
    first: np.ndarray
        1 つ目の方針の行ごとの得
    second: np.ndarray
        2 つ目の方針の行ごとの得
    groups: Optional[np.ndarray] = None
        行のまとまりの番号 (0 から. group_labels の結果). 渡すと
        まとまりごとに選び直す (同じまとまりの行は似ているので, 行ごと
        では幅が狭く出る)

    Returns
    ----------
    bounds: tuple[float, float]
        選び直した差の合計の 5 % と 95 % の点
    """
    # 行 (かまとまりごとの差の合計) の, 重複を許した選び直し (乱数は固定)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gap = first - second
    if groups is not None:
        gap = np.bincount(groups, weights=gap)
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


def group_codes(values: Any) -> np.ndarray:
    """
    まとまりの値を 0 からの番号にする関数 (欠けた値の行は 1 行ずつ別の
    まとまり)

    Parameters
    ----------
    values: Any
        行ごとのまとまりの値

    Returns
    ----------
    codes: np.ndarray
        行ごとの番号
    """
    # 欠けた値 (-1) には, 使っていない番号を 1 つずつ
    codes = pd.factorize(pd.Series(values))[0].astype(int)
    missing = codes < 0
    codes[missing] = codes.max(initial=-1) + 1 + np.arange(missing.sum())

    return codes


def group_labels(
    bundle: Bundle, part: pd.DataFrame, rows: np.ndarray, size: int,
    group: Any
) -> tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """
    たどり直しの幅を選び直すまとまりを返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    part: pd.DataFrame
        数えて決めた行の表
    rows: np.ndarray
        part の行の, 渡したデータの行の位置
    size: int
        渡したデータの行の数
    group: Any
        "auto" (時刻でずらす一式は時刻の列, 他は行ごと), None (行ごと),
        列の名前, 行ごとの値の並び (渡したデータの行の数)

    Returns
    ----------
    result: tuple[Optional[np.ndarray], Optional[str], Optional[str]]
        part の行ごとのまとまりの番号 (行ごとなら None), まとまりの名前
        (列の名前か「渡したまとまり」), 文に添える選び直し方 (「date
        ごとに選び直した」など. まとまりが MIN_GROUPS より少なければ,
        行ごとに戻したことを述べる)

    Raises
    ----------
    ValueError
        列が無い場合と, 並びの数が行の数と違う場合
    """
    # 自動は時刻の列 (同じ時刻の系列どうしは似ている. 名前の順で最初)
    if group is None:
        return None, None, None
    if isinstance(group, str) and group == "auto":
        times = sorted(
            c for c in lag_settings(bundle)["time"] if c in part.columns
        )
        if not times:
            return None, None, None
        group = times[0]
    if np.ndim(group) == 0:
        if group not in part.columns:
            raise ValueError(f"まとまりの列 {group} がありません")
        values, name, way = part[group].to_numpy(), str(group), (
            f"{group} ごと"
        )
    else:
        given = np.asarray(group)
        if given.ndim != 1 or len(given) != size:
            raise ValueError(
                f"まとまりの並びの数 ({len(given):,}) が, 渡したデータの"
                f"行の数 ({size:,}) と違います"
            )
        values, name, way = given[rows], "渡したまとまり", (
            "渡したまとまりごと"
        )
    codes = group_codes(values)
    count = int(codes.max(initial=-1)) + 1
    if count < MIN_GROUPS:
        result = (None, name, f"{way}ではまとまりが {count} しか無いので,"
                  " 行ごとに選び直した")
        return result
    result = (codes, name, f"{way}に選び直した")

    return result


def current_column(
    bundle: Bundle, table: pd.DataFrame, current: Any, keep: np.ndarray
) -> tuple[pd.DataFrame, Optional[tuple[str, bool]], np.ndarray]:
    """
    たどり直しで比べる今のやり方の列を, 予測の表から選ぶか決まりから作る
    関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    table: pd.DataFrame
        全ての行の予測の表
    current: Any
        今のやり方の列の名前か決まり (無ければ None)
    keep: np.ndarray
        数える行の印

    Returns
    ----------
    result: tuple[pd.DataFrame, Optional[tuple[str, bool]], np.ndarray]
        (決まりから作った列を足した) 予測の表, 今のやり方の列の名前と
        置いた後の量の列か (無ければ None), 今のやり方の値が無いので
        数えない行の印

    Raises
    ----------
    ValueError
        今のやり方の列が無い場合と, 決まりから作れない場合
    """
    # 列の名前 (決まりの column も同じ)
    dropped = np.zeros(len(table), dtype=bool)
    if not current:
        return table, None, dropped
    if isinstance(current, dict) and current.get("type") == "column":
        current = current.get("column")
    if not isinstance(current, dict):
        if current not in table.columns:
            raise ValueError(f"今のやり方の列 {current} が過去のデータに"
                             "ありません")
        result = (table, (str(current), False), dropped)
        return result

    # k 期前と平均: 時刻と系列を分ける列を省いた決まりは一式の過去の値の
    # 手順から補い, 全ての行で求めて (前の期の値のため), 値の無い数える
    # 行は除く
    rule = dict(current)
    settings = lag_settings(bundle)
    if "time" not in rule and len(settings["time"]) == 1:
        rule["time"] = next(iter(settings["time"]))
    if "by" not in rule:
        rule["by"] = sorted(settings["by"])
    values = current_from_baseline(
        table, rule, bundle.dataset.info.get("target")
    )
    name = BASELINE_NAME
    while name in table.columns:
        name = f"_{name}"
    table = table.assign(**{name: values.to_numpy()})
    dropped = keep & values.isna().to_numpy()
    result = (table, (name, True), dropped)

    return result


def backtest(
    bundle: Union[str, Path, Bundle],
    source: Union[str, Path, pd.DataFrame, Data], decision: dict,
    options: Optional[dict] = None, count: Optional[Any] = None,
    group: Any = "auto"
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
        意思決定の決め方 (current は今のやり方の列の名前か決まり.
        決まりなら全ての行で値を作り, 値の無い行は数えない)
    options: Optional[dict] = None
        読み込みの設定
    count: Optional[Any] = None
        数える行の印 (行ごとの真偽値. 過去の値の列を作るために前の行も
        渡し, 比べる期間だけを数える場合など. 省略は全ての行)
    group: Any = "auto"
        差の幅を選び直すまとまり. "auto" は時刻でずらす一式なら時刻の
        列 (同じ時刻の系列どうしは似ているので, 行ごとでは幅が狭く出る),
        他は行ごと. None は行ごと. 列の名前か, 行ごとの値の並びも書ける

    Returns
    ----------
    result: dict
        rows (数えた行), policies (方針ごとの実際の得 (正) と損 (負) の
        合計), gaps (この方針と他の方針の差の合計と 90 % の幅),
        group (幅を選び直したまとまりの名前と数. 行ごとなら None),
        sentences, decision

    Raises
    ----------
    ValueError
        目的の列が無いデータの場合と, 数える行が無い場合と, 効果を確率の差
        で渡した対象を選ぶ型の場合
    """
    # 効果を確率の差で渡した選ぶ型は, 過去の結果から実際の得を数えられない
    # (行動しなかった場合の結果が分からない)
    if is_uplift(decision):
        raise ValueError(
            "効果を確率の差 (uplift) で渡した対象を選ぶ型は, たどり直せ"
            "ません. 効果のモデルは, 行動を無作為に割り当てた記録で確かめ"
            "てください"
        )

    # 全ての行で予測し (過去の値の列のため), 数える行だけで方針を決める
    # (上限のある選び方を, 数えない行に使わないように)
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    bundle.require_column("たどり直し")
    errors = check_decision(decision, bundle, with_current=False)
    if errors:
        raise ValueError("意思決定の決め方の誤り: " + " / ".join(errors))
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
    table, current, dropped = current_column(
        bundle, table, decision.get("current"), keep
    )
    keep &= ~dropped
    if not keep.any():
        raise ValueError("今のやり方の値のある, たどり直せる行がありません")
    decided = decide(bundle, table[keep].reset_index(drop=True), decision)
    frame, decision = decided["frame"], decided["decision"]
    acted = frame["action"].notna().to_numpy()
    part = frame[acted]
    groups, group_name, group_note = group_labels(
        bundle, part, np.flatnonzero(keep)[acted], len(table), group
    )

    # 方針ごとの行ごとの実際の得と, この方針との差の幅
    kind = decision["type"]
    actual = pd.to_numeric(part[target], errors="coerce").to_numpy(
        float
    ) if kind == "quantity" else (
        label_text(part[target]) == str(decision["positive"])
    ).to_numpy()
    rows = {
        name: realized(kind, flag, part, decision, actual)
        for name, flag in policies_of(kind, part, decision, current).items()
    }
    policies = {name: float(values.sum()) for name, values in rows.items()}
    gaps = {
        name: {"gap": policies["この方針"] - value,
               "range": bootstrap_gap(rows["この方針"], rows[name], groups)}
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
        # 改善 (得の増え方, 損の減り方) と, 選び直した 90 % の幅
        low, high = item["range"]
        better = "多い" if kind == "select" else "少ない"
        worse = "少ない" if kind == "select" else "多い"
        verdict = "この方針が良い" if low > 0 else (
            "この方針が悪い" if high < 0 else "差は選び直しの揺れの範囲"
        )
        sentences.append(
            f"{name}と比べて, この方針は{word}が "
            f"{amount_text(abs(item['gap']))} "
            f"{better if item['gap'] >= 0 else worse} (改善の 90 % の幅 "
            f"{amount_text(low)} から {amount_text(high)}"
            + (f". {group_note}" if group_note else "")
            + f". {verdict})"
        )
    if kind == "select":
        sentences.append(
            "得は行動の効果の仮定のもとでの値 (行動しなかった過去の結果"
            "しか分からないため)"
        )
    rule = decision.get("current")
    if isinstance(rule, dict) and rule.get("type") != "column":
        sentences.append(
            f"今のやり方は, 決まり ({baseline_text(rule)}) から作った"
            + (f". 今のやり方の値が無い {int(dropped.sum()):,} 行 (前の期の"
               "値が無い始めの行など) は, どの方針でも数えない"
               if dropped.any() else "")
        )
    result = {
        "rows": int(len(part)),
        "policies": policies,
        "gaps": gaps,
        "group": None if groups is None else {
            "name": group_name,
            "count": int(groups.max()) + 1,
        },
        "sentences": sentences,
        "decision": decision,
    }

    return result
