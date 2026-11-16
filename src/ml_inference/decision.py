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
               "current"),
}


def check_amount(
    decision: dict, name: str, columns: Optional[list[str]],
    errors: list[str], required: bool = True
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

    Returns
    ----------
    None
    """
    # 列の名前なら列があるか, 数なら 0 以上か
    value = decision.get(name)
    if value is None:
        if required:
            errors.append(f"{name} を書いてください")
        return
    if isinstance(value, str):
        if columns is not None and value not in columns:
            errors.append(f"{name} の列 {value} が予測の表にありません")
        return
    if not is_number(value) or value < 0:
        errors.append(f"{name} は 0 以上の数か列の名前にしてください")


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
    classify = bundle.task == "classification"

    # 型ごとの項目
    if kind == "quantity":
        if classify:
            errors.append("quantity は回帰と予測の一式にだけ使えます")
        check_amount(decision, "under_cost", columns, errors)
        check_amount(decision, "over_cost", columns, errors)
        check_amount(decision, "stock", columns, errors, False)
        step = decision.get("step", 1)
        if not is_number(step) or step <= 0:
            errors.append("step は正の数にしてください")
        if all(is_number(decision.get(k)) for k in ("under_cost",
                                                     "over_cost")):
            if decision["under_cost"] + decision["over_cost"] <= 0:
                errors.append("under_cost と over_cost の和を正にしてください")
    else:
        if not classify or len(bundle.classes) != 2:
            errors.append(f"{kind} は 2 クラスの分類の一式にだけ使えます")
        elif str(decision.get("positive")) not in bundle.classes:
            errors.append(
                f"positive は {', '.join(bundle.classes)} のどれかにして"
                "ください"
            )
    if kind == "select":
        check_amount(decision, "value", columns, errors)
        check_amount(decision, "cost", columns, errors)
        effect = decision.get("effect")
        if not is_number(effect) or not 0 <= effect <= 1:
            errors.append(
                "effect (行動で当たりの損を防げる割合. 仮定) を 0 から 1 の"
                "数で書いてください"
            )
        for name in ("budget", "budget_cost"):
            value = decision.get(name)
            if value is not None and (not is_number(value) or value < 0):
                errors.append(f"{name} は 0 以上の数にしてください")
    if kind == "triage":
        check_amount(decision, "miss_cost", columns, errors)
        check_amount(decision, "false_alarm_cost", columns, errors)
        if all(is_number(decision.get(k)) for k in ("miss_cost",
                                                     "false_alarm_cost")):
            if decision["miss_cost"] + decision["false_alarm_cost"] <= 0:
                errors.append(
                    "miss_cost と false_alarm_cost の和を正にしてください"
                )
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
        行ごとの数 (列の数にできない値は 0)
    """
    # 列は数にし, 数はそろえる
    if value is None:
        return np.zeros(len(frame))
    if isinstance(value, str):
        values = pd.to_numeric(frame[value], errors="coerce").fillna(0)
        return values.to_numpy(dtype=float)
    values = np.full(len(frame), float(value))

    return values


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
        (行の数, 99) のありうる実際の値 (同じ重み)
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


def decide_quantity(
    bundle: Bundle, frame: pd.DataFrame, decision: dict, ok: np.ndarray
) -> tuple[dict, dict]:
    """
    足りない損と余る損の比で, 置く量を決める関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    frame: pd.DataFrame
        予測の表
    decision: dict
        意思決定の決め方
    ok: np.ndarray
        予測した行の印

    Returns
    ----------
    result: tuple[dict, dict]
        足す列 (action, level, expected, baseline_expected) と, まとめ
    """
    # 損の比で決まる分位点まで置く (新聞売りの問題)
    kind, quantiles = residual_samples(bundle)
    prediction = pd.to_numeric(
        frame["prediction"], errors="coerce"
    ).to_numpy(dtype=float)
    under = values_of(frame, decision["under_cost"])
    over = values_of(frame, decision["over_cost"])
    stock = values_of(frame, decision.get("stock"))
    step = float(decision.get("step", 1))
    tau = under / np.maximum(under + over, 1e-12)
    index = np.clip(np.round(tau * 100).astype(int), 1, 99) - 1
    shift = quantiles[index]
    target = prediction * np.exp(shift) if kind == "relative" else (
        prediction + shift
    )
    action = np.maximum(np.ceil((target - stock) / step - 1e-9) * step, 0)
    level = stock + action

    # 見込みの損 (外れ方の分位点を, ありうる実際の値として平均)
    safe = np.where(ok, prediction, 0.0)
    possible = outcomes(safe, kind, quantiles)
    expected = quantity_cost(level, possible, under, over).mean(axis=1)
    point = np.maximum(np.ceil((safe - stock) / step - 1e-9) * step, 0)
    baseline = quantity_cost(stock + point, possible, under, over).mean(
        axis=1
    )
    columns = {
        "action": np.where(ok, action, np.nan),
        "level": np.where(ok, level, np.nan),
        "expected": np.where(ok, -expected, np.nan),
        "baseline_expected": np.where(ok, -baseline, np.nan),
    }
    summary = {
        "rows": int(ok.sum()),
        "acting": int((action[ok] > 0).sum()),
        "amount": float(action[ok].sum()),
        "expected_cost": float(expected[ok].sum()),
        "baseline_cost": float(baseline[ok].sum()),
        "baseline": "予測の値どおりに置く",
        "tau": float(np.mean(tau[ok])) if ok.any() else None,
        "kind": kind,
    }
    result = (columns, summary)

    return result


def decide_select(
    bundle: Bundle, frame: pd.DataFrame, decision: dict, ok: np.ndarray
) -> tuple[dict, dict]:
    """
    見込みの得の大きい順に, 上限まで行動する行を選ぶ関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    frame: pd.DataFrame
        予測の表
    decision: dict
        意思決定の決め方
    ok: np.ndarray
        予測した行の印

    Returns
    ----------
    result: tuple[dict, dict]
        足す列 (action, expected) と, まとめ
    """
    # 見込みの得 = 確率 × 価値 × 効果 − 費用
    probability = pd.to_numeric(
        frame[f"p:{decision['positive']}"], errors="coerce"
    ).fillna(0).to_numpy(dtype=float)
    value = values_of(frame, decision["value"])
    cost = values_of(frame, decision["cost"])
    gain = probability * value * float(decision["effect"]) - cost
    gain = np.where(ok, gain, -np.inf)

    # 得の大きい順に, 正のものを上限まで
    order = np.argsort(-gain, kind="stable")
    chosen = np.zeros(len(frame), dtype=bool)
    spent, count = 0.0, 0
    for row in order:
        if gain[row] <= 0:
            break
        budget = decision.get("budget")
        if budget is not None and count >= budget:
            break
        limit = decision.get("budget_cost")
        if limit is not None and spent + cost[row] > limit:
            continue
        chosen[row] = True
        spent += cost[row]
        count += 1

    # 比べ: 確率の高い順に同じ件数 (上限があっても同じ件数なので守る).
    # 全てに行動は上限の無い場合だけ (上限を破る比べは公平でない)
    top = np.zeros(len(frame), dtype=bool)
    ranked = np.argsort(-np.where(ok, probability, -1), kind="stable")
    top[ranked[:count]] = True
    finite = np.where(ok, gain, 0.0)
    limited = decision.get("budget") is not None or (
        decision.get("budget_cost") is not None
    )
    baselines = {"確率の高い順に同じ件数": float(finite[top].sum())}
    if not limited:
        baselines["全てに行動"] = float(finite[ok].sum())
    columns = {
        "action": np.where(ok, chosen, None),
        "expected": np.where(ok, finite, np.nan),
    }
    summary = {
        "rows": int(ok.sum()),
        "acting": int(count),
        "amount": float(spent),
        "expected_gain": float(finite[chosen].sum()),
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


def realized_gain(
    flag: np.ndarray, hit: np.ndarray, decision: dict, part: pd.DataFrame
) -> float:
    """
    行動した行の実際の当たりから, 方針の得 (選ぶ) か損 (振り分け) を
    求める関数

    Parameters
    ----------
    flag: np.ndarray
        行動したかの印
    hit: np.ndarray
        実際に当たりだったか
    decision: dict
        意思決定の決め方
    part: pd.DataFrame
        数える行の予測の表

    Returns
    ----------
    total: float
        得の合計 (選ぶ. 効果は仮定のまま) か, 損の合計の負 (振り分け)
    """
    # 選ぶは当たりの価値 × 効果 − 費用, 振り分けは空振りと見逃しの損
    if decision["type"] == "select":
        value = values_of(part, decision["value"])
        cost = values_of(part, decision["cost"])
        gains = hit * value * float(decision["effect"]) - cost
        total = float(gains[flag].sum())
        return total
    miss = values_of(part, decision["miss_cost"])
    alarm = values_of(part, decision["false_alarm_cost"])
    total = -float(
        (alarm * (flag & ~hit)).sum() + (miss * (~flag & hit)).sum()
    )

    return total


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
        予測した行の印

    Returns
    ----------
    result: tuple[dict, dict]
        足す列 (action, expected) と, まとめ
    """
    # 回すと空振りの損, 回さないと見逃しの損の見込み
    probability = pd.to_numeric(
        frame[f"p:{decision['positive']}"], errors="coerce"
    ).fillna(0).to_numpy(dtype=float)
    miss = values_of(frame, decision["miss_cost"])
    alarm = values_of(frame, decision["false_alarm_cost"])
    threshold = alarm / np.maximum(miss + alarm, 1e-12)
    act = ok & (probability >= threshold)

    expected = triage_loss(act, probability, miss, alarm)
    columns = {
        "action": np.where(ok, act, None),
        "expected": np.where(ok, -expected, np.nan),
    }
    summary = {
        "rows": int(ok.sum()),
        "acting": int(act.sum()),
        "threshold": float(np.mean(threshold[ok])) if ok.any() else None,
        "expected_cost": float(expected[ok].sum()),
        "baselines": {
            "確率 0.5 で分ける": float(triage_loss(
                ok & (probability >= 0.5), probability, miss, alarm
            )[ok].sum()),
            "全てを回す": float(
                triage_loss(ok, probability, miss, alarm)[ok].sum()
            ),
            "どれも回さない": float(triage_loss(
                np.zeros(len(frame), dtype=bool), probability, miss, alarm
            )[ok].sum()),
        },
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


def caveats(bundle: Bundle, kind: str, decision: dict) -> list[str]:
    """
    意思決定の前提 (外れ方の記録, 確率のずれ, 仮定) を述べる文を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    kind: str
        意思決定の型
    decision: dict
        意思決定の決め方

    Returns
    ----------
    notes: list[str]
        前提の文
    """
    # 外れ方と確率の記録
    uncertainty = bundle.info.get("uncertainty") or {}
    notes = []
    if kind == "quantity":
        coverage = (uncertainty.get("coverage") or {}).get("90")
        if coverage is not None:
            notes.append(
                f"外れ方は検証用の期間で測った. その 90 % の幅は, テスト用の"
                f"期間の実際の値の {coverage * 100:.0f} % を覆った"
                + (" (外れ方を小さく見積もっている. 足りない損が大きいなら"
                   "多めに見る)" if coverage < 0.85 else "")
            )
    else:
        calibration = uncertainty.get("calibration") or {}
        ece = uncertainty.get("test_ece", calibration.get("ece"))
        if ece is not None:
            notes.append(
                f"確率と実際に当たった割合のずれは平均 {ece * 100:.1f} "
                "ポイント" + (" (大きいので, 確率を額面どおりに使う判断は"
                              "粗い)" if ece > 0.05 else "")
            )
    if kind == "select":
        notes.append(
            f"行動で当たりの損を防げる割合 {decision['effect']} は仮定 (確か"
            "めるには, 行動した組としない組を比べる試験が要る)"
        )

    return notes


def sentences_of(kind: str, summary: dict, decision: dict) -> list[str]:
    """
    意思決定のまとめを日本語の文にする関数

    Parameters
    ----------
    kind: str
        意思決定の型
    summary: dict
        まとめ
    decision: dict
        意思決定の決め方

    Returns
    ----------
    sentences: list[str]
        文
    """
    # 何をどれだけするかと, 素朴なやり方との比べ
    sentences = []
    if kind == "quantity":
        sentences.append(
            f"{summary['rows']:,} 行のうち {summary['acting']:,} 行に, 合計 "
            f"{amount_text(summary['amount'])} を足すことを勧める (足りない"
            f"損 {decision['under_cost']}, 余る損 {decision['over_cost']} "
            f"から, 予測の外れ方の {summary['tau'] * 100:.0f} % の点まで置く)"
        )
        saved = summary["baseline_cost"] - summary["expected_cost"]
        sentences.append(
            f"見込みの損は {amount_text(summary['expected_cost'])} で, "
            f"{summary['baseline']}場合の "
            f"{amount_text(summary['baseline_cost'])} より "
            f"{amount_text(saved)} 少ない"
        )
    elif kind == "select":
        sentences.append(
            f"{summary['rows']:,} 件のうち {summary['acting']:,} 件に行動"
            f"することを勧める (費用の合計 {amount_text(summary['amount'])})"
        )
        sentences.append(
            f"見込みの得は {amount_text(summary['expected_gain'])}. "
            + ", ".join(
                f"{name}なら {amount_text(value)}"
                for name, value in summary["baselines"].items()
            )
        )
    else:
        sentences.append(
            f"{summary['rows']:,} 件のうち {summary['acting']:,} 件を回す"
            f"ことを勧める (確率 {summary['threshold']:.2f} 以上. 見逃しの"
            f"損 {decision['miss_cost']}, 空振りの損 "
            f"{decision['false_alarm_cost']} から)"
        )
        sentences.append(
            f"見込みの損は {amount_text(summary['expected_cost'])}. "
            + ", ".join(
                f"{name}なら {amount_text(value)}"
                for name, value in summary["baselines"].items()
            )
        )

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
        frame (action, expected などを足した表), summary, sentences,
        notes (前提), decision (使った決め方)

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

    # 型ごとの行動と見込み
    kind = decision["type"]
    ok = (prediction["status"] == "ok").to_numpy()
    maker = {"quantity": decide_quantity, "select": decide_select,
             "triage": decide_triage}[kind]
    columns, summary = maker(bundle, prediction, decision, ok)
    frame = prediction.copy()
    for name in columns:
        if name in frame.columns:
            frame = frame.rename(columns={name: f"{name} (入力)"})
    for name, values in columns.items():
        frame[name] = values
    result = {
        "frame": frame,
        "summary": summary,
        "sentences": sentences_of(kind, summary, decision),
        "notes": caveats(bundle, kind, decision),
        "decision": decision,
    }

    return result


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
        合計), sentences

    Raises
    ----------
    ValueError
        目的の列が無いデータの場合
    """
    # 予測して, 方針の行動を決める
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    table = predict(bundle, source, options)
    target = bundle.dataset.info.get("target")
    if target not in table.columns:
        raise ValueError(f"たどり直すには目的の列 {target} が要ります")
    decided = decide(bundle, table, decision)
    frame = decided["frame"]
    decision = decided["decision"]
    known = np.array(
        (frame["status"] == "ok") & frame[target].notna(), dtype=bool
    )
    if count is not None:
        known &= np.asarray(count, dtype=bool)
    part = frame[known]
    kind = decision["type"]
    policies: dict[str, float] = {}

    # 量: 実際の値で足りない損と余る損
    if kind == "quantity":
        actual = pd.to_numeric(part[target], errors="coerce").to_numpy(float)
        under = values_of(part, decision["under_cost"])
        over = values_of(part, decision["over_cost"])
        stock = values_of(part, decision.get("stock"))
        step = float(decision.get("step", 1))
        point = pd.to_numeric(part["prediction"]).to_numpy(float)
        point = stock + np.maximum(
            np.ceil((point - stock) / step - 1e-9) * step, 0
        )
        level = part["level"].to_numpy(float)
        policies["この方針"] = -float(
            quantity_cost(level, actual, under, over).sum()
        )
        policies["予測の値どおりに置く"] = -float(
            quantity_cost(point, actual, under, over).sum()
        )
        if decision.get("current"):
            current = stock + values_of(part, decision["current"])
            policies["今のやり方"] = -float(
                quantity_cost(current, actual, under, over).sum()
            )
    else:
        # 選ぶと振り分け: 実際に当たりだったか
        hit = (part[target].astype(str) == str(decision["positive"])
               ).to_numpy()
        probability = pd.to_numeric(
            part[f"p:{decision['positive']}"]
        ).to_numpy(float)
        acted = part["action"].astype(bool).to_numpy()
        count = int(acted.sum())
        top = np.zeros(len(part), dtype=bool)
        top[np.argsort(-probability, kind="stable")[:count]] = True
        flags = {"この方針": acted}
        if kind == "select":
            flags["確率の高い順に同じ件数"] = top
            if decision.get("budget") is None and decision.get(
                "budget_cost"
            ) is None:
                flags["全てに行動"] = np.ones(len(part), dtype=bool)
        else:
            flags.update({
                "確率 0.5 で分ける": probability >= 0.5,
                "全てを回す": np.ones(len(part), dtype=bool),
                "どれも回さない": np.zeros(len(part), dtype=bool),
            })
        if decision.get("current"):
            flags["今のやり方"] = part[decision["current"]].astype(
                bool
            ).to_numpy()
        policies = {
            name: realized_gain(flag, hit, decision, part)
            for name, flag in flags.items()
        }

    # 文 (得は正, 損は負)
    best = max(policies, key=lambda name: policies[name])
    sentences = [
        f"過去の {len(part):,} 行でたどり直すと, この方針の"
        + ("得" if kind == "select" else "損") + " は "
        + amount_text(abs(policies["この方針"]))
        + ", " + ", ".join(
            f"{name}なら {amount_text(abs(value))}"
            for name, value in policies.items() if name != "この方針"
        ),
        f"最も良いのは「{best}」" + (
            "" if best == "この方針" else " (この方針より良いやり方がある. "
            "損得の値や仮定, 予測の外れ方を見直す)"
        ),
    ]
    if kind == "select":
        sentences.append("得は, 行動の効果の仮定のもとでの値")
    result = {
        "rows": int(len(part)),
        "policies": policies,
        "sentences": sentences,
        "decision": decision,
    }

    return result
