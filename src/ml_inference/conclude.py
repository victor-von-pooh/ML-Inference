from typing import Any, Optional

import numpy as np
import pandas as pd

# 判断の名前
UNDECIDED, NO_PREDICTION = "保留", "予測なし"

# 結論の決め方の項目
RULE_KEYS = ("min_confidence", "positive", "threshold", "bands", "group_by")


def is_number(value: Any) -> bool:
    """
    真偽値でない数かを確かめる関数

    Parameters
    ----------
    value: Any
        確かめる値

    Returns
    ----------
    result: bool
        int か float (bool を除く) なら True
    """
    # bool は int の子なので除く
    result = isinstance(value, (int, float)) and not isinstance(value, bool)

    return result


def check_rules(
    rules: Optional[dict], task: str, classes: list[str],
    columns: Optional[list[str]] = None
) -> list[str]:
    """
    結論の決め方の誤りを一覧で返す関数

    Parameters
    ----------
    rules: Optional[dict]
        結論の決め方
    task: str
        タスクの種類
    classes: list[str]
        分類のクラスの名前
    columns: Optional[list[str]] = None
        予測の表の列 (渡すと group_by の列があるかも確かめる)

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 形と知らない項目
    if not rules:
        return []
    if not isinstance(rules, dict):
        return ["結論の決め方は辞書にしてください"]
    errors = [
        f"知らない項目です: {key}" for key in rules if key not in RULE_KEYS
    ]
    classify = task == "classification"

    # 分類だけの決め方
    floor = rules.get("min_confidence")
    if floor is not None and (not is_number(floor) or not 0 <= floor <= 1):
        errors.append("min_confidence は 0 から 1 の数にしてください")
    positive, threshold = rules.get("positive"), rules.get("threshold")
    if (floor is not None or positive is not None) and not classify:
        errors.append("min_confidence と positive は分類にだけ使えます")
    if positive is not None and classify:
        if len(classes) != 2:
            errors.append("positive は 2 クラスの分類にだけ使えます")
        elif str(positive) not in classes:
            errors.append(
                f"positive のクラス {positive} がありません "
                f"(クラス: {', '.join(classes)})"
            )
        if not is_number(threshold) or not 0 <= threshold <= 1:
            errors.append("threshold は 0 から 1 の数にしてください")
    elif threshold is not None:
        errors.append("threshold は positive と一緒に書いてください")

    # 回帰と予測だけの値の帯 (上限の小さい順, 最後は上限無し)
    bands = rules.get("bands")
    if bands is not None:
        if classify:
            errors.append("bands は回帰と予測にだけ使えます")
        errors.extend(check_bands(bands))

    # まとめる列
    group = rules.get("group_by")
    if group is not None:
        if not isinstance(group, str) or not group:
            errors.append("group_by は列の名前にしてください")
        elif columns is not None and group not in columns:
            errors.append(f"group_by の列 {group} が予測の表にありません")

    return errors


def check_bands(bands: Any) -> list[str]:
    """
    値の帯の並びの誤りを一覧で返す関数

    Parameters
    ----------
    bands: Any
        帯の並び ({label, upper} の辞書の並び)

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 名前と上限, 上限の順
    if not isinstance(bands, list) or not bands:
        return ["bands は {label, upper} の並びにしてください"]
    errors = []
    previous = None
    labels = []
    for index, band in enumerate(bands):
        last = index == len(bands) - 1
        if not isinstance(band, dict) or not str(band.get("label", "")):
            errors.append(f"bands の {index + 1} つ目に label がありません")
            continue
        labels.append(str(band["label"]))
        upper = band.get("upper")
        if last:
            if upper is not None:
                errors.append("bands の最後は上限 (upper) 無しにしてください")
            continue
        if not is_number(upper):
            errors.append(f"bands の {band['label']} の upper が数ではありません")
        elif previous is not None and upper <= previous:
            errors.append("bands は upper の小さい順に並べてください")
        else:
            previous = upper
    if len(set(labels)) != len(labels):
        errors.append("bands の label が重なっています")

    return errors


def decide(
    prediction: pd.DataFrame, task: str, classes: list[str], rules: dict
) -> pd.Series:
    """
    行ごとの判断を決める関数

    Parameters
    ----------
    prediction: pd.DataFrame
        predict の表
    task: str
        タスクの種類
    classes: list[str]
        分類のクラスの名前
    rules: dict
        結論の決め方

    Returns
    ----------
    decision: pd.Series
        判断の列
    """
    # 予測しなかった行
    ok = prediction["status"] == "ok"
    decision = pd.Series(NO_PREDICTION, index=prediction.index, dtype=object)
    if task == "classification":
        chosen = prediction["prediction"].astype(object)
        positive = rules.get("positive")
        if positive is not None:
            other = next(c for c in classes if c != str(positive))
            hit = prediction[f"p:{positive}"].astype(float) >= float(
                rules["threshold"]
            )
            chosen = pd.Series(
                np.where(hit, str(positive), other), index=prediction.index
            )
            confidence = np.where(
                hit, prediction[f"p:{positive}"], prediction[f"p:{other}"]
            ).astype(float)
        else:
            confidence = prediction["confidence"].astype(float).to_numpy()
        floor = rules.get("min_confidence")
        if floor is not None:
            chosen = chosen.where(confidence >= float(floor), UNDECIDED)
        decision[ok] = chosen[ok]
        return decision

    # 回帰と予測は値の帯 (帯が無ければ空)
    bands = rules.get("bands")
    decision[ok] = ""
    if bands:
        values = prediction["prediction"].astype(float)
        labels = pd.Series(str(bands[-1]["label"]), index=prediction.index)
        for band in reversed(bands[:-1]):
            labels = labels.where(values >= band["upper"], str(band["label"]))
        decision[ok] = labels[ok]

    return decision


def summarize(
    frame: pd.DataFrame, task: str, rules: dict
) -> dict:
    """
    予測と判断のまとめを求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        decision を足した予測の表
    task: str
        タスクの種類
    rules: dict
        結論の決め方

    Returns
    ----------
    summary: dict
        件数, 判断ごとの件数と割合, 値の合計など (グループごとも)
    """
    # 全体と, まとめる列の値ごと
    summary = count_up(frame, task)
    group = rules.get("group_by")
    if group:
        summary["groups"] = [
            {"group": str(key), **count_up(part, task)}
            for key, part in frame.groupby(group, sort=True, dropna=False)
        ]

    return summary


def count_up(frame: pd.DataFrame, task: str) -> dict:
    """
    件数, 判断ごとの件数と割合, 値の要約を求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        decision を足した予測の表
    task: str
        タスクの種類

    Returns
    ----------
    counts: dict
        rows, predicted, skipped, decisions, (回帰と予測は values)
    """
    # 予測した行での判断の内訳 (多い順)
    ok = frame[frame["status"] == "ok"]
    counts: dict = {
        "rows": int(len(frame)),
        "predicted": int(len(ok)),
        "skipped": int(len(frame) - len(ok)),
    }
    decided = ok["decision"][ok["decision"] != ""]
    counts["decisions"] = [
        {"decision": str(name), "count": int(number),
         "share": float(number / len(ok)) if len(ok) else 0.0}
        for name, number in decided.value_counts(sort=True).items()
    ]
    if task != "classification":
        values = ok["prediction"].astype(float)
        counts["values"] = {
            "sum": float(values.sum()),
            "mean": float(values.mean()) if len(values) else None,
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }

    return counts


def number_text(value: float) -> str:
    """
    値を桁に合わせた短い文字にする関数

    Parameters
    ----------
    value: float
        値

    Returns
    ----------
    text: str
        3 桁ごとのカンマ付き (大きな値) か有効数字 4 桁の文字
    """
    # 大きな値は整数, 小さな値は有効数字
    text = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"

    return text


def sentences_of(summary: dict, task: str) -> list[str]:
    """
    まとめを日本語の文にする関数

    Parameters
    ----------
    summary: dict
        summarize のまとめ
    task: str
        タスクの種類

    Returns
    ----------
    sentences: list[str]
        結論の文
    """
    # 件数と, 判断の内訳
    head = f"{summary['rows']:,} 件のうち {summary['predicted']:,} 件を予測"
    if summary["skipped"]:
        head += f" ({summary['skipped']:,} 件は予測できず)"
    sentences = [head]
    if summary["decisions"]:
        sentences.append(", ".join(
            f"{item['decision']} {item['count']:,} 件 "
            f"({item['share'] * 100:.1f} %)"
            for item in summary["decisions"]
        ))
    values = summary.get("values")
    if task != "classification" and values and values["mean"] is not None:
        sentences.append(
            f"予測の合計 {number_text(values['sum'])}, 平均 "
            f"{number_text(values['mean'])} (最小 "
            f"{number_text(values['min'])}, 最大 "
            f"{number_text(values['max'])})"
        )

    # グループごと (多い場合は先頭の 10 個)
    for item in summary.get("groups", [])[:10]:
        if task == "classification":
            detail = ", ".join(
                f"{d['decision']} {d['count']:,} 件"
                for d in item["decisions"]
            )
        else:
            detail = (
                f"合計 {number_text(item['values']['sum'])}"
                if item["values"]["mean"] is not None else "予測なし"
            )
        sentences.append(f"{item['group']}: {detail or '予測なし'}")

    return sentences


def conclude(
    prediction: pd.DataFrame, task: str, classes: list[str],
    rules: Optional[dict] = None
) -> dict:
    """
    予測から行ごとの判断と全体の結論を出す関数

    Parameters
    ----------
    prediction: pd.DataFrame
        predict の表
    task: str
        タスクの種類
    classes: list[str]
        分類のクラスの名前
    rules: Optional[dict] = None
        結論の決め方 (min_confidence, positive, threshold, bands,
        group_by)

    Returns
    ----------
    conclusion: dict
        frame (decision を足した表), summary (まとめ), sentences (文),
        rules (使った決め方)

    Raises
    ----------
    ValueError
        決め方に誤りがある場合
    """
    # 決め方の確認
    rules = dict(rules or {})
    errors = check_rules(rules, task, classes, list(prediction.columns))
    if errors:
        raise ValueError("結論の決め方の誤り: " + " / ".join(errors))

    # 判断, まとめ, 文
    frame = prediction.copy()
    frame["decision"] = decide(frame, task, classes, rules)
    summary = summarize(frame, task, rules)
    conclusion = {
        "frame": frame,
        "summary": summary,
        "sentences": sentences_of(summary, task),
        "rules": rules,
    }

    return conclusion
