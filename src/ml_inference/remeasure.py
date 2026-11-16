import datetime as dt
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import yaml
from ml_eda import Data

from ml_inference.bundle import Bundle, load_bundle
from ml_inference.decision import label_text
from ml_inference.predict import predict
from ml_inference.uncertainty import (
    class_calibration, interval, residual_quantiles
)

# 測り直すのに要る行の数の下限
REMEASURE_MIN_ROWS = 20


def counted_rows(
    bundle: Bundle, table: pd.DataFrame, count: Optional[Any]
) -> np.ndarray:
    """
    測る行 (予測でき, 目的の値が分かり, 数える印のある行) を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    table: pd.DataFrame
        predict の表
    count: Optional[Any]
        数える行の印 (省略は全ての行)

    Returns
    ----------
    keep: np.ndarray
        行ごとの真偽値

    Raises
    ----------
    ValueError
        目的の列が無い場合と, 測る行が少ない場合
    """
    # 目的の列と, backtest と同じ行の選び方
    target = bundle.dataset.info.get("target")
    if target not in table.columns:
        raise ValueError(f"測り直すには目的の列 {target} が要ります")
    keep = np.array(
        (table["status"] == "ok") & table[target].notna(), dtype=bool
    )
    if count is not None:
        keep &= np.asarray(count, dtype=bool)
    if keep.sum() < REMEASURE_MIN_ROWS:
        raise ValueError(
            f"測り直せる行 (予測し, 目的の値が分かる行) が {int(keep.sum())}"
            f" 行です. {REMEASURE_MIN_ROWS} 行以上要ります"
        )

    return keep


def regression_remeasure(
    old: dict, prediction: np.ndarray, answers: np.ndarray
) -> tuple[dict, dict]:
    """
    回帰の外れ方を測り直し, 前の記録の幅が覆った割合を求める関数

    Parameters
    ----------
    old: dict
        前の uncertainty (無ければ空)
    prediction: np.ndarray
        予測
    answers: np.ndarray
        実際の値

    Returns
    ----------
    result: tuple[dict, dict]
        前の記録の当たり具合 (coverage. 記録が無ければ空) と, 新しい
        誤りの分位点
    """
    # 前の記録の 80 % と 90 % の幅 (比の形で正でない予測は覆わない扱い)
    before: dict = {}
    if old.get("kind") and old.get(old["kind"]):
        before["coverage"] = {}
        for name, (low, high) in (("80", (10, 90)), ("90", (5, 95))):
            lower, upper = interval(prediction, old, low, high)
            inside = (answers >= lower) & (answers <= upper)
            before["coverage"][name] = float(inside.mean())
    quantiles = residual_quantiles(prediction, answers)
    result = (before, quantiles)

    return result


def class_remeasure(
    bundle: Bundle, old: dict, table: pd.DataFrame, target: str
) -> tuple[dict, dict]:
    """
    分類の確率の当たり具合を測り直す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    old: dict
        前の uncertainty (無ければ空)
    table: pd.DataFrame
        測る行の predict の表
    target: str
        目的の列

    Returns
    ----------
    result: tuple[dict, dict]
        前の記録のずれ (ece) と, 新しい calibration と calibration_of

    Raises
    ----------
    ValueError
        一式のクラスに無いラベルしか無い場合
    """
    # ラベルをクラスの番号に (一式に無いラベルの行は除く)
    index = {name: number for number, name in enumerate(bundle.classes)}
    labels = label_text(table[target]).map(index)
    known = labels.notna().to_numpy()
    if known.sum() < REMEASURE_MIN_ROWS:
        raise ValueError(
            "目的の列のラベルが一式のクラス ("
            + ", ".join(bundle.classes) + ") と合う行が少なすぎます"
        )
    part = table[known]
    columns = {name: part[name].to_numpy() for name in part.columns
               if name in ("prediction", "confidence")
               or str(name).startswith("p:")}
    calibration, of = class_calibration(
        bundle, columns, labels[known].to_numpy(dtype=int)
    )
    recorded = old.get("test_ece", (old.get("calibration") or {}).get("ece"))
    before = {} if recorded is None else {"ece": float(recorded)}
    result = (before, {"calibration": calibration, "calibration_of": of})

    return result


def remeasure_sentences(
    bundle: Bundle, rows: int, before: dict, uncertainty: dict
) -> list[str]:
    """
    測り直しの結果を日本語の文にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    rows: int
        測った行の数
    before: dict
        前の記録の当たり具合
    uncertainty: dict
        新しい uncertainty

    Returns
    ----------
    sentences: list[str]
        文
    """
    # 前の記録の当たり具合と, 新しい外れ方
    sentences = []
    if bundle.task != "classification":
        coverage = before.get("coverage")
        if coverage:
            low = coverage["80"] < 0.75
            sentences.append(
                f"今の記録の 80 % の幅は, 渡した実績の {rows:,} 行の "
                f"{coverage['80'] * 100:.0f} % を覆った (90 % の幅は "
                f"{coverage['90'] * 100:.0f} %)"
                + (". 外れ方を小さく見積もっていた" if low else "")
            )
        kind = uncertainty["kind"]
        lower, upper = uncertainty[kind][9], uncertainty[kind][89]
        width = (
            f"予測の {np.exp(lower):.2f} 倍から {np.exp(upper):.2f} 倍"
            if kind == "relative"
            else f"予測に {lower:+,.4g} から {upper:+,.4g}"
        )
        sentences.append(
            f"測り直した外れ方の 80 % の幅は, {width} ({rows:,} 行で測った)"
        )
        return sentences
    ece = uncertainty["calibration"]["ece"]
    recorded = before.get("ece")
    sentences.append(
        f"渡した実績の {rows:,} 行で, 確率と実際に当たった割合のずれは平均"
        f" {ece * 100:.1f} ポイント"
        + ("" if recorded is None
           else f" (今の記録は {recorded * 100:.1f} ポイント)")
    )

    return sentences


def write_uncertainty(bundle: Bundle, uncertainty: dict) -> None:
    """
    一式の inference.yaml の uncertainty を書き換える関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式 (info も書き換える)
    uncertainty: dict
        新しい uncertainty

    Returns
    ----------
    None
    """
    # 一時のファイルに書いてから置き換える (途中で止まっても壊さない)
    path = bundle.path / "inference.yaml"
    info = yaml.safe_load(path.read_text(encoding="utf-8"))
    info["uncertainty"] = uncertainty
    temporary = path.with_name("inference.yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )
    temporary.replace(path)
    bundle.info["uncertainty"] = uncertainty


def remeasure(
    bundle: Union[str, Path, Bundle],
    source: Union[str, Path, pd.DataFrame, Data],
    options: Optional[dict] = None, count: Optional[Any] = None,
    write: bool = True
) -> dict:
    """
    目的の列が分かる新しい実績で, 一式の予測の外れ方を測り直す関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ
    source: Union[str, Path, pd.DataFrame, Data]
        目的の列を持つ新しい実績
    options: Optional[dict] = None
        読み込みの設定
    count: Optional[Any] = None
        測る行の印 (行ごとの真偽値. 過去の値の列を作るために前の行も
        渡し, 新しい期間だけで測る場合など. 省略は全ての行)
    write: bool = True
        一式に書き戻すか

    Returns
    ----------
    result: dict
        rows (測った行), before (今の記録の当たり具合. 回帰は coverage,
        分類は ece), uncertainty (新しい記録), sentences, written

    Raises
    ----------
    ValueError
        目的の列が無い場合と, 測れる行が少ない場合
    """
    # 予測と, 測る行
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    table = predict(bundle, source, options)
    keep = counted_rows(bundle, table, count)
    target = str(bundle.dataset.info.get("target"))
    old = dict(bundle.info.get("uncertainty") or {})
    part = table[keep]

    # 回帰は誤りの分位点, 分類は確率の当たり具合
    uncertainty: dict[str, Any] = {"split": "recent"}
    if bundle.task != "classification":
        prediction = pd.to_numeric(part["prediction"], errors="coerce")
        answers = pd.to_numeric(part[target], errors="coerce")
        finite = np.isfinite(prediction) & np.isfinite(answers)
        if finite.sum() < REMEASURE_MIN_ROWS:
            raise ValueError("数の予測と目的の値がそろった行が少なすぎます")
        before, measured = regression_remeasure(
            old, prediction[finite].to_numpy(float),
            answers[finite].to_numpy(float)
        )
        rows = int(finite.sum())
    else:
        before, measured = class_remeasure(bundle, old, part, target)
        rows = int(sum(b["rows"] for b in measured["calibration"]["bins"]))
    uncertainty.update({"rows": rows, **measured})
    uncertainty["measured_at"] = dt.datetime.now().astimezone().isoformat()
    uncertainty["previous"] = {
        key: old.get(key) for key in ("split", "rows", "kind")
        if old.get(key) is not None
    }

    # 文と, 書き戻し
    sentences = remeasure_sentences(bundle, rows, before, uncertainty)
    if write:
        write_uncertainty(bundle, uncertainty)
    result = {
        "rows": rows,
        "before": before,
        "uncertainty": uncertainty,
        "sentences": sentences,
        "written": bool(write),
    }

    return result
