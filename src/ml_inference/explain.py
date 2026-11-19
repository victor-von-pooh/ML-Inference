from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from ml_eda import Data

from ml_inference.bundle import Bundle, load_bundle
from ml_inference.predict import (
    TABLE_KINDS, lag_settings, predict, read_source, series_keys
)

# 系列ごとに選ぶときの, 系列ごとの行の数の下限
MIN_SERIES_ROWS = 30


def sample_rows(
    frame: pd.DataFrame, max_rows: int, settings: dict,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    列の効きを求めるのに使う行を選ぶ関数

    Parameters
    ----------
    frame: pd.DataFrame
        渡したデータ
    max_rows: int
        行の上限
    settings: dict
        lag_settings の結果
    rng: np.random.Generator
        乱数

    Returns
    ----------
    chosen: pd.DataFrame
        選んだ行 (元の並び)
    """
    # 過去の値の列が無ければ行を無作為に選ぶ
    if len(frame) <= max_rows:
        return frame
    if not settings["columns"]:
        rows = np.sort(rng.choice(len(frame), max_rows, replace=False))
        chosen = frame.iloc[rows].reset_index(drop=True)
        return chosen

    # 過去の値の列は前の行が要るので, 系列ごとに最後の続いた行を選ぶ (全て
    # の系列を残す. 系列を分ける列が無ければ 1 つの系列)
    by = [c for c in settings["by"] if c in frame.columns]
    if not by:
        chosen = frame.iloc[-max_rows:].reset_index(drop=True)
        return chosen
    keys = series_keys(frame, settings)
    count = max(max_rows // max(keys.nunique(), 1), MIN_SERIES_ROWS)
    tail = keys.groupby(keys).cumcount(ascending=False) < count
    chosen = frame[tail.to_numpy()].reset_index(drop=True)

    return chosen


def measure_of(
    bundle: Bundle, table: pd.DataFrame, chosen: np.ndarray
) -> np.ndarray:
    """
    予測の表から, 並べ替える前に予測したクラスの確率か予測の値を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    table: pd.DataFrame
        predict の表
    chosen: np.ndarray
        並べ替える前に予測したクラス (分類)

    Returns
    ----------
    values: np.ndarray
        行ごとの値 (予測しなかった行は欠損)
    """
    # 回帰と予測は値, 分類は行ごとに並べ替える前に予測したクラスの確率
    if bundle.task != "classification":
        values = pd.to_numeric(
            table["prediction"], errors="coerce"
        ).to_numpy(dtype=float)
        return values
    values = np.full(len(table), np.nan)
    for name in bundle.classes:
        hit = chosen == name
        values[hit] = pd.to_numeric(
            table[f"p:{name}"], errors="coerce"
        ).to_numpy(dtype=float)[hit]

    return values


def shuffled(
    frame: pd.DataFrame, column: str, target: Optional[str],
    series: bool, rng: np.random.Generator
) -> pd.DataFrame:
    """
    列を並べ替えた表を返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        渡したデータ
    column: str
        並べ替える列
    target: Optional[str]
        目的の列
    series: bool
        系列を分ける列か (系列の名前だけを入れ替え, 系列の中の前後関係は
        残す)
    rng: np.random.Generator
        乱数

    Returns
    ----------
    result: pd.DataFrame
        並べ替えた表
    """
    # 系列を分ける列は名前の入れ替え
    result = frame.copy()
    values = frame[column]
    if series:
        names = values.dropna().unique()
        mapping = dict(zip(names, rng.permutation(names)))
        result[column] = values.map(lambda value: mapping.get(value, value))
        return result

    # 目的の列は分かる行の中で並べ替え (先の予測の行の欠損は動かさない)
    rows = np.flatnonzero(values.notna().to_numpy()) if column == target else (
        np.arange(len(frame))
    )
    column_values = values.to_numpy(copy=True)
    column_values[rows] = column_values[rng.permutation(rows)]
    result[column] = column_values

    return result


def column_effect(
    bundle: Bundle, frame: pd.DataFrame, column: str, reference: np.ndarray,
    chosen: np.ndarray, repeats: int, series: bool,
    rng: np.random.Generator, batch_size: int, device: str
) -> dict:
    """
    1 つの列を並べ替えて推論し直し, 予測の変わり方を求める関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    frame: pd.DataFrame
        渡したデータ
    column: str
        並べ替える列
    reference: np.ndarray
        並べ替える前の値 (予測しなかった行は欠損)
    chosen: np.ndarray
        並べ替える前に予測したクラス
    repeats: int
        並べ替える回数
    series: bool
        系列を分ける列か
    rng: np.random.Generator
        乱数
    batch_size: int
        1 回に渡す件数
    device: str
        使うデバイス

    Returns
    ----------
    effect: dict
        effect (変わり方の平均), spread (並べ替えの乱数による揺れ. 回数の
        標準偏差), rows (平均に使った行の数の平均), lost (並べ替えた後に
        予測できなくなった行の数の平均), error (推論できなかった理由)
    """
    # 予測できた行での変わり方と, 予測できなくなった行の数
    target = bundle.dataset.info.get("target")
    known = ~np.isnan(reference)
    effects, used, lost = [], [], []
    for _ in range(repeats):
        table = shuffled(frame, column, target, series, rng)
        try:
            changed = measure_of(
                bundle, predict(bundle, table, None, batch_size, device),
                chosen
            )
        except ValueError as error:
            return {"effect": None, "spread": None, "rows": 0.0,
                    "lost": float(known.sum()), "error": str(error)}
        both = known & ~np.isnan(changed)
        used.append(int(both.sum()))
        lost.append(int((known & np.isnan(changed)).sum()))
        if both.any():
            effects.append(
                float(np.mean(np.abs(changed[both] - reference[both])))
            )
    effect = {
        "effect": float(np.mean(effects)) if effects else None,
        "spread": float(np.std(effects, ddof=1)) if len(effects) > 1
        else None,
        "rows": float(np.mean(used)),
        "lost": float(np.mean(lost)),
        "error": None,
    }

    return effect


def explain(
    bundle: Union[str, Path, Bundle],
    source: Union[str, Path, pd.DataFrame, Data],
    options: Optional[dict] = None, repeats: int = 3,
    max_rows: int = 2000, seed: int = 0, batch_size: int = 256,
    device: str = "cpu"
) -> dict:
    """
    渡したデータの元の列ごとに, 予測を動かす大きさ (並べ替えの重要度) を
    求める関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ
    source: Union[str, Path, pd.DataFrame, Data]
        推論するデータ (表か時系列)
    options: Optional[dict] = None
        読み込みの設定 (学習のときの設定に上書きする)
    repeats: int = 3
        列ごとに並べ替える回数
    max_rows: int = 2000
        求めるのに使う行の上限 (多い場合は選ぶ. 過去の値の列を作る
        レシピは系列ごとに選ぶ)
    seed: int = 0
        乱数のシード
    batch_size: int = 256
        1 回に渡す件数
    device: str = "cpu"
        使うデバイス

    Returns
    ----------
    result: dict
        measure (分類は probability, 回帰と予測は value), rows (使った
        行の数), columns (列ごとの column, kind, effect, spread, rows,
        lost, error の並び. 並べ替えると推論できない列が先頭, 次に効きの
        大きい順, 求めない時刻の列 (kind が time) は最後)

    Raises
    ----------
    ValueError
        表と時系列でない一式の場合と, 予測できる行が無い場合
    """
    # 表と時系列の元の列だけ
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    bundle.require_column("列の効き")
    if bundle.kind not in TABLE_KINDS:
        raise ValueError(
            f"{bundle.kind} の一式は, 1 件の中身が列ではないので列の効きを"
            "求めません"
        )
    rng = np.random.default_rng(seed)
    settings = lag_settings(bundle)
    data = read_source(bundle, source, options)
    frame = sample_rows(
        data.frame.reset_index(drop=True), max_rows, settings, rng
    )

    # 並べ替える前の予測
    base = predict(bundle, frame, None, batch_size, device)
    if not (base["status"] == "ok").any():
        raise ValueError("予測できる行が無いので, 列の効きを求められません")
    chosen = base["prediction"].to_numpy()
    reference = measure_of(bundle, base, chosen)

    # 列ごとの効き. 目的の列は, 過去の値の列が読む場合だけ「過去の値」.
    # 系列を分ける列は名前の入れ替え, 時刻でずらす時刻の列は求めない
    target = bundle.dataset.info.get("target")
    columns: list[dict[str, Any]] = []
    for column in frame.columns:
        if column == target and target not in settings["columns"]:
            continue
        kind = "past" if column == target else (
            "series" if column in settings["by"] else (
                "time" if column in settings["time"] else "column"
            )
        )
        name = f"{column} (過去の値)" if kind == "past" else str(column)
        if kind == "time":
            columns.append({
                "column": name, "kind": kind, "effect": None,
                "spread": None, "rows": 0.0, "lost": 0.0,
                "error": "過去の値の前後関係を決める時刻の列は並べ替えない",
            })
            continue
        effect = column_effect(
            bundle, frame, column, reference, chosen, repeats,
            kind == "series", rng, batch_size, device
        )
        columns.append({"column": name, "kind": kind, **effect})

    # 並べ替えると推論できない列を先頭 (最も頼っている), 残りは効きの大きい
    # 順, 求めない時刻の列は最後
    columns.sort(key=lambda item: (
        2 if item["kind"] == "time" else 1 if item["effect"] is not None
        else 0,
        -(item["effect"] or 0.0)
    ))
    result = {
        "measure": "probability" if bundle.task == "classification"
        else "value",
        "rows": int((base["status"] == "ok").sum()),
        "columns": columns,
    }

    return result
