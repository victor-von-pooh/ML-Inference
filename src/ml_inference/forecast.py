from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from ml_eda import Data

from ml_inference.bundle import Bundle, load_bundle
from ml_inference.predict import lag_settings, predict, read_source

# 先の行を予測できなかった理由
BLOCKED = "前の先の行を予測できないので, 過去の値が揃いません"


def target_values(frame: pd.DataFrame, target: str) -> pd.Series:
    """
    目的の列を数にする関数 (空は欠損, 数にできない値は誤り)

    Parameters
    ----------
    frame: pd.DataFrame
        渡したデータ
    target: str
        目的の列

    Returns
    ----------
    values: pd.Series
        数の目的の列

    Raises
    ----------
    ValueError
        数にできない値がある場合
    """
    # 空の文字も欠損にし, それ以外で数にできない値は読み込みの誤りとする
    raw = frame[target]
    blank = raw.isna() | (raw.astype(str).str.strip() == "")
    values = pd.to_numeric(raw.where(~blank), errors="coerce")
    bad = values.isna() & ~blank
    if bad.any():
        examples = ", ".join(map(str, raw[bad].unique()[:3]))
        raise ValueError(
            f"目的の列 {target} に数にできない値があります ({examples}). "
            "load_data で読んだ形にそろえてください"
        )

    return values


def time_order(frame: pd.DataFrame, settings: dict) -> np.ndarray:
    """
    時刻でずらす場合の, 時刻の順の行の並びを返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        渡したデータ
    settings: dict
        lag_settings の結果

    Returns
    ----------
    order: np.ndarray
        行の位置の並び (時刻でずらさない場合は元の並び)
    """
    # 文字の時刻も日時にしてから並べる (前処理の時刻のずらしと同じ読み方)
    times = [c for c in settings["time"] if c in frame.columns]
    if not times:
        return np.arange(len(frame))
    when = pd.to_datetime(frame[times[0]], errors="coerce", format="mixed")
    order = when.sort_values(kind="stable").index.to_numpy()

    return order


def series_keys(frame: pd.DataFrame, settings: dict) -> pd.Series:
    """
    行ごとの系列の名前を返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        渡したデータ
    settings: dict
        lag_settings の結果

    Returns
    ----------
    keys: pd.Series
        系列を分ける列の値をつないだ文字 (列が無ければ 1 つの系列)
    """
    # 系列を分ける列の値のまま (表記のゆれはそろえて渡す)
    by = [c for c in settings["by"] if c in frame.columns]
    keys = frame[by].astype(str).agg("\u0000".join, axis=1) if by else (
        pd.Series("", index=frame.index)
    )

    return keys


def future_rows(values: pd.Series, keys: pd.Series) -> np.ndarray:
    """
    系列ごとに, 最後の分かる値より後の空の行 (先の行) を返す関数

    Parameters
    ----------
    values: pd.Series
        目的の列
    keys: pd.Series
        系列の名前

    Returns
    ----------
    future: np.ndarray
        先の行の印 (過去の途中の空の行は含まない)
    """
    # 後ろから数えて, 分かる値がまだ出てこない行
    known = values.notna().astype(int)
    later = known[::-1].groupby(keys[::-1]).cumsum()[::-1]
    future = (values.isna() & (later == 0)).to_numpy()

    return future


def forecast(
    bundle: Union[str, Path, Bundle],
    source: Union[str, Path, pd.DataFrame, Data],
    options: Optional[dict] = None, batch_size: int = 256,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    目的の列が空の先の行を, 予測した値を次の入力に使いながら順に予測する
    関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ (目的の列の過去の値の列を作る一式)
    source: Union[str, Path, pd.DataFrame, Data]
        過去の行 (目的の列が分かる) と先の行 (目的の列が空) を持つデータ
    options: Optional[dict] = None
        読み込みの設定 (学習のときの設定に上書きする)
    batch_size: int = 256
        1 回に渡す件数
    device: str = "cpu"
        使うデバイス

    Returns
    ----------
    result: pd.DataFrame
        predict の表に, step (何段目の予測か. 先の行でなければ欠損) を
        足した表. 先の行の目的の列には予測した値が入る

    Raises
    ----------
    ValueError
        目的の列が無い場合, 数にできない場合, 分類の一式の場合
    """
    # 読み込みと, 数の目的の列
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    if bundle.task == "classification":
        raise ValueError("何段も先の予測は, 回帰と予測の一式だけです")
    data = read_source(bundle, source, options)
    target = bundle.dataset.info.get("target")
    if not target or target not in data.frame.columns:
        raise ValueError(
            f"目的の列 {target} が要ります (過去の行は値を, 先の行は空を書く)"
        )
    settings = lag_settings(bundle)
    order = time_order(data.frame.reset_index(drop=True), settings)
    frame = data.frame.reset_index(drop=True).iloc[order]
    frame = frame.reset_index(drop=True)
    frame[target] = target_values(frame, target)
    keys = series_keys(frame, settings)
    future = future_rows(frame[target], keys)

    # 予測と, 系列ごとに最も前の先の行を埋めることの繰り返し
    result, step = advance(bundle, frame, target, keys, future,
                           target in settings["columns"], batch_size, device)

    # 先の行の結果と, 元の並び
    result["step"] = step
    left = future & np.isnan(step)
    result.loc[left, "status"] = "skipped"
    result.loc[left, "prediction"] = np.nan
    # 系列で最初に止まった先の行は自分の理由, その後ろは止まったことが理由
    behind = pd.Series(left).groupby(keys).cumsum().to_numpy() > 1
    blocked = left & (behind | (result["reason"] == "").to_numpy())
    result.loc[blocked, "reason"] = BLOCKED
    result.index = order
    result = result.sort_index().reset_index(drop=True)

    return result


def advance(
    bundle: Bundle, frame: pd.DataFrame, target: str, keys: pd.Series,
    future: np.ndarray, recursive: bool, batch_size: int, device: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    先の行を系列ごとに 1 段ずつ予測して, 目的の列に入れる関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    frame: pd.DataFrame
        時刻の順の表 (目的の列を書き換える)
    target: str
        目的の列
    keys: pd.Series
        系列の名前
    future: np.ndarray
        先の行の印
    recursive: bool
        目的の列の過去の値を使う一式か (使わなければ 1 回で予測する)
    batch_size: int
        1 回に渡す件数
    device: str
        使うデバイス

    Returns
    ----------
    result: tuple[pd.DataFrame, np.ndarray]
        最後の予測の表と, 行ごとの何段目か
    """
    # 過去の値を使わない一式は 1 回の予測
    table = predict(bundle, frame, None, batch_size, device)
    step = np.full(len(frame), np.nan)
    if not recursive:
        step[future & (table["status"] == "ok").to_numpy()] = 1.0
        result = (table, step)
        return result

    # 系列ごとに最も前の, まだ埋めていない先の行 (止まった系列は除く)
    stuck = np.zeros(len(frame), dtype=bool)
    number = 0
    while True:
        open_rows = pd.Series(future & np.isnan(step) & ~stuck)
        first = (open_rows & (open_rows.groupby(keys).cumsum() == 1))
        first = first.to_numpy()
        values = pd.to_numeric(table["prediction"], errors="coerce")
        good = (table["status"] == "ok").to_numpy() & np.isfinite(values)
        fill = first & good
        stuck |= first & ~good
        if not fill.any():
            break
        number += 1
        frame.loc[fill, target] = values[fill].to_numpy()
        step[fill] = number
        table = predict(bundle, frame, None, batch_size, device)

    # 埋めた行の予測は埋めた値, 数でない予測の行は理由付き
    filled = ~np.isnan(step)
    table.loc[filled, "prediction"] = frame.loc[filled, target].to_numpy()
    odd = stuck & (table["status"] == "ok").to_numpy()
    table.loc[odd, "status"] = "skipped"
    table.loc[odd, "reason"] = "予測の値が数ではありません (入力に無限大などが無いか)"
    result = (table, step)

    return result
