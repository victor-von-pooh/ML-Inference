from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from ml_eda import Data

from ml_inference.bundle import Bundle, load_bundle
from ml_inference.predict import lag_settings, predict, read_source


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
        predict の表に, step (何段目の予測か. 過去の行は欠損) を足した表.
        先の行の目的の列には予測した値が入る

    Raises
    ----------
    ValueError
        目的の列が無いデータの場合
    """
    # 読み込みと目的の列
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    data = read_source(bundle, source, options)
    frame = data.frame.reset_index(drop=True).copy()
    target = bundle.dataset.info.get("target")
    if not target or target not in frame.columns:
        raise ValueError(
            f"目的の列 {target} が要ります (過去の行は値を, 先の行は空を書く)"
        )
    settings = lag_settings(bundle)
    if target not in settings["columns"]:
        # 過去の値を使わない一式は, 先の行も 1 回で予測できる
        result = predict(bundle, frame, None, batch_size, device)
        result["step"] = np.where(
            frame[target].isna() & (result["status"] == "ok"), 1.0, np.nan
        )
        return result

    # 時刻でずらす場合は時刻の順に並べて進め, 最後に元の並びに戻す
    times = [c for c in settings["time"] if c in frame.columns]
    order = np.arange(len(frame))
    if times:
        order = frame.sort_values(times[0], kind="stable").index.to_numpy()
        frame = frame.iloc[order].reset_index(drop=True)
    by = [c for c in settings["by"] if c in frame.columns]
    keys = frame[by].astype(str).agg("\u0000".join, axis=1) if by else (
        pd.Series("", index=frame.index)
    )
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    future = frame[target].isna().to_numpy()
    step = np.full(len(frame), np.nan)

    # 系列ごとに最も前の空の行を予測して埋めることの繰り返し
    number = 0
    result = predict(bundle, frame, None, batch_size, device)
    while True:
        pending = frame[target].isna()
        first = pending & (pending.groupby(keys).cumsum() == 1)
        fill = (first & (result["status"] == "ok")).to_numpy()
        if not fill.any():
            break
        number += 1
        values = pd.to_numeric(result["prediction"], errors="coerce")
        frame.loc[fill, target] = values[fill].to_numpy()
        step[fill] = number
        result = predict(bundle, frame, None, batch_size, device)

    # 先の行の予測は埋めた値, 埋められなかった行は理由付き
    result["step"] = step
    filled = ~np.isnan(step)
    result.loc[filled, "prediction"] = frame.loc[filled, target].to_numpy()
    left = future & ~filled
    result.loc[left, "status"] = "skipped"
    result.loc[left, "reason"] = result.loc[left, "reason"].where(
        result.loc[left, "reason"] != "",
        "前の先の行を予測できないので, 過去の値が揃わない"
    )
    result.loc[left, "prediction"] = np.nan
    if times:
        result.index = order
        result = result.sort_index().reset_index(drop=True)

    return result
