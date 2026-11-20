from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from ml_eda import STEPS, Data

# 今のやり方の決まりの種類 (ML-Quest の action.current と同じ名前). lag は
# k 期前の目的の値, mean は直前の k 期の目的の値の平均, column はデータの列
BASELINE_TYPES = ("lag", "mean", "column")

# 決まりに書ける項目 (type と periods か column は ML-Quest の形. 他は
# 求めるのに要る目的の列, 時刻の列, 系列を分ける列, 1 期の間隔)
BASELINE_KEYS = ("type", "periods", "column", "target", "time", "by",
                 "freq")

# 作る列の名前
BASELINE_NAME = "今のやり方"


def baseline_text(baseline: dict) -> str:
    """
    今のやり方の決まりを, 文に添える短い言葉にする関数

    Parameters
    ----------
    baseline: dict
        今のやり方の決まり

    Returns
    ----------
    text: str
        決まりの言葉 (例: 7 期前の値と同じ量)
    """
    # 種類ごとの言葉
    kind = baseline.get("type")
    if kind == "lag":
        text = f"{baseline.get('periods')} 期前の値と同じ量"
        return text
    if kind == "mean":
        text = f"直前の {baseline.get('periods')} 期の値の平均の量"
        return text
    text = f"列 {baseline.get('column')} の値"

    return text


def check_baseline(
    baseline: Any, columns: Optional[list[str]] = None
) -> list[str]:
    """
    今のやり方の決まりの誤りを一覧で返す関数

    Parameters
    ----------
    baseline: Any
        今のやり方の決まり
    columns: Optional[list[str]] = None
        データの列 (渡すと, 決まりの列があるかも確かめる)

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 種類と知らない項目
    if not isinstance(baseline, dict):
        return ["今のやり方の決まりは辞書にしてください"]
    kind = baseline.get("type")
    if kind not in BASELINE_TYPES:
        return [f"今のやり方の type は {', '.join(BASELINE_TYPES)} の"
                "どれかにしてください"]
    errors = [
        f"今のやり方に知らない項目です: {key}"
        for key in baseline if key not in BASELINE_KEYS
    ]

    # 列は名前, k 期前と平均は 1 以上の整数の期と, 時刻と間隔
    if kind == "column":
        column = baseline.get("column")
        if not isinstance(column, str) or not column:
            errors.append("今のやり方の column は列の名前にしてください")
        elif columns is not None and column not in columns:
            errors.append(f"今のやり方の列 {column} がデータにありません")
        return errors
    periods = baseline.get("periods")
    if isinstance(periods, bool) or not isinstance(
        periods, (int, np.integer)
    ) or periods < 1:
        errors.append("今のやり方の periods は 1 以上の整数にしてください")
    for name in ("target", "time", "freq"):
        value = baseline.get(name)
        if value is not None and (not isinstance(value, str) or not value):
            errors.append(f"今のやり方の {name} は文字にしてください")
    by = baseline.get("by")
    if by is not None and not isinstance(by, str) and not (
        isinstance(by, list) and all(isinstance(b, str) for b in by)
    ):
        errors.append("今のやり方の by は列の名前の並びにしてください")
    if isinstance(baseline.get("freq"), str):
        try:
            step = pd.to_timedelta(baseline["freq"])
        except ValueError:
            step = pd.NaT
        if pd.isna(step) or step <= pd.Timedelta(0):
            errors.append(
                f"今のやり方の freq を正の間隔として読めません: "
                f"{baseline['freq']} (86400s, 1D, 1h など)"
            )
    if columns is not None:
        wanted = [baseline.get("target"), baseline.get("time"),
                  *series_columns(baseline)]
        errors += [
            f"今のやり方の列 {name} がデータにありません"
            for name in wanted if isinstance(name, str) and name not in columns
        ]

    return errors


def series_columns(baseline: dict) -> list[str]:
    """
    決まりの系列を分ける列を並びにする関数

    Parameters
    ----------
    baseline: dict
        今のやり方の決まり

    Returns
    ----------
    by: list[str]
        系列を分ける列 (無ければ空)
    """
    # 1 つの名前も並びに
    by = baseline.get("by") or []
    by = [by] if isinstance(by, str) else list(by)

    return by


def interval_freq(times: pd.Series) -> Optional[str]:
    """
    時刻の列の 1 期の間隔を, 重ならない時刻の差の最頻値から求める関数

    ML-Quest がデータを入れたときに求める間隔と同じ決まり

    Parameters
    ----------
    times: pd.Series
        時刻の列

    Returns
    ----------
    freq: Optional[str]
        間隔 (秒の数の文字. 例: 86400s). 求められなければ None
    """
    # 重ならない時刻の差の最頻値
    parsed = pd.to_datetime(times, errors="coerce", format="mixed").dropna()
    steps = pd.Series(np.sort(parsed.unique())).diff().dropna()
    steps = steps[steps > pd.Timedelta(0)]
    if steps.empty:
        return None
    freq = f"{int(steps.mode().iloc[0].total_seconds())}s"

    return freq


def current_from_baseline(
    data: Union[pd.DataFrame, Data], baseline: dict,
    target: Optional[str] = None
) -> pd.Series:
    """
    今のやり方の決まりから, 行ごとの今のやり方の値の列を作る関数

    問題の設定 (ML-Quest の action.current) と同じ決まりで求める. k 期前
    と平均は ML-EDA の時刻でずらす手順 (lag の time, freq, match: exact)
    で求める

    Parameters
    ----------
    data: Union[pd.DataFrame, Data]
        データ (k 期前と平均は目的の列と時刻の列を持つ)
    baseline: dict
        今のやり方の決まり. type (lag, mean, column) と, periods (期の数)
        か column (列の名前). k 期前と平均は time (時刻の列), by (系列を
        分ける列. 省略は 1 つの系列), freq (1 期の間隔. 省略は時刻の差の
        最頻値), target (目的の列) も書ける
    target: Optional[str] = None
        目的の列 (決まりに target が無い場合に使う)

    Returns
    ----------
    values: pd.Series
        今のやり方の値 (データと同じ行の並びと index. 名前は 今のやり方).
        k 期前と平均は数で, 前の期の値が無い行は欠損. 列はその列の値の
        まま

    Raises
    ----------
    ValueError
        決まりの誤り, 列が無い場合, 間隔を求められない場合
    """
    # 決まりの確かめ (列はデータの列で)
    frame = data.frame if isinstance(data, Data) else data
    errors = check_baseline(baseline, [str(c) for c in frame.columns])
    if errors:
        raise ValueError(". ".join(errors))
    if baseline["type"] == "column":
        values = frame[baseline["column"]].rename(BASELINE_NAME)
        return values

    # k 期前と平均は目的の列, 時刻の列, 間隔が要る
    name = baseline.get("target") or target
    time = baseline.get("time")
    if not name or name not in frame.columns:
        raise ValueError(f"今のやり方を求める目的の列 {name} がデータに"
                         "ありません")
    if not time:
        raise ValueError(
            "k 期前と平均は時刻の列で数えるので, 今のやり方の time に時刻の"
            "列を書いてください (無ければ今のやり方の値の列で書く)"
        )
    freq = baseline.get("freq") or interval_freq(frame[time])
    if freq is None:
        raise ValueError(f"時刻の列 {time} から 1 期の間隔を求められません")

    # ML-EDA の時刻でずらす手順 (暦の k 期前. その時刻が無ければ欠損)
    k = int(baseline["periods"])
    numbers = frame.assign(**{name: pd.to_numeric(
        frame[name], errors="coerce"
    )})
    params: dict = {"columns": [name], "by": series_columns(baseline),
                    "time": time, "freq": freq, "match": "exact"}
    if baseline["type"] == "lag":
        params["lags"] = [k]
        column = f"{name}_lag{k}"
    else:
        params.update(lags=[], windows=[k])
        column = f"{name}_mean{k}"
    shifted = STEPS["lag"].apply(numbers, params, {})
    values = pd.to_numeric(shifted[column], errors="coerce").rename(
        BASELINE_NAME
    )

    return values
