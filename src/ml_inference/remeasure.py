import datetime as dt
from pathlib import Path
import uuid
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import yaml
from ml_eda import Data

from ml_inference.bundle import Bundle, load_bundle
from ml_inference.decision import amount_text, group_key, label_text
from ml_inference.forecast import forecast
from ml_inference.predict import lag_settings, predict, read_source
from ml_inference.uncertainty import (
    adjusted_levels, class_calibration, interval, residual_quantiles,
    residual_values
)

# 測り直すのに要る行の数の下限
REMEASURE_MIN_ROWS = 20

# 何段も先の外れ方で, 予測を始める時刻の数の上限 (新しい方から)
MAX_ORIGINS = 30


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
        目的の列が無い場合, 印の数が行の数と違う場合, 測る行が少ない
        場合
    """
    # 目的の列と, backtest と同じ行の選び方
    target = bundle.dataset.info.get("target")
    if target not in table.columns:
        raise ValueError(f"測り直すには目的の列 {target} が要ります")
    keep = np.array(
        (table["status"] == "ok") & table[target].notna(), dtype=bool
    )
    if count is not None:
        mark = np.asarray(count, dtype=bool)
        if mark.shape != keep.shape:
            raise ValueError(
                f"count の印の数 ({mark.size:,}) が行の数 ({len(keep):,}) "
                "と違います"
            )
        keep &= mark
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
        前の記録の当たり具合 (coverage と, 比の形で幅を出せずに除いた
        行の数 excluded. 記録が無ければ空) と, 新しい誤りの分位点
    """
    # 前の記録の 80 % と 90 % の幅 (比の形は正の予測の行だけ)
    before: dict = {}
    if old.get("kind") and old.get(old["kind"]):
        usable = prediction > 0 if old["kind"] == "relative" else (
            np.ones(len(prediction), dtype=bool)
        )
        before["excluded"] = int((~usable).sum())
        if usable.any():
            before["coverage"] = {}
            for name, (low, high) in (("80", (10, 90)), ("90", (5, 95))):
                lower, upper = interval(prediction[usable], old, low, high)
                inside = (answers[usable] >= lower) & (
                    answers[usable] <= upper
                )
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
        前の記録のずれ (ece. 記録したときの値で, 渡した実績で測った
        値ではない) と一式のクラスに無いラベルで除いた行の数 (dropped),
        新しい calibration と calibration_of

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
    before: dict = {"dropped": int((~known).sum())}
    if recorded is not None:
        before["ece"] = float(recorded)
    result = (before, {"calibration": calibration, "calibration_of": of})

    return result


def measure_groups(
    part: pd.DataFrame, column: str, prediction: np.ndarray,
    answers: np.ndarray, kind: str
) -> dict:
    """
    回帰の外れ方を, 分ける列の値 (行の種類) ごとに測る関数

    Parameters
    ----------
    part: pd.DataFrame
        測る行の予測の表 (予測と目的の値がそろった行)
    column: str
        分ける列
    prediction: np.ndarray
        予測
    answers: np.ndarray
        実際の値
    kind: str
        全ての行で選んだ外れ方の形 (種類ごとも同じ形)

    Returns
    ----------
    groups: dict
        column, values (種類ごとの誤りの 1 % から 99 % の分位点), rows
        (種類ごとの行の数), few (行が少なく測らなかった種類の数)

    Raises
    ----------
    ValueError
        測れる種類が無い場合
    """
    # 20 行以上の種類だけ測る (少ない種類は全ての行の外れ方)
    keys = pd.Series([group_key(v) for v in part[column].tolist()])
    values: dict[str, list[float]] = {}
    rows: dict[str, int] = {}
    few = 0
    for key in sorted(k for k in keys.dropna().unique()):
        inside = (keys == key).to_numpy()
        if inside.sum() < REMEASURE_MIN_ROWS:
            few += 1
            continue
        values[key] = residual_values(
            prediction[inside], answers[inside], kind
        )
        rows[key] = int(inside.sum())
    if not values:
        raise ValueError(
            f"{column} の種類に {REMEASURE_MIN_ROWS} 行以上のものが無いので,"
            " 種類ごとの外れ方を測れません"
        )
    groups = {"column": column, "values": values, "rows": rows, "few": few}

    return groups


def width_text(values: list[float], kind: str) -> str:
    """
    誤りの分位点の 80 % の幅を日本語にする関数

    Parameters
    ----------
    values: list[float]
        誤りの 1 % から 99 % の分位点
    kind: str
        relative か absolute

    Returns
    ----------
    text: str
        比は「a 倍から b 倍」, 差は「-a から +b」
    """
    # 10 % と 90 % の点
    lower, upper = values[9], values[89]
    if kind == "relative":
        text = f"{np.exp(lower):.2f} 倍から {np.exp(upper):.2f} 倍"
    else:
        text = (
            f"{'+' if lower >= 0 else '-'}{amount_text(abs(lower))} から "
            f"{'+' if upper >= 0 else '-'}{amount_text(abs(upper))}"
        )

    return text


def group_sentence(groups: dict, kind: str) -> str:
    """
    種類ごとの外れ方を日本語の文にする関数

    Parameters
    ----------
    groups: dict
        measure_groups の結果
    kind: str
        外れ方の形

    Returns
    ----------
    sentence: str
        文 (80 % の幅の最も狭い種類と広い種類)
    """
    # 80 % の幅 (10 % と 90 % の点の離れ) で並べる
    widths = {
        key: values[89] - values[9] for key, values in groups["values"].items()
    }
    narrow = min(widths, key=widths.get)
    wide = max(widths, key=widths.get)
    sentence = (
        f"{groups['column']} ごとの外れ方を {len(widths)} 種類で測った ("
        f"80 % の幅の最も狭い {narrow} は "
        f"{width_text(groups['values'][narrow], kind)}, 最も広い {wide} は "
        f"{width_text(groups['values'][wide], kind)}"
        + (f". {REMEASURE_MIN_ROWS} 行より少ない {groups['few']} 種類は全ての"
           "行の外れ方" if groups["few"] else "")
        + ")"
    )

    return sentence


def history_length(bundle: Bundle) -> int:
    """
    過去の値の列を作るのに要る, 予測を始める時刻より前の時刻の数を返す
    関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    count: int
        過去の値の手順の最も長いずらしと窓 (と移動平均の始まりの gap) の
        和の 2 倍と 2 (日の抜けなどへの余裕)
    """
    # lag の手順の lags と windows の最大 (移動平均を gap つ前から始める
    # 手順は, その分だけ長く)
    longest = 1
    for spec in bundle.dataset.recipe.get("steps", []):
        if spec["op"] != "lag":
            continue
        lags = [int(v) for v in spec.get("lags") or [1]]
        windows = [int(v) for v in spec.get("windows") or [0]]
        gap = int(spec.get("gap", 1))
        longest = max(longest, max(lags) + max(windows) + gap - 1)
    count = 2 * longest + 2

    return count


def step_errors(
    bundle: Bundle, frame: pd.DataFrame, keep: np.ndarray, horizon: int,
    time: str
) -> pd.DataFrame:
    """
    予測を始める時刻をずらしながら何段も先まで予測し, 段ごとの予測と実際
    の値を集める関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式 (目的の列の過去の値を使う回帰と予測の一式)
    frame: pd.DataFrame
        読み込んだ実績 (目的の列が分かる)
    keep: np.ndarray
        測る行の印 (予測を始める時刻と, 誤りを数える行はこの中だけ)
    horizon: int
        何段先まで
    time: str
        時刻の列

    Returns
    ----------
    errors: pd.DataFrame
        origin (予測を始めた時刻), step, prediction, actual
    """
    # 予測を始める時刻は, 測る行の時刻を horizon おきに (新しい方から
    # MAX_ORIGINS まで)
    target = str(bundle.dataset.info.get("target"))
    when = pd.to_datetime(frame[time], errors="coerce", format="mixed")
    times = np.sort(when[keep].dropna().unique())
    origins = times[::horizon][-MAX_ORIGINS:]
    every = np.sort(when.dropna().unique())
    history = history_length(bundle)
    parts = []
    for origin in origins:
        # 始めの時刻から horizon 個の時刻の目的の値は空にし, それより後の
        # 行と, 過去の値に要らない前の行は除く (計算を軽くするため)
        later = every[every >= origin]
        end = later[min(horizon, len(later)) - 1]
        earlier = every[every < origin]
        first = earlier[-history] if len(earlier) >= history else every[0]
        part = frame[((when <= end) & (when >= first)).to_numpy()].copy()
        inside = (when[part.index] >= origin).to_numpy()
        actual = pd.to_numeric(part[target], errors="coerce").to_numpy(
            dtype=float, copy=True
        )
        part.loc[inside, target] = np.nan
        result = forecast(bundle, part.reset_index(drop=True))
        step = result["step"].to_numpy(dtype=float)
        counted = inside & keep[part.index] & ~np.isnan(step) & np.isfinite(
            actual
        )
        parts.append(pd.DataFrame({
            "origin": origin,
            "step": step[counted].astype(int),
            "prediction": pd.to_numeric(
                result["prediction"], errors="coerce"
            ).to_numpy(dtype=float)[counted],
            "actual": actual[counted],
        }))
    errors = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["origin", "step", "prediction", "actual"]
    )

    return errors


def step_time(
    bundle: Bundle, table: pd.DataFrame, horizon: Any, time: Optional[str]
) -> str:
    """
    何段も先の外れ方を測れるかを確かめ, 時刻の列を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    table: pd.DataFrame
        predict の表
    horizon: Any
        段の数
    time: Optional[str]
        時刻の列 (省略は前処理の時刻でずらす列)

    Returns
    ----------
    column: str
        時刻の列

    Raises
    ----------
    ValueError
        分類の一式, 目的の列の過去の値を使わない一式, 段の数が 2 より
        小さい場合, 時刻の列が無い場合と日時として読めない場合
    """
    # 何段も先まで予測できる一式と, 時刻の列
    settings = lag_settings(bundle)
    target = bundle.dataset.info.get("target")
    if bundle.task == "classification" or target not in settings["columns"]:
        raise ValueError("何段も先の外れ方は, 目的の列の過去の値を使う回帰"
                         "と予測の一式だけで測れます")
    if not isinstance(horizon, (int, np.integer)) or horizon < 2:
        raise ValueError("horizon は 2 以上の整数にしてください")
    times = sorted(c for c in settings["time"] if c in table.columns)
    column = time or (times[0] if times else None)
    if column is None or column not in table.columns:
        raise ValueError("予測を始める時刻の列 (time) を指定してください")
    # 整数の列 (期の番号) は順番として認め, 他の数の列と, 日時として
    # 読めない値の多い列の除外
    values = table[column]
    if pd.api.types.is_integer_dtype(values) and not (
        pd.api.types.is_bool_dtype(values)
    ):
        return str(column)
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if pd.api.types.is_numeric_dtype(values) or parsed.notna().mean() < 0.5:
        raise ValueError(f"列 {column} は日時として読めません. 予測を始める"
                         "時刻の列を選んでください")

    return str(column)


def step_quantiles(errors: pd.DataFrame, kind: str) -> dict:
    """
    段ごとの誤りの分位点を求める関数

    Parameters
    ----------
    errors: pd.DataFrame
        step_errors の結果
    kind: str
        relative か absolute (1 段目の記録と同じ形)

    Returns
    ----------
    steps: dict
        段 (文字) ごとの, 1 % から 99 % の分位点 (誤りが 20 より少ない段は
        除く) と, rows (段ごとの誤りの数)
    """
    # 比の形は, 予測と実際の値が正の行だけ
    usable = errors
    if kind == "relative":
        usable = errors[(errors["prediction"] > 0) & (errors["actual"] > 0)]
        gap = np.log(usable["actual"] / usable["prediction"])
    else:
        gap = usable["actual"] - usable["prediction"]
    steps: dict = {"rows": {}}
    for step, values in gap.groupby(usable["step"]):
        steps["rows"][str(step)] = int(len(values))
        if len(values) >= REMEASURE_MIN_ROWS:
            steps[str(step)] = [
                float(v) for v in np.percentile(
                    values, adjusted_levels(len(values))
                )
            ]

    return steps


def step_sentence(uncertainty: dict) -> str:
    """
    段ごとの外れ方の 80 % の幅を述べる文を作る関数

    Parameters
    ----------
    uncertainty: dict
        steps と kind と origins を持つ記録

    Returns
    ----------
    sentence: str
        文
    """
    # 段ごとの 10 % と 90 % の点
    kind = uncertainty["kind"]
    steps = uncertainty["steps"]
    measured = sorted(int(k) for k in steps if k != "rows")
    parts = []
    for step in measured:
        lower, upper = steps[str(step)][9], steps[str(step)][89]
        parts.append(
            f"{step} 段目 {np.exp(lower):.2f} から {np.exp(upper):.2f} 倍"
            if kind == "relative"
            else f"{step} 段目 {lower:+.4g} から {upper:+.4g}"
        )
    sentence = (
        f"何段も先の外れ方 (予測を始める時刻 {uncertainty['origins']} 個): "
        "80 % の幅は " + ", ".join(parts)
        if parts else "何段も先の外れ方は, 誤りが少なく測れませんでした"
    )

    return sentence


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
        excluded = before.get("excluded", 0)
        if coverage:
            low = coverage["80"] < 0.75
            sentences.append(
                f"今の記録の 80 % の幅は, 渡した実績の {rows - excluded:,} "
                f"行の {coverage['80'] * 100:.0f} % を覆った (90 % の幅は "
                f"{coverage['90'] * 100:.0f} %)"
                + (". 外れ方を小さく見積もっていた" if low else "")
            )
        if excluded:
            sentences.append(
                f"予測が 0 以下の {excluded:,} 行は, 今の記録 (比の形) では"
                "幅を出せないので確かめから除いた"
            )
        kind = uncertainty["kind"]
        lower, upper = uncertainty[kind][9], uncertainty[kind][89]
        width = (
            f"予測の {np.exp(lower):.2f} 倍から {np.exp(upper):.2f} 倍"
            if kind == "relative"
            else f"予測に {'+' if lower >= 0 else '-'}"
            f"{amount_text(abs(lower))} から "
            f"{'+' if upper >= 0 else '-'}{amount_text(abs(upper))}"
        )
        sentences.append(
            f"測り直した外れ方の 80 % の幅は, {width} ({rows:,} 行で測った)"
        )
        if uncertainty.get("groups"):
            sentences.append(group_sentence(uncertainty["groups"], kind))
        steps = uncertainty.get("steps")
        if steps:
            sentences.append(step_sentence(uncertainty))
        return sentences
    ece = uncertainty["calibration"]["ece"]
    recorded = before.get("ece")
    sentences.append(
        f"渡した実績の {rows:,} 行で, 確率と実際に当たった割合のずれは平均"
        f" {ece * 100:.1f} ポイント"
        + ("" if recorded is None
           else f" (記録したときは {recorded * 100:.1f} ポイント)")
    )
    if before.get("dropped"):
        sentences.append(
            f"一式のクラスに無いラベルの {before['dropped']:,} 行は除いた"
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
    # 一時のファイル (呼び出しごとに別の名前) への書き込みと置き換え
    # (途中で止まっても壊さないため)
    path = bundle.path / "inference.yaml"
    info = yaml.safe_load(path.read_text(encoding="utf-8"))
    info["uncertainty"] = uncertainty
    temporary = path.with_name(f"inference.yaml.{uuid.uuid4().hex}.tmp")
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
    write: bool = True, horizon: Optional[int] = None,
    time: Optional[str] = None, groups: Optional[str] = None
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
    horizon: Optional[int] = None
        何段も先の外れ方も測る場合の段の数 (目的の列の過去の値を使う
        回帰と予測の一式だけ. 予測を始める時刻を horizon おきにずらし
        ながら何段も先まで予測し, 段ごとの誤りの分位点を steps に残す)
    time: Optional[str] = None
        horizon で予測を始める時刻の列 (省略は前処理の時刻でずらす列)
    groups: Optional[str] = None
        外れ方を種類ごとに分けて測る列 (店や品など. 回帰と予測の一式
        だけ. 量の判断はその種類の行に種類ごとの外れ方を当てる)

    Returns
    ----------
    result: dict
        rows (測った行), before (回帰は今の記録の幅が渡した実績を
        覆った割合 coverage と除いた行の数 excluded, 分類は記録した
        ときの確率のずれ ece と除いた行の数 dropped. 分類の確率は
        直さないので, 渡した実績でのずれは uncertainty の方),
        uncertainty (新しい記録), sentences, written

    Raises
    ----------
    ValueError
        目的の列が無い場合, 印の数が行の数と違う場合, 測れる行が少ない
        場合, 一式のクラスに合うラベルの行が少ない場合, horizon が使え
        ない場合, groups の列が無いか分類の一式か目的の列の場合, 20 行
        以上の種類が無い場合
    """
    # 予測と, 測る行
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    bundle.require_column("外れ方の測り直し")
    data = read_source(bundle, source, options)
    table = predict(bundle, data)
    keep = counted_rows(bundle, table, count)
    target = str(bundle.dataset.info.get("target"))
    old = dict(bundle.info.get("uncertainty") or {})
    part = table[keep]
    if groups is not None:
        if bundle.task == "classification":
            raise ValueError("種類ごとの外れ方 (groups) は回帰と予測の一式"
                             "だけです")
        if groups == target or groups not in table.columns:
            raise ValueError(
                f"groups の列 {groups} が渡した実績に無いか, 目的の列です"
            )

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
        if groups is not None:
            measured["groups"] = measure_groups(
                part[finite.to_numpy()], groups,
                prediction[finite].to_numpy(float),
                answers[finite].to_numpy(float), measured["kind"]
            )
    else:
        before, measured = class_remeasure(bundle, old, part, target)
        rows = int(sum(b["rows"] for b in measured["calibration"]["bins"]))
    uncertainty.update({"rows": rows, **measured})
    uncertainty["measured_at"] = dt.datetime.now().astimezone().isoformat()
    uncertainty["previous"] = {
        key: old.get(key) for key in ("split", "rows", "kind")
        if old.get(key) is not None
    }
    # 一式を作ったときの記録は丸ごと残す (restore_uncertainty で戻すため.
    # 何度測り直しても最初の記録)
    original = old.get("original") if old.get("split") == "recent" else (
        {key: value for key, value in old.items() if key != "original"}
        if old else None
    )
    if original:
        uncertainty["original"] = original

    # 何段も先の外れ方 (1 段目の記録と同じ形)
    if horizon is not None:
        time = step_time(bundle, table, horizon, time)
        frame = data.frame.reset_index(drop=True)
        errors = step_errors(bundle, frame, keep, int(horizon), time)
        uncertainty["steps"] = step_quantiles(errors, uncertainty["kind"])
        uncertainty["horizon"] = int(horizon)
        uncertainty["origins"] = int(errors["origin"].nunique())

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


def restore_uncertainty(bundle: Union[str, Path, Bundle]) -> dict:
    """
    測り直した一式の外れ方を, 一式を作ったときの記録に戻す関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ

    Returns
    ----------
    uncertainty: dict
        戻した記録

    Raises
    ----------
    ValueError
        測り直していない一式と, 作ったときの記録が残っていない場合
    """
    # 測り直した記録の original を書き戻す
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    now = bundle.info.get("uncertainty") or {}
    if now.get("split") != "recent" or not now.get("original"):
        raise ValueError("戻せる記録がありません (測り直していない一式か,"
                         " 作ったときの記録が残っていない一式です)")
    uncertainty = dict(now["original"])
    write_uncertainty(bundle, uncertainty)

    return uncertainty

