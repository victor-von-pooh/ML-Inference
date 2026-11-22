from typing import Any, Optional

import numpy as np
from ml_eda import Dataset

from ml_inference.bundle import Bundle, split_sources
from ml_inference.calibrate import fit_calibration
from ml_inference.predict import (
    decode, inverse_target, run_model, shape_inputs
)

# 外れ方を測るときに推論する行の上限と, 1 回に推論する行の数
UNCERTAINTY_ROWS = 50000
UNCERTAINTY_BATCH = 256

# 残す分位点 (1 % から 99 %)
LEVELS = list(range(1, 100))

# 確率の当たり具合を見る区間の数
BINS = 10


def split_outputs(
    bundle: Bundle, dataset: Dataset, split: str, rows: int
) -> tuple[dict, np.ndarray]:
    """
    分割の最初の行を少しずつ推論し, 予測の列と目的の値を返す関数

    Parameters
    ----------
    bundle: Bundle
        一式
    dataset: Dataset
        分割の表を持つデータセット
    split: str
        分割の名前
    rows: int
        推論する行の数

    Returns
    ----------
    outputs: tuple[dict, np.ndarray]
        decode の列 (行の順につないだもの) と, 目的の値 (回帰は元の単位,
        分類はクラスの番号)
    """
    # 256 行ずつ推論してつなぐ
    _, y = dataset.arrays(split)
    parts: dict[str, list] = {}
    for start in range(0, rows, UNCERTAINTY_BATCH):
        stop = min(rows, start + UNCERTAINTY_BATCH)
        sources = split_sources(bundle, dataset, split, start, stop)
        columns = decode(
            bundle,
            run_model(bundle, shape_inputs(bundle, sources), stop, "cpu")
        )
        for name, values in columns.items():
            parts.setdefault(name, []).extend(list(values))
    answers = np.asarray(y[:rows])
    if bundle.task != "classification":
        answers = inverse_target(bundle, answers.astype(float))
    outputs = ({k: np.asarray(v) for k, v in parts.items()}, answers)

    return outputs


def adjusted_levels(count: int) -> np.ndarray:
    """
    行の少ない場合に幅が狭くなりすぎないように, 分位点の水準を外へずらす
    関数 (共形予測の補正)

    Parameters
    ----------
    count: int
        誤りを測った行の数

    Returns
    ----------
    levels: np.ndarray
        1 % から 99 % に対応する, ずらした水準 (0 から 100)
    """
    # 中央 (50 %) からの離れを (n + 1) / n 倍に広げる (中央は動かさない.
    # 量の判断の τ = 0.5 が偏らないように)
    q = np.asarray(LEVELS, dtype=float) / 100
    levels = np.clip(0.5 + (q - 0.5) * (count + 1) / count, 0.0, 1.0) * 100

    return levels


def coverage_spread(
    prediction: np.ndarray, answers: np.ndarray, kind: str
) -> float:
    """
    片方の半分で求めた 90 % の幅が, もう片方の半分を予測の水準ごとに
    どれだけ均等に覆うかを求める関数

    Parameters
    ----------
    prediction: np.ndarray
        予測
    answers: np.ndarray
        実際の値
    kind: str
        relative か absolute

    Returns
    ----------
    spread: float
        水準の 4 つの区間ごとの覆う割合と 90 % のずれの平均 (小さいほど
        良い. 2 つの半分を入れ替えた平均)
    """
    # 偶数番と奇数番の行に分けて, 互いに確かめる
    errors = np.log(answers / prediction) if kind == "relative" else (
        answers - prediction
    )
    gaps = []
    for fit, check in ((0, 1), (1, 0)):
        low, high = np.percentile(errors[fit::2], [5, 95])
        part = errors[check::2]
        level = prediction[check::2]
        edges = np.percentile(level, [25, 50, 75])
        group = np.digitize(level, edges)
        for number in range(4):
            inside = part[group == number]
            if len(inside):
                gaps.append(abs(((inside >= low) & (inside <= high)).mean()
                                - 0.9))
    spread = float(np.mean(gaps)) if gaps else 0.0

    return spread


def residual_values(
    prediction: np.ndarray, answers: np.ndarray, kind: str
) -> list[float]:
    """
    決めた形で, 誤りの 1 % から 99 % の分位点を求める関数

    Parameters
    ----------
    prediction: np.ndarray
        予測
    answers: np.ndarray
        実際の値
    kind: str
        relative (比の対数) か absolute (差)

    Returns
    ----------
    values: list[float]
        行の少ない場合の補正をした分位点
    """
    # 比は対数, 差はそのまま
    errors = np.log(answers / prediction) if kind == "relative" else (
        answers - prediction
    )
    values = [
        float(v) for v in np.percentile(errors, adjusted_levels(len(errors)))
    ]

    return values


def residual_quantiles(
    prediction: np.ndarray, answers: np.ndarray
) -> dict:
    """
    誤りの分位点を, 差と比の 2 つの形で求め, 使う形を選ぶ関数

    Parameters
    ----------
    prediction: np.ndarray
        予測
    answers: np.ndarray
        実際の値

    Returns
    ----------
    quantiles: dict
        absolute (差の分位点), relative (比の対数の分位点. 予測か実際の
        値に正でないものがあれば None), kind (使う形. 両方あれば, 半分で
        求めた幅が残りの半分を予測の水準ごとに均等に覆う方)
    """
    # 行の少ない場合の補正をした分位点
    levels = adjusted_levels(len(answers))
    absolute = np.percentile(answers - prediction, levels)
    positive = bool((prediction > 0).all() and (answers > 0).all())
    relative = np.percentile(
        np.log(answers / prediction), levels
    ) if positive else None
    kind = "absolute"
    if positive and len(answers) >= 40:
        kind = "relative" if coverage_spread(
            prediction, answers, "relative"
        ) <= coverage_spread(prediction, answers, "absolute") else "absolute"
    elif positive:
        kind = "relative"
    quantiles = {
        "kind": kind,
        "absolute": [float(v) for v in absolute],
        "relative": None if relative is None else [float(v) for v in relative],
    }

    return quantiles


def interval(
    prediction: np.ndarray, quantiles: dict, low: int, high: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    予測と誤りの分位点から, 予測の幅を求める関数

    Parameters
    ----------
    prediction: np.ndarray
        予測
    quantiles: dict
        residual_quantiles の結果
    low: int
        下の分位点 (%)
    high: int
        上の分位点 (%)

    Returns
    ----------
    bounds: tuple[np.ndarray, np.ndarray]
        幅の下と上
    """
    # 比の形は予測に掛け, 差の形は予測に足す
    values = quantiles[quantiles["kind"]]
    lower, upper = values[low - 1], values[high - 1]
    if quantiles["kind"] == "relative":
        bounds = (prediction * np.exp(lower), prediction * np.exp(upper))
    else:
        bounds = (prediction + lower, prediction + upper)

    return bounds


def calibration(confidence: np.ndarray, hit: np.ndarray) -> dict:
    """
    予測したクラスの確率が, 実際に当たった割合と合っているかを求める関数

    Parameters
    ----------
    confidence: np.ndarray
        予測したクラスの確率
    hit: np.ndarray
        当たったか

    Returns
    ----------
    result: dict
        bins (区間ごとの確率の平均, 当たった割合, 行の数), ece (ずれの
        平均. 行の数で重み付け)
    """
    # 確率を 10 の区間に分ける
    edges = np.linspace(0, 1, BINS + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, BINS - 1)
    bins, gap = [], 0.0
    for number in range(BINS):
        inside = index == number
        if not inside.any():
            continue
        mean = float(confidence[inside].mean())
        rate = float(hit[inside].mean())
        bins.append({"low": float(edges[number]), "high": float(
            edges[number + 1]
        ), "confidence": mean, "accuracy": rate, "rows": int(inside.sum())})
        gap += inside.sum() * abs(rate - mean)
    result = {"bins": bins, "ece": float(gap / len(confidence))}

    return result


def class_calibration(
    bundle: Bundle, columns: dict, answers: np.ndarray
) -> tuple[dict, Optional[str]]:
    """
    分類の確率の当たり具合を求める関数

    Parameters
    ----------
    bundle: Bundle
        一式
    columns: dict
        decode の列
    answers: np.ndarray
        クラスの番号

    Returns
    ----------
    result: tuple[dict, Optional[str]]
        calibration の結果と, 測ったクラス (2 クラスは 2 つ目のクラス,
        3 クラス以上は None (予測したクラス))
    """
    # 2 クラスは当たりの確率, 他は予測したクラスの確率
    classes = np.asarray(bundle.classes)
    if len(classes) == 2:
        name = str(classes[1])
        result = (calibration(
            columns[f"p:{name}"].astype(float), answers.astype(int) == 1
        ), name)
        return result
    hit = columns["prediction"] == classes[answers.astype(int)]
    result = (calibration(columns["confidence"].astype(float), hit), None)

    return result


def fit_column_calibration(
    bundle: Bundle, dataset: Dataset, method: str
) -> dict[str, Any]:
    """
    分類の一式の確率の補正を, 検証用の分割で当てはめる関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式 (まだ補正を持たないもの)
    dataset: Dataset
        分割の表を持つデータセット
    method: str
        補正の選び方 (platt か isotonic)

    Returns
    ----------
    calibrate: dict[str, Any]
        一式の calibrate (method, split, rows, classes (クラスごとの当て
        はめた値), before と after (補正の前と後の検証用のずれ ece と
        テスト用のずれ test_ece. テスト用の分割が無ければ test_ece は
        None)). 補正は bundle.info にもかける

    Raises
    ----------
    ValueError
        目的の列か検証用の分割が無い場合, 検証用の分割に出ないクラスが
        ある場合, Platt 法の傾きが 0 以下になった場合
    """
    # 当てはめるのは検証用の分割だけ (学習用は学習に使ったので確率が
    # 当たりすぎて見える)
    frames = dataset.frames or {}
    if not dataset.target or "valid" not in frames or len(
        frames["valid"]
    ) == 0:
        raise ValueError(
            "確率の補正には, 目的の列のある検証用の分割 (valid) が要ります"
        )
    rows = min(len(dataset.arrays("valid")[1]), UNCERTAINTY_ROWS)
    columns, answers = split_outputs(bundle, dataset, "valid", rows)
    classes = bundle.classes
    probability = np.stack(
        [columns[f"p:{name}"].astype(float) for name in classes], axis=1
    )
    calibrate = {
        "method": method, "split": "valid", "rows": int(rows),
        "classes": fit_calibration(method, classes, probability, answers),
        "before": split_eces(bundle, dataset, rows),
    }

    # 補正をかけて, 同じ求め方で補正の後のずれも
    bundle.info["calibrate"] = calibrate
    calibrate["after"] = split_eces(bundle, dataset, rows)

    return calibrate


def split_eces(
    bundle: Bundle, dataset: Dataset, rows: int
) -> dict[str, Any]:
    """
    検証用とテスト用の分割で, 今の一式の確率のずれ (ECE) を求める関数

    Parameters
    ----------
    bundle: Bundle
        一式
    dataset: Dataset
        分割の表を持つデータセット
    rows: int
        検証用の分割で推論する行の数

    Returns
    ----------
    eces: dict[str, Any]
        ece (検証用) と test_ece (テスト用. テスト用の分割が無ければ None)
    """
    # 測り方は外れ方の記録 (class_calibration) と同じ
    frames = dataset.frames or {}
    columns, answers = split_outputs(bundle, dataset, "valid", rows)
    eces: dict[str, Any] = {
        "ece": class_calibration(bundle, columns, answers)[0]["ece"],
        "test_ece": None,
    }
    if "test" in frames and len(frames["test"]) > 0:
        count = min(len(dataset.arrays("test")[1]), UNCERTAINTY_ROWS)
        got, truth = split_outputs(bundle, dataset, "test", count)
        eces["test_ece"] = class_calibration(bundle, got, truth)[0]["ece"]

    return eces


def measure_uncertainty(
    bundle: Bundle, dataset: Dataset
) -> Optional[dict[str, Any]]:
    """
    一式の予測の外れ方を, 検証用の分割で測ってテスト用の分割で確かめる
    関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    dataset: Dataset
        分割の表を持つデータセット

    Returns
    ----------
    uncertainty: Optional[dict[str, Any]]
        split, rows と, 回帰は誤りの分位点とテスト用で幅が覆った割合
        (coverage), 分類は確率の当たり具合 (calibration) とテスト用の
        ずれ (test_ece). 目的の値の無いデータセットは None
    """
    # 検証用の分割 (無ければ学習用)
    if not dataset.target or not dataset.frames:
        return None
    split = "valid" if "valid" in dataset.frames else (
        "train" if "train" in dataset.frames else None
    )
    if split is None:
        return None
    _, y = dataset.arrays(split)
    rows = min(len(y), UNCERTAINTY_ROWS)
    if rows == 0:
        return None
    columns, answers = split_outputs(bundle, dataset, split, rows)
    test = "test" in dataset.frames and split != "test" and len(
        dataset.frames["test"]
    ) > 0
    uncertainty: dict[str, Any] = {"split": split, "rows": int(rows)}

    # 回帰と予測は誤りの分位点と, テスト用で覆った割合
    if bundle.task != "classification":
        prediction = columns["prediction"].astype(float)
        quantiles = residual_quantiles(prediction, answers)
        uncertainty.update(quantiles)
        if test:
            count = min(len(dataset.arrays("test")[1]), UNCERTAINTY_ROWS)
            got, truth = split_outputs(bundle, dataset, "test", count)
            got_prediction = got["prediction"].astype(float)
            uncertainty["coverage"] = {}
            for name, (low, high) in (("80", (10, 90)), ("90", (5, 95))):
                lower, upper = interval(got_prediction, quantiles, low, high)
                inside = (truth >= lower) & (truth <= upper)
                uncertainty["coverage"][name] = float(inside.mean())
            # 片側の点ごとに, テスト用の実際の値がそれ以下だった割合 (量の
            # 判断は 1 つの分位点を使うので)
            uncertainty["test_below"] = [
                float((truth <= interval(got_prediction, quantiles, level,
                                         level)[1]).mean())
                for level in LEVELS
            ]
        return uncertainty

    # 分類は確率の当たり具合. 2 クラスは 2 つ目のクラスの確率で測る
    # (判断に使う当たりの確率. 予測したクラスの確率では, 少ない当たりの
    # 低い確率の区間のずれが折り返して見えない), 3 クラス以上は予測した
    # クラスの確率で測る
    uncertainty["calibration"], uncertainty["calibration_of"] = (
        class_calibration(bundle, columns, answers)
    )
    if test:
        count = min(len(dataset.arrays("test")[1]), UNCERTAINTY_ROWS)
        got, truth = split_outputs(bundle, dataset, "test", count)
        uncertainty["test_ece"] = class_calibration(
            bundle, got, truth
        )[0]["ece"]

    return uncertainty
