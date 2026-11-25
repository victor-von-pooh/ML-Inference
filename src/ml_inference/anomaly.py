import copy
import datetime as dt
import json
import math
from pathlib import Path
import shutil
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from ml_eda import Dataset
from ml_modelcontrol import dump_spec, load_spec

from ml_inference.bundle import (
    ANOMALY_BUNDLE_VERSION, Bundle, bundle_notes, inside, load_bundle,
    plan_weights, run_summary
)
from ml_inference.conclude import NO_PREDICTION, is_number, plain
from ml_inference.grouping import read_table
from ml_inference.units import check_units

# 実験の中の成果物 (ML-Quest の異常の検知の骨組みが書く名前)
SCORER_FILE = "anomaly_scorer.json"
RECIPE_FILE = "anomaly_recipe.yaml"
SCORES_FILE = "anomaly_scores.csv"
EVALUATION_FILE = "evaluations/anomaly_{split}.json"

# 読める度合いの決め方の成果物の版
SCORER_VERSION = 1

# 度合いの決め方と, 文に使う名前 (骨組みの SCORE_NAMES と同じ)
SCORE_NAMES = {"autoencoder": "自己符号化器の再構成の誤差",
               "distance": "学習用の正常の行の中心からの距離"}

# 度合いの決め方の成果物の鍵
SCORER_KEYS = ("format_version", "score", "recipe", "scores", "features",
               "id", "capacity", "k", "normal", "model", "distance")

# 学習用の正常の行の度合いの分布の鍵
NORMAL_KEYS = ("rows", "mean", "p50", "p90", "p95", "p99", "max")

# 行ごとの度合いの表の, 行の id の列の他の列
SCORES_COLUMNS = ("split", "score", "anomaly")

# 実験の評価を写す分割と, 評価の行の鍵
EVALUATION_SPLITS = ("valid", "test", "unlabeled")
EVALUATION_KEYS = ("rows", "anomalies", "k", "precision_at_k",
                   "average_precision")

# 分割の日本語の名前
SPLIT_NAMES = {"valid": "検証用", "test": "テスト用",
               "unlabeled": "正解の欠けた"}

# anomaly の欄
ANOMALY_KEYS = ("scorer", "recipe", "score", "features", "id", "input",
                "output", "capacity", "k", "rows", "normal", "evaluations",
                "check")

# 一式を作るときに再構成を確かめる 0 の行の数
PROBE_ROWS = 2

# 実験の度合いと同じとみなす差 (絶対と相対) と, 知らせる一致の割合の下限
ABSOLUTE_GAP = 1e-6
RELATIVE_GAP = 1e-4
AGREEMENT = 0.99

# 予測の表に足す列
ANOMALY_FRAME_COLUMNS = ("status", "reason", "score", "rank", "column",
                         "column_share")

# 予測の表の status の値
OK = "ok"
SKIPPED = "skipped"

# 度合いが数でない行の理由
NOT_FINITE = "度合いが数ではありません"

# 振り分けで書ける項目と, 足す列
TRIAGE_KEYS = ("type", "capacity")
ACTION_COLUMN = "action"


def is_whole(value: Any, lowest: int) -> bool:
    """
    下限以上の整数 (真偽値を除く) かを確かめる関数

    Parameters
    ----------
    value: Any
        確かめる値
    lowest: int
        下限

    Returns
    ----------
    result: bool
        下限以上の整数なら True
    """
    # bool は int の子なので除く
    result = isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ) and int(value) >= lowest

    return result


def finite_matrix(value: Any, shape: tuple) -> bool:
    """
    決まった形の有限の数の並びかを確かめる関数

    Parameters
    ----------
    value: Any
        確かめる値 (JSON の並び)
    shape: tuple
        形

    Returns
    ----------
    result: bool
        形が同じで, 全てが有限の数なら True
    """
    # 数にできない値と真偽値は誤り
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    result = array.shape == shape and bool(np.isfinite(array).all()) and (
        not any(isinstance(v, bool) for v in np.ravel(
            np.asarray(value, dtype=object)
        ))
    )

    return result


def check_scorer(scorer: Any) -> list[str]:
    """
    度合いの決め方の成果物 (anomaly_scorer.json) の誤りを一覧で返す関数

    Parameters
    ----------
    scorer: Any
        読み込んだ成果物

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 版と決め方
    if not isinstance(scorer, dict):
        return ["度合いの決め方が辞書ではありません"]
    version = scorer.get("format_version")
    if version != SCORER_VERSION:
        return [f"度合いの決め方の版 {version} は読めません (版 "
                f"{SCORER_VERSION} だけ)"]
    errors = [f"度合いの決め方に {key} がありません"
              for key in SCORER_KEYS if key not in scorer]
    if errors:
        return errors
    score = scorer["score"]
    if score not in SCORE_NAMES:
        errors.append(f"score は {', '.join(SCORE_NAMES)} のどちらかにして"
                      "ください")

    # 列, 行の id, 件数の上限と K, 正常の行の分布
    features = scorer["features"]
    if not isinstance(features, list) or not features or not all(
        isinstance(name, str) and name for name in features
    ) or len(set(features)) != len(features):
        errors.append("features は重ならない列の名前の並びにしてください")
        features = []
    if not isinstance(scorer["id"], str) or not scorer["id"]:
        errors.append("id は行の id の列の名前にしてください")
    for key in ("capacity", "k"):
        if not is_whole(scorer[key], 1):
            errors.append(f"{key} は 1 以上の整数にしてください")
    for key in ("recipe", "scores"):
        if not isinstance(scorer[key], str) or not scorer[key]:
            errors.append(f"{key} はファイルの名前にしてください")
    normal = scorer["normal"]
    if not isinstance(normal, dict) or not all(
        is_number(normal.get(key)) and math.isfinite(normal[key])
        for key in NORMAL_KEYS
    ):
        errors.append(f"normal は {', '.join(NORMAL_KEYS)} の数にして"
                      "ください")

    # 決め方ごとのモデルの入力と出力か, 距離の中心と精度行列
    model, distance = scorer["model"], scorer["distance"]
    if score == "autoencoder":
        if distance is not None:
            errors.append("自己符号化器の distance は null にしてください")
        if not isinstance(model, dict) or not all(
            isinstance(model.get(key), str) and model[key]
            for key in ("input", "output")
        ):
            errors.append("自己符号化器の model は input と output の名前"
                          "にしてください")
    elif score == "distance":
        if model is not None:
            errors.append("距離に基づく度合いの model は null にして"
                          "ください")
        width = len(features)
        if not isinstance(distance, dict) or not finite_matrix(
            distance.get("center"), (width,)
        ) or not finite_matrix(distance.get("precision"), (width, width)):
            errors.append(
                f"distance は列の数 ({width}) の center と, 列の数 × 列の"
                "数の precision の有限の数にしてください"
            )

    return errors


def refuse_constant(value: str) -> None:
    """
    JSON の外の NaN と Infinity を誤りにする関数

    Parameters
    ----------
    value: str
        NaN, Infinity, -Infinity のどれか

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        いつも
    """
    # json.loads の parse_constant に渡す
    raise ValueError(f"{value} は数にできません")


def read_scorer(path: Path) -> dict:
    """
    度合いの決め方の成果物を読み, 形を確かめる関数

    Parameters
    ----------
    path: Path
        anomaly_scorer.json

    Returns
    ----------
    scorer: dict
        成果物の中身

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        JSON でない場合と, 形が違う場合
    """
    # JSON の外の NaN と Infinity も誤り
    if not path.is_file():
        raise FileNotFoundError(f"度合いの決め方 {path} がありません")
    try:
        scorer = json.loads(path.read_text(encoding="utf-8"),
                            parse_constant=refuse_constant)
    except ValueError as error:
        raise ValueError(f"{path} を読めません ({error})") from error
    errors = check_scorer(scorer)
    if errors:
        raise ValueError(f"{path} の誤り: " + " / ".join(errors))

    return scorer


def read_recipe(path: Path) -> dict:
    """
    前処理のレシピの写しを読み, 異常の検知のレシピかを確かめる関数

    Parameters
    ----------
    path: Path
        anomaly_recipe.yaml

    Returns
    ----------
    recipe: dict
        当てはめた値を含むレシピ

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        異常の検知のレシピでない場合
    """
    # 手順の並びを持つ, task が anomaly のレシピ
    if not path.is_file():
        raise FileNotFoundError(f"前処理のレシピ {path} がありません")
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or recipe.get("task") != "anomaly" or (
        not isinstance(recipe.get("steps"), list)
    ):
        raise ValueError(
            f"{path} は異常の検知のレシピ (task: anomaly と steps) では"
            "ありません"
        )

    return recipe


def read_scores(path: Path, key: str) -> pd.DataFrame:
    """
    実験の行ごとの度合いの表を読む関数

    Parameters
    ----------
    path: Path
        anomaly_scores.csv
    key: str
        行の id の列の名前

    Returns
    ----------
    scores: pd.DataFrame
        行の id の列 (文字), split, score, anomaly の表

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        列が違う場合と, 度合いが数でない場合
    """
    # 行の id は文字 (先頭の 0 などを残すため)
    if not path.is_file():
        raise FileNotFoundError(f"行ごとの度合いの表 {path} がありません")
    scores = pd.read_csv(path, dtype={key: str}, keep_default_na=False,
                         na_values=[""])
    expected = [key, *SCORES_COLUMNS]
    if list(scores.columns) != expected:
        raise ValueError(
            f"{path} の列は {', '.join(expected)} にしてください (今は "
            f"{', '.join(map(str, scores.columns))})"
        )
    values = pd.to_numeric(scores["score"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path} の score に数でない値があります")
    scores["score"] = values.astype(np.float64)

    return scores


def read_evaluations(root: Path) -> dict:
    """
    実験の上位 K 件の適合率と PR-AUC の評価を読む関数

    Parameters
    ----------
    root: Path
        実験のディレクトリ

    Returns
    ----------
    evaluations: dict
        分割の名前から rows, anomalies, k, precision_at_k,
        average_precision (評価の無い分割は無い)
    """
    # ML-Found の evaluations の形の最初の行 (形の違う評価は写さない)
    evaluations = {}
    for split in EVALUATION_SPLITS:
        path = root / EVALUATION_FILE.format(split=split)
        if not path.is_file():
            continue
        table = json.loads(path.read_text(encoding="utf-8"))
        rows = table.get("rows") if isinstance(table, dict) else None
        row = rows[0] if isinstance(rows, list) and rows else None
        if isinstance(row, dict) and all(
            is_number(row.get(key)) for key in EVALUATION_KEYS
        ):
            evaluations[split] = {key: plain(row[key])
                                  for key in EVALUATION_KEYS}

    return evaluations


def run_sources(run: Union[str, Path]) -> dict:
    """
    異常の検知の実験のディレクトリから, 一式に要るものを読む関数

    Parameters
    ----------
    run: Union[str, Path]
        ML-Found の実験のディレクトリ

    Returns
    ----------
    sources: dict
        root, scorer, recipe_file, recipe, scores, evaluations, weights と
        spec (距離は None), run (実験の状態)

    Raises
    ----------
    ValueError
        実験のディレクトリでない場合, 自己符号化器の重みか定義が無い場合,
        成果物の形が違う場合
    FileNotFoundError
        成果物が無い場合
    """
    # 度合いの決め方, レシピ, 行ごとの度合い, 評価
    root = Path(run).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"実験のディレクトリ {root} がありません")
    scorer = read_scorer(root / SCORER_FILE)
    recipe_file = inside(root, scorer["recipe"])
    recipe = read_recipe(recipe_file)
    scores = read_scores(inside(root, scorer["scores"]), scorer["id"])

    # 自己符号化器の重みと定義 (仮のモデルで学習した実験には定義が無い)
    weights = spec = None
    if scorer["score"] == "autoencoder":
        weights = root / "final_model.pt"
        if not weights.is_file():
            raise ValueError(f"重み {weights} がありません")
        spec = root / "model_spec.yaml"
        if not spec.is_file():
            raise ValueError(
                f"モデルの定義 {spec} がありません (仮のモデル "
                "PlaceholderModel で学習した実験は一式にできません. 定義を"
                "置いて学習し直してください)"
            )
    sources = {
        "root": root, "scorer": scorer, "recipe_file": recipe_file,
        "recipe": recipe, "scores": scores,
        "evaluations": read_evaluations(root), "weights": weights,
        "spec": spec, "run": run_summary(root),
    }

    return sources


def check_anomaly_info(info: dict) -> list[str]:
    """
    版 11 の一式の anomaly の欄と, 決め方と model の食い違いを確かめる関数

    Parameters
    ----------
    info: dict
        inference.yaml の中身

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 欄と出力の形
    anomaly = info.get("anomaly")
    if not isinstance(anomaly, dict):
        return [f"版 {ANOMALY_BUNDLE_VERSION} の一式には anomaly が要ります"]
    errors = [f"anomaly に {key} がありません"
              for key in ANOMALY_KEYS if key not in anomaly]
    unknown = [str(key) for key in anomaly if key not in ANOMALY_KEYS]
    if unknown:
        errors.append(f"anomaly に知らない項目があります: "
                      f"{', '.join(unknown)}")
    output = info.get("output") if isinstance(info.get("output"),
                                              dict) else {}
    if output.get("form") != "value":
        errors.append(f"版 {ANOMALY_BUNDLE_VERSION} の一式の出力の形は "
                      "value (行ごとの異常の度合い) にしてください")
    if errors:
        return errors

    # 決め方とモデルの有無, ファイルの名前, 件数
    score = anomaly["score"]
    model = info.get("model")
    if score not in SCORE_NAMES:
        errors.append(f"anomaly.score は {', '.join(SCORE_NAMES)} の"
                      "どちらかにしてください")
    elif score == "autoencoder" and not isinstance(model, dict):
        errors.append("自己符号化器の一式には model が要ります")
    elif score == "distance" and model is not None:
        errors.append("距離に基づく度合いの一式の model は null にして"
                      "ください")
    for key in ("scorer", "recipe"):
        if not isinstance(anomaly[key], str) or not anomaly[key]:
            errors.append(f"anomaly.{key} はファイルの名前にしてください")
    for key in ("capacity", "k"):
        if not is_whole(anomaly[key], 1):
            errors.append(f"anomaly.{key} は 1 以上の整数にしてください")

    return errors


def check_anomaly_spec(spec: dict, names: dict) -> None:
    """
    自己符号化器の定義の入力と出力が約束に合うかを確かめる関数

    Parameters
    ----------
    spec: dict
        モデルの定義 (ML-ModelControl の形)
    names: dict
        input と output (成果物の model)

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        入力が成果物の入力の名前の 1 つでない場合と, 出力の名前が無い
        場合
    """
    # 入力は行の特徴の 1 つ, 出力に再構成
    inputs = list(spec.get("inputs") or {})
    if inputs != [names["input"]]:
        raise ValueError(
            f"自己符号化器の入力は {names['input']} の 1 つにしてください "
            f"(今は {', '.join(inputs) or '無し'})"
        )
    outputs = spec.get("outputs") or {}
    if names["output"] not in outputs:
        raise ValueError(
            f"自己符号化器の出力に {names['output']} がありません (出力: "
            f"{', '.join(map(str, outputs)) or '無し'})"
        )


def anomaly_dataset(root: Path, recipe: dict, features: list) -> Dataset:
    """
    レシピの写しと度合いに使う列から, 前処理だけの ML-EDA のデータセット
    を作る関数

    Parameters
    ----------
    root: Path
        一式 (か実験) のディレクトリ
    recipe: dict
        当てはめた値を含むレシピ
    features: list
        度合いに使う列

    Returns
    ----------
    dataset: Dataset
        分割の表を持たないデータセット (transform だけに使う)
    """
    # 目的の列 (異常の正解) は前処理の手順の外
    info = {"kind": "table", "task": "anomaly",
            "target": recipe.get("target"),
            "features": [{"name": name} for name in features]}
    dataset = Dataset(path=root, info=info, frames={}, recipe=recipe)

    return dataset


def reconstruct(model: Any, x: torch.Tensor, names: dict) -> torch.Tensor:
    """
    行の特徴を入力の名前でモデルに渡し, 再構成を返す関数

    Parameters
    ----------
    model: Any
        自己符号化器
    x: torch.Tensor
        (件数, 列の数) の行の特徴
    names: dict
        input と output

    Returns
    ----------
    recon: torch.Tensor
        再構成 (出力が複数のモデルは出力の名前の値)
    """
    # 出力が 1 つの定義はテンソル, 複数の定義は辞書
    outputs = model(**{names["input"]: x})
    recon = outputs[names["output"]] if isinstance(outputs, dict) else (
        outputs
    )

    return recon


def autoencoder_parts(
    model: Any, x: np.ndarray, names: dict, batch_size: int, device: str
) -> np.ndarray:
    """
    行ごと, 列ごとの再構成の二乗誤差を, まとまりごとに求める関数

    Parameters
    ----------
    model: Any
        自己符号化器
    x: np.ndarray
        (件数, 列の数) の float32 の行の特徴
    names: dict
        input と output
    batch_size: int
        1 回に渡す行の数
    device: str
        使うデバイス

    Returns
    ----------
    parts: np.ndarray
        (件数, 列の数) の float64 (行の平均が度合い)

    Raises
    ----------
    ValueError
        再構成の形が入力と違う場合
    """
    # 二乗誤差は float32 で求めてから float64 に (骨組みと同じ)
    chunks = []
    model = model.to(device)
    try:
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                chunk = torch.tensor(x[start:start + batch_size]).to(device)
                recon = reconstruct(model, chunk, names)
                if tuple(recon.shape) != tuple(chunk.shape):
                    raise ValueError(
                        f"再構成の形 {tuple(recon.shape)} が入力の形 "
                        f"{tuple(chunk.shape)} と違います"
                    )
                chunks.append(((recon - chunk) ** 2).cpu())
    finally:
        model.to("cpu")
    parts = torch.cat(chunks).double().numpy() if chunks else np.zeros(
        (0, x.shape[1])
    )

    return parts


def distance_parts(x: np.ndarray, distance: dict) -> np.ndarray:
    """
    行ごと, 列ごとの, 二乗の Mahalanobis 距離の分を求める関数

    Parameters
    ----------
    x: np.ndarray
        (件数, 列の数) の float32 の行の特徴
    distance: dict
        center と precision (成果物の distance)

    Returns
    ----------
    parts: np.ndarray
        (件数, 列の数) の float64 (行の平均が度合い)
    """
    # 差と, 精度行列をかけた差の要素ごとの積 (骨組みと同じ)
    center = np.asarray(distance["center"], dtype=np.float64)
    precision = np.asarray(distance["precision"], dtype=np.float64)
    diff = x.astype(np.float64) - center
    parts = diff * (diff @ precision)

    return parts


def score_parts(
    bundle: Bundle, x: np.ndarray, batch_size: int, device: str
) -> np.ndarray:
    """
    一式の度合いの決め方で, 行ごと, 列ごとの度合いの分を求める関数

    Parameters
    ----------
    bundle: Bundle
        異常の度合いの一式
    x: np.ndarray
        (件数, 列の数) の float32 の行の特徴
    batch_size: int
        1 回に渡す行の数 (自己符号化器)
    device: str
        使うデバイス (自己符号化器)

    Returns
    ----------
    parts: np.ndarray
        (件数, 列の数) の float64
    """
    # 自己符号化器は重みのモデル, 距離は成果物の中心と精度行列
    scorer = bundle.cache["anomaly_scorer"]
    if scorer["score"] == "autoencoder":
        parts = autoencoder_parts(bundle.model(), x, scorer["model"],
                                  batch_size, device)
        return parts
    parts = distance_parts(x, scorer["distance"])

    return parts


def row_features(
    bundle: Bundle, frame: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """
    表にレシピの前処理をかけ, 度合いに使う列の配列と行ごとの理由を返す
    関数

    Parameters
    ----------
    bundle: Bundle
        異常の度合いの一式
    frame: pd.DataFrame
        元の列の表

    Returns
    ----------
    found: tuple[np.ndarray, list[str]]
        (件数, 列の数) の float32 の配列と, 行ごとの度合いを出さない理由
        (出す行は空の文字)

    Raises
    ----------
    ValueError
        前処理に要る列が無い場合 (ML-EDA の誤り)
    """
    # 当てはめた値のままの前処理 (行を落とす手順で落ちた行は欠損)
    transformed = bundle.cache["anomaly_dataset"].transform(frame)
    reasons = [""] * len(transformed)
    for row, op in (transformed.attrs.get("dropped") or {}).items():
        reasons[int(row)] = (
            f"前処理の手順 {op} で除く行です (学習でも使わない行)"
        )

    # 数にできない値も欠損として, 欠けた列を理由に
    numbers = transformed.apply(lambda c: pd.to_numeric(c, errors="coerce"))
    values = numbers.to_numpy(dtype=np.float64, na_value=np.nan)
    with np.errstate(over="ignore"):
        x = values.astype(np.float32)
    for row in range(len(x)):
        if reasons[row]:
            continue
        lost = [name for name, value in zip(numbers.columns, x[row])
                if not np.isfinite(value)]
        if lost:
            reasons[row] = ("前処理の後に欠損: " + ", ".join(lost[:5])
                            + (" など" if len(lost) > 5 else ""))
    found = (x, reasons)

    return found


def predict_anomaly(
    bundle: Bundle, source: Any, options: Optional[dict] = None,
    batch_size: int = 256, device: str = "cpu"
) -> pd.DataFrame:
    """
    新しい行の異常の度合いと, 最も外れた列を返す関数

    Parameters
    ----------
    bundle: Bundle
        異常の度合いの一式
    source: Any
        表のファイル, 表, 表の ML-EDA の Data
    options: Optional[dict] = None
        ファイルの読み込みの設定 (load_data に渡す)
    batch_size: int = 256
        1 回に渡す行の数
    device: str = "cpu"
        使うデバイス

    Returns
    ----------
    result: pd.DataFrame
        元の列 (予測の列と同じ名前は「(入力)」付き) と, status, reason,
        score, rank (Int64), column, column_share

    Raises
    ----------
    ValueError
        表でない場合と, 前処理に要る列が無い場合
    """
    # 表の読み込みと前処理, 出せる行だけの度合い
    frame = read_table(source, options, "異常の度合いを出す表")
    x, reasons = row_features(bundle, frame)
    features = bundle.info["anomaly"]["features"]
    scores = np.full(len(x), math.nan)
    columns: list = [None] * len(x)
    shares: list = [None] * len(x)
    keep = np.flatnonzero([not r for r in reasons])
    if len(keep):
        parts = score_parts(bundle, x[keep], batch_size, device)
        for row, part in zip(keep, parts):
            if not np.isfinite(part).all():
                reasons[row] = NOT_FINITE
                continue
            top = int(np.argmax(part))
            total = float(part.sum())
            scores[row] = float(part.mean())
            columns[row] = features[top]
            shares[row] = float(part[top]) / total if total else None

    # 度合いの高い順の順位 (同じ度合いは前の行が先)
    ok = np.array([not r for r in reasons], dtype=bool)
    ranks: list = [None] * len(x)
    rows = np.flatnonzero(ok)
    order = rows[np.argsort(-scores[rows], kind="stable")]
    for rank, row in enumerate(order, 1):
        ranks[row] = rank

    # 元の列の後ろに予測の列 (同じ名前の元の列は「(入力)」)
    added = list(ANOMALY_FRAME_COLUMNS)
    result = frame.rename(
        columns={c: f"{c} (入力)" for c in frame.columns if c in added}
    )
    result["status"] = np.where(ok, OK, SKIPPED)
    result["reason"] = pd.Series(reasons, dtype=object)
    result["score"] = scores.astype(float)
    result["rank"] = pd.array(ranks, dtype="Int64")
    result["column"] = pd.Series(columns, dtype=object)
    result["column_share"] = pd.Series(
        [math.nan if v is None else v for v in shares], dtype=float
    )

    return result


def id_column(bundle: Bundle, prediction: pd.DataFrame) -> Optional[str]:
    """
    予測の表の行の id の列の名前を返す関数

    Parameters
    ----------
    bundle: Bundle
        異常の度合いの一式
    prediction: pd.DataFrame
        predict_anomaly の表

    Returns
    ----------
    name: Optional[str]
        行の id の列 (予測の列と同じ名前なら「(入力)」付き. 無ければ
        None)
    """
    # 予測の列と同じ名前の元の列は「(入力)」付き
    key = bundle.info["anomaly"]["id"]
    if key in ANOMALY_FRAME_COLUMNS:
        key = f"{key} (入力)"
    name = key if key in prediction.columns else None

    return name


def score_check(
    bundle: Bundle, data: Any, options: Optional[dict], sources: dict,
    batch_size: int
) -> tuple[Optional[dict], list[str]]:
    """
    元の表の度合いを一式で出し, 実験の行ごとの度合いと比べる関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    data: Any
        学習に使った元の表 (ファイル, 表, 表の Data)
    options: Optional[dict]
        ファイルの読み込みの設定
    sources: dict
        run_sources の結果
    batch_size: int
        1 回に渡す行の数

    Returns
    ----------
    found: tuple[Optional[dict], list[str]]
        rows, agreement, max_diff (比べられなければ None) と注意の文
    """
    # 行の id の値で実験の度合いと突き合わせ
    prediction = predict_anomaly(bundle, data, options, batch_size, "cpu")
    key = id_column(bundle, prediction)
    if key is None:
        note = (f"元の表に行の id の列 {bundle.info['anomaly']['id']} が"
                "無いので, 一式の度合いを実験の度合いと比べません")
        return None, [note]
    expected = sources["scores"]
    known = expected.drop_duplicates(expected.columns[0]).set_index(
        expected.columns[0]
    )["score"]
    ok = prediction[prediction["status"] == OK]
    ids = ok[key].astype(str)
    matched = ids.isin(known.index).to_numpy()
    if not matched.any():
        note = ("元の表の行の id が実験の行ごとの度合いの表に無いので, 一式"
                "の度合いを実験の度合いと比べません")
        return None, [note]

    # 差の許し (絶対と相対) の中の割合と, 差の最大
    found_scores = ok["score"].to_numpy(dtype=np.float64)[matched]
    recorded = known.loc[ids[matched]].to_numpy(dtype=np.float64)
    gap = np.abs(found_scores - recorded)
    agreement = float((gap <= ABSOLUTE_GAP + RELATIVE_GAP
                       * np.abs(recorded)).mean())
    check = {"rows": int(matched.sum()), "agreement": agreement,
             "max_diff": float(gap.max())}
    notes = []
    if agreement < AGREEMENT:
        notes.append(
            f"元の表を一式で出した度合いが, 実験の度合いと "
            f"{agreement * 100:.1f} % の行でしか同じになりません (前処理の"
            "レシピか重みが学習と違う可能性があります)"
        )
    if check["rows"] < len(expected):
        notes.append(
            f"実験の行ごとの度合いの {len(expected):,} 行のうち, 比べたのは"
            f" {check['rows']:,} 行です (元の表に無い行か, 度合いを出せない"
            "行があります)"
        )
    found = (check, notes)

    return found


def probe_autoencoder(bundle: Bundle, width: int, batch_size: int) -> None:
    """
    0 の行を自己符号化器に渡し, 再構成の形と値を確かめる関数

    Parameters
    ----------
    bundle: Bundle
        自己符号化器の一式 (cache に anomaly_scorer)
    width: int
        度合いに使う列の数
    batch_size: int
        1 回に渡す行の数

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        再構成が (件数, 列の数) の数の配列でない場合
    """
    # 再構成の形の確かめは autoencoder_parts
    parts = autoencoder_parts(
        bundle.model(), np.zeros((PROBE_ROWS, width), dtype=np.float32),
        bundle.cache["anomaly_scorer"]["model"], batch_size, "cpu"
    )
    if parts.shape != (PROBE_ROWS, width) or not np.isfinite(parts).all():
        raise ValueError(
            f"0 の行の再構成が (件数, 列の数 {width}) の数の配列では"
            "ありません"
        )


def anomaly_section(sources: dict) -> dict:
    """
    成果物から一式の anomaly の欄を作る関数

    Parameters
    ----------
    sources: dict
        run_sources の結果

    Returns
    ----------
    anomaly: dict
        anomaly の欄 (check は None)
    """
    # 成果物の写し (中心と精度行列は anomaly_scorer.json の中だけ)
    scorer = sources["scorer"]
    model = scorer["model"] or {}
    anomaly = {
        "scorer": SCORER_FILE,
        "recipe": RECIPE_FILE,
        "score": scorer["score"],
        "features": list(scorer["features"]),
        "id": scorer["id"],
        "input": model.get("input"),
        "output": model.get("output"),
        "capacity": int(scorer["capacity"]),
        "k": int(scorer["k"]),
        "rows": int(len(sources["scores"])),
        "normal": {key: plain(scorer["normal"][key]) for key in NORMAL_KEYS},
        "evaluations": copy.deepcopy(sources["evaluations"]),
        "check": None,
    }

    return anomaly


def make_anomaly_bundle(
    path: Union[str, Path], run: Union[str, Path],
    name: Optional[str] = None, data: Any = None,
    options: Optional[dict] = None, copy_weights: bool = True,
    overwrite: bool = False, pretrained: Optional[dict] = None,
    batch_size: int = 256
) -> dict:
    """
    新しい行の異常の度合いを出す一式を, 異常の検知の実験から作る関数

    Parameters
    ----------
    path: Union[str, Path]
        作るディレクトリ
    run: Union[str, Path]
        ML-Found の実験のディレクトリ (anomaly_scorer.json,
        anomaly_recipe.yaml, anomaly_scores.csv, evaluations の
        anomaly_<分割>.json と, 自己符号化器は final_model.pt と
        model_spec.yaml を使い, 実験の状態を記録する)
    name: Optional[str] = None
        一式の名前 (省略するとディレクトリの名前)
    data: Any = None
        学習に使った元の表 (渡すと度合いを出して実験の度合いと比べる)
    options: Optional[dict] = None
        data のファイルの読み込みの設定 (load_data に渡す)
    copy_weights: bool = True
        自己符号化器の重みを一式に写すか (写さない場合は絶対パスで指す)
    overwrite: bool = False
        既にある一式を置き換えるか
    pretrained: Optional[dict] = None
        ノードの名前ごとの重みのライセンスの情報 (make_bundle と同じ)
    batch_size: int = 256
        1 回に渡す行の数

    Returns
    ----------
    info: dict
        書き出した inference.yaml の中身

    Raises
    ----------
    FileExistsError
        ディレクトリが既にあり, overwrite が False の場合と, 一式ではない
        ディレクトリを置き換えようとした場合
    FileNotFoundError
        成果物が無い場合
    ValueError
        定義, 重み, 成果物が約束に合わない場合
    """
    # 置き換えられるのは, 前に作った一式だけ
    final = Path(path).expanduser().resolve()
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"{final} は既にあります")
        if not (final / "inference.yaml").is_file():
            raise FileExistsError(f"{final} は推論の一式ではないので置き換えません")
    if batch_size < 1:
        raise ValueError("batch_size は 1 以上にしてください")

    # 実験の成果物と, 自己符号化器の入力と出力の約束
    sources = run_sources(run)
    scorer = sources["scorer"]
    autoencoder = scorer["score"] == "autoencoder"
    if not autoencoder and pretrained is not None:
        raise ValueError("距離に基づく度合いの一式には pretrained を渡せ"
                         "ません (モデルを持たない)")
    spec_dict: dict = {}
    records: list = []
    kept = None
    weight_notes: list = []
    if autoencoder:
        spec_dict = load_spec(sources["spec"])
        check_units(spec_dict)
        check_anomaly_spec(spec_dict, scorer["model"])
        spec_dict, records, kept, weight_notes = plan_weights(
            spec_dict, sources["weights"], pretrained, copy_weights
        )

    # 隣の一時ディレクトリに作り, 確かめてから入れ替え
    writing = final.with_name(f".{final.name}.writing")
    if writing.exists():
        shutil.rmtree(writing)
    writing.mkdir(parents=True)
    try:
        shutil.copy(sources["root"] / SCORER_FILE, writing / SCORER_FILE)
        shutil.copy(sources["recipe_file"], writing / RECIPE_FILE)
        model = None
        if autoencoder:
            dump_spec(spec_dict, writing / "model_spec.yaml")
            if copy_weights and kept is not None:
                torch.save(kept, writing / "final_model.pt")
            elif copy_weights:
                shutil.copy(sources["weights"], writing / "final_model.pt")
            model = {
                "spec": "model_spec.yaml",
                "weights": "final_model.pt" if copy_weights
                else str(sources["weights"]),
                "output": scorer["model"]["output"],
                "kind": "value",
                "target_scale": None,
                "pretrained": records,
            }
        empty = Dataset(path=writing, info={}, frames={}, recipe={})
        info = {
            "format_version": ANOMALY_BUNDLE_VERSION,
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "role": "predict",
            "output": {"form": "value"},
            "target": None,
            "procedure": None,
            "model": model,
            "inputs": {},
            "dataset": None,
            "run": sources["run"],
            "conclusion": {},
            "decision": {"type": "triage",
                         "capacity": int(scorer["capacity"])},
            "check": None,
            "uncertainty": None,
            "notes": bundle_notes(empty, sources["run"]) + weight_notes,
            "anomaly": anomaly_section(sources),
        }

        # 0 の行の再構成と, 元の表の度合いと実験の度合いの比べ
        building = Bundle(path=writing, info=info, spec=spec_dict,
                          dataset=empty)
        building.cache["anomaly_scorer"] = scorer
        building.cache["anomaly_dataset"] = anomaly_dataset(
            writing, sources["recipe"], scorer["features"]
        )
        if autoencoder:
            probe_autoencoder(building, len(scorer["features"]), batch_size)
        if data is not None:
            found, notes = score_check(building, data, options, sources,
                                       batch_size)
            info["anomaly"]["check"] = found
            info["notes"].extend(notes)
        (writing / "inference.yaml").write_text(
            yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
            encoding="utf-8"
        )
        load_bundle(writing)
    except Exception:
        shutil.rmtree(writing, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(final)
    writing.rename(final)

    return info


def check_anomaly_bundle(bundle: Bundle) -> None:
    """
    読んだ異常の度合いの一式の成果物を確かめ, 度合いに要るものを cache に
    置く関数

    Parameters
    ----------
    bundle: Bundle
        異常の度合いの一式

    Returns
    ----------
    None

    Raises
    ----------
    FileNotFoundError
        成果物の写しが無い場合
    ValueError
        成果物の形が違う場合と, 成果物と一式の欄が食い違う場合
    """
    # 度合いの決め方とレシピの写し (一式の外を指せない)
    anomaly = bundle.info["anomaly"]
    scorer = read_scorer(inside(bundle.path, anomaly["scorer"]))
    recipe = read_recipe(inside(bundle.path, anomaly["recipe"]))
    model = scorer["model"] or {}
    mismatch = [
        key for key, value in (
            ("score", scorer["score"]), ("features", scorer["features"]),
            ("id", scorer["id"]), ("input", model.get("input")),
            ("output", model.get("output")),
        ) if anomaly[key] != value
    ]
    if mismatch:
        raise ValueError(
            f"anomaly の {', '.join(mismatch)} が度合いの決め方 "
            f"{anomaly['scorer']} と違います"
        )
    bundle.cache["anomaly_scorer"] = scorer
    bundle.cache["anomaly_dataset"] = anomaly_dataset(
        bundle.path, recipe, scorer["features"]
    )

    # 自己符号化器の定義の入力と出力
    if scorer["score"] == "autoencoder":
        check_anomaly_spec(bundle.spec, scorer["model"])


def distribution_of(scores: np.ndarray) -> Optional[dict]:
    """
    度合いの分位点をまとめる関数

    Parameters
    ----------
    scores: np.ndarray
        予測した行の度合い

    Returns
    ----------
    distribution: Optional[dict]
        p50, p90, p99, max (行が無ければ None)
    """
    # 分位点は線形の補間 (骨組みの normal と同じ)
    if not len(scores):
        return None
    distribution = {f"p{point}": float(np.percentile(scores, point))
                    for point in (50, 90, 99)}
    distribution["max"] = float(scores.max())

    return distribution


def value_of(value: Any) -> Any:
    """
    表の値を result.yaml に書ける値にする関数

    Parameters
    ----------
    value: Any
        表の値

    Returns
    ----------
    result: Any
        欠けは None, 真偽値と数はそのまま, 他は文字
    """
    # numpy の値は Python の値に
    value = plain(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    result = value if isinstance(value, (bool, int, float)) else str(value)

    return result


def top_rows(
    bundle: Bundle, prediction: pd.DataFrame, key: Optional[str]
) -> list[dict]:
    """
    度合いの高い順の上位 K 行の要約を作る関数

    Parameters
    ----------
    bundle: Bundle
        異常の度合いの一式 (anomaly.k)
    prediction: pd.DataFrame
        predict_anomaly の表
    key: Optional[str]
        行の id の列 (無ければ None)

    Returns
    ----------
    rows: list[dict]
        row (表の行の位置), id, score, column, column_share
    """
    # 予測した行の順位の順
    ranked = prediction[prediction["status"] == OK].sort_values(
        "rank", kind="stable"
    ).head(int(bundle.info["anomaly"]["k"]))
    rows = []
    for position, row in ranked.iterrows():
        share = row["column_share"]
        rows.append({
            "row": int(position),
            "id": None if key is None else value_of(row[key]),
            "score": float(row["score"]),
            "column": row["column"],
            "column_share": None if pd.isna(share) else float(share),
        })

    return rows


def evaluation_sentences(evaluations: dict) -> list[str]:
    """
    実験の評価を述べる文を返す関数

    Parameters
    ----------
    evaluations: dict
        分割の名前から評価の行

    Returns
    ----------
    sentences: list[str]
        分割ごとの上位 K 件の適合率と PR-AUC の文 (無ければ, 人が見て
        確かめる文)
    """
    # 正解の無い実験は, 分布と上位の行を人が見る文
    if not evaluations:
        sentences = ["異常の正解の無い実験なので, 度合いの分布と上位の行を"
                     "人が見て確かめる"]
        return sentences
    sentences = [
        f"{SPLIT_NAMES.get(split, split)}の上位 {row['k']} 件の適合率 "
        f"{row['precision_at_k']:.3f}, PR-AUC {row['average_precision']:.3f}"
        f" (異常 {row['anomalies']} 行 / {row['rows']} 行)"
        for split, row in evaluations.items()
    ]

    return sentences


def conclude_anomaly(bundle: Bundle, prediction: pd.DataFrame) -> dict:
    """
    行ごとの度合いから, 件数と度合いの要約の結論を出す関数

    Parameters
    ----------
    bundle: Bundle
        異常の度合いの一式
    prediction: pd.DataFrame
        predict_anomaly の表

    Returns
    ----------
    conclusion: dict
        frame (decision を足した表. 予測した行は空, しなかった行は予測
        なし), summary (rows, predicted, skipped, score, id,
        distribution, normal, beyond_normal, top, evaluations),
        sentences, rules (空)
    """
    # 予測したかと, 度合いの分布
    anomaly = bundle.info["anomaly"]
    frame = prediction.copy()
    ok = (frame["status"] == OK).to_numpy()
    frame["decision"] = np.where(ok, "", NO_PREDICTION)
    scores = frame["score"].to_numpy(dtype=np.float64)[ok]
    normal = anomaly["normal"]
    key = id_column(bundle, prediction)
    summary = {
        "rows": int(len(frame)), "predicted": int(ok.sum()),
        "skipped": int(len(frame) - ok.sum()), "score": anomaly["score"],
        "id": key, "distribution": distribution_of(scores),
        "normal": copy.deepcopy(normal),
        "beyond_normal": int((scores > normal["p99"]).sum()),
        "top": top_rows(bundle, prediction, key),
        "evaluations": copy.deepcopy(anomaly["evaluations"]),
    }

    # 件数, 分布と学習用の正常の行の 99 % 点, 実験の評価の文
    head = (f"{summary['rows']:,} 行のうち {summary['predicted']:,} 行の"
            f"異常の度合い ({SCORE_NAMES[anomaly['score']]}) を出した")
    if summary["skipped"]:
        head += f" ({summary['skipped']:,} 行は出せず)"
    sentences = [head]
    found = summary["distribution"]
    if found is not None:
        share = summary["beyond_normal"] / summary["predicted"] * 100
        sentences.append(
            f"度合いの中央 {found['p50']:.3g}, 90 % 点 {found['p90']:.3g}, "
            f"99 % 点 {found['p99']:.3g}, 最大 {found['max']:.3g}. 学習用の"
            f"正常の行の 99 % 点 {normal['p99']:.3g} を超えた行は "
            f"{summary['beyond_normal']:,} 行 ({share:.1f} %)"
        )
    sentences.extend(evaluation_sentences(anomaly["evaluations"]))
    conclusion = {"frame": frame, "summary": summary,
                  "sentences": sentences, "rules": {}}

    return conclusion


def check_anomaly_decision(decision: dict) -> list[str]:
    """
    異常の度合いの一式の振り分けの決め方の誤りを一覧で返す関数

    Parameters
    ----------
    decision: dict
        意思決定の決め方

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 型は振り分けだけ, 項目は type と capacity だけ
    if decision.get("type") != "triage":
        return ["異常の度合いの一式の意思決定は振り分け (type: triage) "
                "だけです"]
    errors = [
        f"異常の度合いの一式の振り分けに書けない項目です: {key} (書けるの"
        "は type と capacity だけ. 度合いは確率でないので損の値は使わない)"
        for key in decision if key not in TRIAGE_KEYS
    ]
    if not is_whole(decision.get("capacity"), 1):
        errors.append("capacity (人が 1 回に確かめる件数の上限) は 1 以上の"
                      "整数にしてください")

    return errors


def decide_anomaly(
    bundle: Bundle, prediction: pd.DataFrame, decision: dict
) -> dict:
    """
    度合いの高い順に, 件数の上限まで人の確認に回す行を決める関数

    Parameters
    ----------
    bundle: Bundle
        異常の度合いの一式
    prediction: pd.DataFrame
        predict_anomaly の表 (か conclude_anomaly の表)
    decision: dict
        {type: triage, capacity}

    Returns
    ----------
    result: dict
        frame (action を足した表), summary (rows, acted, capacity,
        lowest, normal_p99, left_beyond), sentences, notes (前提),
        decision (使った決め方)

    Raises
    ----------
    ValueError
        決め方に誤りがある場合と, 表に度合いの列が無い場合
    """
    # 決め方と表の確認
    decision = plain(dict(decision))
    errors = check_anomaly_decision(decision)
    missing = [c for c in ("status", "score") if c not in prediction.columns]
    if missing:
        errors.append(f"予測の表に {', '.join(missing)} の列がありません")
    if errors:
        raise ValueError("意思決定の決め方の誤り: " + " / ".join(errors))

    # 予測した行の度合いの高い順 (同じ度合いは前の行が先) に上限まで
    capacity = int(decision["capacity"])
    ok = (prediction["status"] == OK).to_numpy()
    scores = prediction["score"].to_numpy(dtype=np.float64)
    rows = np.flatnonzero(ok)
    order = rows[np.argsort(-scores[rows], kind="stable")]
    chosen = order[:capacity]
    action = pd.array([None] * len(prediction), dtype="boolean")
    action[rows] = False
    action[chosen] = True
    p99 = float(bundle.info["anomaly"]["normal"]["p99"])
    left = order[capacity:]
    summary = {
        "rows": int(len(rows)), "acted": int(len(chosen)),
        "capacity": capacity,
        "lowest": float(scores[chosen].min()) if len(chosen) else None,
        "normal_p99": p99,
        "left_beyond": int((scores[left] > p99).sum()),
    }
    frame = prediction.copy()
    if ACTION_COLUMN in frame.columns:
        frame = frame.rename(columns={ACTION_COLUMN: f"{ACTION_COLUMN} (入力)"})
    frame[ACTION_COLUMN] = action

    # 回す件数と度合いの下限, 上限で回さない外れた行の文と前提
    sentence = (f"度合いの高い順に {summary['acted']:,} 件を人の確認に回す "
                f"(人が 1 回に確かめる件数の上限 {capacity:,} 件)")
    if summary["lowest"] is not None:
        sentence += f". 回す行の度合いは {summary['lowest']:.3g} 以上"
    sentences = [sentence]
    if summary["left_beyond"]:
        sentences.append(
            f"学習用の正常の行の 99 % 点 ({p99:.3g}) を超えたのに上限で回さ"
            f"ない行が {summary['left_beyond']:,} 行ある"
        )
    notes = [
        "度合いは確率ではないので, 回す件数は損得のしきい値でなく人手の上限"
        "で決めた",
        "学習用の正常の行の分布から外れた度合いは異常の目安で, 異常かは人の"
        "確認で決める (機械学習の結果は判断の材料)",
    ]
    result = {"frame": frame, "summary": summary, "sentences": sentences,
              "notes": notes, "decision": decision}

    return result


def anomaly_of_bundle(bundle: Bundle) -> Optional[dict]:
    """
    異常の度合いの一式の anomaly の欄を, 説明の形で返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    anomaly: Optional[dict]
        anomaly の欄の写し (異常の度合いの一式でなければ None)
    """
    # 画面が度合いの読み方と実験の評価を出すための写し
    if not bundle.from_anomaly:
        return None
    anomaly = copy.deepcopy(bundle.info["anomaly"])

    return anomaly
