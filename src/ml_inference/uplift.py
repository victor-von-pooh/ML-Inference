import copy
import datetime as dt
from decimal import Decimal
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

from ml_inference.anomaly import (
    ABSOLUTE_GAP, AGREEMENT, OK, RELATIVE_GAP, SKIPPED, is_whole,
    refuse_constant, row_features, value_of
)
from ml_inference.bundle import (
    SINGLE_UPLIFT_BUNDLE_VERSION, UPLIFT_BUNDLE_VERSION, Bundle,
    bundle_notes, inside, load_bundle, plan_weights, run_summary
)
from ml_inference.conclude import NO_PREDICTION, is_number, plain
from ml_inference.grouping import read_table
from ml_inference.units import check_units

# 実験の中の成果物 (ML-Quest の効果の見積もりの骨組みが書く名前)
ESTIMATOR_FILE = "uplift_estimator.json"
RECIPE_FILE = "uplift_recipe.yaml"
EFFECTS_FILE = "uplift_effects.csv"
EVALUATION_FILE = "evaluations/uplift_{split}.json"
CURVE_FILE = "evaluations/qini_curve_{split}.json"

# 読める効果の決め方の成果物の版と, 効果の決め方 (t_learner は処置の群と
# 対照の群の 2 つの予測, s_learner は処置の列を入力の最後に足した 1 つの
# 予測)
ESTIMATOR_VERSION = 1
LEARNERS = ("t_learner", "s_learner")
SINGLE = "s_learner"

# 効果の決め方ごとの一式の版
LEARNER_VERSIONS = {"t_learner": UPLIFT_BUNDLE_VERSION,
                    SINGLE: SINGLE_UPLIFT_BUNDLE_VERSION}

# 結果の種類と, 結果の種類ごとの出力を結果の予測にする決まりと, 効果の
# 呼び名
OUTCOME_LINKS = {"binary": "sigmoid", "amount": "identity"}
EFFECT_NAMES = {"binary": "確率の差", "amount": "値の差"}

# 効果の決め方の成果物の鍵と, 効果の決め方ごとのモデルの出力の名前の鍵と,
# 学習用の行の分布の鍵
ESTIMATOR_KEYS = ("format_version", "learner", "outcome", "recipe",
                  "effects", "features", "id", "treatment", "budget", "k",
                  "model", "scale", "train")
OUTPUT_NAMES = {"t_learner": ("treated", "control"), SINGLE: ("outcome",)}
TRAIN_KEYS = ("rows", "mean", "min", "p10", "p50", "p90", "max",
              "positive")

# 行ごとの効果の表の, 行の id の列の他の列
EFFECTS_COLUMNS = ("split", "effect", "treated", "control", "treatment",
                   "outcome")

# 実験の評価を写す分割と, 評価の行の鍵 (uplift_at_k は空にもなる)
EVALUATION_SPLITS = ("valid", "test")
EVALUATION_KEYS = ("rows", "treated", "control", "k", "qini", "effect",
                   "estimated")

# Qini の曲線の列 (ML-Found の qini_curve の列)
CURVE_COLUMNS = ("rows", "fraction", "qini", "random")

# 分割の日本語の名前
SPLIT_NAMES = {"valid": "検証用", "test": "テスト用"}

# uplift の欄 (モデルの出力の名前の鍵は効果の決め方ごと. s_learner の
# output は成果物の model.outcome)
UPLIFT_HEAD = ("estimator", "recipe", "learner", "outcome", "features",
               "id", "treatment", "input")
UPLIFT_OUTPUTS = {"t_learner": ("treated", "control"), SINGLE: ("output",)}
UPLIFT_TAIL = ("budget", "k", "rows", "train", "evaluations", "curves",
               "check")

# 一式を作るときに 2 つの予測を確かめる 0 の行の数
PROBE_ROWS = 2

# 予測の表に足す列
UPLIFT_FRAME_COLUMNS = ("status", "reason", "effect", "treated", "control",
                        "rank")

# 効果が数でない行の理由
NOT_FINITE = "効果が数ではありません"

# 結論に並べる効果の高い行の数
TOP_ROWS = 10

# 配る判断で書ける項目と, 足す列
TREAT_KEYS = ("type", "budget")
ACTION_COLUMN = "action"


def is_share(value: Any) -> bool:
    """
    0 より大きく 1 以下の数 (真偽値を除く) かを確かめる関数

    Parameters
    ----------
    value: Any
        確かめる値

    Returns
    ----------
    result: bool
        0 より大きく 1 以下の数なら True
    """
    # 上位 K 割の K
    result = is_number(value) and 0 < float(value) <= 1

    return result


def check_scale(outcome: Any, scale: Any) -> list[str]:
    """
    結果の種類と標準化の値の食い違いを一覧で返す関数

    Parameters
    ----------
    outcome: Any
        結果の種類
    scale: Any
        成果物の scale

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 0 と 1 は null, 量は有限の mean と正の std
    if outcome == "binary":
        errors = [] if scale is None else [
            "結果が 0 と 1 の scale は null にしてください"
        ]
        return errors
    good = isinstance(scale, dict) and all(
        is_number(scale.get(key)) and math.isfinite(scale[key])
        for key in ("mean", "std")
    ) and scale["std"] > 0
    errors = [] if good else [
        "量の結果の scale は有限の mean と 0 より大きい std にしてください"
    ]

    return errors


def check_estimator(estimator: Any) -> list[str]:
    """
    効果の決め方の成果物 (uplift_estimator.json) の誤りを一覧で返す関数

    Parameters
    ----------
    estimator: Any
        読み込んだ成果物

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 版, 鍵, 決め方, 結果の種類
    if not isinstance(estimator, dict):
        return ["効果の決め方が辞書ではありません"]
    version = estimator.get("format_version")
    if version != ESTIMATOR_VERSION:
        return [f"効果の決め方の版 {version} は読めません (版 "
                f"{ESTIMATOR_VERSION} だけ)"]
    errors = [f"効果の決め方に {key} がありません"
              for key in ESTIMATOR_KEYS if key not in estimator]
    if errors:
        return errors
    if estimator["learner"] not in LEARNERS:
        errors.append(f"learner は {', '.join(LEARNERS)} にしてください")
    outcome = estimator["outcome"]
    if not isinstance(outcome, str) or outcome not in OUTCOME_LINKS:
        errors.append(f"outcome は {', '.join(OUTCOME_LINKS)} のどちらか"
                      "にしてください")

    # 列, 行の id と処置の列, ファイルの名前, 配る数の上限と K
    features = estimator["features"]
    if not isinstance(features, list) or not features or not all(
        isinstance(name, str) and name for name in features
    ) or len(set(features)) != len(features):
        errors.append("features は重ならない列の名前の並びにしてください")
    for key in ("id", "treatment"):
        if not isinstance(estimator[key], str) or not estimator[key]:
            errors.append(f"{key} は列の名前にしてください")
    for key in ("recipe", "effects"):
        if not isinstance(estimator[key], str) or not estimator[key]:
            errors.append(f"{key} はファイルの名前にしてください")
    budget = estimator["budget"]
    if budget is not None and not is_whole(budget, 1):
        errors.append("budget は null か 1 以上の整数にしてください")
    if not is_share(estimator["k"]):
        errors.append("k は 0 より大きく 1 以下の数にしてください")

    # モデルの名前 (効果の決め方ごと) と結果の予測にする決まり, 標準化,
    # 学習用の行の分布
    model = estimator["model"]
    learner = estimator["learner"]
    keys = ("input", *OUTPUT_NAMES[learner]) if learner in LEARNERS else ()
    if keys and (not isinstance(model, dict) or not all(
        isinstance(model.get(key), str) and model[key] for key in keys
    )):
        errors.append(f"learner が {learner} の model は "
                      f"{', '.join(keys)} の名前にしてください")
    elif isinstance(model, dict) and isinstance(outcome, str) and (
        outcome in OUTCOME_LINKS
    ) and (
        model.get("link") != OUTCOME_LINKS[outcome]
    ):
        errors.append(f"結果の種類 {outcome} の model.link は "
                      f"{OUTCOME_LINKS[outcome]} にしてください")
    if isinstance(outcome, str) and outcome in OUTCOME_LINKS:
        errors.extend(check_scale(outcome, estimator["scale"]))
    train = estimator["train"]
    if not isinstance(train, dict) or not all(
        is_number(train.get(key)) and math.isfinite(train[key])
        for key in TRAIN_KEYS
    ):
        errors.append(f"train は {', '.join(TRAIN_KEYS)} の数にして"
                      "ください")

    return errors


def read_estimator(path: Path) -> dict:
    """
    効果の決め方の成果物を読み, 形を確かめる関数

    Parameters
    ----------
    path: Path
        uplift_estimator.json

    Returns
    ----------
    estimator: dict
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
        raise FileNotFoundError(f"効果の決め方 {path} がありません")
    try:
        estimator = json.loads(path.read_text(encoding="utf-8"),
                               parse_constant=refuse_constant)
    except ValueError as error:
        raise ValueError(f"{path} を読めません ({error})") from error
    errors = check_estimator(estimator)
    if errors:
        raise ValueError(f"{path} の誤り: " + " / ".join(errors))

    return estimator


def read_recipe(path: Path) -> dict:
    """
    前処理のレシピの写しを読み, 介入の記録の表のレシピかを確かめる関数

    Parameters
    ----------
    path: Path
        uplift_recipe.yaml

    Returns
    ----------
    recipe: dict
        当てはめた値を含むレシピ

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        介入の記録の表のレシピでない場合
    """
    # 手順の並びを持つ, task が uplift のレシピ
    if not path.is_file():
        raise FileNotFoundError(f"前処理のレシピ {path} がありません")
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or recipe.get("task") != "uplift" or (
        not isinstance(recipe.get("steps"), list)
    ):
        raise ValueError(
            f"{path} は介入の記録の表のレシピ (task: uplift と steps) では"
            "ありません"
        )

    return recipe


def read_effects(path: Path, key: str) -> pd.DataFrame:
    """
    実験の行ごとの効果の表を読む関数

    Parameters
    ----------
    path: Path
        uplift_effects.csv
    key: str
        行の id の列の名前

    Returns
    ----------
    effects: pd.DataFrame
        行の id の列 (文字), split, effect, treated, control, treatment,
        outcome の表

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        列が違う場合と, 効果が数でない場合
    """
    # 行の id は文字 (先頭の 0 などを残すため)
    if not path.is_file():
        raise FileNotFoundError(f"行ごとの効果の表 {path} がありません")
    effects = pd.read_csv(path, dtype={key: str}, keep_default_na=False,
                          na_values=[""])
    expected = [key, *EFFECTS_COLUMNS]
    if list(effects.columns) != expected:
        raise ValueError(
            f"{path} の列は {', '.join(expected)} にしてください (今は "
            f"{', '.join(map(str, effects.columns))})"
        )
    values = pd.to_numeric(effects["effect"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path} の effect に数でない値があります")
    effects["effect"] = values.astype(np.float64)

    return effects


def read_json(path: Path) -> Any:
    """
    評価のファイルを読む関数

    Parameters
    ----------
    path: Path
        evaluations の中の JSON のファイル

    Returns
    ----------
    table: Any
        読んだ中身

    Raises
    ----------
    ValueError
        JSON でない場合と, JSON の外の NaN と Infinity がある場合
    """
    # JSON の外の NaN と Infinity も誤り (他の成果物の読み込みと同じ)
    try:
        table = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=refuse_constant)
    except ValueError as error:
        raise ValueError(f"{path} を読めません ({error})") from error

    return table


def read_evaluations(root: Path) -> dict:
    """
    実験の Qini の係数と上位 K 割の効果の差の評価を読む関数

    Parameters
    ----------
    root: Path
        実験のディレクトリ

    Returns
    ----------
    evaluations: dict
        分割の名前から rows, treated, control, k, qini, uplift_at_k
        (無ければ None), effect, estimated (評価の無い分割は無い)
    """
    # ML-Found の evaluations の形の最初の行 (形の違う評価は写さずに除外)
    evaluations = {}
    for split in EVALUATION_SPLITS:
        path = root / EVALUATION_FILE.format(split=split)
        if not path.is_file():
            continue
        table = read_json(path)
        rows = table.get("rows") if isinstance(table, dict) else None
        row = rows[0] if isinstance(rows, list) and rows else None
        if not isinstance(row, dict) or not all(
            is_number(row.get(key)) for key in EVALUATION_KEYS
        ):
            continue
        lift = row.get("uplift_at_k")
        if lift is not None and not is_number(lift):
            continue
        found = {key: plain(row[key]) for key in EVALUATION_KEYS[:5]}
        found["uplift_at_k"] = plain(lift)
        found.update({key: plain(row[key]) for key in EVALUATION_KEYS[5:]})
        evaluations[split] = found

    return evaluations


def read_curves(root: Path) -> dict:
    """
    実験の Qini の曲線を読む関数

    Parameters
    ----------
    root: Path
        実験のディレクトリ

    Returns
    ----------
    curves: dict
        分割の名前から, rows, fraction, qini, random の値の並び (曲線の
        無い分割は無い)
    """
    # ML-Found の evaluations の曲線の形 (形の違う曲線は写さずに除外)
    curves = {}
    for split in EVALUATION_SPLITS:
        path = root / CURVE_FILE.format(split=split)
        if not path.is_file():
            continue
        table = read_json(path)
        rows = table.get("rows") if isinstance(table, dict) else None
        if not isinstance(rows, list) or not rows or not all(
            isinstance(row, dict) and all(
                is_number(row.get(key)) for key in CURVE_COLUMNS
            ) for row in rows
        ):
            continue
        curves[split] = {key: [plain(row[key]) for row in rows]
                         for key in CURVE_COLUMNS}

    return curves


def run_sources(run: Union[str, Path]) -> dict:
    """
    効果の見積もりの実験のディレクトリから, 一式に要るものを読む関数

    Parameters
    ----------
    run: Union[str, Path]
        ML-Found の実験のディレクトリ

    Returns
    ----------
    sources: dict
        root, estimator, recipe_file, recipe, effects, evaluations,
        curves, weights, spec, run (実験の状態)

    Raises
    ----------
    ValueError
        実験のディレクトリでない場合, 重みか定義が無い場合, 成果物の形が
        違う場合
    FileNotFoundError
        成果物が無い場合
    """
    # 効果の決め方, レシピ, 行ごとの効果, 評価
    root = Path(run).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"実験のディレクトリ {root} がありません")
    estimator = read_estimator(root / ESTIMATOR_FILE)
    recipe_file = inside(root, estimator["recipe"])
    recipe = read_recipe(recipe_file)
    effects = read_effects(inside(root, estimator["effects"]),
                           estimator["id"])

    # 効果のモデルの重みと定義 (仮のモデルで学習した実験は定義無し)
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
        "root": root, "estimator": estimator, "recipe_file": recipe_file,
        "recipe": recipe, "effects": effects,
        "evaluations": read_evaluations(root),
        "curves": read_curves(root), "weights": weights,
        "spec": spec, "run": run_summary(root),
    }

    return sources


def uplift_keys(learner: str) -> tuple:
    """
    効果の決め方ごとの uplift の欄の鍵を返す関数

    Parameters
    ----------
    learner: str
        効果の決め方 (t_learner か s_learner)

    Returns
    ----------
    keys: tuple
        uplift の欄の鍵の並び
    """
    # 共通の鍵の間にモデルの出力の名前の鍵
    keys = (*UPLIFT_HEAD, *UPLIFT_OUTPUTS[learner], *UPLIFT_TAIL)

    return keys


def check_uplift_info(info: dict) -> list[str]:
    """
    版 12 と 18 の一式の uplift の欄と, 受け持ちと出力の形を確かめる関数

    Parameters
    ----------
    info: dict
        inference.yaml の中身 (format_version は 12 か 18)

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 版ごとの効果の決め方と欄, 受け持ち, 出力の形
    version = info.get("format_version")
    learner = next(key for key, value in LEARNER_VERSIONS.items()
                   if value == version)
    uplift = info.get("uplift")
    if not isinstance(uplift, dict):
        return [f"版 {version} の一式には uplift が要ります"]
    keys = uplift_keys(learner)
    errors = [f"uplift に {key} がありません"
              for key in keys if key not in uplift]
    unknown = [str(key) for key in uplift if key not in keys]
    if unknown:
        errors.append(f"uplift に知らない項目があります: "
                      f"{', '.join(unknown)}")
    if info.get("role") != "effect":
        errors.append(f"版 {version} の一式の受け持ちは effect (効果を"
                      "見積もる) にしてください")
    if errors:
        return errors
    if uplift["learner"] != learner:
        errors.append(f"版 {version} の一式の uplift.learner は {learner} "
                      "にしてください")

    # 結果の種類, ファイルの名前, 配る数の上限と K
    if not isinstance(uplift["outcome"], str) or uplift["outcome"] not in (
        OUTCOME_LINKS
    ):
        errors.append(f"uplift.outcome は {', '.join(OUTCOME_LINKS)} の"
                      "どちらかにしてください")
    for key in ("estimator", "recipe"):
        if not isinstance(uplift[key], str) or not uplift[key]:
            errors.append(f"uplift.{key} はファイルの名前にしてください")
    if uplift["budget"] is not None and not is_whole(uplift["budget"], 1):
        errors.append("uplift.budget は null か 1 以上の整数にしてください")
    if not is_share(uplift["k"]):
        errors.append("uplift.k は 0 より大きく 1 以下の数にしてください")

    return errors


def input_width(estimator: dict) -> int:
    """
    効果のモデルの入力の幅を返す関数

    Parameters
    ----------
    estimator: dict
        効果の決め方 (learner, features)

    Returns
    ----------
    width: int
        見積もりに使う列の数 (s_learner は処置の列の 1 を足した数)
    """
    # s_learner は入力の最後に処置の列
    width = len(estimator["features"]) + (
        1 if estimator["learner"] == SINGLE else 0
    )

    return width


def check_uplift_spec(spec: dict, estimator: dict) -> None:
    """
    効果のモデルの定義の入力と出力が約束に合うかを確かめる関数

    Parameters
    ----------
    spec: dict
        モデルの定義 (ML-ModelControl の形)
    estimator: dict
        効果の決め方 (learner, features, model)

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        入力が成果物の入力の名前の 1 つでない場合と, 入力の形の最後の
        大きさが入力の幅と違う場合と, 出力の名前 (t_learner は 2 つ,
        s_learner は 1 つ) が無い場合
    """
    # 入力は行の特徴 (s_learner は最後に処置の列) の 1 つ
    names = estimator["model"]
    inputs = spec.get("inputs") or {}
    if list(inputs) != [names["input"]]:
        raise ValueError(
            f"効果のモデルの入力は {names['input']} の 1 つにしてください "
            f"(今は {', '.join(map(str, inputs)) or '無し'})"
        )
    item = inputs[names["input"]]
    shape = item.get("shape") if isinstance(item, dict) else None
    width = input_width(estimator)
    if isinstance(shape, list) and shape and is_whole(shape[-1], 1) and (
        shape[-1] != width
    ):
        what = ("見積もりに使う列の数と処置の列の 1 の和"
                if estimator["learner"] == SINGLE else "見積もりに使う列の数")
        raise ValueError(
            f"効果のモデルの入力 {names['input']} の幅 {shape[-1]} が, "
            f"{what} {width} と違います"
        )

    # 出力に処置したときとしないときの予測 (s_learner は結果の予測)
    outputs = spec.get("outputs") or {}
    lost = [names[key] for key in OUTPUT_NAMES[estimator["learner"]]
            if names[key] not in outputs]
    if lost:
        raise ValueError(
            f"効果のモデルの出力に {', '.join(lost)} がありません (出力: "
            f"{', '.join(map(str, outputs)) or '無し'})"
        )


def uplift_dataset(
    root: Path, recipe: dict, estimator: dict
) -> Dataset:
    """
    前処理だけの ML-EDA のデータセットを作る関数

    Parameters
    ----------
    root: Path
        一式 (か実験) のディレクトリ
    recipe: dict
        当てはめた値を含むレシピ
    estimator: dict
        効果の決め方 (features, id, treatment, outcome)

    Returns
    ----------
    dataset: Dataset
        分割の表を持たないデータセット (transform だけに使う)
    """
    # 結果の列と処置の列は前処理の手順の外
    info = {"kind": "table", "task": "uplift",
            "target": recipe.get("target"),
            "features": [{"name": name} for name in estimator["features"]],
            "uplift": {"id": estimator["id"],
                       "treatment": estimator["treatment"],
                       "outcome": estimator["outcome"]}}
    dataset = Dataset(path=root, info=info, frames={}, recipe=recipe)

    return dataset


def chunk_outputs(model: Any, chunk: torch.Tensor, estimator: dict) -> dict:
    """
    まとまりの行の特徴をモデルに渡し, 2 つの群の予測を返す関数

    Parameters
    ----------
    model: Any
        効果のモデル
    chunk: torch.Tensor
        (件数, 見積もりに使う列の数) の float32 の行の特徴 (処置の列は
        入れない)
    estimator: dict
        効果の決め方 (learner, model)

    Returns
    ----------
    found: dict
        treated, control から (誤りの文の予測の呼び名, 予測のテンソル) の
        組 (呼び名は t_learner は出力の名前, s_learner は出力の名前と処置
        の列の値)

    Raises
    ----------
    ValueError
        t_learner の出力が名前の辞書でない場合
    """
    # t_learner は出力の名前の辞書の 2 つの予測
    names = estimator["model"]
    if estimator["learner"] != SINGLE:
        outputs = model(**{names["input"]: chunk})
        if not isinstance(outputs, dict):
            raise ValueError("効果のモデルの出力が名前の辞書ではありません")
        found = {key: (names[key], outputs[names[key]])
                 for key in ("treated", "control")}
        return found

    # s_learner は処置の列を 1 と 0 にして入力の最後に足した 2 回の予測
    # (出力が 1 つの定義はテンソル. 骨組みと同じ)
    found = {}
    for key, value in (("treated", 1.0), ("control", 0.0)):
        column = torch.full((len(chunk), 1), value, dtype=chunk.dtype,
                            device=chunk.device)
        outputs = model(**{names["input"]: torch.cat([chunk, column],
                                                     dim=1)})
        if isinstance(outputs, dict):
            outputs = outputs[names["outcome"]]
        found[key] = (f"{names['outcome']} (処置の列が {int(value)})",
                      outputs)

    return found


def raw_outputs(
    model: Any, x: np.ndarray, estimator: dict, batch_size: int,
    device: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    行の特徴をモデルに渡し, 2 つの群の予測を, まとまりごとに求める関数

    Parameters
    ----------
    model: Any
        効果のモデル (t_learner は 2 つの予測, s_learner は処置の列を
        入力の最後に足した 1 つの予測)
    x: np.ndarray
        (件数, 見積もりに使う列の数) の float32 の行の特徴 (処置の列は
        入れない)
    estimator: dict
        効果の決め方 (learner, model)
    batch_size: int
        1 回に渡す行の数
    device: str
        使うデバイス

    Returns
    ----------
    found: tuple[np.ndarray, np.ndarray]
        処置したときとしないときの (件数,) の float64 の予測 (0 と 1 なら
        ロジット, 量なら標準化した値)

    Raises
    ----------
    ValueError
        t_learner の出力が名前の辞書でない場合と, 予測の形が (件数,) か
        (件数, 1) でない場合
    """
    # 2 つの予測 (float32 から float64 へ. 骨組みと同じ)
    pairs: dict = {"treated": [], "control": []}
    model = model.to(device)
    try:
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                chunk = torch.tensor(x[start:start + batch_size]).to(device)
                found = chunk_outputs(model, chunk, estimator)
                for key in pairs:
                    label, value = found[key]
                    if tuple(value.shape) not in ((len(chunk),),
                                                  (len(chunk), 1)):
                        raise ValueError(
                            f"予測 {label} の形 {tuple(value.shape)} "
                            f"が ({len(chunk)},) か ({len(chunk)}, 1) では"
                            "ありません"
                        )
                    pairs[key].append(value.reshape(-1).cpu())
    finally:
        model.to("cpu")
    treated, control = (
        torch.cat(pairs[key]).double().numpy() if pairs[key]
        else np.zeros(0) for key in ("treated", "control")
    )
    found = (treated, control)

    return found


def estimate(
    bundle: Bundle, x: np.ndarray, batch_size: int, device: str
) -> dict:
    """
    一式の効果の決め方で, 行ごとの 2 つの群の予測と効果を求める関数

    Parameters
    ----------
    bundle: Bundle
        効果の一式
    x: np.ndarray
        (件数, 列の数) の float32 の行の特徴
    batch_size: int
        1 回に渡す行の数
    device: str
        使うデバイス

    Returns
    ----------
    found: dict
        treated, control (結果の予測. 0 と 1 は確率, 量は元の単位),
        effect (treated − control) の (件数,) の float64 の配列
    """
    # 0 と 1 はシグモイドで確率にしてから, 量は元の単位に戻してからの差
    estimator = bundle.cache["uplift_estimator"]
    treated, control = raw_outputs(bundle.model(), x, estimator,
                                   batch_size, device)
    if estimator["model"]["link"] == "sigmoid":
        treated = torch.sigmoid(torch.tensor(treated)).numpy()
        control = torch.sigmoid(torch.tensor(control)).numpy()
    else:
        mean = float(estimator["scale"]["mean"])
        std = float(estimator["scale"]["std"])
        treated, control = treated * std + mean, control * std + mean
    found = {"treated": treated, "control": control,
             "effect": treated - control}

    return found


def frame_name(name: str) -> str:
    """
    元の列の, 予測の表での名前を返す関数

    Parameters
    ----------
    name: str
        元の列の名前

    Returns
    ----------
    found: str
        予測の列と同じ名前なら「(入力)」付き, 他はそのまま
    """
    # 予測の列と同じ名前の元の列の「(入力)」付きの名前
    found = f"{name} (入力)" if name in UPLIFT_FRAME_COLUMNS else name

    return found


def predict_uplift(
    bundle: Bundle, source: Any, options: Optional[dict] = None,
    batch_size: int = 256, device: str = "cpu"
) -> pd.DataFrame:
    """
    新しい行の効果と, 処置したときとしないときの結果の予測を返す関数

    Parameters
    ----------
    bundle: Bundle
        効果の一式
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
        effect, treated, control, rank (Int64)

    Raises
    ----------
    ValueError
        表でない場合と, 前処理に要る列が無い場合
    """
    # 表の読み込みと前処理, 出せる行だけの効果
    frame = read_table(source, options, "効果を見積もる表")
    x, reasons = row_features(bundle.cache["uplift_dataset"], frame)
    values = {key: np.full(len(x), math.nan)
              for key in ("effect", "treated", "control")}
    keep = np.flatnonzero([not r for r in reasons])
    if len(keep):
        found = estimate(bundle, x[keep], batch_size, device)
        finite = np.isfinite(found["treated"]) & np.isfinite(
            found["control"]
        ) & np.isfinite(found["effect"])
        for key in values:
            values[key][keep[finite]] = found[key][finite]
        for row in keep[~finite]:
            reasons[row] = NOT_FINITE

    # 効果の高い順の順位 (同じ効果は前の行が先)
    ok = np.array([not r for r in reasons], dtype=bool)
    ranks: list = [None] * len(x)
    rows = np.flatnonzero(ok)
    order = rows[np.argsort(-values["effect"][rows], kind="stable")]
    for rank, row in enumerate(order, 1):
        ranks[row] = rank

    # 元の列の後ろに予測の列 (同じ名前の元の列は「(入力)」)
    result = frame.rename(columns={c: frame_name(c) for c in frame.columns})
    result["status"] = np.where(ok, OK, SKIPPED)
    result["reason"] = pd.Series(reasons, dtype=object)
    for key in ("effect", "treated", "control"):
        result[key] = values[key].astype(float)
    result["rank"] = pd.array(ranks, dtype="Int64")

    return result


def id_column(bundle: Bundle, prediction: pd.DataFrame) -> Optional[str]:
    """
    予測の表の行の id の列の名前を返す関数

    Parameters
    ----------
    bundle: Bundle
        効果の一式
    prediction: pd.DataFrame
        predict_uplift の表

    Returns
    ----------
    name: Optional[str]
        行の id の列 (予測の列と同じ名前なら「(入力)」付き. 無ければ
        None)
    """
    # 予測の列と同じ名前の元の列の「(入力)」付きの名前
    key = frame_name(bundle.info["uplift"]["id"])
    name = key if key in prediction.columns else None

    return name


def effect_check(
    bundle: Bundle, data: Any, options: Optional[dict], sources: dict,
    batch_size: int
) -> tuple[Optional[dict], list[str]]:
    """
    元の表の効果を一式で出し, 実験の行ごとの効果と比べる関数

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
    # 行の id の値による実験の効果との突き合わせ
    prediction = predict_uplift(bundle, data, options, batch_size, "cpu")
    key = id_column(bundle, prediction)
    if key is None:
        note = (f"元の表に行の id の列 {bundle.info['uplift']['id']} が"
                "無いので, 一式の効果を実験の効果と比べません")
        found = (None, [note])
        return found
    expected = sources["effects"]
    known = expected.drop_duplicates(expected.columns[0]).set_index(
        expected.columns[0]
    )["effect"]
    ok = prediction[prediction["status"] == OK]
    ids = ok[key].astype(str)
    matched = ids.isin(known.index).to_numpy()
    if not matched.any():
        note = ("元の表の行の id が実験の行ごとの効果の表に無いので, 一式"
                "の効果を実験の効果と比べません")
        found = (None, [note])
        return found

    # 差の許し (絶対と相対) の中の割合と, 差の最大
    found_effects = ok["effect"].to_numpy(dtype=np.float64)[matched]
    recorded = known.loc[ids[matched]].to_numpy(dtype=np.float64)
    gap = np.abs(found_effects - recorded)
    agreement = float((gap <= ABSOLUTE_GAP + RELATIVE_GAP
                       * np.abs(recorded)).mean())
    check = {"rows": int(matched.sum()), "agreement": agreement,
             "max_diff": float(gap.max())}
    notes = []
    if agreement < AGREEMENT:
        notes.append(
            f"元の表を一式で出した効果が, 実験の効果と "
            f"{agreement * 100:.1f} % の行でしか同じになりません (前処理の"
            "レシピか重みが学習と違う可能性があります)"
        )
    if check["rows"] < len(expected):
        notes.append(
            f"実験の行ごとの効果の {len(expected):,} 行のうち, 比べたのは"
            f" {check['rows']:,} 行です (元の表に無い行か, 効果を出せない"
            "行があります)"
        )
    found = (check, notes)

    return found


def probe_model(bundle: Bundle, width: int, batch_size: int) -> None:
    """
    0 の行をモデルに渡し, 2 つの予測の形と値を確かめる関数

    Parameters
    ----------
    bundle: Bundle
        効果の一式 (cache に uplift_estimator)
    width: int
        見積もりに使う列の数
    batch_size: int
        1 回に渡す行の数

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        2 つの予測が件数の数でない場合
    """
    # 予測の形の確かめ (raw_outputs の中) と, 値が数かの確かめ
    treated, control = raw_outputs(
        bundle.model(), np.zeros((PROBE_ROWS, width), dtype=np.float32),
        bundle.cache["uplift_estimator"], batch_size, "cpu"
    )
    if not (np.isfinite(treated).all() and np.isfinite(control).all()):
        raise ValueError("0 の行の 2 つの予測が数ではありません")


def section_names(estimator: dict) -> dict:
    """
    成果物のモデルの名前を, 一式の uplift の欄の鍵で返す関数

    Parameters
    ----------
    estimator: dict
        効果の決め方 (learner, model)

    Returns
    ----------
    names: dict
        input と, t_learner は treated と control, s_learner は output
        (成果物の model.outcome)
    """
    # s_learner の結果の予測の名前は output (uplift.outcome は結果の種類)
    model = estimator["model"]
    names = {"input": model["input"]}
    if estimator["learner"] == SINGLE:
        names["output"] = model["outcome"]
    else:
        names.update(treated=model["treated"], control=model["control"])

    return names


def uplift_section(sources: dict) -> dict:
    """
    成果物から一式の uplift の欄を作る関数

    Parameters
    ----------
    sources: dict
        run_sources の結果

    Returns
    ----------
    uplift: dict
        uplift の欄 (check は None)
    """
    # 成果物の写し (標準化の値と結果の予測にする決まりは
    # uplift_estimator.json の中だけ)
    estimator = sources["estimator"]
    budget = estimator["budget"]
    uplift = {
        "estimator": ESTIMATOR_FILE,
        "recipe": RECIPE_FILE,
        "learner": estimator["learner"],
        "outcome": estimator["outcome"],
        "features": list(estimator["features"]),
        "id": estimator["id"],
        "treatment": estimator["treatment"],
        **section_names(estimator),
        "budget": None if budget is None else int(budget),
        "k": float(estimator["k"]),
        "rows": int(len(sources["effects"])),
        "train": {key: plain(estimator["train"][key]) for key in TRAIN_KEYS},
        "evaluations": copy.deepcopy(sources["evaluations"]),
        "curves": copy.deepcopy(sources["curves"]),
        "check": None,
    }

    return uplift


def make_uplift_bundle(
    path: Union[str, Path], run: Union[str, Path],
    name: Optional[str] = None, data: Any = None,
    options: Optional[dict] = None, copy_weights: bool = True,
    overwrite: bool = False, pretrained: Optional[dict] = None,
    batch_size: int = 256
) -> dict:
    """
    新しい行の効果を見積もる一式を, 効果の見積もりの実験から作る関数

    Parameters
    ----------
    path: Union[str, Path]
        作るディレクトリ
    run: Union[str, Path]
        ML-Found の実験のディレクトリ (uplift_estimator.json,
        uplift_recipe.yaml, uplift_effects.csv, evaluations の
        uplift_<分割>.json, final_model.pt, model_spec.yaml を使い, 実験
        の状態を記録する)
    name: Optional[str] = None
        一式の名前 (省略するとディレクトリの名前)
    data: Any = None
        学習に使った元の表 (渡すと効果を出して実験の効果と比べる)
    options: Optional[dict] = None
        data のファイルの読み込みの設定 (load_data に渡す)
    copy_weights: bool = True
        重みを一式に写すか (写さない場合は絶対パスで指す)
    overwrite: bool = False
        既にある一式を上書きするか
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
        ディレクトリを上書きしようとした場合
    FileNotFoundError
        成果物が無い場合
    ValueError
        定義, 重み, 成果物が約束に合わない場合
    """
    # 前に作った一式だけの置き換え
    final = Path(path).expanduser().resolve()
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"{final} は既にあります")
        if not (final / "inference.yaml").is_file():
            raise FileExistsError(f"{final} は推論の一式ではないので上書きしません")
    if batch_size < 1:
        raise ValueError("batch_size は 1 以上にしてください")

    # 実験の成果物と, モデルの入力と出力の約束
    sources = run_sources(run)
    estimator = sources["estimator"]
    spec_dict = load_spec(sources["spec"])
    check_units(spec_dict)
    check_uplift_spec(spec_dict, estimator)
    spec_dict, records, kept, weight_notes = plan_weights(
        spec_dict, sources["weights"], pretrained, copy_weights
    )

    # 隣の一時ディレクトリに作り, 確かめてから入れ替え
    writing = final.with_name(f".{final.name}.writing")
    if writing.exists():
        shutil.rmtree(writing)
    writing.mkdir(parents=True)
    try:
        shutil.copy(sources["root"] / ESTIMATOR_FILE,
                    writing / ESTIMATOR_FILE)
        shutil.copy(sources["recipe_file"], writing / RECIPE_FILE)
        dump_spec(spec_dict, writing / "model_spec.yaml")
        if copy_weights and kept is not None:
            torch.save(kept, writing / "final_model.pt")
        elif copy_weights:
            shutil.copy(sources["weights"], writing / "final_model.pt")
        budget = estimator["budget"]
        empty = Dataset(path=writing, info={}, frames={}, recipe={})
        info = {
            "format_version": LEARNER_VERSIONS[estimator["learner"]],
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "role": "effect",
            "output": {"form": "effect"},
            "target": None,
            "procedure": None,
            "model": {
                "spec": "model_spec.yaml",
                "weights": "final_model.pt" if copy_weights
                else str(sources["weights"]),
                "output": None,
                "kind": "value",
                "target_scale": None,
                "pretrained": records,
            },
            "inputs": {},
            "dataset": None,
            "run": sources["run"],
            "conclusion": {},
            "decision": None if budget is None else {
                "type": "treat", "budget": int(budget)
            },
            "check": None,
            "uncertainty": None,
            "notes": bundle_notes(empty, sources["run"]) + weight_notes,
            "uplift": uplift_section(sources),
        }

        # 0 の行の 2 つの予測と, 元の表の効果と実験の効果の比べ
        building = Bundle(path=writing, info=info, spec=spec_dict,
                          dataset=empty)
        building.cache["uplift_estimator"] = estimator
        building.cache["uplift_dataset"] = uplift_dataset(
            writing, sources["recipe"], estimator
        )
        probe_model(building, len(estimator["features"]), batch_size)
        if data is not None:
            found, notes = effect_check(building, data, options, sources,
                                        batch_size)
            info["uplift"]["check"] = found
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


def check_uplift_bundle(bundle: Bundle) -> None:
    """
    効果の一式の成果物を確かめて cache に置く関数

    Parameters
    ----------
    bundle: Bundle
        効果の一式

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
    # 効果の決め方とレシピの写し (一式の外は誤り)
    uplift = bundle.info["uplift"]
    estimator = read_estimator(inside(bundle.path, uplift["estimator"]))
    recipe = read_recipe(inside(bundle.path, uplift["recipe"]))
    copied = {key: estimator[key]
              for key in ("learner", "outcome", "features", "id",
                          "treatment")}
    copied.update(section_names(estimator)
                  if estimator["learner"] == uplift["learner"] else {})
    mismatch = [key for key, value in copied.items()
                if uplift[key] != value]
    if mismatch:
        raise ValueError(
            f"uplift の {', '.join(mismatch)} が効果の決め方 "
            f"{uplift['estimator']} と違います"
        )
    bundle.cache["uplift_estimator"] = estimator
    bundle.cache["uplift_dataset"] = uplift_dataset(bundle.path, recipe,
                                                    estimator)

    # モデルの定義の入力と出力
    check_uplift_spec(bundle.spec, estimator)


def tie_order(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    見積もりの高い順の並びと, 同点のまとまりの終わりを返す関数

    Parameters
    ----------
    scores: np.ndarray
        行ごとの見積もり (float64)

    Returns
    ----------
    found: tuple[np.ndarray, np.ndarray]
        高い順の行の位置と, 並べた後の同点のまとまりの最後の位置
    """
    # 同じ値の行は同点
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    ends = np.append(np.flatnonzero(ranked[1:] != ranked[:-1]),
                     len(ranked) - 1)
    found = (order, ends)

    return found


def qini_points(
    scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Qini の曲線の点 (上位の行の数と Qini の値) を求める関数

    Parameters
    ----------
    scores: np.ndarray
        行ごとの見積もり (float64)
    treatment: np.ndarray
        処置の群なら 1, 対照の群なら 0 (float64)
    outcome: np.ndarray
        結果 (float64)

    Returns
    ----------
    points: tuple[np.ndarray, np.ndarray]
        (0, 0) から始め, 同点のまとまりの終わりごとの上位の行の数と,
        Y_t − Y_c × n_t / n_c (n_c が 0 の間は Y_t)
    """
    # 上位の行の群ごとの数と結果の和
    order, ends = tie_order(scores)
    treated = treatment[order]
    result = outcome[order]
    n_t = np.cumsum(treated)[ends]
    n_c = np.cumsum(1 - treated)[ends]
    y_t = np.cumsum(treated * result)[ends]
    y_c = np.cumsum((1 - treated) * result)[ends]
    ratio = np.divide(n_t, n_c, out=np.zeros_like(n_t), where=n_c > 0)
    values = np.where(n_c > 0, y_t - y_c * ratio, y_t)
    points = (np.append(0.0, ends + 1.0), np.append(0.0, values))

    return points


def curve_area(rows: np.ndarray, values: np.ndarray) -> float:
    """
    曲線の下の面積を台形で求める関数

    Parameters
    ----------
    rows: np.ndarray
        横軸 (上位の行の数)
    values: np.ndarray
        縦軸

    Returns
    ----------
    area: float
        台形の面積の和
    """
    # 隣の点の間の台形
    area = float(((values[1:] + values[:-1]) / 2 * np.diff(rows)).sum())

    return area


def qini_coefficient(
    scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
) -> float:
    """
    Qini の係数を求める関数

    ML-Found の qini_coefficient と同じ決まり

    Parameters
    ----------
    scores: np.ndarray
        行ごとの効果の見積もり
    treatment: np.ndarray
        処置の群なら 1, 対照の群なら 0
    outcome: np.ndarray
        結果 (0 と 1 か額)

    Returns
    ----------
    coefficient: float
        (A − B) / (P − B). P − B が 0 なら 0
    """
    # 見積もりの並び, 無作為の直線, 理想の並びの面積
    scores = np.asarray(scores, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    rows, values = qini_points(scores, treatment, outcome)
    area = curve_area(rows, values)
    random = len(scores) * values[-1] / 2
    ideal = curve_area(*qini_points(
        np.where(treatment == 1, outcome, -outcome), treatment, outcome
    ))
    denominator = ideal - random
    coefficient = 0.0 if denominator == 0 else float(
        (area - random) / denominator
    )

    return coefficient


def uplift_at_k(
    scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray,
    k: float
) -> float:
    """
    上位 K 割の効果の差を求める関数

    ML-Found の uplift_at_k と同じ決まり

    Parameters
    ----------
    scores: np.ndarray
        行ごとの効果の見積もり
    treatment: np.ndarray
        処置の群なら 1, 対照の群なら 0
    outcome: np.ndarray
        結果 (0 と 1 か額)
    k: float
        上位の割合 (0 より大きく 1 以下)

    Returns
    ----------
    lift: float
        上位の行の処置の群の結果の平均 − 対照の群の結果の平均 (上位の行に
        片方の群が無ければ nan)
    """
    # 上位の行の数は 10 進の表記どおりの K × 行の数の切り捨て
    scores = np.asarray(scores, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    count = int(Decimal(str(k)) * len(scores))
    if count == 0:
        return math.nan

    # 境より上は全て, 境の同点のまとまりは残りの枠 / 行の数の重み
    order, ends = tie_order(scores)
    weights = np.zeros(len(scores))
    start, left = 0, count
    for end in ends:
        size = end - start + 1
        weights[start:end + 1] = min(1.0, left / size)
        left -= min(left, size)
        start = end + 1
        if not left:
            break
    treated = treatment[order]
    result = outcome[order]
    w_t = (weights * treated).sum()
    w_c = (weights * (1 - treated)).sum()
    if w_t == 0 or w_c == 0:
        return math.nan
    lift = float((weights * treated * result).sum() / w_t
                 - (weights * (1 - treated) * result).sum() / w_c)

    return lift


def distribution_of(effects: np.ndarray) -> Optional[dict]:
    """
    効果の分布をまとめる関数

    Parameters
    ----------
    effects: np.ndarray
        予測した行の効果

    Returns
    ----------
    distribution: Optional[dict]
        mean, min, p10, p50, p90, max, positive (効果が 0 より大きい行の
        割合. 行が無ければ None)
    """
    # 分位点は線形の補間 (骨組みの train と同じ)
    if not len(effects):
        return None
    distribution = {"mean": float(effects.mean()),
                    "min": float(effects.min())}
    for point in (10, 50, 90):
        distribution[f"p{point}"] = float(np.percentile(effects, point))
    distribution["max"] = float(effects.max())
    distribution["positive"] = float((effects > 0).mean())

    return distribution


def ranked_rows(prediction: pd.DataFrame) -> pd.DataFrame:
    """
    予測した行を効果の高い順に並べる関数

    Parameters
    ----------
    prediction: pd.DataFrame
        predict_uplift の表

    Returns
    ----------
    ranked: pd.DataFrame
        予測した行の順位の順の表 (行の名前は予測の表の位置)
    """
    # 順位の順 (同じ効果は前の行が先)
    ranked = prediction[prediction["status"] == OK].sort_values(
        "rank", kind="stable"
    )

    return ranked


def top_share(bundle: Bundle, ranked: pd.DataFrame) -> Optional[dict]:
    """
    効果の高い順の上位 K 割の行の要約を作る関数

    Parameters
    ----------
    bundle: Bundle
        効果の一式 (uplift.k)
    ranked: pd.DataFrame
        ranked_rows の表

    Returns
    ----------
    top: Optional[dict]
        k, rows, mean, lowest (上位の行が無ければ None)
    """
    # 行の数は 10 進の表記どおりの K × 予測した行の数の切り捨て
    k = float(bundle.info["uplift"]["k"])
    count = int(Decimal(str(k)) * len(ranked))
    if not count:
        return None
    effects = ranked["effect"].to_numpy(dtype=np.float64)[:count]
    top = {"k": k, "rows": count, "mean": float(effects.mean()),
           "lowest": float(effects.min())}

    return top


def top_rows(ranked: pd.DataFrame, key: Optional[str]) -> list[dict]:
    """
    効果の高い順の上位の行の要約を作る関数

    Parameters
    ----------
    ranked: pd.DataFrame
        ranked_rows の表
    key: Optional[str]
        行の id の列 (無ければ None)

    Returns
    ----------
    rows: list[dict]
        row (表の行の位置), id, effect, treated, control
    """
    # 上位 TOP_ROWS 行
    rows = [
        {"row": int(position),
         "id": None if key is None else value_of(row[key]),
         "effect": float(row["effect"]), "treated": float(row["treated"]),
         "control": float(row["control"])}
        for position, row in ranked.head(TOP_ROWS).iterrows()
    ]

    return rows


def binary_values(values: pd.Series, whole: bool) -> np.ndarray:
    """
    処置か結果の列の値を数にする関数

    Parameters
    ----------
    values: pd.Series
        列の値
    whole: bool
        処置の列か (0 と 1 だけを読む)

    Returns
    ----------
    numbers: np.ndarray
        float64 の値 (読めない値は nan)
    """
    # 真偽値は 0 と 1, 数にできない値は nan
    numbers = np.array([
        float(v) if isinstance(v, (bool, np.bool_)) or is_number(v)
        else math.nan for v in values.tolist()
    ], dtype=np.float64)
    numbers[~np.isfinite(numbers)] = math.nan
    if whole:
        numbers[(numbers != 0) & (numbers != 1)] = math.nan

    return numbers


def measure(
    bundle: Bundle, prediction: pd.DataFrame
) -> tuple[Optional[dict], Optional[str]]:
    """
    渡した表の処置と結果で, 効果の見積もりの評価を数え直す関数

    Parameters
    ----------
    bundle: Bundle
        効果の一式
    prediction: pd.DataFrame
        predict_uplift の表

    Returns
    ----------
    found: tuple[Optional[dict], Optional[str]]
        rows, treated, control, excluded, k, qini, uplift_at_k, effect
        (数えなければ None) と, 数えなかった理由 (列が無ければ None)
    """
    # 処置の列と結果の列 (無ければ数えない)
    treatment = frame_name(bundle.info["uplift"]["treatment"])
    outcome = bundle.cache["uplift_dataset"].recipe.get("target")
    outcome = frame_name(str(outcome)) if outcome else None
    if treatment not in prediction.columns or outcome not in (
        prediction.columns
    ):
        found = (None, None)
        return found

    # 予測した行のうち, 処置が 0 か 1 で結果が有限の数の行
    ok = prediction[prediction["status"] == OK]
    t = binary_values(ok[treatment], True)
    y = binary_values(ok[outcome], False)
    use = np.isfinite(t) & np.isfinite(y)
    effects = ok["effect"].to_numpy(dtype=np.float64)[use]
    t, y = t[use], y[use]
    if not (t == 1).any() or not (t == 0).any():
        found = (None, "渡した表の処置と結果では, 処置の群か対照の群の行が"
                 "無いので, 効果の見積もりの評価を数え直さない")
        return found
    k = float(bundle.info["uplift"]["k"])
    lift = uplift_at_k(effects, t, y, k)
    measured = {
        "rows": int(use.sum()), "treated": int((t == 1).sum()),
        "control": int((t == 0).sum()), "excluded": int((~use).sum()),
        "k": k, "qini": qini_coefficient(effects, t, y),
        "uplift_at_k": None if math.isnan(lift) else lift,
        "effect": float(y[t == 1].mean() - y[t == 0].mean()),
    }
    found = (measured, None)

    return found


def lift_text(lift: Optional[float], k: float) -> str:
    """
    上位 K 割の効果の差を述べる文を返す関数

    Parameters
    ----------
    lift: Optional[float]
        上位 K 割の効果の差 (無ければ None)
    k: float
        K

    Returns
    ----------
    text: str
        上位 K 割の効果の差の句 (K は % で書く)
    """
    # 上位に片方の群が無い場合
    if lift is None:
        text = (f"上位 {k * 100:g} % の効果の差は無し (上位に片方の群が"
                "無い)")
        return text
    text = f"上位 {k * 100:g} % の効果の差 {lift:.3g}"

    return text


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
        分割ごとの Qini の係数と上位 K 割の効果の差の文 (無ければ, 分布と
        上位の行を見て確かめる文)
    """
    # 評価の無い実験の, 分布と上位の行を見る文
    if not evaluations:
        sentences = ["実験の評価が無いので, 効果の分布と上位の行を見て"
                     "確かめる"]
        return sentences
    sentences = [
        f"{SPLIT_NAMES.get(split, split)}の Qini の係数 {row['qini']:.3f}, "
        f"{lift_text(row['uplift_at_k'], row['k'])} (分割の全ての行の結果の"
        f"差 {row['effect']:.3g}. 処置 {row['treated']} 行 / 対照 "
        f"{row['control']} 行)"
        for split, row in evaluations.items()
    ]

    return sentences


def conclude_uplift(bundle: Bundle, prediction: pd.DataFrame) -> dict:
    """
    行ごとの効果から, 件数と効果の要約の結論を出す関数

    Parameters
    ----------
    bundle: Bundle
        効果の一式
    prediction: pd.DataFrame
        predict_uplift の表

    Returns
    ----------
    conclusion: dict
        frame (decision を足した表. 予測した行は空, しなかった行は予測
        なし), summary (rows, predicted, skipped, outcome, id,
        distribution, train, top_k, top, evaluations, measured),
        sentences, rules (空)
    """
    # 予測したかと, 効果の分布と上位の行
    uplift = bundle.info["uplift"]
    frame = prediction.copy()
    ok = (frame["status"] == OK).to_numpy()
    frame["decision"] = np.where(ok, "", NO_PREDICTION)
    effects = frame["effect"].to_numpy(dtype=np.float64)[ok]
    key = id_column(bundle, prediction)
    ranked = ranked_rows(prediction)
    measured, unmeasured = measure(bundle, prediction)
    summary = {
        "rows": int(len(frame)), "predicted": int(ok.sum()),
        "skipped": int(len(frame) - ok.sum()), "outcome": uplift["outcome"],
        "id": key, "distribution": distribution_of(effects),
        "train": copy.deepcopy(uplift["train"]),
        "top_k": top_share(bundle, ranked), "top": top_rows(ranked, key),
        "evaluations": copy.deepcopy(uplift["evaluations"]),
        "measured": measured,
    }

    # 件数, 分布, 上位 K 割, 実験の評価, 数え直しの文
    head = (f"{summary['rows']:,} 行のうち {summary['predicted']:,} 行の"
            "効果 (処置したときの結果 − しないときの結果. "
            f"{EFFECT_NAMES[uplift['outcome']]}) を見積もった")
    if summary["skipped"]:
        head += f" ({summary['skipped']:,} 行は出せず)"
    sentences = [head]
    found = summary["distribution"]
    if found is not None:
        positive = int((effects > 0).sum())
        sentences.append(
            f"効果の中央 {found['p50']:.3g}, 10 % 点 {found['p10']:.3g}, "
            f"90 % 点 {found['p90']:.3g}, 平均 {found['mean']:.3g}. 効果が"
            f" 0 より大きい行は {positive:,} 行 "
            f"({found['positive'] * 100:.1f} %)"
        )
    top = summary["top_k"]
    if top is not None:
        sentences.append(
            f"効果の高い上位 {top['k'] * 100:g} % ({top['rows']:,} 行) の"
            f"効果の平均 {top['mean']:.3g}, 最も低い効果 "
            f"{top['lowest']:.3g}"
        )
    sentences.extend(evaluation_sentences(uplift["evaluations"]))
    if measured is not None:
        sentences.append(
            f"渡した表の処置と結果で数えた Qini の係数 {measured['qini']:.3f}"
            f", {lift_text(measured['uplift_at_k'], measured['k'])} (処置 "
            f"{measured['treated']:,} 行 / 対照 {measured['control']:,} 行. "
            "配る相手を無作為に分けた記録でだけ効果の確かめになる)"
        )
    elif unmeasured is not None:
        sentences.append(unmeasured)
    conclusion = {"frame": frame, "summary": summary,
                  "sentences": sentences, "rules": {}}

    return conclusion


def check_uplift_decision(decision: dict) -> list[str]:
    """
    効果の一式の配る判断の決め方の誤りを一覧で返す関数

    Parameters
    ----------
    decision: dict
        意思決定の決め方

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 配る判断の型と書ける項目 (type, budget)
    if decision.get("type") != "treat":
        return ["効果の一式の意思決定は配る判断 (type: treat) だけです"]
    errors = [
        f"効果の一式の配る判断に書けない項目です: {key} (書けるのは type "
        "と budget だけ. 効果の高い順に上限まで配るので損の値は使わない)"
        for key in decision if key not in TREAT_KEYS
    ]
    if not is_whole(decision.get("budget"), 1):
        errors.append("budget (配る数の上限) は 1 以上の整数にしてください")

    return errors


def decide_uplift(
    bundle: Bundle, prediction: pd.DataFrame, decision: dict
) -> dict:
    """
    効果の高い順に, 配る数の上限まで配る行を決める関数

    Parameters
    ----------
    bundle: Bundle
        効果の一式
    prediction: pd.DataFrame
        predict_uplift の表 (か conclude_uplift の表)
    decision: dict
        {type: treat, budget}

    Returns
    ----------
    result: dict
        frame (action を足した表), summary (rows, acted, budget, lowest,
        gain, left_positive, nonpositive), sentences, notes (前提),
        decision (使った決め方)

    Raises
    ----------
    ValueError
        決め方に誤りがある場合と, 表に効果の列が無い場合
    """
    # 決め方と表の確認
    decision = plain(dict(decision))
    errors = check_uplift_decision(decision)
    missing = [c for c in ("status", "effect")
               if c not in prediction.columns]
    if missing:
        errors.append(f"予測の表に {', '.join(missing)} の列がありません")
    if errors:
        raise ValueError("意思決定の決め方の誤り: " + " / ".join(errors))

    # 予測した行の効果の高い順 (同じ効果は前の行が先) の, 上限までの効果
    # が正の行
    budget = int(decision["budget"])
    ok = (prediction["status"] == OK).to_numpy()
    effects = prediction["effect"].to_numpy(dtype=np.float64)
    rows = np.flatnonzero(ok)
    order = rows[np.argsort(-effects[rows], kind="stable")]
    within = order[:budget]
    chosen = within[effects[within] > 0]
    action = pd.array([None] * len(prediction), dtype="boolean")
    action[rows] = False
    action[chosen] = True
    summary = {
        "rows": int(len(rows)), "acted": int(len(chosen)),
        "budget": budget,
        "lowest": float(effects[chosen].min()) if len(chosen) else None,
        "gain": float(effects[chosen].sum()),
        "left_positive": int((effects[order[budget:]] > 0).sum()),
        "nonpositive": int(len(within) - len(chosen)),
    }
    frame = prediction.copy()
    if ACTION_COLUMN in frame.columns:
        frame = frame.rename(columns={ACTION_COLUMN: f"{ACTION_COLUMN} (入力)"})
    frame[ACTION_COLUMN] = action

    # 前提, 配る件数と効果の下限と和, 上限で配らない行と効果が 0 以下の
    # 行の文 (前提は notes にも)
    notes = [
        "効果は処置したときとしないときの 2 つの予測の差の見積もりで, 処置を"
        "無作為に配った記録を前提にしている (配る相手を選んだ記録から学ぶと"
        "偏る)",
        "配った効果は, 配る相手を無作為に分けた試験で確かめる (機械学習の"
        "結果は判断の材料)",
        "配る数は損得のしきい値でなく, 配る数の上限で決めた",
    ]
    sentence = (f"効果の高い順に {summary['acted']:,} 件に配る (配る数の"
                f"上限 {budget:,} 件)")
    if summary["lowest"] is not None:
        sentence += (f". 配る行の効果は {summary['lowest']:.3g} 以上で, 効果"
                     f"の和は {summary['gain']:.3g} (見積もりで増える結果)")
    sentences = [*notes, sentence]
    if summary["left_positive"]:
        sentences.append(f"効果が正なのに上限で配らない行が "
                         f"{summary['left_positive']:,} 行ある")
    if summary["nonpositive"]:
        sentences.append(f"効果が 0 以下の {summary['nonpositive']:,} 行は"
                         "上限の中でも配らない (見積もりでは配っても結果が"
                         "増えない)")
    result = {"frame": frame, "summary": summary, "sentences": sentences,
              "notes": notes, "decision": decision}

    return result


def uplift_of_bundle(bundle: Bundle) -> Optional[dict]:
    """
    効果の一式の uplift の欄を, 説明の形で返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    uplift: Optional[dict]
        uplift の欄の写し (効果の一式でなければ None)
    """
    # 画面が効果の読み方と実験の評価を出すための写し
    if not bundle.from_uplift:
        return None
    uplift = copy.deepcopy(bundle.info["uplift"])

    return uplift
