import copy
import datetime as dt
from decimal import Decimal
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
    AGREEMENT, OK, SKIPPED, is_whole, row_features
)
from ml_inference.bundle import (
    PRICE_BUNDLE_VERSION, Bundle, bundle_notes, inside, load_bundle,
    plan_weights, run_summary
)
from ml_inference.conclude import NO_PREDICTION, is_number, plain
from ml_inference.grouping import read_table
from ml_inference.units import check_units
from ml_inference.uplift import read_json

# 実験の中の成果物 (ML-Quest の値段を変える判断の骨組みが書く名前)
ESTIMATOR_FILE = "price_estimator.json"
RECIPE_FILE = "price_recipe.yaml"
EVALUATION_FILE = "evaluations/price_{split}.json"

# 読める値段の決め方の成果物の版と, 今の値段の決め方 (商品の id の列が
# あれば median, 無ければ row) と, 出力の決まり
ESTIMATOR_VERSION = 1
CURRENT_RULES = {True: "median", False: "row"}
LINK = "softplus"

# 弾力性を数える値段の上げ幅 (今の値段の 1 %)
ELASTICITY_STEP = 0.01

# 値段の決め方の成果物の鍵と, モデルの名前の鍵と, 学習用の分割の要約の鍵
ESTIMATOR_KEYS = ("format_version", "recipe", "choices", "features",
                  "price", "cost", "product", "id", "current", "candidates",
                  "max_change", "elasticity_step", "model", "scale", "train")
MODEL_NAMES = ("input", "units")
TRAIN_KEYS = ("products", "changed", "elasticity")

# 選んだ値段の表の, 鍵の列の他の列と, 空にならない数の列
CHOICES_COLUMNS = ("split", "rows", "current_price", "cost", "change",
                   "price", "profit", "current_profit", "elasticity")
CHOICES_NUMBERS = ("rows", "current_price", "change", "price", "profit",
                   "current_profit")

# 実験の評価を写す分割と, 評価の行の鍵 (null にもなる鍵も含む順)
EVALUATION_SPLITS = ("valid", "test")
EVALUATION_KEYS = ("rows", "products", "mae", "rmse", "elasticity",
                   "elasticity_counted", "profit", "current_profit",
                   "profit_gain", "profit_ratio", "changed", "max_change")
NULLABLE_KEYS = ("elasticity", "profit_ratio")

# 分割の日本語の名前
SPLIT_NAMES = {"valid": "検証用", "test": "テスト用"}

# price の欄
PRICE_KEYS = ("estimator", "recipe", "features", "price", "units", "cost",
              "product", "id", "input", "output", "candidates",
              "max_change", "elasticity_step", "current", "rows", "train",
              "evaluations", "check")

# 一式を作るときに予測を確かめる 0 の行の数
PROBE_ROWS = 2

# 予測の表に足す列, 値段の判断が足す列, 同じ名前の元の列を「(入力)」に
# する列 (結論の decision も)
PRICE_FRAME_COLUMNS = ("status", "reason", "current_price", "cost", "units",
                       "current_profit", "elasticity")
DECISION_COLUMNS = ("action", "change", "price", "price_units", "profit")
ADDED_COLUMNS = (*PRICE_FRAME_COLUMNS, *DECISION_COLUMNS, "decision")
INPUT_MARK = " (入力)"

# 飛ばす行の理由
BAD_PRICE = "値段が 0 より大きい数ではありません"
BAD_COST = "原価が 0 以上の数ではありません"
NOT_FINITE = "予測が数ではありません"

# 値段の判断で書ける項目, 同じ利益とみなす差の割合, 予測し直す行の数
PRICE_DECISION_KEYS = ("type", "candidates", "max_change")
TIE = 1e-9
DECIDE_BATCH = 256

# 選んだ割合が同じとみなす差
SAME_CHANGE = 1e-9


def decimal_of(value: Any) -> Decimal:
    """
    数を 10 進の表記どおりの Decimal にする関数

    Parameters
    ----------
    value: Any
        数 (真偽値を除く)

    Returns
    ----------
    number: Decimal
        表記どおりの値 (0.1 は 1/10)
    """
    # float の短い表記から (2 進の丸めを持ち込まない)
    number = Decimal(str(plain(value)))

    return number


def check_candidates(candidates: Any, where: str) -> list[str]:
    """
    値段の候補の誤りを一覧で返す関数

    Parameters
    ----------
    candidates: Any
        今の値段に対する変える割合の並び
    where: str
        誤りの文に使う項目の名前

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # -1 より大きい有限の数の並びで, 重ならず, 0 を含む
    if not isinstance(candidates, (list, tuple)) or not candidates or not (
        all(is_number(c) and math.isfinite(c) and c > -1 for c in candidates)
    ):
        errors = [f"{where} は -1 より大きい数の並び (今の値段に対する変える"
                  "割合) にしてください"]
        return errors
    changes = [decimal_of(c) for c in candidates]
    errors = []
    if len(set(changes)) != len(changes):
        errors.append(f"{where} に同じ割合が 2 つあります")
    if Decimal(0) not in changes:
        errors.append(f"{where} には今の値段 (0) を含めてください")

    return errors


def check_max_change(value: Any, where: str) -> list[str]:
    """
    値段を変える幅の上限の誤りを一覧で返す関数

    Parameters
    ----------
    value: Any
        変える割合の絶対値の上限
    where: str
        誤りの文に使う項目の名前

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 0 以上の有限の数 (守る指標なので省けない)
    good = is_number(value) and math.isfinite(value) and value >= 0
    errors = [] if good else [
        f"{where} (値段を変える幅の上限) は 0 以上の数にしてください (守る"
        "指標なので省けません)"
    ]

    return errors


def check_columns(estimator: dict) -> list[str]:
    """
    値段の決め方の成果物の列の誤りを一覧で返す関数

    Parameters
    ----------
    estimator: dict
        読み込んだ成果物

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 予測に使う列 (最後が値段の列), 値段と行の id の列, 原価と商品の id
    features = estimator["features"]
    errors = []
    if not isinstance(features, list) or not features or not all(
        isinstance(name, str) and name for name in features
    ) or len(set(features)) != len(features):
        errors.append("features は重ならない列の名前の並びにしてください")
        features = []
    for key in ("price", "id"):
        if not isinstance(estimator[key], str) or not estimator[key]:
            errors.append(f"{key} は列の名前にしてください")
    for key in ("cost", "product"):
        value = estimator[key]
        if value is not None and (not isinstance(value, str) or not value):
            errors.append(f"{key} は null か列の名前にしてください")
    if errors:
        return errors
    if features[-1] != estimator["price"]:
        errors.append(f"features の最後は値段の列 {estimator['price']} に"
                      "してください")

    # 値段, 原価, 商品の id, 行の id の列は互いに違い, 予測に使わない
    names = [estimator[key] for key in ("price", "cost", "product", "id")
             if estimator[key] is not None]
    if len(set(names)) != len(names):
        errors.append("price, cost, product, id は違う列にしてください")
    used = [name for name in names[1:] if name in features]
    if used:
        errors.append(f"予測に使う列に {', '.join(used)} があります")

    return errors


def check_estimator(estimator: Any) -> list[str]:
    """
    値段の決め方の成果物 (price_estimator.json) の誤りを一覧で返す関数

    Parameters
    ----------
    estimator: Any
        読み込んだ成果物

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 版, 鍵, 列, ファイルの名前
    if not isinstance(estimator, dict):
        return ["値段の決め方が辞書ではありません"]
    version = estimator.get("format_version")
    if version != ESTIMATOR_VERSION:
        return [f"値段の決め方の版 {version} は読めません (版 "
                f"{ESTIMATOR_VERSION} だけ)"]
    errors = [f"値段の決め方に {key} がありません"
              for key in ESTIMATOR_KEYS if key not in estimator]
    if errors:
        return errors
    errors.extend(check_columns(estimator))
    for key in ("recipe", "choices"):
        if not isinstance(estimator[key], str) or not estimator[key]:
            errors.append(f"{key} はファイルの名前にしてください")

    # 今の値段の決め方, 候補と上限, 弾力性の上げ幅
    product = estimator["product"] is not None
    if estimator["current"] != CURRENT_RULES[product]:
        errors.append(
            f"current は商品の id の列が{'ある' if product else '無い'}ので "
            f"{CURRENT_RULES[product]} にしてください"
        )
    errors.extend(check_candidates(estimator["candidates"], "candidates"))
    errors.extend(check_max_change(estimator["max_change"], "max_change"))
    step = estimator["elasticity_step"]
    if not is_number(step) or float(step) != ELASTICITY_STEP:
        errors.append(f"elasticity_step は {ELASTICITY_STEP} にしてください")

    # モデルの名前と出力の決まり, 損失で割った値, 学習用の分割の要約
    model = estimator["model"]
    if not isinstance(model, dict) or not all(
        isinstance(model.get(key), str) and model[key]
        for key in MODEL_NAMES
    ):
        errors.append("model は input と units の名前にしてください")
    elif model.get("link") != LINK:
        errors.append(f"model.link は {LINK} にしてください (出力がそのまま"
                      "売れた数)")
    scale = estimator["scale"]
    if not (is_number(scale) and math.isfinite(scale) and scale > 0):
        errors.append("scale は 0 より大きい数にしてください")
    train = estimator["train"]
    if not isinstance(train, dict) or not all(
        is_whole(train.get(key), 0) for key in ("products", "changed")
    ) or not (train.get("elasticity") is None or (
        is_number(train.get("elasticity"))
        and math.isfinite(train["elasticity"])
    )):
        errors.append("train は products と changed の 0 以上の整数と, "
                      "elasticity の数か null にしてください")

    return errors


def read_estimator(path: Path) -> dict:
    """
    値段の決め方の成果物を読み, 形を確かめる関数

    Parameters
    ----------
    path: Path
        price_estimator.json

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
        raise FileNotFoundError(f"値段の決め方 {path} がありません")
    estimator = read_json(path)
    errors = check_estimator(estimator)
    if errors:
        raise ValueError(f"{path} の誤り: " + " / ".join(errors))

    return estimator


def read_recipe(path: Path, estimator: dict) -> dict:
    """
    値段を変えた記録の表のレシピの写しを読んで確かめる関数

    Parameters
    ----------
    path: Path
        price_recipe.yaml
    estimator: dict
        値段の決め方 (price, features)

    Returns
    ----------
    recipe: dict
        当てはめた値を含むレシピ

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        値段を変えた記録の表のレシピでない場合と, 値段か売れた数の列が
        値段の決め方と食い違う場合
    """
    # 手順の並びを持つ, task が price のレシピ
    if not path.is_file():
        raise FileNotFoundError(f"前処理のレシピ {path} がありません")
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or recipe.get("task") != "price" or (
        not isinstance(recipe.get("steps"), list)
    ):
        raise ValueError(
            f"{path} は値段を変えた記録の表のレシピ (task: price と steps) "
            "ではありません"
        )

    # 値段の列と売れた数の列 (予測に使わない列)
    if recipe.get("price") != estimator["price"]:
        raise ValueError(
            f"{path} の値段の列 {recipe.get('price')} が値段の決め方の "
            f"{estimator['price']} と違います"
        )
    target = recipe.get("target")
    others = [*estimator["features"],
              *(estimator[key] for key in ("cost", "product", "id"))]
    if not isinstance(target, str) or not target or target in others:
        raise ValueError(f"{path} の target (売れた数の列) は予測に使わない"
                         "他と重ならない列の名前にしてください")

    return recipe


def read_choices(path: Path, key: str) -> pd.DataFrame:
    """
    実験の選んだ値段の表を読む関数

    Parameters
    ----------
    path: Path
        price_choices.csv
    key: str
        鍵の列 (商品の id の列. 無ければ行の id の列) の名前

    Returns
    ----------
    choices: pd.DataFrame
        鍵の列 (文字), split, rows, current_price, cost, change, price,
        profit, current_profit, elasticity の表

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        列が違う場合と, 数の列が数でない場合
    """
    # 鍵は文字 (先頭の 0 などを残すため)
    if not path.is_file():
        raise FileNotFoundError(f"選んだ値段の表 {path} がありません")
    choices = pd.read_csv(path, dtype={key: str}, keep_default_na=False,
                          na_values=[""])
    expected = [key, *CHOICES_COLUMNS]
    if list(choices.columns) != expected:
        raise ValueError(
            f"{path} の列は {', '.join(expected)} にしてください (今は "
            f"{', '.join(map(str, choices.columns))})"
        )

    # 空にならない数の列と, 空にもなる原価と弾力性
    for name in CHOICES_COLUMNS[1:]:
        values = pd.to_numeric(choices[name], errors="coerce")
        numbers = values.to_numpy(dtype=np.float64)
        empty = choices[name].isna().to_numpy()
        if name in CHOICES_NUMBERS and not np.isfinite(numbers).all():
            raise ValueError(f"{path} の {name} に数でない値があります")
        if not (np.isfinite(numbers) | empty).all():
            raise ValueError(f"{path} の {name} に数でない値があります")
        choices[name] = values.astype(np.float64)

    return choices


def read_evaluations(root: Path) -> dict:
    """
    実験の予測の誤差と値段の選び方の評価を読む関数

    Parameters
    ----------
    root: Path
        実験のディレクトリ

    Returns
    ----------
    evaluations: dict
        分割の名前から rows, products, mae, rmse, elasticity (無ければ
        None), elasticity_counted, profit, current_profit, profit_gain,
        profit_ratio (無ければ None), changed, max_change (評価の無い分割
        は無い)
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
            is_number(row.get(key)) or (
                key in NULLABLE_KEYS and key in row and row[key] is None
            ) for key in EVALUATION_KEYS
        ):
            continue
        evaluations[split] = {key: plain(row[key]) for key in EVALUATION_KEYS}

    return evaluations


def run_sources(run: Union[str, Path]) -> dict:
    """
    値段を変える判断の実験のディレクトリから, 一式に要るものを読む関数

    Parameters
    ----------
    run: Union[str, Path]
        ML-Found の実験のディレクトリ

    Returns
    ----------
    sources: dict
        root, estimator, recipe_file, recipe, choices, evaluations,
        weights, spec, run (実験の状態)

    Raises
    ----------
    ValueError
        実験のディレクトリでない場合, 重みか定義が無い場合, 成果物の形が
        違う場合
    FileNotFoundError
        成果物が無い場合
    """
    # 値段の決め方, レシピ, 選んだ値段, 評価
    root = Path(run).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"実験のディレクトリ {root} がありません")
    estimator = read_estimator(root / ESTIMATOR_FILE)
    recipe_file = inside(root, estimator["recipe"])
    recipe = read_recipe(recipe_file, estimator)
    choices = read_choices(inside(root, estimator["choices"]),
                           estimator["product"] or estimator["id"])

    # 売れ行きのモデルの重みと定義 (仮のモデルで学習した実験は定義無し)
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
        "recipe": recipe, "choices": choices,
        "evaluations": read_evaluations(root), "weights": weights,
        "spec": spec, "run": run_summary(root),
    }

    return sources


def check_price_info(info: dict) -> list[str]:
    """
    版 13 の一式の price の欄と, 受け持ちと出力の形を確かめる関数

    Parameters
    ----------
    info: dict
        inference.yaml の中身

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 欄, 受け持ち
    price = info.get("price")
    if not isinstance(price, dict):
        return [f"版 {PRICE_BUNDLE_VERSION} の一式には price が要ります"]
    errors = [f"price に {key} がありません"
              for key in PRICE_KEYS if key not in price]
    unknown = [str(key) for key in price if key not in PRICE_KEYS]
    if unknown:
        errors.append(f"price に知らない項目があります: {', '.join(unknown)}")
    if info.get("role") != "effect":
        errors.append(f"版 {PRICE_BUNDLE_VERSION} の一式の受け持ちは "
                      "effect (効果を見積もる) にしてください")
    if errors:
        return errors

    # ファイルの名前, 値段の列, 候補と上限
    for key in ("estimator", "recipe"):
        if not isinstance(price[key], str) or not price[key]:
            errors.append(f"price.{key} はファイルの名前にしてください")
    if not isinstance(price["price"], str) or not price["price"]:
        errors.append("price.price は列の名前にしてください")
    errors.extend(check_candidates(price["candidates"], "price.candidates"))
    errors.extend(check_max_change(price["max_change"], "price.max_change"))

    return errors


def check_price_spec(spec: dict, names: dict) -> None:
    """
    売れ行きのモデルの定義の入力と出力が約束に合うかを確かめる関数

    Parameters
    ----------
    spec: dict
        モデルの定義 (ML-ModelControl の形)
    names: dict
        input, units (成果物の model)

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        入力が成果物の入力の名前の 1 つでない場合と, 売れた数の出力の名前が
        無い場合
    """
    # 入力は前処理の後の行の 1 つ, 出力に売れた数の予測
    inputs = list(spec.get("inputs") or {})
    if inputs != [names["input"]]:
        raise ValueError(
            f"売れ行きのモデルの入力は {names['input']} の 1 つにしてください"
            f" (今は {', '.join(inputs) or '無し'})"
        )
    outputs = spec.get("outputs") or {}
    if names["units"] not in outputs:
        raise ValueError(
            f"売れ行きのモデルの出力に {names['units']} がありません (出力: "
            f"{', '.join(map(str, outputs)) or '無し'})"
        )


def price_dataset(root: Path, recipe: dict, estimator: dict) -> Dataset:
    """
    前処理だけの ML-EDA のデータセットを作る関数

    Parameters
    ----------
    root: Path
        一式 (か実験) のディレクトリ
    recipe: dict
        当てはめた値を含むレシピ
    estimator: dict
        値段の決め方 (features, price, cost, product, id)

    Returns
    ----------
    dataset: Dataset
        分割の表を持たないデータセット (transform だけに使う)
    """
    # 売れた数, 原価, 商品の id, 行の id の列は前処理の手順の外
    info = {"kind": "table", "task": "price",
            "target": recipe.get("target"),
            "features": [{"name": name} for name in estimator["features"]],
            "price": {key: estimator[key]
                      for key in ("id", "price", "cost", "product")}}
    dataset = Dataset(path=root, info=info, frames={}, recipe=recipe)

    return dataset


def raw_units(
    model: Any, x: np.ndarray, names: dict, batch_size: int, device: str
) -> np.ndarray:
    """
    前処理の後の行をモデルに渡し, 売れた数の予測をまとまりごとに求める関数

    Parameters
    ----------
    model: Any
        売れ行きのモデル
    x: np.ndarray
        (件数, 列の数) の float32 の行
    names: dict
        input, units
    batch_size: int
        1 回に渡す行の数
    device: str
        使うデバイス

    Returns
    ----------
    units: np.ndarray
        (件数,) の float64 の予測 (負の値は 0 に切る. 数でない値は nan)

    Raises
    ----------
    ValueError
        名前の辞書の出力に売れた数の名前が無い場合と, 予測の形が (件数,) か
        (件数, 1) でない場合
    """
    # 出力が 1 つの定義のテンソルと, 名前の辞書の売れた数 (float32 から
    # float64 へ. 骨組みと同じ)
    found = []
    model = model.to(device)
    try:
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                chunk = torch.tensor(x[start:start + batch_size]).to(device)
                value = model(**{names["input"]: chunk})
                if isinstance(value, dict):
                    if names["units"] not in value:
                        raise ValueError(
                            f"売れ行きのモデルの出力に {names['units']} が"
                            "ありません"
                        )
                    value = value[names["units"]]
                if tuple(value.shape) not in ((len(chunk),),
                                              (len(chunk), 1)):
                    raise ValueError(
                        f"予測 {names['units']} の形 {tuple(value.shape)} が"
                        f" ({len(chunk)},) か ({len(chunk)}, 1) ではありません"
                    )
                found.append(value.reshape(-1).cpu())
    finally:
        model.to("cpu")
    units = torch.cat(found).double().numpy() if found else np.zeros(0)
    units = np.where(np.isfinite(units), np.clip(units, 0.0, None), math.nan)

    return units


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
        予測の列, 値段の判断の列, 結論の列と同じ名前なら「(入力)」付き,
        他はそのまま
    """
    # 足す列と同じ名前の元の列の「(入力)」付きの名前
    found = f"{name}{INPUT_MARK}" if name in ADDED_COLUMNS else name

    return found


def original_frame(prediction: pd.DataFrame) -> pd.DataFrame:
    """
    予測の表から元の列の表を戻す関数

    Parameters
    ----------
    prediction: pd.DataFrame
        predict_price の表 (か結論と値段の判断の列を足した表)

    Returns
    ----------
    frame: pd.DataFrame
        足した列を除き,「(入力)」を付けた列を元の名前に戻した表 (添字は
        0 から)
    """
    # 足した列の除外と, 同じ名前の元の列の名前の戻し
    names = {}
    for name in prediction.columns:
        if name in ADDED_COLUMNS:
            continue
        base = str(name)[:-len(INPUT_MARK)] if str(name).endswith(
            INPUT_MARK
        ) else None
        names[name] = base if base in ADDED_COLUMNS else name
    frame = prediction[list(names)].rename(columns=names).reset_index(
        drop=True
    )

    return frame


def number_values(values: pd.Series, lowest: float, open_low: bool) -> tuple:
    """
    値段か原価の列の値を数にする関数

    Parameters
    ----------
    values: pd.Series
        列の値
    lowest: float
        下限
    open_low: bool
        下限を含まないか (値段は 0 より大きい)

    Returns
    ----------
    found: tuple
        float64 の値 (読めない値は nan) と, 読めた行の印
    """
    # 真偽値, 数でない値, 範囲の外の値の除外
    numbers = np.array([
        float(v) if is_number(v) else math.nan for v in values.tolist()
    ], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        good = np.isfinite(numbers) & (
            numbers > lowest if open_low else numbers >= lowest
        )
    numbers[~good] = math.nan
    found = (numbers, good)

    return found


def candidate_units(
    bundle: Bundle, frame: pd.DataFrame, current: np.ndarray,
    rows: np.ndarray, changes: list, batch_size: int, device: str
) -> tuple[np.ndarray, list[str]]:
    """
    値段を置き換えた行で, 割合ごとの売れた数を予測する関数

    Parameters
    ----------
    bundle: Bundle
        値段の一式
    frame: pd.DataFrame
        元の列の表
    current: np.ndarray
        行ごとの今の値段
    rows: np.ndarray
        予測する行の印
    changes: list
        今の値段に対する変える割合の並び
    batch_size: int
        1 回に渡す行の数
    device: str
        使うデバイス

    Returns
    ----------
    found: tuple[np.ndarray, list[str]]
        (件数, 割合の数) の予測した数 (予測しない行は nan) と, 予測する行の
        うち, どれかの割合で予測できない理由 (できる行は空の文字)
    """
    # 割合ごとの値段の列の置き換えと前処理 (落ちる行と欠ける行は理由)
    estimator = bundle.cache["price_estimator"]
    name = estimator["price"]
    units = np.full((len(frame), len(changes)), math.nan)
    reasons = [""] * len(frame)
    for column, change in enumerate(changes):
        changed = frame.copy()
        changed[name] = np.where(rows, current * (1 + float(change)),
                                 math.nan)
        x, lost = row_features(bundle.cache["price_dataset"], changed)
        keep = np.flatnonzero(rows & np.array([not r for r in lost],
                                              dtype=bool))
        for row in np.flatnonzero(rows):
            if lost[row] and not reasons[row]:
                reasons[row] = lost[row]
        if len(keep):
            units[keep, column] = raw_units(
                bundle.model(), x[keep], estimator["model"], batch_size,
                device
            )

    # 数でない予測
    for row in np.flatnonzero(rows):
        if not reasons[row] and not np.isfinite(units[row]).all():
            reasons[row] = NOT_FINITE
    found = (units, reasons)

    return found


def cost_values(
    bundle: Bundle, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    """
    行ごとの原価を返す関数

    Parameters
    ----------
    bundle: Bundle
        値段の一式
    frame: pd.DataFrame
        元の列の表

    Returns
    ----------
    found: tuple[np.ndarray, np.ndarray, Optional[str]]
        float64 の原価 (原価の列が無い一式と表は 0), 読めた行の印, 使った
        原価の列の名前 (無ければ None)
    """
    # 原価の列が無い一式か表は, 利益を売上で数える 0
    name = bundle.info["price"]["cost"]
    if name is None or name not in frame.columns:
        found = (np.zeros(len(frame)), np.ones(len(frame), dtype=bool),
                 None)
        return found
    numbers, good = number_values(frame[name], 0.0, False)
    found = (numbers, good, name)

    return found


def elasticity_of(now: np.ndarray, raised: np.ndarray) -> np.ndarray:
    """
    行ごとの弾力性を返す関数

    Parameters
    ----------
    now: np.ndarray
        今の値段での予測した数
    raised: np.ndarray
        1 % 上げた値段での予測した数

    Returns
    ----------
    elasticity: np.ndarray
        100 × (raised − now) / now (now が 0 の行は nan)
    """
    # 今の予測した数が 0 の行は変わる割合を決められない
    with np.errstate(divide="ignore", invalid="ignore"):
        elasticity = np.where(now != 0, 100 * (raised - now) / now,
                              math.nan)

    return elasticity


def price_elasticity(now: Any, raised: Any) -> dict:
    """
    値段を 1 % 上げたときの弾力性を数える関数

    Parameters
    ----------
    now: Any
        今の値段での予測した数 (N,)
    raised: Any
        1 % 上げた値段での予測した数 (N,)

    Returns
    ----------
    found: dict
        items (列 elasticity の並び), mean (nan でない行の平均. 無ければ
        nan), counted (平均に数えた行の数)

    Raises
    ----------
    ValueError
        長さが 0 か違う場合と, 予測した数が 0 以上の有限の数でない場合
    """
    # 0 以上の有限の数の, 同じ長さの並び (ML-Found の price_elasticity と
    # 同じ決まり)
    now = np.asarray(now, dtype=np.float64).reshape(-1)
    raised = np.asarray(raised, dtype=np.float64).reshape(-1)
    if not len(now) or now.shape != raised.shape:
        raise ValueError("予測した数は同じ長さの 1 つ以上の並びにして"
                         "ください")
    if not (np.isfinite(now).all() and np.isfinite(raised).all()) or (
        (now < 0).any() or (raised < 0).any()
    ):
        raise ValueError("予測した数は 0 以上の有限の数にしてください")

    # 行ごとの弾力性と, nan でない行の平均
    values = elasticity_of(now, raised)
    counted = int(np.isfinite(values).sum())
    found = {
        "items": {"elasticity": values.tolist()},
        "mean": float(values[np.isfinite(values)].mean()) if counted
        else math.nan,
        "counted": counted,
    }

    return found


def choose(
    profits: np.ndarray, changes: list, max_change: Optional[float]
) -> np.ndarray:
    """
    行ごとに, 上限の中で見込みの利益が最も大きい候補の番号を返す関数

    Parameters
    ----------
    profits: np.ndarray
        (行の数, 候補の数) の見込みの利益
    changes: list
        候補の割合
    max_change: Optional[float]
        変える幅の上限 (None は上限無し)

    Returns
    ----------
    chosen: np.ndarray
        行ごとの選んだ候補の番号
    """
    # 上限の中の候補 (10 進の表記どおりに比べる. 今の値段はいつも選べる)
    exact = [decimal_of(c) for c in changes]
    allowed = np.array([
        max_change is None or abs(c) <= decimal_of(max_change)
        for c in exact
    ], dtype=bool)
    masked = np.where(allowed, profits, -math.inf)
    best = masked.max(axis=1)

    # 最も大きい利益との差が小さい候補のうち, 今の値段に近い候補
    tolerance = TIE * np.maximum(1.0, np.abs(best))
    near = allowed & (profits >= (best - tolerance)[:, None])
    order = sorted(range(len(exact)), key=lambda j: (abs(exact[j]),
                                                      exact[j]))
    chosen = np.full(len(profits), -1, dtype=np.int64)
    for j in reversed(order):
        chosen[near[:, j]] = j

    return chosen


def profit_lift(
    price: Any, candidates: list, units: Any, cost: Any = None,
    max_change: Optional[float] = None
) -> dict:
    """
    選んだ値段と今の値段の見込みの利益を比べる関数

    Parameters
    ----------
    price: Any
        行ごとの今の値段 (N,)
    candidates: list
        今の値段に対する変える割合の候補 (0 を含む)
    units: Any
        (N, 候補の数) の候補ごとの予測した数
    cost: Any = None
        行ごとの原価 (None は全て 0. 利益は売上)
    max_change: Optional[float] = None
        変える幅の上限 (None は上限無し)

    Returns
    ----------
    found: dict
        items (change, price, profit, current_profit の並び), total,
        current_total, gain, ratio (current_total が 0 以下なら nan),
        changed

    Raises
    ----------
    ValueError
        形が合わない場合と, 値が決まりの外の場合
    """
    # 形と値の確かめ (ML-Found の profit_lift と同じ決まり)
    price = np.asarray(price, dtype=np.float64).reshape(-1)
    units = np.asarray(units, dtype=np.float64)
    cost = np.zeros(len(price)) if cost is None else np.asarray(
        cost, dtype=np.float64
    ).reshape(-1)
    errors = check_candidates(list(candidates), "candidates")
    if max_change is not None:
        errors.extend(check_max_change(max_change, "max_change"))
    if not len(price) or units.shape != (len(price), len(candidates)) or (
        cost.shape != price.shape
    ):
        errors.append("値段, 予測した数, 原価の形が合いません")
    elif not (np.isfinite(price).all() and (price > 0).all()):
        errors.append("今の値段は 0 より大きい有限の数にしてください")
    elif not (np.isfinite(units).all() and (units >= 0).all()):
        errors.append("予測した数は 0 以上の有限の数にしてください")
    elif not (np.isfinite(cost).all() and (cost >= 0).all()):
        errors.append("原価は 0 以上の有限の数にしてください")
    if errors:
        raise ValueError("見込みの利益の比べの誤り: " + " / ".join(errors))

    # 候補ごとの見込みの利益と, 選んだ候補
    changes = [float(c) for c in candidates]
    prices = price[:, None] * (1 + np.array(changes))[None, :]
    profits = (prices - cost[:, None]) * units
    chosen = choose(profits, list(candidates), max_change)
    rows = np.arange(len(price))
    now = [decimal_of(c) for c in candidates].index(Decimal(0))
    profit = profits[rows, chosen]
    current = profits[:, now]
    total, current_total = float(profit.sum()), float(current.sum())
    found = {
        "items": {
            "change": [changes[j] for j in chosen],
            "price": prices[rows, chosen].tolist(),
            "profit": profit.tolist(), "current_profit": current.tolist(),
        },
        "total": total, "current_total": current_total,
        "gain": total - current_total,
        "ratio": total / current_total if current_total > 0 else math.nan,
        "changed": int((chosen != now).sum()),
    }

    return found


def predict_price(
    bundle: Bundle, source: Any, options: Optional[dict] = None,
    batch_size: int = 256, device: str = "cpu"
) -> pd.DataFrame:
    """
    新しい行の今の値段での売れた数と弾力性を返す関数

    Parameters
    ----------
    bundle: Bundle
        値段の一式
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
        元の列 (足す列と同じ名前は「(入力)」付き) と, status, reason,
        current_price, cost, units, current_profit, elasticity

    Raises
    ----------
    ValueError
        表でない場合と, 値段の列か前処理に要る列が無い場合
    """
    # 表の読み込みと, 値段と原価の確かめ
    frame = read_table(source, options, "値段を選ぶ表")
    price = bundle.info["price"]
    if price["price"] not in frame.columns:
        raise ValueError(f"表に値段の列 {price['price']} がありません (今の"
                         "値段に使います)")
    current, priced = number_values(frame[price["price"]], 0.0, True)
    cost, costed, _ = cost_values(bundle, frame)
    reasons = [
        "" if p and c else (BAD_PRICE if not p else BAD_COST)
        for p, c in zip(priced, costed)
    ]

    # 今の値段と, 弾力性の上げ幅だけ上げた値段での予測した数
    rows = np.array([not r for r in reasons], dtype=bool)
    units, lost = candidate_units(
        bundle, frame, current, rows, [0, price["elasticity_step"]],
        batch_size, device
    )
    for row in np.flatnonzero(rows):
        reasons[row] = lost[row]
    ok = np.array([not r for r in reasons], dtype=bool)

    # 元の列の後ろに予測の列 (同じ名前の元の列は「(入力)」)
    now = np.where(ok, units[:, 0], math.nan)
    result = frame.rename(columns={c: frame_name(c) for c in frame.columns})
    result["status"] = np.where(ok, OK, SKIPPED)
    result["reason"] = pd.Series(reasons, dtype=object)
    result["current_price"] = np.where(ok, current, math.nan)
    result["cost"] = np.where(ok, cost, math.nan)
    result["units"] = now
    result["current_profit"] = (result["current_price"]
                                - result["cost"]) * now
    result["elasticity"] = math.nan
    if ok.any():
        found = price_elasticity(units[ok, 0], units[ok, 1])
        result.loc[ok, "elasticity"] = found["items"]["elasticity"]

    return result


def column_in(name: Optional[str], prediction: pd.DataFrame) -> Optional[str]:
    """
    元の列の, 予測の表での名前を返す関数

    Parameters
    ----------
    name: Optional[str]
        元の列の名前 (無ければ None)
    prediction: pd.DataFrame
        predict_price の表

    Returns
    ----------
    found: Optional[str]
        予測の表の列の名前 (「(入力)」付きも. 表に無ければ None)
    """
    # 足す列と同じ名前の元の列の「(入力)」付きの名前
    if name is None:
        return None
    key = frame_name(name)
    found = key if key in prediction.columns else None

    return found


def group_choices(
    bundle: Bundle, frame: pd.DataFrame, keys: pd.Series,
    batch_size: int
) -> tuple[pd.Index, dict, np.ndarray]:
    """
    元の表の行を鍵でまとめ, 実験と同じまとめ方で鍵ごとの値段を選ぶ関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    frame: pd.DataFrame
        元の表の, 一式で出せる行
    keys: pd.Series
        行ごとの鍵 (文字)
    batch_size: int
        1 回に渡す行の数

    Returns
    ----------
    found: tuple[pd.Index, dict, np.ndarray]
        比べられる鍵, その鍵の profit_lift の結果, 鍵ごとの行の数
    """
    # 鍵ごとの今の値段 (元の値段の中央値. 行ごとなら行の値段) と原価の平均
    price = bundle.info["price"]
    codes, names = pd.factorize(keys, sort=False)
    raw, _ = number_values(frame[price["price"]], 0.0, True)
    cost, _, _ = cost_values(bundle, frame)
    medians = pd.Series(raw).groupby(codes).median().to_numpy()
    current = medians[codes]
    rows = np.ones(len(frame), dtype=bool)

    # 候補ごとの行の予測した数の鍵ごとの平均 (予測できない行のある鍵は除く)
    units, lost = candidate_units(bundle, frame, current, rows,
                                  price["candidates"], batch_size, "cpu")
    counts = np.bincount(codes, minlength=len(names))
    bad = np.bincount(codes, weights=np.array([bool(r) for r in lost],
                                              dtype=np.float64),
                      minlength=len(names)) > 0
    sums = np.zeros((len(names), len(price["candidates"])))
    np.add.at(sums, codes, np.nan_to_num(units))
    means = sums / counts[:, None]
    costs = np.bincount(codes, weights=cost, minlength=len(names)) / counts
    use = ~bad
    lift = profit_lift(medians[use], price["candidates"], means[use],
                       costs[use], price["max_change"]) if use.any() else {}
    found = (pd.Index(names[use]), lift, counts[use])

    return found


def choice_check(
    bundle: Bundle, data: Any, options: Optional[dict], sources: dict,
    batch_size: int
) -> tuple[Optional[dict], list[str]]:
    """
    元の表で選んだ値段を, 実験の選んだ値段と比べる関数

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
    # 鍵の列 (商品の id の列. 無ければ行の id の列) と一式で出せる行
    estimator = sources["estimator"]
    key = estimator["product"] or estimator["id"]
    frame = read_table(data, options, "元の表")
    if key not in frame.columns:
        note = (f"元の表に鍵の列 {key} が無いので, 一式で選んだ値段を実験の"
                "選んだ値段と比べません")
        found = (None, [note])
        return found
    prediction = predict_price(bundle, frame, None, batch_size, "cpu")
    ok = (prediction["status"] == OK).to_numpy()
    frame = frame[ok].reset_index(drop=True)
    keys = frame[key].astype(str)

    # 実験の表に 1 つの分割の 1 行だけがあり, 行の数が同じ鍵
    expected = sources["choices"]
    single = expected.drop_duplicates(expected.columns[0], keep=False)
    single = single.set_index(single.columns[0])
    names, lift, counts = group_choices(bundle, frame, keys, batch_size)
    matched = np.array([
        name in single.index and int(single.loc[name, "rows"]) == count
        for name, count in zip(names, counts)
    ], dtype=bool)
    if not matched.any():
        note = ("元の表の鍵に, 実験と同じ行で選んだ鍵が無いので (商品が分割"
                "をまたぐ表など), 一式で選んだ値段を実験の選んだ値段と比べ"
                "ません")
        found = (None, [note])
        return found

    # 選んだ割合が同じ割合と, 1 行あたりの見込みの利益の差の最大
    recorded = single.loc[names[matched]]
    change = np.array(lift["items"]["change"])[matched]
    profit = np.array(lift["items"]["profit"])[matched]
    same = np.abs(change - recorded["change"].to_numpy()) <= SAME_CHANGE
    check = {"rows": int(matched.sum()), "agreement": float(same.mean()),
             "max_diff": float(np.abs(
                 profit - recorded["profit"].to_numpy()
             ).max())}
    notes = []
    if check["agreement"] < AGREEMENT:
        notes.append(
            f"元の表で一式が選んだ値段が, 実験の選んだ値段と "
            f"{check['agreement'] * 100:.1f} % の鍵でしか同じになりません "
            "(前処理のレシピか重みが学習と違う可能性があります)"
        )
    if check["rows"] < len(expected):
        notes.append(
            f"実験の選んだ値段の {len(expected):,} 行のうち, 比べたのは "
            f"{check['rows']:,} 行です (分割をまたぐ商品と, 元の表に無いか"
            "出せない行のある鍵は比べません)"
        )
    found = (check, notes)

    return found


def probe_model(bundle: Bundle, width: int, batch_size: int) -> None:
    """
    0 の行をモデルに渡し, 売れた数の予測の形と値を確かめる関数

    Parameters
    ----------
    bundle: Bundle
        値段の一式 (cache に price_estimator)
    width: int
        予測に使う列の数
    batch_size: int
        1 回に渡す行の数

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        予測が件数の数でない場合
    """
    # 予測の形の確かめ (raw_units の中) と, 値が数かの確かめ
    units = raw_units(
        bundle.model(), np.zeros((PROBE_ROWS, width), dtype=np.float32),
        bundle.cache["price_estimator"]["model"], batch_size, "cpu"
    )
    if not np.isfinite(units).all():
        raise ValueError("0 の行の売れた数の予測が数ではありません")


def price_section(sources: dict) -> dict:
    """
    成果物から一式の price の欄を作る関数

    Parameters
    ----------
    sources: dict
        run_sources の結果

    Returns
    ----------
    price: dict
        price の欄 (check は None)
    """
    # 成果物の写し (損失で割った値と出力の決まりは price_estimator.json
    # の中だけ)
    estimator = sources["estimator"]
    model = estimator["model"]
    price = {
        "estimator": ESTIMATOR_FILE,
        "recipe": RECIPE_FILE,
        "features": list(estimator["features"]),
        "price": estimator["price"],
        "units": sources["recipe"]["target"],
        "cost": estimator["cost"],
        "product": estimator["product"],
        "id": estimator["id"],
        "input": model["input"],
        "output": model["units"],
        "candidates": [plain(c) for c in estimator["candidates"]],
        "max_change": plain(estimator["max_change"]),
        "elasticity_step": float(estimator["elasticity_step"]),
        "current": estimator["current"],
        "rows": int(len(sources["choices"])),
        "train": {key: plain(estimator["train"][key]) for key in TRAIN_KEYS},
        "evaluations": copy.deepcopy(sources["evaluations"]),
        "check": None,
    }

    return price


def make_price_bundle(
    path: Union[str, Path], run: Union[str, Path],
    name: Optional[str] = None, data: Any = None,
    options: Optional[dict] = None, copy_weights: bool = True,
    overwrite: bool = False, pretrained: Optional[dict] = None,
    batch_size: int = 256
) -> dict:
    """
    新しい行の値段を選ぶ一式を, 値段を変える判断の実験から作る関数

    Parameters
    ----------
    path: Union[str, Path]
        作るディレクトリ
    run: Union[str, Path]
        ML-Found の実験のディレクトリ (price_estimator.json,
        price_recipe.yaml, price_choices.csv, evaluations の
        price_<分割>.json, final_model.pt, model_spec.yaml を使い, 実験
        の状態を記録する)
    name: Optional[str] = None
        一式の名前 (省略するとディレクトリの名前)
    data: Any = None
        学習に使った元の表 (渡すと値段を選んで実験の選んだ値段と比べる)
    options: Optional[dict] = None
        data のファイルの読み込みの設定 (load_data に渡す)
    copy_weights: bool = True
        重みを一式に写すか (写さない場合は絶対パスで指す)
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
    # 前に作った一式だけの置き換え
    final = Path(path).expanduser().resolve()
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"{final} は既にあります")
        if not (final / "inference.yaml").is_file():
            raise FileExistsError(f"{final} は推論の一式ではないので置き換えません")
    if batch_size < 1:
        raise ValueError("batch_size は 1 以上にしてください")

    # 実験の成果物と, モデルの入力と出力の約束
    sources = run_sources(run)
    estimator = sources["estimator"]
    spec_dict = load_spec(sources["spec"])
    check_units(spec_dict)
    check_price_spec(spec_dict, estimator["model"])
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
        empty = Dataset(path=writing, info={}, frames={}, recipe={})
        section = price_section(sources)
        info = {
            "format_version": PRICE_BUNDLE_VERSION,
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
            "decision": {"type": "price",
                         "candidates": list(section["candidates"]),
                         "max_change": section["max_change"]},
            "check": None,
            "uncertainty": None,
            "notes": bundle_notes(empty, sources["run"]) + weight_notes,
            "price": section,
        }

        # 0 の行の予測と, 元の表で選んだ値段と実験の選んだ値段の比べ
        building = Bundle(path=writing, info=info, spec=spec_dict,
                          dataset=empty)
        building.cache["price_estimator"] = estimator
        building.cache["price_dataset"] = price_dataset(
            writing, sources["recipe"], estimator
        )
        probe_model(building, len(estimator["features"]), batch_size)
        if data is not None:
            found, notes = choice_check(building, data, options, sources,
                                        batch_size)
            info["price"]["check"] = found
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


def check_price_bundle(bundle: Bundle) -> None:
    """
    値段の一式の成果物を確かめて cache に置く関数

    Parameters
    ----------
    bundle: Bundle
        値段の一式

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
    # 値段の決め方とレシピの写し (一式の外は誤り)
    price = bundle.info["price"]
    estimator = read_estimator(inside(bundle.path, price["estimator"]))
    recipe = read_recipe(inside(bundle.path, price["recipe"]), estimator)
    model = estimator["model"]
    mismatch = [
        key for key, value in (
            ("features", estimator["features"]),
            ("price", estimator["price"]), ("units", recipe["target"]),
            ("cost", estimator["cost"]), ("product", estimator["product"]),
            ("id", estimator["id"]), ("input", model["input"]),
            ("output", model["units"]),
            ("candidates", estimator["candidates"]),
            ("max_change", estimator["max_change"]),
            ("elasticity_step", estimator["elasticity_step"]),
            ("current", estimator["current"]),
        ) if price[key] != value
    ]
    if mismatch:
        raise ValueError(
            f"price の {', '.join(mismatch)} が値段の決め方 "
            f"{price['estimator']} と違います"
        )
    bundle.cache["price_estimator"] = estimator
    bundle.cache["price_dataset"] = price_dataset(bundle.path, recipe,
                                                  estimator)

    # モデルの定義の入力と出力
    check_price_spec(bundle.spec, model)


def measure(bundle: Bundle, prediction: pd.DataFrame) -> Optional[dict]:
    """
    渡した表の売れた数で, 今の値段での予測の誤差を数え直す関数

    Parameters
    ----------
    bundle: Bundle
        値段の一式
    prediction: pd.DataFrame
        predict_price の表

    Returns
    ----------
    measured: Optional[dict]
        rows, excluded, mae, rmse (売れた数の列が無いか数える行が無ければ
        None)
    """
    # 売れた数の列 (無ければ数えない)
    name = column_in(bundle.info["price"]["units"], prediction)
    if name is None:
        return None

    # 予測した行のうち, 売れた数が 0 以上の有限の数の行
    ok = prediction[prediction["status"] == OK]
    actual, good = number_values(ok[name], 0.0, False)
    if not good.any():
        return None
    errors = ok["units"].to_numpy(dtype=np.float64)[good] - actual[good]
    measured = {
        "rows": int(good.sum()), "excluded": int((~good).sum()),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt((errors ** 2).mean())),
    }

    return measured


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
        分割ごとの誤差, 弾力性, 見込みの利益の比べの文 (無ければ, 予測した
        数と弾力性を見て確かめる文)
    """
    # 評価の無い実験の, 予測した数と弾力性を見る文
    if not evaluations:
        sentences = ["実験の評価が無いので, 予測した数と弾力性を見て確かめる"]
        return sentences
    sentences = []
    for split, row in evaluations.items():
        elasticity = ("弾力性は無し" if row["elasticity"] is None
                      else f"弾力性の平均 {row['elasticity']:.3g} %")
        lift = (
            f"選んだ値段の見込みの利益の差 {row['profit_gain']:.4g} (今の値段"
            "の利益の和が 0 以下なので比は無し)"
            if row["profit_ratio"] is None else
            f"選んだ値段の見込みの利益は今の値段の {row['profit_ratio']:.4g}"
            f" 倍 (差 {row['profit_gain']:.4g}"
        )
        if row["profit_ratio"] is not None:
            lift += (f". 値段を変える商品 {row['changed']} / "
                     f"{row['products']})")
        sentences.append(
            f"{SPLIT_NAMES.get(split, split)}の売れた数の平均絶対誤差 "
            f"{row['mae']:.4g}, {elasticity}, {lift}"
        )

    return sentences


def cost_sentence(bundle: Bundle, prediction: pd.DataFrame) -> Optional[str]:
    """
    原価を 0 として数えることを述べる文を返す関数

    Parameters
    ----------
    bundle: Bundle
        値段の一式
    prediction: pd.DataFrame
        predict_price の表

    Returns
    ----------
    sentence: Optional[str]
        原価の列が無い一式か表の文 (原価の列を使えば None)
    """
    # 一式に原価の列が無いか, 表に原価の列が無い
    name = bundle.info["price"]["cost"]
    if name is None:
        sentence = "原価の列が無いので, 利益は売上 (原価 0) で数える"
        return sentence
    if column_in(name, prediction) is None:
        sentence = (f"表に原価の列 {name} が無いので, 利益は売上 (原価 0) "
                    "で数える")
        return sentence

    return None


def conclude_price(bundle: Bundle, prediction: pd.DataFrame) -> dict:
    """
    今の値段での売れた数と弾力性から, 件数と要約の結論を出す関数

    Parameters
    ----------
    bundle: Bundle
        値段の一式
    prediction: pd.DataFrame
        predict_price の表

    Returns
    ----------
    conclusion: dict
        frame (decision を足した表. 予測した行は空, しなかった行は予測
        なし), summary (rows, predicted, skipped, id, product, cost,
        units, elasticity, train, evaluations, measured), sentences, rules
        (空)
    """
    # 予測したかと, 今の値段での予測した数と弾力性
    price = bundle.info["price"]
    frame = prediction.copy()
    ok = (frame["status"] == OK).to_numpy()
    frame["decision"] = np.where(ok, "", NO_PREDICTION)
    units = frame["units"].to_numpy(dtype=np.float64)[ok]
    values = frame["elasticity"].to_numpy(dtype=np.float64)[ok]
    counted = int(np.isfinite(values).sum())
    elasticity = {
        "mean": float(values[np.isfinite(values)].mean()) if counted
        else None,
        "counted": counted,
    } if ok.any() else None
    summary = {
        "rows": int(len(frame)), "predicted": int(ok.sum()),
        "skipped": int(len(frame) - ok.sum()),
        "id": column_in(price["id"], prediction),
        "product": column_in(price["product"], prediction),
        "cost": column_in(price["cost"], prediction),
        "units": {"mean": float(units.mean()), "total": float(units.sum())}
        if ok.any() else None,
        "elasticity": elasticity,
        "train": copy.deepcopy(price["train"]),
        "evaluations": copy.deepcopy(price["evaluations"]),
        "measured": measure(bundle, prediction),
    }

    # 件数, 予測した数, 弾力性, 原価, 実験の評価, 数え直しの文
    head = (f"{summary['rows']:,} 行のうち {summary['predicted']:,} 行の"
            "売れた数を今の値段 (行の値段の列) で予測した")
    if summary["skipped"]:
        head += f" ({summary['skipped']:,} 行は出せず)"
    sentences = [head]
    if summary["units"] is not None:
        sentences.append(
            f"今の値段での予測した数の平均 {summary['units']['mean']:.4g} "
            f"(和 {summary['units']['total']:.4g})"
        )
    if elasticity is not None and elasticity["mean"] is not None:
        sentences.append(
            f"値段を {price['elasticity_step'] * 100:g} % 上げたときに予測"
            f"した数が変わる割合 (弾力性) の平均 {elasticity['mean']:.3g} % "
            f"({elasticity['counted']:,} 行)"
        )
    cost = cost_sentence(bundle, prediction)
    if cost is not None:
        sentences.append(cost)
    sentences.extend(evaluation_sentences(price["evaluations"]))
    measured = summary["measured"]
    if measured is not None:
        sentences.append(
            f"渡した表の売れた数で数えた平均絶対誤差 {measured['mae']:.4g}, "
            f"二乗平均平方根誤差 {measured['rmse']:.4g} "
            f"({measured['rows']:,} 行. 記録の値段での予測の誤差)"
        )
    conclusion = {"frame": frame, "summary": summary,
                  "sentences": sentences, "rules": {}}

    return conclusion


def check_price_decision(decision: dict) -> list[str]:
    """
    値段の一式の値段の判断の決め方の誤りを一覧で返す関数

    Parameters
    ----------
    decision: dict
        意思決定の決め方

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 値段の判断の型と書ける項目 (type, candidates, max_change)
    if decision.get("type") != "price":
        return ["値段の一式の意思決定は値段の判断 (type: price) だけです"]
    errors = [
        f"値段の一式の値段の判断に書けない項目です: {key} (書けるのは type, "
        "candidates, max_change だけ. 原価は表の原価の列から読む)"
        for key in decision if key not in PRICE_DECISION_KEYS
    ]
    errors.extend(check_candidates(decision.get("candidates"),
                                   "candidates (値段の候補)"))
    errors.extend(check_max_change(decision.get("max_change"),
                                   "max_change"))

    return errors


def limited_rows(
    profits: np.ndarray, chosen: np.ndarray
) -> int:
    """
    上限を超える候補の方が見込みの利益の大きい行の数を返す関数

    Parameters
    ----------
    profits: np.ndarray
        (行の数, 候補の数) の見込みの利益
    chosen: np.ndarray
        行ごとの選んだ値段の見込みの利益

    Returns
    ----------
    count: int
        全ての候補の中の最も大きい利益が, 同じ利益とみなす差を超えて選んだ
        値段の利益より大きい行の数
    """
    # 同じ利益とみなす差 (選び方と同じ)
    best = profits.max(axis=1) if len(profits) else np.zeros(0)
    count = int((best - chosen > TIE * np.maximum(1.0, np.abs(best))).sum())

    return count


def decision_sentences(summary: dict) -> list[str]:
    """
    値段の判断の件数と見込みの利益を述べる文を返す関数

    Parameters
    ----------
    summary: dict
        decide_price の summary

    Returns
    ----------
    sentences: list[str]
        値段を変える行の数と見込みの利益, 上限で選ばない行, 決めない行の文
    """
    # 値段を変える行の数, 候補と上限, 見込みの利益の和と比べ
    candidates = ", ".join(f"{c:g}" for c in summary["candidates"])
    ratio = (f"比 {summary['ratio']:.4g}" if summary["ratio"] is not None
             else "今の値段の利益の和が 0 以下なので比は無し")
    sentences = [
        f"{summary['rows']:,} 行のうち {summary['changed']:,} 行の値段を変え"
        f"る (上げる {summary['raised']:,} 行, 下げる "
        f"{summary['lowered']:,} 行. 候補 [{candidates}], 変える幅の上限 "
        f"{summary['max_change']:g}). 見込みの利益の和は "
        f"{summary['total']:.4g} (今の値段のままなら "
        f"{summary['current_total']:.4g}. 差 {summary['gain']:.4g}, {ratio})"
    ]
    if summary["limited"]:
        sentences.append(
            f"上限を超える候補の方が見込みの利益の大きい行が "
            f"{summary['limited']:,} 行ある (上限は守る指標なので選ばない)"
        )
    if summary["undecided"]:
        sentences.append(f"候補の値段で予測できない {summary['undecided']:,}"
                         " 行は決めない")

    return sentences


def decide_price(
    bundle: Bundle, prediction: pd.DataFrame, decision: dict
) -> dict:
    """
    予測した行ごとに, 上限の中で見込みの利益が最も大きい値段を選ぶ関数

    Parameters
    ----------
    bundle: Bundle
        値段の一式
    prediction: pd.DataFrame
        predict_price の表 (か conclude_price の表)
    decision: dict
        {type: price, candidates, max_change}

    Returns
    ----------
    result: dict
        frame (action, change, price, price_units, profit を足した表),
        summary (rows, undecided, changed, raised, lowered, total,
        current_total, gain, ratio, candidates, max_change, limited),
        sentences, notes (前提), decision (使った決め方)

    Raises
    ----------
    ValueError
        決め方に誤りがある場合と, 表に予測の列が無い場合
    """
    # 決め方と表の確認
    decision = plain(dict(decision))
    errors = check_price_decision(decision)
    missing = [c for c in ("status", "current_price", "cost")
               if c not in prediction.columns]
    if missing:
        errors.append(f"予測の表に {', '.join(missing)} の列がありません")
    if errors:
        raise ValueError("意思決定の決め方の誤り: " + " / ".join(errors))

    # 予測した行の候補ごとの予測した数 (元の列から求め直す)
    changes = list(decision["candidates"])
    ok = (prediction["status"] == OK).to_numpy()
    current = prediction["current_price"].to_numpy(dtype=np.float64)
    cost = prediction["cost"].to_numpy(dtype=np.float64)
    units, lost = candidate_units(bundle, original_frame(prediction),
                                  current, ok, changes, DECIDE_BATCH, "cpu")
    rows = np.flatnonzero(ok & np.array([not r for r in lost], dtype=bool))

    # 上限の中の見込みの利益が最も大きい値段 (決めない行は空)
    values = {key: np.full(len(prediction), math.nan)
              for key in DECISION_COLUMNS[1:]}
    action = pd.array([None] * len(prediction), dtype="boolean")
    lift = {"total": 0.0, "current_total": 0.0, "gain": 0.0,
            "ratio": math.nan}
    limited = 0
    if len(rows):
        lift = profit_lift(current[rows], changes, units[rows], cost[rows],
                           decision["max_change"])
        index = [decimal_of(c) for c in changes]
        picked = [index.index(decimal_of(c)) for c in lift["items"]["change"]]
        values["change"][rows] = lift["items"]["change"]
        values["price"][rows] = lift["items"]["price"]
        values["price_units"][rows] = units[rows, picked]
        values["profit"][rows] = lift["items"]["profit"]
        action[rows] = values["change"][rows] != 0
        prices = current[rows, None] * (1 + np.array(changes, dtype=float))
        limited = limited_rows((prices - cost[rows, None]) * units[rows],
                               np.asarray(lift["items"]["profit"]))
    frame = prediction.copy()
    frame["action"] = action
    for key, column in values.items():
        frame[key] = column
    chosen = values["change"][rows]
    summary = {
        "rows": int(len(rows)), "undecided": int(ok.sum() - len(rows)),
        "changed": int((chosen != 0).sum()),
        "raised": int((chosen > 0).sum()),
        "lowered": int((chosen < 0).sum()),
        "total": float(lift["total"]),
        "current_total": float(lift["current_total"]),
        "gain": float(lift["gain"]),
        "ratio": None if math.isnan(lift["ratio"]) else float(lift["ratio"]),
        "candidates": changes, "max_change": decision["max_change"],
        "limited": limited,
    }

    # 前提, 値段を変える行の数と見込みの利益の文 (前提は notes にも)
    notes = [
        "売れた数は値段を入力に入れたモデルの予測で, 試しに値段を変えた記録"
        " (無作為に近い) を前提にしている (売れ行きに合わせて値段を決めた記録"
        "から学ぶと偏る)",
        "選んだ値段の利益は見込みで, 値段を変える試験で確かめる (機械学習の"
        "結果は判断の材料)",
        f"候補は今の値段に対する変える割合で, 変える幅の上限 "
        f"{decision['max_change']:g} (守る指標) を超える候補は選ばない",
    ]
    cost_note = cost_sentence(bundle, prediction)
    if cost_note is not None:
        notes.append(cost_note.replace("数える", "比べた"))
    sentences = [*notes, *decision_sentences(summary)]
    result = {"frame": frame, "summary": summary, "sentences": sentences,
              "notes": notes, "decision": decision}

    return result


def price_of_bundle(bundle: Bundle) -> Optional[dict]:
    """
    値段の一式の price の欄を, 説明の形で返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    price: Optional[dict]
        price の欄の写し (値段の一式でなければ None)
    """
    # 画面が値段の候補と実験の評価を出すための写し
    if not bundle.from_price:
        return None
    price = copy.deepcopy(bundle.info["price"])

    return price

