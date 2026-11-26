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
    DEMAND_BUNDLE_VERSION, Bundle, bundle_notes, inside, is_level,
    load_bundle, plan_weights, run_summary
)
from ml_inference.conclude import NO_PREDICTION, is_number, plain
from ml_inference.grouping import read_table
from ml_inference.price import decimal_of, number_values
from ml_inference.units import check_units
from ml_inference.uplift import read_json

# 実験の中の成果物 (ML-Quest の時間をまたぐ量の判断の骨組みが書く名前)
ESTIMATOR_FILE = "demand_estimator.json"
RECIPE_FILE = "demand_recipe.yaml"
EVALUATION_FILE = "evaluations/demand_{split}.json"

# 読める需要の決め方の成果物の版と, 行の id を書かないレシピの行の id の列
ESTIMATOR_VERSION = 1
ROW_ID = "row_id"

# 需要の決め方の成果物の鍵と, モデルの名前の鍵と, 学習用の分割の要約の鍵
ESTIMATOR_KEYS = ("format_version", "recipe", "orders", "features",
                  "product", "date", "units", "stock", "id", "horizon",
                  "interval", "levels", "shortage_cost", "holding_cost",
                  "max_order", "model", "train")
MODEL_NAMES = ("input", "quantiles")
TRAIN_KEYS = ("rows", "products", "demand_mean")

# 発注の数の表の, 商品の id と日付の列の他の列と, 空にならない数の列
ORDERS_COLUMNS = ("split", "stock", "quantile", "order", "previous",
                  "demand")
ORDERS_NUMBERS = ("quantile", "order")

# 実験の評価を写す分割と, 評価の行の鍵 (null にもなる鍵も含む順)
EVALUATION_SPLITS = ("valid", "test")
EVALUATION_KEYS = ("rows", "products", "mae", "pinball_loss", "order_cost",
                   "baseline_cost", "order_cost_saving", "order_cost_ratio",
                   "capped")
NULLABLE_KEYS = ("order_cost_ratio",)

# 分割の日本語の名前
SPLIT_NAMES = {"valid": "検証用", "test": "テスト用"}

# demand の欄
DEMAND_KEYS = ("estimator", "recipe", "features", "product", "date",
               "units", "stock", "id", "horizon", "interval", "input",
               "output", "shortage_cost", "holding_cost", "max_order",
               "rows", "train", "evaluations", "check")

# 一式を作るときに予測を確かめる 0 の行の数
PROBE_ROWS = 2

# 予測の表に足す列 (分位点の列 q:<水準> も), 発注の判断が足す列, 同じ名前
# の元の列を「(入力)」にする列 (結論の decision も)
DEMAND_FRAME_COLUMNS = ("status", "reason", "order_date", "on_hand",
                        "previous_demand", "prediction")
DECISION_COLUMNS = ("action", "quantile", "order", "capped",
                    "expected_cost", "baseline_order", "baseline_cost")
ADDED_COLUMNS = (*DEMAND_FRAME_COLUMNS, *DECISION_COLUMNS, "decision")
QUANTILE_MARK = "q:"
INPUT_MARK = " (入力)"

# 飛ばす行の理由
BAD_STOCK = "在庫の数が 0 以上の数ではありません"
NOT_FINITE = "予測が数ではありません"

# 発注の判断で書ける項目
ORDER_DECISION_KEYS = ("type", "shortage_cost", "holding_cost", "max_order")


def quantile_name(level: float) -> str:
    """
    水準の分位点の列の名前を返す関数

    Parameters
    ----------
    level: float
        分位点の水準

    Returns
    ----------
    name: str
        q:<水準> (列の予測の分布の一式と同じ書き方)
    """
    # 短い表記の水準
    name = f"{QUANTILE_MARK}{float(level):g}"

    return name


def is_added(name: Any) -> bool:
    """
    予測の表に足す列の名前かを返す関数

    Parameters
    ----------
    name: Any
        列の名前

    Returns
    ----------
    added: bool
        足す列か分位点の列 (q: で始まる) の名前なら True
    """
    # 決まった名前と分位点の列
    added = name in ADDED_COLUMNS or str(name).startswith(QUANTILE_MARK)

    return added


def frame_name(name: Any) -> Any:
    """
    元の列の, 予測の表での名前を返す関数

    Parameters
    ----------
    name: Any
        元の列の名前

    Returns
    ----------
    found: Any
        足す列と同じ名前なら「(入力)」付き, 他はそのまま
    """
    # 足す列と同じ名前の元の列の「(入力)」付きの名前
    found = f"{name}{INPUT_MARK}" if is_added(name) else name

    return found


def column_in(name: Optional[str], prediction: pd.DataFrame) -> Optional[str]:
    """
    元の列の, 予測の表での名前を返す関数

    Parameters
    ----------
    name: Optional[str]
        元の列の名前 (無ければ None)
    prediction: pd.DataFrame
        predict_demand の表

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


def positive_cost(value: Any) -> bool:
    """
    商品 1 つあたりの損 (有限の正の数) かを確かめる関数

    Parameters
    ----------
    value: Any
        確かめる値

    Returns
    ----------
    good: bool
        真偽値を除いた有限の正の数なら True
    """
    # 真偽値を除いた有限の正の数
    good = is_number(value) and math.isfinite(value) and value > 0

    return good


def check_order_values(values: dict, where: str) -> list[str]:
    """
    品切れの損, 売れ残りの損, 発注の数の上限の誤りを一覧で返す関数

    Parameters
    ----------
    values: dict
        shortage_cost, holding_cost, max_order を持つ辞書
    where: str
        誤りの文に使う項目の頭 (空の文字も)

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 2 つの損は正の数, 上限は 0 以上の数 (どれも省けない)
    errors = []
    for key, label in (("shortage_cost", "品切れの損"),
                       ("holding_cost", "売れ残りの損")):
        if not positive_cost(values.get(key)):
            errors.append(
                f"{where}{key} ({label}) は商品 1 つあたりの正の数にして"
                "ください (省けません)"
            )
    limit = values.get("max_order")
    if not (is_number(limit) and math.isfinite(limit) and limit >= 0):
        errors.append(
            f"{where}max_order (発注の数の上限) は 0 以上の数にしてください "
            "(守る指標なので省けません)"
        )

    return errors


def check_levels(levels: Any, where: str) -> list[str]:
    """
    分位点の水準の誤りを一覧で返す関数

    Parameters
    ----------
    levels: Any
        水準の並び
    where: str
        誤りの文に使う項目の名前

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 0 より大きく 1 より小さい数の, 小さい順の並び (モデルの出力の順)
    good = isinstance(levels, list) and levels and all(
        is_level(level) for level in levels
    ) and all(
        decimal_of(a) < decimal_of(b) for a, b in zip(levels, levels[1:])
    )
    errors = [] if good else [
        f"{where} は 0 より大きく 1 より小さい数の, 小さい順の並びにして"
        "ください"
    ]

    return errors


def check_columns(estimator: dict) -> list[str]:
    """
    需要の決め方の成果物の列の誤りを一覧で返す関数

    Parameters
    ----------
    estimator: dict
        読み込んだ成果物

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 予測に使う列と, 商品の id, 日付, 売れた数, 行の id, 在庫の数の列
    features = estimator["features"]
    errors = []
    if not isinstance(features, list) or not features or not all(
        isinstance(name, str) and name for name in features
    ) or len(set(features)) != len(features):
        errors.append("features は重ならない列の名前の並びにしてください")
        features = []
    for key in ("product", "date", "units", "id"):
        if not isinstance(estimator[key], str) or not estimator[key]:
            errors.append(f"{key} は列の名前にしてください")
    stock = estimator["stock"]
    if stock is not None and (not isinstance(stock, str) or not stock):
        errors.append("stock は null か列の名前にしてください")
    if errors:
        return errors

    # 役割の列は互いに違い, 予測に使わない
    names = [estimator[key]
             for key in ("product", "date", "units", "id", "stock")
             if estimator[key] is not None]
    if len(set(names)) != len(names):
        errors.append("product, date, units, id, stock は違う列にして"
                      "ください")
    used = [name for name in names if name in features]
    if used:
        errors.append(f"予測に使う列に {', '.join(used)} があります")

    return errors


def check_estimator(estimator: Any) -> list[str]:
    """
    需要の決め方の成果物 (demand_estimator.json) の誤りを一覧で返す関数

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
        return ["需要の決め方が辞書ではありません"]
    version = estimator.get("format_version")
    if version != ESTIMATOR_VERSION:
        return [f"需要の決め方の版 {version} は読めません (版 "
                f"{ESTIMATOR_VERSION} だけ)"]
    errors = [f"需要の決め方に {key} がありません"
              for key in ESTIMATOR_KEYS if key not in estimator]
    if errors:
        return errors
    errors.extend(check_columns(estimator))
    for key in ("recipe", "orders"):
        if not isinstance(estimator[key], str) or not estimator[key]:
            errors.append(f"{key} はファイルの名前にしてください")

    # 期間, 水準, 損と上限
    for key in ("horizon", "interval"):
        if not is_whole(estimator[key], 1):
            errors.append(f"{key} は 1 以上の整数 (日数) にしてください")
    errors.extend(check_levels(estimator["levels"], "levels"))
    errors.extend(check_order_values(estimator, ""))

    # モデルの名前と, 学習用の分割の要約
    model = estimator["model"]
    if not isinstance(model, dict) or not all(
        isinstance(model.get(key), str) and model[key]
        for key in MODEL_NAMES
    ):
        errors.append("model は input と quantiles の名前にしてください")
    train = estimator["train"]
    mean = train.get("demand_mean") if isinstance(train, dict) else None
    if not isinstance(train, dict) or not all(
        is_whole(train.get(key), 0) for key in ("rows", "products")
    ) or not (mean is None or (is_number(mean) and math.isfinite(mean))):
        errors.append("train は rows と products の 0 以上の整数と, "
                      "demand_mean の数か null にしてください")

    return errors


def read_estimator(path: Path) -> dict:
    """
    需要の決め方の成果物を読み, 形を確かめる関数

    Parameters
    ----------
    path: Path
        demand_estimator.json

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
        raise FileNotFoundError(f"需要の決め方 {path} がありません")
    estimator = read_json(path)
    errors = check_estimator(estimator)
    if errors:
        raise ValueError(f"{path} の誤り: " + " / ".join(errors))

    return estimator


def read_recipe(path: Path, estimator: dict) -> dict:
    """
    需要の記録の表のレシピの写しを読んで確かめる関数

    Parameters
    ----------
    path: Path
        demand_recipe.yaml
    estimator: dict
        需要の決め方 (product, date, units, stock, id, horizon, interval)

    Returns
    ----------
    recipe: dict
        当てはめた値を含むレシピ

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        需要の記録の表のレシピでない場合と, 役割の列か期間が需要の決め方と
        食い違う場合
    """
    # 手順の並びを持つ, task が demand のレシピ
    if not path.is_file():
        raise FileNotFoundError(f"前処理のレシピ {path} がありません")
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or recipe.get("task") != "demand" or (
        not isinstance(recipe.get("steps"), list)
    ):
        raise ValueError(
            f"{path} は需要の記録の表のレシピ (task: demand と steps) では"
            "ありません"
        )

    # 役割の列と期間 (書かない項目は ML-EDA の既定)
    horizon = recipe.get("horizon", 7)
    written = {
        "units": recipe.get("target"), "product": recipe.get("product"),
        "date": recipe.get("date"), "stock": recipe.get("stock"),
        "id": recipe.get("id", ROW_ID), "horizon": horizon,
        "interval": recipe.get("interval", horizon),
    }
    mismatch = [key for key, value in written.items()
                if value != estimator[key]]
    if mismatch:
        raise ValueError(
            f"{path} の {', '.join(mismatch)} が需要の決め方と違います"
        )

    return recipe


def read_orders(path: Path, estimator: dict) -> pd.DataFrame:
    """
    実験の発注の数の表を読む関数

    Parameters
    ----------
    path: Path
        demand_orders.csv
    estimator: dict
        需要の決め方 (product, date)

    Returns
    ----------
    orders: pd.DataFrame
        商品の id の列 (文字), 日付の列 (日ごとの日時), split, stock,
        quantile, order, previous, demand の表

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        列が違う場合と, 数の列が数でない場合と, 日付が読めない場合
    """
    # 商品の id は文字 (先頭の 0 などを残すため)
    if not path.is_file():
        raise FileNotFoundError(f"発注の数の表 {path} がありません")
    product, date = estimator["product"], estimator["date"]
    orders = pd.read_csv(path, dtype={product: str, date: str},
                         keep_default_na=False, na_values=[""])
    expected = [product, date, *ORDERS_COLUMNS]
    if list(orders.columns) != expected:
        raise ValueError(
            f"{path} の列は {', '.join(expected)} にしてください (今は "
            f"{', '.join(map(str, orders.columns))})"
        )
    orders[date] = parse_days(orders[date], f"{path} の {date}")

    # 空にならない数の列と, 空にもなる在庫, 前の期間, 先の期間
    for name in ORDERS_COLUMNS[1:]:
        values = pd.to_numeric(orders[name], errors="coerce")
        numbers = values.to_numpy(dtype=np.float64)
        empty = orders[name].isna().to_numpy()
        if name in ORDERS_NUMBERS and not np.isfinite(numbers).all():
            raise ValueError(f"{path} の {name} に数でない値があります")
        if not (np.isfinite(numbers) | empty).all():
            raise ValueError(f"{path} の {name} に数でない値があります")
        orders[name] = values.astype(np.float64)

    return orders


def read_evaluations(root: Path) -> dict:
    """
    実験の需要の予測と発注の数の評価を読む関数

    Parameters
    ----------
    root: Path
        実験のディレクトリ

    Returns
    ----------
    evaluations: dict
        分割の名前から rows, products, mae, pinball_loss, order_cost,
        baseline_cost, order_cost_saving, order_cost_ratio (無ければ
        None), capped (評価の無い分割は無い)
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
    時間をまたぐ量の判断の実験のディレクトリから, 一式に要るものを読む関数

    Parameters
    ----------
    run: Union[str, Path]
        ML-Found の実験のディレクトリ

    Returns
    ----------
    sources: dict
        root, estimator, recipe_file, recipe, orders, evaluations,
        weights, spec, run (実験の状態)

    Raises
    ----------
    ValueError
        実験のディレクトリでない場合, 重みか定義が無い場合, 成果物の形が
        違う場合
    FileNotFoundError
        成果物が無い場合
    """
    # 需要の決め方, レシピ, 発注の数, 評価
    root = Path(run).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"実験のディレクトリ {root} がありません")
    estimator = read_estimator(root / ESTIMATOR_FILE)
    recipe_file = inside(root, estimator["recipe"])
    recipe = read_recipe(recipe_file, estimator)
    orders = read_orders(inside(root, estimator["orders"]), estimator)

    # 需要のモデルの重みと定義 (仮のモデルで学習した実験は定義無し)
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
        "recipe": recipe, "orders": orders,
        "evaluations": read_evaluations(root), "weights": weights,
        "spec": spec, "run": run_summary(root),
    }

    return sources


def check_demand_info(info: dict) -> list[str]:
    """
    版 14 の一式の demand の欄と, 受け持ちと出力の形を確かめる関数

    Parameters
    ----------
    info: dict
        inference.yaml の中身

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 欄, 受け持ち, 出力の形
    demand = info.get("demand")
    if not isinstance(demand, dict):
        return [f"版 {DEMAND_BUNDLE_VERSION} の一式には demand が要ります"]
    errors = [f"demand に {key} がありません"
              for key in DEMAND_KEYS if key not in demand]
    unknown = [str(key) for key in demand if key not in DEMAND_KEYS]
    if unknown:
        errors.append(f"demand に知らない項目があります: {', '.join(unknown)}")
    if info.get("role") != "predict":
        errors.append(f"版 {DEMAND_BUNDLE_VERSION} の一式の受け持ちは "
                      "predict (予測する) にしてください")
    output = info.get("output") if isinstance(info.get("output"), dict) else {}
    if output.get("form") != "distribution":
        errors.append(f"版 {DEMAND_BUNDLE_VERSION} の一式の出力の形は "
                      "distribution (分位点) にしてください")
    if errors:
        return errors

    # ファイルの名前, 列, 損と上限
    for key in ("estimator", "recipe"):
        if not isinstance(demand[key], str) or not demand[key]:
            errors.append(f"demand.{key} はファイルの名前にしてください")
    for key in ("product", "date"):
        if not isinstance(demand[key], str) or not demand[key]:
            errors.append(f"demand.{key} は列の名前にしてください")
    errors.extend(check_order_values(demand, "demand."))

    return errors


def check_demand_spec(spec: dict, names: dict) -> None:
    """
    需要のモデルの定義の入力と出力が約束に合うかを確かめる関数

    Parameters
    ----------
    spec: dict
        モデルの定義 (ML-ModelControl の形)
    names: dict
        input, quantiles (成果物の model)

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        入力が成果物の入力の名前の 1 つでない場合と, 分位点の出力の名前が
        無い場合
    """
    # 入力は前処理の後の行の 1 つ, 出力に分位点の予測
    inputs = list(spec.get("inputs") or {})
    if inputs != [names["input"]]:
        raise ValueError(
            f"需要のモデルの入力は {names['input']} の 1 つにしてください"
            f" (今は {', '.join(inputs) or '無し'})"
        )
    outputs = spec.get("outputs") or {}
    if names["quantiles"] not in outputs:
        raise ValueError(
            f"需要のモデルの出力に {names['quantiles']} がありません (出力: "
            f"{', '.join(map(str, outputs)) or '無し'})"
        )


def demand_dataset(root: Path, recipe: dict, estimator: dict) -> Dataset:
    """
    前処理だけの ML-EDA のデータセットを作る関数

    Parameters
    ----------
    root: Path
        一式 (か実験) のディレクトリ
    recipe: dict
        当てはめた値を含むレシピ
    estimator: dict
        需要の決め方 (features)

    Returns
    ----------
    dataset: Dataset
        分割の表を持たないデータセット (transform だけに使う. 目的の列は
        ML-EDA の書き出しと同じ先の期間の合計の列)
    """
    # 売れた数, 商品の id, 日付, 在庫の数, 行の id の列は前処理の手順の外
    info = {"kind": "table", "task": "demand", "target": "demand",
            "features": [{"name": name} for name in estimator["features"]]}
    dataset = Dataset(path=root, info=info, frames={}, recipe=recipe)

    return dataset


def raw_quantiles(
    model: Any, x: np.ndarray, names: dict, width: int, batch_size: int,
    device: str
) -> np.ndarray:
    """
    前処理の後の行をモデルに渡し, 分位点の予測をまとまりごとに求める関数

    Parameters
    ----------
    model: Any
        需要のモデル
    x: np.ndarray
        (件数, 列の数) の float32 の行
    names: dict
        input, quantiles
    width: int
        水準の数
    batch_size: int
        1 回に渡す行の数
    device: str
        使うデバイス

    Returns
    ----------
    values: np.ndarray
        (件数, 水準の数) の float64 の予測 (負の値は 0 に切り, 行ごとに
        小さい順に並べ直す. 数でない値を含む行は nan)

    Raises
    ----------
    ValueError
        名前の辞書の出力に分位点の名前が無い場合と, 予測の形が (件数,
        水準の数) でない場合
    """
    # 出力が 1 つの定義のテンソルと, 名前の辞書の分位点 (float32 から
    # float64 へ)
    found = []
    model = model.to(device)
    try:
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                chunk = torch.tensor(x[start:start + batch_size]).to(device)
                value = model(**{names["input"]: chunk})
                if isinstance(value, dict):
                    if names["quantiles"] not in value:
                        raise ValueError(
                            f"需要のモデルの出力に {names['quantiles']} が"
                            "ありません"
                        )
                    value = value[names["quantiles"]]
                if tuple(value.shape) != (len(chunk), width):
                    raise ValueError(
                        f"予測 {names['quantiles']} の形 "
                        f"{tuple(value.shape)} が ({len(chunk)}, {width}) "
                        "ではありません"
                    )
                found.append(value.cpu())
    finally:
        model.to("cpu")
    values = torch.cat(found).double().numpy() if found else np.zeros(
        (0, width)
    )

    # 0 で切り, 分位点の交差を行ごとの並べ直しで直す
    bad = ~np.isfinite(values).all(axis=1)
    values = np.sort(np.clip(np.nan_to_num(values), 0.0, None), axis=1)
    values[bad] = math.nan

    return values


def parse_days(values: pd.Series, name: str) -> pd.Series:
    """
    日付の列を日ごとの日時に読む関数

    Parameters
    ----------
    values: pd.Series
        日付の列の値
    name: str
        誤りの文に使う列の名前

    Returns
    ----------
    days: pd.Series
        日ごとの日時 (0 時)

    Raises
    ----------
    ValueError
        数か真偽値の列の場合と, 欠けたか読めないか 0 時の他の時刻のある値が
        ある場合
    """
    # 数は日付として読まない (ML-EDA の需要の記録の表と同じ決まり)
    if pd.api.types.is_bool_dtype(values) or (
        pd.api.types.is_numeric_dtype(values)
    ):
        raise ValueError(f"日付の列 {name} が日付ではありません "
                         f"({values.dtype})")
    parsed = values
    if not pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        if not pd.api.types.is_datetime64_any_dtype(parsed):
            parsed = pd.to_datetime(values, errors="coerce",
                                    format="mixed", utc=True)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    days = parsed.dt.normalize()
    bad = int((parsed.isna() | (parsed != days)).sum())
    if bad:
        raise ValueError(
            f"日付の列 {name} に, 欠けたか読めないか時刻のある値が {bad:,} 行"
            "あります (日ごとの日付にしてください)"
        )

    return days


def table_keys(bundle: Bundle, frame: pd.DataFrame) -> tuple:
    """
    表の商品の id と日付を確かめ, 行ごとの商品と日を返す関数

    Parameters
    ----------
    bundle: Bundle
        需要の一式
    frame: pd.DataFrame
        日ごとの記録の表

    Returns
    ----------
    found: tuple
        行ごとの商品の id (文字) の Series と, 日の Series

    Raises
    ----------
    ValueError
        商品の id か日付の列が無い場合, 商品の id が欠けた行か, 日付を読め
        ない行か, 同じ商品の同じ日の行がある場合
    """
    # 商品の id と日付の列 (どちらも全ての行で要る)
    demand = bundle.info["demand"]
    for key, label in (("product", "商品の id"), ("date", "日付")):
        if demand[key] not in frame.columns:
            raise ValueError(f"表に{label}の列 {demand[key]} がありません")
    products = frame[demand["product"]]
    if int(products.isna().sum()):
        raise ValueError(
            f"商品の id の列 {demand['product']} に欠けた値が "
            f"{int(products.isna().sum()):,} 行あります"
        )
    days = parse_days(frame[demand["date"]], demand["date"])

    # 同じ商品の同じ日の行は, どちらを使うか決められない
    keys = products.astype(str)
    twins = int(pd.DataFrame({"p": keys, "d": days}).duplicated().sum())
    if twins:
        raise ValueError(
            f"同じ商品の同じ日の行が {twins:,} 行あります (1 日 1 行に"
            "まとめてください)"
        )
    found = (keys.reset_index(drop=True), days.reset_index(drop=True))

    return found


def stock_values(
    bundle: Bundle, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    """
    行ごとの在庫の数を返す関数

    Parameters
    ----------
    bundle: Bundle
        需要の一式
    frame: pd.DataFrame
        元の列の表

    Returns
    ----------
    found: tuple[np.ndarray, np.ndarray, Optional[str]]
        float64 の在庫の数 (在庫の数の列が無い一式と表は 0), 読めた行の
        印, 使った在庫の数の列の名前 (無ければ None)
    """
    # 在庫の数の列が無い一式か表は 0
    name = bundle.info["demand"]["stock"]
    if name is None or name not in frame.columns:
        found = (np.zeros(len(frame)), np.ones(len(frame), dtype=bool),
                 None)
        return found
    numbers, good = number_values(frame[name], 0.0, False)
    found = (numbers, good, name)

    return found


def previous_sums(
    bundle: Bundle, frame: pd.DataFrame, keys: pd.Series, days: pd.Series,
    positions: np.ndarray
) -> np.ndarray:
    """
    発注を決める行ごとに, 前の期間の売れた数の合計を返す関数

    Parameters
    ----------
    bundle: Bundle
        需要の一式
    frame: pd.DataFrame
        日ごとの記録の表
    keys: pd.Series
        行ごとの商品の id (文字)
    days: pd.Series
        行ごとの日
    positions: np.ndarray
        発注を決める行の位置

    Returns
    ----------
    sums: np.ndarray
        前の horizon 日の売れた数の合計 (1 日でも行か読める売れた数が
        無ければ nan)
    """
    # 売れた数の列 (無ければ全て nan) と, 商品と日から行の位置
    demand = bundle.info["demand"]
    sums = np.full(len(positions), math.nan)
    if demand["units"] not in frame.columns:
        return sums
    units, good = number_values(frame[demand["units"]], 0.0, False)
    where = {(key, day): row for row, (key, day) in enumerate(zip(keys, days))}

    # 前の horizon 日 (発注を決める日を含まない) が全て揃う行の合計
    for number, row in enumerate(positions):
        found = [where.get((keys[row], days[row] - pd.Timedelta(days=step)))
                 for step in range(1, int(demand["horizon"]) + 1)]
        if all(f is not None and good[f] for f in found):
            sums[number] = float(units[found].sum())

    return sums


def position_quantiles(
    bundle: Bundle, frame: pd.DataFrame, positions: np.ndarray,
    batch_size: int, device: str
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    表の全ての行に前処理をかけ, 選んだ行の分位点を予測する関数

    Parameters
    ----------
    bundle: Bundle
        需要の一式
    frame: pd.DataFrame
        日ごとの記録の表
    positions: np.ndarray
        予測する行の位置
    batch_size: int
        1 回に渡す行の数
    device: str
        使うデバイス

    Returns
    ----------
    found: tuple[np.ndarray, list[str], np.ndarray]
        (行の数, 水準の数) の分位点 (予測しない行は nan), 行ごとの予測
        しない理由 (する行は空の文字), 行ごとの在庫の数
    """
    # 日ごとの全ての行の前処理 (過去の売れた数の列を日ごとにずらすため)
    x, lost = row_features(bundle.cache["demand_dataset"], frame)
    stock, stocked, _ = stock_values(bundle, frame)
    reasons = [lost[row] or ("" if stocked[row] else BAD_STOCK)
               for row in positions]

    # 理由の無い行の予測と, 数でない予測
    levels = bundle.output_info["quantiles"]
    values = np.full((len(positions), len(levels)), math.nan)
    keep = np.array([not r for r in reasons], dtype=bool)
    if keep.any():
        values[keep] = raw_quantiles(
            bundle.model(), x[positions[keep]],
            bundle.cache["demand_estimator"]["model"], len(levels),
            batch_size, device
        )
    for number in np.flatnonzero(keep):
        if not np.isfinite(values[number]).all():
            reasons[number] = NOT_FINITE
    found = (values, reasons, stock[positions])

    return found


def level_values(levels: list, values: np.ndarray, tau: float) -> np.ndarray:
    """
    水準 τ の分位点を, 学んだ水準の分位点から求める関数

    Parameters
    ----------
    levels: list
        分位点の水準 (小さい順)
    values: np.ndarray
        (行の数, 水準の数) の分位点 (行ごとに小さい順)
    tau: float
        求める水準

    Returns
    ----------
    found: np.ndarray
        行ごとの τ の分位点 (隣の 2 つの水準の間は直線で補い, 学んだ水準の
        範囲の外は端の水準の値)
    """
    # 端の水準の外
    points = [float(level) for level in levels]
    if tau <= points[0]:
        found = values[:, 0].copy()
        return found
    if tau >= points[-1]:
        found = values[:, -1].copy()
        return found

    # 隣の 2 つの水準の直線の補い
    upper = int(np.searchsorted(points, tau, side="left"))
    low, high = points[upper - 1], points[upper]
    weight = (tau - low) / (high - low)
    found = (1 - weight) * values[:, upper - 1] + weight * values[:, upper]

    return found


def row_costs(
    demand: np.ndarray, supply: np.ndarray, shortage: float, holding: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    行ごとの足りない数, 余った数, 損を数える関数

    Parameters
    ----------
    demand: np.ndarray
        行ごとの需要
    supply: np.ndarray
        行ごとの売れる数 (発注の数. 在庫を含む場合は在庫 + 発注の数)
    shortage: float
        1 つ足りないときの損
    holding: float
        1 つ余ったときの損

    Returns
    ----------
    short: np.ndarray
        足りない数 max(需要 - 発注の数, 0)
    excess: np.ndarray
        余った数 max(発注の数 - 需要, 0)
    cost: np.ndarray
        shortage × 足りない数 + holding × 余った数
    """
    # 足りない数と余った数は片方だけが正 (ML-Found の order_cost と同じ)
    short = np.clip(demand - supply, 0.0, None)
    excess = np.clip(supply - demand, 0.0, None)
    cost = shortage * short + holding * excess

    return short, excess, cost


def expected_costs(
    levels: list, values: np.ndarray, supply: np.ndarray, shortage: float,
    holding: float
) -> np.ndarray:
    """
    予測した分位点の分布での, 行ごとの見込みの損を数える関数

    Parameters
    ----------
    levels: list
        分位点の水準 (小さい順)
    values: np.ndarray
        (行の数, 水準の数) の分位点 (行ごとに小さい順)
    supply: np.ndarray
        行ごとの在庫 + 発注の数
    shortage: float
        1 つ足りないときの損
    holding: float
        1 つ余ったときの損

    Returns
    ----------
    cost: np.ndarray
        行ごとの損の, 分位点の関数 (水準の間は直線, 外は端の水準の値) で
        決まる需要の分布での期待値
    """
    # 水準 u の区間ごとの, 分位点の関数の両端 (端の外は平らな区間)
    points = [0.0, *[float(level) for level in levels], 1.0]
    ends = np.concatenate([values[:, :1], values, values[:, -1:]], axis=1)
    cost = np.zeros(len(values))

    # 区間の中で損は需要の直線の 1 つか 2 つの部分 (売れる数を超える点で
    # 分ける) なので, 台形で正しく積む
    for low, high, start, end in zip(points, points[1:], ends.T,
                                     ends.T[1:]):
        width = high - low
        if width <= 0:
            continue
        start_cost = row_costs(start, supply, shortage, holding)[2]
        end_cost = row_costs(end, supply, shortage, holding)[2]
        crossing = (start < supply) & (supply < end)
        with np.errstate(divide="ignore", invalid="ignore"):
            cut = np.where(crossing, (supply - start) / (end - start), 0.0)
        whole = width * (start_cost + end_cost) / 2
        split = width * (cut * start_cost + (1 - cut) * end_cost) / 2
        cost += np.where(crossing, split, whole)

    return cost


def best_orders(
    levels: list, values: np.ndarray, stock: np.ndarray, tau: float,
    shortage: float, holding: float, max_order: float
) -> dict:
    """
    行ごとに, 水準 τ の分位点から在庫を引いた発注の数を決める関数

    Parameters
    ----------
    levels: list
        分位点の水準 (小さい順)
    values: np.ndarray
        (行の数, 水準の数) の分位点
    stock: np.ndarray
        行ごとの在庫の数
    tau: float
        水準 τ (shortage / (shortage + holding))
    shortage: float
        1 つ足りないときの損
    holding: float
        1 つ余ったときの損
    max_order: float
        発注の数の上限

    Returns
    ----------
    found: dict
        quantile (τ の分位点), order (0 から上限の間の整数の発注の数),
        capped (上限が無ければ上限を超える数を発注した印),
        expected_cost (発注した後の見込みの損)
    """
    # τ の分位点から在庫を引いた数の切り捨てと切り上げ (0 より下は 0)
    quantile = level_values(levels, values, tau)
    wanted = quantile - stock
    low = np.clip(np.floor(wanted), 0, None)
    high = np.clip(np.ceil(wanted), 0, None)

    # 見込みの損の小さい方 (同じなら少ない方) と, 上限での切り (損は発注の
    # 数の凸な関数なので, 上限を超えるなら上限が上限の中で最も良い)
    low_cost = expected_costs(levels, values, stock + low, shortage, holding)
    high_cost = expected_costs(levels, values, stock + high, shortage,
                               holding)
    wished = np.where(high_cost < low_cost, high, low)
    top = math.floor(max_order)
    order = np.minimum(wished, top)
    found = {
        "quantile": quantile,
        "order": order,
        "capped": wished > top,
        "expected_cost": expected_costs(levels, values, stock + order,
                                        shortage, holding),
    }

    return found


def baseline_orders(
    previous: np.ndarray, stock: np.ndarray, max_order: float
) -> np.ndarray:
    """
    前の期間と同じ数を発注する決め方の発注の数を返す関数

    Parameters
    ----------
    previous: np.ndarray
        行ごとの前の期間の売れた数の合計 (無ければ nan)
    stock: np.ndarray
        行ごとの在庫の数
    max_order: float
        発注の数の上限

    Returns
    ----------
    orders: np.ndarray
        前の期間の合計から在庫を引き, 四捨五入して 0 から上限の間に収めた
        数 (前の期間の合計が無ければ nan)
    """
    # 発注の判断と同じ在庫と上限の扱い
    rounded = np.floor(previous - stock + 0.5)
    orders = np.clip(rounded, 0, math.floor(max_order))

    return orders


def tau_of(shortage: Any, holding: Any) -> tuple[float, Decimal]:
    """
    品切れの損と売れ残りの損から水準 τ を返す関数

    Parameters
    ----------
    shortage: Any
        1 つ足りないときの損
    holding: Any
        1 つ余ったときの損

    Returns
    ----------
    found: tuple[float, Decimal]
        τ = shortage / (shortage + holding) の数と, 10 進の表記どおりの値
    """
    # 10 進の表記どおりの比 (学んだ水準との比べに使う)
    exact = decimal_of(shortage) / (decimal_of(shortage)
                                    + decimal_of(holding))
    found = (float(exact), exact)

    return found


def pinball_loss(actual: Any, quantiles: Any, levels: Any) -> dict:
    """
    時間をまたぐ量の判断の分位点の損失を数える関数

    Parameters
    ----------
    actual: Any
        行ごとの実際の値 (0 以上) の 1 次元の並び
    quantiles: Any
        (行の数, 水準の数) の分位点の予測
    levels: Any
        分位点の水準 (0 より大きく 1 より小さい) の 1 次元の並び

    Returns
    ----------
    found: dict
        levels (列 level, loss の水準ごとの損失の平均), mean (全ての行と
        水準の損失の平均)

    Raises
    ----------
    ValueError
        形が合わない場合と, 値が決まりの外の場合
    """
    # 形と値の確かめ (ML-Found の pinball_loss と同じ決まり)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    points = [float(decimal_of(level)) for level in list(levels)]
    if not points or not all(0 < level < 1 for level in points) or (
        len(set(points)) != len(points)
    ):
        raise ValueError("分位点の水準は 0 より大きく 1 より小さい, 重なら"
                         "ない数の並びにしてください")
    if not len(actual) or quantiles.shape != (len(actual), len(points)):
        raise ValueError("分位点の予測は (行の数, 水準の数) の形にして"
                         "ください")
    if not (np.isfinite(actual).all() and (actual >= 0).all()):
        raise ValueError("実際の値は 0 以上の有限の数にしてください")
    if not np.isfinite(quantiles).all():
        raise ValueError("分位点の予測に nan か無限大があります")

    # 水準 q の損 max(q (y - ŷ), (q - 1) (y - ŷ))
    level = np.array(points)[None, :]
    diff = actual[:, None] - quantiles
    loss = np.maximum(level * diff, (level - 1) * diff)
    found = {"levels": {"level": points, "loss": loss.mean(axis=0).tolist()},
             "mean": float(loss.mean())}

    return found


def order_cost(
    demand: Any, order: Any, shortage_cost: Any, holding_cost: Any,
    baseline: Any = None
) -> dict:
    """
    時間をまたぐ量の判断の発注の数の見込みの損を数える関数

    Parameters
    ----------
    demand: Any
        行ごとの実際の需要 (0 以上) の 1 次元の並び
    order: Any
        行ごとの発注の数 (0 以上) の 1 次元の並び
    shortage_cost: Any
        1 つ足りないときの損 (正の数)
    holding_cost: Any
        1 つ余ったときの損 (正の数)
    baseline: Any = None
        比べる発注の数 (0 以上) の並び (None は比べない)

    Returns
    ----------
    found: dict
        items (shortage, excess, cost と, 比べる場合は baseline_cost),
        total, baseline_total, saving, ratio (比べない場合と baseline_total
        が 0 の場合の比は nan. 比べない場合の baseline_total と saving も
        nan)

    Raises
    ----------
    ValueError
        形が合わない場合と, 値が決まりの外の場合
    """
    # 形と値の確かめ (ML-Found の order_cost と同じ決まり)
    errors = check_order_values({"shortage_cost": shortage_cost,
                                 "holding_cost": holding_cost,
                                 "max_order": 0}, "")
    if errors:
        raise ValueError("見込みの損の誤り: " + " / ".join(errors))
    shortage = float(decimal_of(shortage_cost))
    holding = float(decimal_of(holding_cost))
    demand = np.asarray(demand, dtype=np.float64).reshape(-1)
    named = {"発注の数": order}
    if baseline is not None:
        named["比べる発注の数"] = baseline
    arrays = {}
    for name, value in named.items():
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if not len(demand) or array.shape != demand.shape:
            raise ValueError(f"実際の需要と{name}は同じ長さの 1 つ以上の"
                             "並びにしてください")
        arrays[name] = array
    for name, array in (("実際の需要", demand), *arrays.items()):
        if not (np.isfinite(array).all() and (array >= 0).all()):
            raise ValueError(f"{name}は 0 以上の有限の数にしてください")

    # 行ごとの損と, 比べる発注の数での損, 差と比
    short, excess, cost = row_costs(demand, arrays["発注の数"], shortage,
                                    holding)
    total = float(cost.sum())
    found: dict = {
        "items": {"shortage": short.tolist(), "excess": excess.tolist(),
                  "cost": cost.tolist()},
        "total": total, "baseline_total": math.nan, "saving": math.nan,
        "ratio": math.nan,
    }
    if baseline is not None:
        compared = row_costs(demand, arrays["比べる発注の数"], shortage,
                             holding)[2]
        baseline_total = float(compared.sum())
        found["items"]["baseline_cost"] = compared.tolist()
        found["baseline_total"] = baseline_total
        found["saving"] = baseline_total - total
        if baseline_total > 0:
            found["ratio"] = total / baseline_total

    return found


def predict_demand(
    bundle: Bundle, source: Any, options: Optional[dict] = None,
    batch_size: int = 256, device: str = "cpu"
) -> pd.DataFrame:
    """
    商品ごとに, 最後の日を発注を決める日として先の需要の分位点を返す関数

    Parameters
    ----------
    bundle: Bundle
        需要の一式
    source: Any
        日ごとの記録の表のファイル, 表, 表の ML-EDA の Data
    options: Optional[dict] = None
        ファイルの読み込みの設定 (load_data に渡す)
    batch_size: int = 256
        1 回に渡す行の数
    device: str = "cpu"
        使うデバイス

    Returns
    ----------
    result: pd.DataFrame
        商品ごとの発注を決める日の行の元の列 (足す列と同じ名前は「(入力)」
        付き) と, status, reason, order_date, on_hand, previous_demand,
        prediction, q:<水準>

    Raises
    ----------
    ValueError
        表でない場合と, 商品の id と日付の列か前処理に要る列が無いか, 値が
        決まりに合わない場合
    """
    # 表の読み込みと, 商品ごとの最も遅い日の行 (表の商品の順)
    frame = read_table(source, options, "需要を予測する表")
    keys, days = table_keys(bundle, frame)
    order = pd.DataFrame({"key": keys, "day": days, "row": range(len(keys))})
    last = order.sort_values(["day", "row"]).groupby("key", sort=False).tail(1)
    first = pd.unique(keys)
    positions = last.set_index("key").loc[first, "row"].to_numpy(np.int64)

    # 分位点, 在庫の数, 前の期間の合計
    values, reasons, stock = position_quantiles(bundle, frame, positions,
                                                batch_size, device)
    previous = previous_sums(bundle, frame, keys, days, positions)
    ok = np.array([not r for r in reasons], dtype=bool)

    # 元の列の後ろに予測の列 (同じ名前の元の列は「(入力)」)
    levels = bundle.output_info["quantiles"]
    picked = frame.iloc[positions].reset_index(drop=True)
    result = picked.rename(columns={c: frame_name(c)
                                    for c in picked.columns})
    result["status"] = np.where(ok, OK, SKIPPED)
    result["reason"] = pd.Series(reasons, dtype=object)
    result["order_date"] = days[positions].dt.strftime("%Y-%m-%d").to_numpy()
    result["on_hand"] = np.where(ok, stock, math.nan)
    result["previous_demand"] = previous
    result["prediction"] = np.where(ok, level_values(levels, values, 0.5),
                                    math.nan)
    for index, level in enumerate(levels):
        result[quantile_name(level)] = values[:, index]

    return result


def order_check(
    bundle: Bundle, data: Any, options: Optional[dict], sources: dict,
    batch_size: int
) -> tuple[Optional[dict], list[str]]:
    """
    元の表で決めた発注の数を, 実験の発注の数と比べる関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    data: Any
        学習に使った元の日ごとの記録の表 (ファイル, 表, 表の Data)
    options: Optional[dict]
        ファイルの読み込みの設定
    sources: dict
        run_sources の結果
    batch_size: int
        1 回に渡す行の数

    Returns
    ----------
    found: tuple[Optional[dict], list[str]]
        rows, agreement, max_diff, pinball_loss (比べられなければ None) と
        注意の文
    """
    # 商品の id と日付の列と, 実験の行と同じ商品と日の元の行
    estimator = sources["estimator"]
    frame = read_table(data, options, "元の表")
    missing = [estimator[key] for key in ("product", "date")
               if estimator[key] not in frame.columns]
    if missing:
        note = (f"元の表に {', '.join(missing)} の列が無いので, 一式で決めた"
                "発注の数を実験の発注の数と比べません")
        found = (None, [note])
        return found
    keys, days = table_keys(bundle, frame)
    where = {(key, day): row for row, (key, day) in enumerate(zip(keys, days))}
    orders = sources["orders"]
    matched = [where.get((key, day)) for key, day in zip(
        orders[estimator["product"]], orders[estimator["date"]]
    )]
    use = np.array([row is not None for row in matched], dtype=bool)
    positions = np.array([row for row in matched if row is not None],
                         dtype=np.int64)
    values, reasons, stock = position_quantiles(bundle, frame, positions,
                                                batch_size, "cpu")
    good = np.array([not r for r in reasons], dtype=bool)
    if not good.any():
        note = ("元の表に, 実験の発注の数の行と同じ商品と日の出せる行が無い"
                "ので, 一式で決めた発注の数を実験の発注の数と比べません")
        found = (None, [note])
        return found

    # 実験と同じ損と上限での発注の数と, τ の分位点の差
    levels = bundle.output_info["quantiles"]
    tau, _ = tau_of(estimator["shortage_cost"], estimator["holding_cost"])
    best = best_orders(levels, values[good], stock[good], tau,
                       float(estimator["shortage_cost"]),
                       float(estimator["holding_cost"]),
                       float(estimator["max_order"]))
    recorded = orders[use].iloc[np.flatnonzero(good)]
    same = best["order"] == recorded["order"].to_numpy()
    labeled = recorded["demand"].notna().to_numpy()
    check = {
        "rows": int(good.sum()), "agreement": float(same.mean()),
        "max_diff": float(np.abs(
            best["quantile"] - recorded["quantile"].to_numpy()
        ).max()),
        "pinball_loss": pinball_loss(
            recorded["demand"].to_numpy()[labeled], values[good][labeled],
            levels
        )["mean"] if labeled.any() else None,
    }
    notes = []
    if check["agreement"] < AGREEMENT:
        notes.append(
            f"元の表で一式が決めた発注の数が, 実験の発注の数と "
            f"{check['agreement'] * 100:.1f} % の行でしか同じになりません "
            "(前処理のレシピか重みが学習と違う可能性があります)"
        )
    if check["rows"] < len(orders):
        notes.append(
            f"実験の発注の数の {len(orders):,} 行のうち, 比べたのは "
            f"{check['rows']:,} 行です (元の表に無いか出せない行は比べ"
            "ません)"
        )
    found = (check, notes)

    return found


def probe_model(bundle: Bundle, width: int, batch_size: int) -> None:
    """
    0 の行をモデルに渡し, 分位点の予測の形と値を確かめる関数

    Parameters
    ----------
    bundle: Bundle
        需要の一式 (cache に demand_estimator)
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
        予測が (件数, 水準の数) の数でない場合
    """
    # 予測の形の確かめ (raw_quantiles の中) と, 値が数かの確かめ
    levels = bundle.output_info["quantiles"]
    values = raw_quantiles(
        bundle.model(), np.zeros((PROBE_ROWS, width), dtype=np.float32),
        bundle.cache["demand_estimator"]["model"], len(levels), batch_size,
        "cpu"
    )
    if not np.isfinite(values).all():
        raise ValueError("0 の行の分位点の予測が数ではありません")


def demand_section(sources: dict) -> dict:
    """
    成果物から一式の demand の欄を作る関数

    Parameters
    ----------
    sources: dict
        run_sources の結果

    Returns
    ----------
    demand: dict
        demand の欄 (check は None)
    """
    # 成果物の写し (水準は output.quantiles)
    estimator = sources["estimator"]
    model = estimator["model"]
    demand = {
        "estimator": ESTIMATOR_FILE,
        "recipe": RECIPE_FILE,
        "features": list(estimator["features"]),
        "product": estimator["product"],
        "date": estimator["date"],
        "units": estimator["units"],
        "stock": estimator["stock"],
        "id": estimator["id"],
        "horizon": int(estimator["horizon"]),
        "interval": int(estimator["interval"]),
        "input": model["input"],
        "output": model["quantiles"],
        "shortage_cost": plain(estimator["shortage_cost"]),
        "holding_cost": plain(estimator["holding_cost"]),
        "max_order": plain(estimator["max_order"]),
        "rows": int(len(sources["orders"])),
        "train": {key: plain(estimator["train"][key]) for key in TRAIN_KEYS},
        "evaluations": copy.deepcopy(sources["evaluations"]),
        "check": None,
    }

    return demand


def make_demand_bundle(
    path: Union[str, Path], run: Union[str, Path],
    name: Optional[str] = None, data: Any = None,
    options: Optional[dict] = None, copy_weights: bool = True,
    overwrite: bool = False, pretrained: Optional[dict] = None,
    batch_size: int = 256
) -> dict:
    """
    先の需要の分位点から発注の数を決める一式を, 実験から作る関数

    Parameters
    ----------
    path: Union[str, Path]
        作るディレクトリ
    run: Union[str, Path]
        ML-Found の実験のディレクトリ (demand_estimator.json,
        demand_recipe.yaml, demand_orders.csv, evaluations の
        demand_<分割>.json, final_model.pt, model_spec.yaml を使い, 実験
        の状態を記録する)
    name: Optional[str] = None
        一式の名前 (省略するとディレクトリの名前)
    data: Any = None
        学習に使った元の日ごとの記録の表 (渡すと発注の数を決めて実験の
        発注の数と比べる)
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
    check_demand_spec(spec_dict, estimator["model"])
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
        section = demand_section(sources)
        info = {
            "format_version": DEMAND_BUNDLE_VERSION,
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "role": "predict",
            "output": {"form": "distribution",
                       "quantiles": [plain(level)
                                     for level in estimator["levels"]]},
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
            "decision": {"type": "order",
                         "shortage_cost": section["shortage_cost"],
                         "holding_cost": section["holding_cost"],
                         "max_order": section["max_order"]},
            "check": None,
            "uncertainty": None,
            "notes": bundle_notes(empty, sources["run"]) + weight_notes,
            "demand": section,
        }

        # 0 の行の予測と, 元の表で決めた発注の数と実験の発注の数の比べ
        building = Bundle(path=writing, info=info, spec=spec_dict,
                          dataset=empty)
        building.cache["demand_estimator"] = estimator
        building.cache["demand_dataset"] = demand_dataset(
            writing, sources["recipe"], estimator
        )
        probe_model(building, len(estimator["features"]), batch_size)
        if data is not None:
            found, notes = order_check(building, data, options, sources,
                                       batch_size)
            info["demand"]["check"] = found
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


def check_demand_bundle(bundle: Bundle) -> None:
    """
    需要の一式の成果物を確かめて cache に置く関数

    Parameters
    ----------
    bundle: Bundle
        需要の一式

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
    # 需要の決め方とレシピの写し (一式の外は誤り)
    demand = bundle.info["demand"]
    estimator = read_estimator(inside(bundle.path, demand["estimator"]))
    recipe = read_recipe(inside(bundle.path, demand["recipe"]), estimator)
    model = estimator["model"]
    mismatch = [
        key for key, value in (
            ("features", estimator["features"]),
            ("product", estimator["product"]), ("date", estimator["date"]),
            ("units", estimator["units"]), ("stock", estimator["stock"]),
            ("id", estimator["id"]), ("horizon", estimator["horizon"]),
            ("interval", estimator["interval"]),
            ("input", model["input"]), ("output", model["quantiles"]),
            ("shortage_cost", estimator["shortage_cost"]),
            ("holding_cost", estimator["holding_cost"]),
            ("max_order", estimator["max_order"]),
        ) if demand[key] != value
    ]
    if bundle.output_info.get("quantiles") != estimator["levels"]:
        mismatch.append("output.quantiles")
    if mismatch:
        raise ValueError(
            f"demand の {', '.join(mismatch)} が需要の決め方 "
            f"{demand['estimator']} と違います"
        )
    bundle.cache["demand_estimator"] = estimator
    bundle.cache["demand_dataset"] = demand_dataset(bundle.path, recipe,
                                                    estimator)

    # モデルの定義の入力と出力
    check_demand_spec(bundle.spec, model)


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
        分割ごとの誤差, 分位点の損失, 見込みの損の比べの文 (無ければ,
        分位点の予測を見て確かめる文)
    """
    # 評価の無い実験の, 分位点の予測を見る文
    if not evaluations:
        sentences = ["実験の評価が無いので, 分位点の予測を見て確かめる"]
        return sentences
    sentences = []
    for split, row in evaluations.items():
        ratio = ("比は無し" if row["order_cost_ratio"] is None
                 else f"比 {row['order_cost_ratio']:.4g}")
        sentences.append(
            f"{SPLIT_NAMES.get(split, split)}の需要の 0.5 の分位点の平均絶対"
            f"誤差 {row['mae']:.4g}, 分位点の損失 {row['pinball_loss']:.4g},"
            f" 発注の数の損 {row['order_cost']:.4g} (前の期間と同じ数の発注"
            f"なら {row['baseline_cost']:.4g}. 差 "
            f"{row['order_cost_saving']:.4g}, {ratio})"
        )

    return sentences


def stock_sentence(
    bundle: Bundle, prediction: pd.DataFrame
) -> Optional[str]:
    """
    在庫を 0 として数えることを述べる文を返す関数

    Parameters
    ----------
    bundle: Bundle
        需要の一式
    prediction: pd.DataFrame
        predict_demand の表

    Returns
    ----------
    sentence: Optional[str]
        在庫の数の列が無い一式か表の文 (在庫の数の列を使えば None)
    """
    # 一式に在庫の数の列が無いか, 表に在庫の数の列が無い
    name = bundle.info["demand"]["stock"]
    if name is None:
        sentence = "在庫の数の列が無いので, 在庫は 0 として数える"
        return sentence
    if column_in(name, prediction) is None:
        sentence = f"表に在庫の数の列 {name} が無いので, 在庫は 0 として数える"
        return sentence

    return None


def conclude_demand(bundle: Bundle, prediction: pd.DataFrame) -> dict:
    """
    先の需要の分位点から, 件数と要約の結論を出す関数

    Parameters
    ----------
    bundle: Bundle
        需要の一式
    prediction: pd.DataFrame
        predict_demand の表

    Returns
    ----------
    conclusion: dict
        frame (decision を足した表. 予測した行は空, しなかった行は予測
        なし), summary (rows, predicted, skipped, id, product, date,
        stock, horizon, levels, dates, prediction, previous, train,
        evaluations), sentences, rules (空)
    """
    # 予測したかと, 0.5 の分位点と前の期間の合計
    demand = bundle.info["demand"]
    frame = prediction.copy()
    ok = (frame["status"] == OK).to_numpy()
    frame["decision"] = np.where(ok, "", NO_PREDICTION)
    median = frame["prediction"].to_numpy(dtype=np.float64)[ok]
    previous = frame["previous_demand"].to_numpy(dtype=np.float64)[ok]
    dates = sorted(frame.loc[ok, "order_date"].astype(str))
    summary = {
        "rows": int(len(frame)), "predicted": int(ok.sum()),
        "skipped": int(len(frame) - ok.sum()),
        "id": column_in(demand["id"], prediction),
        "product": column_in(demand["product"], prediction),
        "date": column_in(demand["date"], prediction),
        "stock": column_in(demand["stock"], prediction),
        "horizon": int(demand["horizon"]),
        "levels": list(bundle.output_info["quantiles"]),
        "dates": {"min": dates[0], "max": dates[-1]} if dates else None,
        "prediction": {"mean": float(median.mean()),
                       "total": float(median.sum())} if ok.any() else None,
        "previous": {
            "mean": float(previous[np.isfinite(previous)].mean())
            if np.isfinite(previous).any() else None,
            "counted": int(np.isfinite(previous).sum()),
        } if ok.any() else None,
        "train": copy.deepcopy(demand["train"]),
        "evaluations": copy.deepcopy(demand["evaluations"]),
    }

    # 件数, 発注を決める日, 予測, 前の期間, 在庫, 実験の評価の文
    head = (f"{summary['rows']:,} 商品のうち {summary['predicted']:,} 商品の"
            f"先の {summary['horizon']} 日の需要の分位点を予測した (発注を"
            "決める日は商品ごとの表の最後の日)")
    if summary["skipped"]:
        head += f" ({summary['skipped']:,} 商品は出せず)"
    sentences = [head]
    if summary["dates"] is not None:
        span = summary["dates"]
        sentences.append(
            f"発注を決める日は {span['min']}" if span["min"] == span["max"]
            else f"発注を決める日は {span['min']} から {span['max']}"
        )
    if summary["prediction"] is not None:
        sentences.append(
            f"先の {summary['horizon']} 日の需要の 0.5 の分位点の平均 "
            f"{summary['prediction']['mean']:.4g} (和 "
            f"{summary['prediction']['total']:.4g})"
        )
    if summary["previous"] is not None and summary["previous"]["counted"]:
        sentences.append(
            f"前の {summary['horizon']} 日の売れた数の合計の平均 "
            f"{summary['previous']['mean']:.4g} "
            f"({summary['previous']['counted']:,} 商品)"
        )
    stock = stock_sentence(bundle, prediction)
    if stock is not None:
        sentences.append(stock)
    sentences.extend(evaluation_sentences(demand["evaluations"]))
    conclusion = {"frame": frame, "summary": summary,
                  "sentences": sentences, "rules": {}}

    return conclusion


def check_order_decision(decision: dict) -> list[str]:
    """
    需要の一式の発注の判断の決め方の誤りを一覧で返す関数

    Parameters
    ----------
    decision: dict
        意思決定の決め方

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 発注の判断の型と書ける項目 (type, shortage_cost, holding_cost,
    # max_order)
    if decision.get("type") != "order":
        return ["需要の一式の意思決定は発注の判断 (type: order) だけです"]
    errors = [
        f"需要の一式の発注の判断に書けない項目です: {key} (書けるのは type, "
        "shortage_cost, holding_cost, max_order だけ. 在庫の数は表の在庫の"
        "数の列から読む)"
        for key in decision if key not in ORDER_DECISION_KEYS
    ]
    errors.extend(check_order_values(decision, ""))

    return errors


def decision_notes(
    bundle: Bundle, prediction: pd.DataFrame, decision: dict, tau: Decimal
) -> list[str]:
    """
    発注の判断の前提の文を返す関数

    Parameters
    ----------
    bundle: Bundle
        需要の一式
    prediction: pd.DataFrame
        predict_demand の表
    decision: dict
        発注の判断の決め方
    tau: Decimal
        水準 τ

    Returns
    ----------
    notes: list[str]
        分布の前提, 水準, 学んだ水準の外, 在庫, 比べの前提の文
    """
    # 予測した分布と損の前提, 水準と上限
    levels = bundle.output_info["quantiles"]
    horizon = bundle.info["demand"]["horizon"]
    notes = [
        f"書いた損 (品切れ {decision['shortage_cost']:g}, 売れ残り "
        f"{decision['holding_cost']:g}. 商品 1 つあたり) と, 先の {horizon} "
        "日の需要の分位点の予測が正しいとすれば, 見込みの損が最も小さいのは"
        f"水準 τ = {float(tau):.4g} の分位点から在庫を引いた数で, 発注の数は"
        f"整数にして 0 から上限 {decision['max_order']:g} (守る指標) に"
        "収める",
        "需要は在庫が 0 の日の売れた数 (需要より少ない) も含む記録から"
        "学んでいるので, 品切れの多い商品では少なめに出ることがある",
    ]

    # 学んだ水準の範囲の外
    low, high = decimal_of(levels[0]), decimal_of(levels[-1])
    if tau < low or tau > high:
        end = levels[0] if tau < low else levels[-1]
        notes.append(
            f"τ = {float(tau):.4g} は学んだ水準の範囲 [{levels[0]:g}, "
            f"{levels[-1]:g}] の外なので, 端の水準 {end:g} の分位点を使った "
            "(分布の端は学んでいないので, 見込みの損は少なめに出る)"
        )
    stock = stock_sentence(bundle, prediction)
    if stock is not None:
        notes.append(stock.replace("数える", "発注の数を決めた"))
    notes.append(
        "見込みの損と前の期間と同じ数の発注との比べは同じ予測の分布から数え"
        "たもので, 必ずこの決め方が良く出る. 実際の損は発注の後の売れた数で"
        "確かめる (機械学習の結果は判断の材料)"
    )

    return notes


def decision_sentences(summary: dict) -> list[str]:
    """
    発注の判断の件数と見込みの損を述べる文を返す関数

    Parameters
    ----------
    summary: dict
        decide_order の summary

    Returns
    ----------
    sentences: list[str]
        発注する行の数と見込みの損, 前の期間との比べ, 上限で切った行, 決め
        ない行の文
    """
    # 発注する行の数と見込みの損
    sentences = [
        f"{summary['rows']:,} 商品のうち {summary['ordered']:,} 商品で発注"
        f"する (発注の数の和 {summary['total']:,.0f}. 水準 "
        f"{summary['level']:.4g}, 上限 {summary['max_order']:g}). 見込みの"
        f"損の和は {summary['expected_cost']:.4g}"
    ]

    # 前の期間と同じ数の発注との比べ
    if summary["compared"]:
        ratio = (f"比 {summary['ratio']:.4g}" if summary["ratio"] is not None
                 else "比は無し")
        sentences.append(
            f"前の期間と同じ数を発注すると, 見込みの損は "
            f"{summary['baseline_cost']:.4g} (この決め方なら "
            f"{summary['compared_cost']:.4g}. {summary['compared']:,} 商品. "
            f"差 {summary['saving']:.4g}, {ratio})"
        )
    if summary["capped"]:
        sentences.append(
            f"上限で切った商品が {summary['capped']:,} 商品ある (τ の分位点"
            "から在庫を引いた数が上限を超える. 上限は守る指標なので超えない)"
        )
    if summary["undecided"]:
        sentences.append(f"分位点を予測できない {summary['undecided']:,} 商品"
                         "は決めない")

    return sentences


def decide_order(
    bundle: Bundle, prediction: pd.DataFrame, decision: dict
) -> dict:
    """
    予測した商品ごとに, 水準 τ の分位点から在庫を引いた発注の数を決める関数

    Parameters
    ----------
    bundle: Bundle
        需要の一式
    prediction: pd.DataFrame
        predict_demand の表 (か conclude_demand の表)
    decision: dict
        {type: order, shortage_cost, holding_cost, max_order}

    Returns
    ----------
    result: dict
        frame (action, quantile, order, capped, expected_cost,
        baseline_order, baseline_cost を足した表), summary (rows,
        undecided, ordered, total, capped, level, used_level, outside,
        expected_cost, compared, compared_cost, baseline_cost, saving,
        ratio, shortage_cost, holding_cost, max_order), sentences, notes
        (前提), decision (使った決め方)

    Raises
    ----------
    ValueError
        決め方に誤りがある場合と, 表に予測の列が無い場合
    """
    # 決め方と表の確認
    decision = plain(dict(decision))
    errors = check_order_decision(decision)
    levels = bundle.output_info["quantiles"]
    names = [quantile_name(level) for level in levels]
    missing = [c for c in ("status", "on_hand", "previous_demand", *names)
               if c not in prediction.columns]
    if missing:
        errors.append(f"予測の表に {', '.join(missing)} の列がありません")
    if errors:
        raise ValueError("意思決定の決め方の誤り: " + " / ".join(errors))

    # 予測した行の τ の分位点と発注の数
    shortage = float(decision["shortage_cost"])
    holding = float(decision["holding_cost"])
    limit = float(decision["max_order"])
    tau, exact = tau_of(decision["shortage_cost"], decision["holding_cost"])
    values = prediction[names].to_numpy(dtype=np.float64)
    stock = prediction["on_hand"].to_numpy(dtype=np.float64)
    ok = (prediction["status"] == OK).to_numpy() & np.isfinite(
        values
    ).all(axis=1) & np.isfinite(stock)
    rows = np.flatnonzero(ok)
    columns = {key: np.full(len(prediction), math.nan)
               for key in DECISION_COLUMNS if key not in ("action", "capped")}
    action = pd.array([None] * len(prediction), dtype="boolean")
    capped = pd.array([None] * len(prediction), dtype="boolean")
    best = best_orders(levels, values[rows], stock[rows], tau, shortage,
                       holding, limit)
    for key in ("quantile", "order", "expected_cost"):
        columns[key][rows] = best[key]
    action[rows] = best["order"] > 0
    capped[rows] = best["capped"]

    # 前の期間と同じ数の発注 (前の期間の合計のある行)
    previous = prediction["previous_demand"].to_numpy(dtype=np.float64)
    compared = rows[np.isfinite(previous[rows])]
    columns["baseline_order"][compared] = baseline_orders(
        previous[compared], stock[compared], limit
    )
    columns["baseline_cost"][compared] = expected_costs(
        levels, values[compared], stock[compared]
        + columns["baseline_order"][compared], shortage, holding
    )
    frame = prediction.copy()
    frame["action"] = action
    frame["quantile"] = columns["quantile"]
    frame["order"] = columns["order"]
    frame["capped"] = capped
    for key in ("expected_cost", "baseline_order", "baseline_cost"):
        frame[key] = columns[key]

    # 要約 (比は前の期間と同じ数の発注の損が 0 なら無し)
    points = [decimal_of(level) for level in levels]
    baseline = float(columns["baseline_cost"][compared].sum())
    policy = float(columns["expected_cost"][compared].sum())
    summary = {
        "rows": int(len(rows)),
        "undecided": int((prediction["status"] == OK).sum() - len(rows)),
        "ordered": int((best["order"] > 0).sum()),
        "total": float(best["order"].sum()),
        "capped": int(best["capped"].sum()),
        "level": tau,
        "used_level": float(min(max(exact, points[0]), points[-1])),
        "outside": bool(exact < points[0] or exact > points[-1]),
        "expected_cost": float(best["expected_cost"].sum()),
        "compared": int(len(compared)),
        "compared_cost": policy,
        "baseline_cost": baseline,
        "saving": baseline - policy,
        "ratio": policy / baseline if baseline > 0 else None,
        "shortage_cost": decision["shortage_cost"],
        "holding_cost": decision["holding_cost"],
        "max_order": decision["max_order"],
    }

    # 前提, 発注する行の数と見込みの損の文 (前提は notes にも)
    notes = decision_notes(bundle, prediction, decision, exact)
    sentences = [*notes, *decision_sentences(summary)]
    result = {"frame": frame, "summary": summary, "sentences": sentences,
              "notes": notes, "decision": decision}

    return result


def demand_of_bundle(bundle: Bundle) -> Optional[dict]:
    """
    需要の一式の demand の欄を, 説明の形で返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    demand: Optional[dict]
        demand の欄の写しと水準 levels (需要の一式でなければ None)
    """
    # 画面が水準, 損, 上限と実験の評価を出すための写し
    if not bundle.from_demand:
        return None
    demand = copy.deepcopy(bundle.info["demand"])
    demand["levels"] = list(bundle.output_info["quantiles"])

    return demand
