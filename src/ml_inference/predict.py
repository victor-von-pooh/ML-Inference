from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from ml_eda import Data, load_data

from ml_inference.bundle import Bundle, load_bundle

# 予測しなかった行の印
OK, SKIPPED = "ok", "skipped"

# 表として前処理をかける種類
TABLE_KINDS = ("table", "timeseries")


def read_source(
    bundle: Bundle, source: Union[str, Path, pd.DataFrame, Data],
    options: Optional[dict] = None
) -> Data:
    """
    推論するデータを, 学習のときと同じ読み込みで読む関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    source: Union[str, Path, pd.DataFrame, Data]
        ファイルかディレクトリ, 表, ML-EDA で読んだデータ
    options: Optional[dict] = None
        読み込みの設定 (学習のときの設定に上書きする)

    Returns
    ----------
    data: Data
        読み込んだデータ

    Raises
    ----------
    ValueError
        表で渡せない種類の場合と, 文の列が無い場合
    """
    # 読んだデータと表はそのまま (文は文の列を持つ表)
    kind = bundle.kind
    if isinstance(source, Data):
        return source
    if isinstance(source, pd.DataFrame):
        frame = source.reset_index(drop=True)
        if kind in TABLE_KINDS:
            return Data(kind=kind, frame=frame)
        column = bundle.dataset.info.get("text_column")
        if kind != "text" or not column:
            raise ValueError(
                f"{kind} のデータは表では渡せません. ファイルかディレクトリの"
                "場所を渡してください"
            )
        if column not in frame.columns:
            raise ValueError(f"文の列 {column} がありません")
        # 欠損の文は空の文 (ML-EDA の読み込みと同じ. 空の文は予測しない)
        texts = frame[column].fillna("").astype(str)
        return Data(
            kind="text", frame=frame, roles={"text": column},
            item_loader=lambda i: texts.iloc[i]
        )

    # 場所は学習のときの読み込みと設定で
    recorded = bundle.dataset.info.get("source") or {}
    merged = {**(recorded.get("options") or {}), **(options or {})}
    data = load_data(source, loader=recorded.get("loader"), **merged)

    return data


def features_of(
    bundle: Bundle, data: Data
) -> tuple[np.ndarray, list[str]]:
    """
    表の列に前処理をかけ, 学習に使う列の配列と行ごとの問題を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    data: Data
        読み込んだデータ

    Returns
    ----------
    x: np.ndarray
        (件数, 列の数) の float32 の配列
    reasons: list[str]
        行ごとの予測できない理由 (予測できる行は空の文字)
    """
    # 当てはめた値のままの前処理 (行は増減しない). 一部の行の値で全体が
    # 失敗した場合は, 行ごとにかけて失敗した行を飛ばす (過去の値の列は
    # 前の行が要るので行ごとにはかけない)
    try:
        frame = bundle.dataset.transform(data.frame)
        failed = [""] * len(frame)
        # 行を落とす手順で落ちた行 (ML-EDA が手順の名前を返す場合)
        for row, op in (frame.attrs.get("dropped") or {}).items():
            failed[int(row)] = (
                f"前処理の手順 {op} で除く行です (学習でも使わない行)"
            )
    except ValueError as error:
        steps = [s["op"] for s in bundle.dataset.recipe.get("steps", [])]
        if "lag" in steps or "前処理の後に, 学習に使う列が足りません" in str(
            error
        ) or "にかける列がありません" in str(error):
            raise
        frame, failed = transform_rows(bundle, data.frame)
    try:
        x = frame.to_numpy(dtype=np.float32, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"前処理の後に数でない列があります ({error}). レシピで数に"
            "してください"
        ) from error
    reasons = []
    for row in frame.itertuples(index=False):
        lost = [name for name, value in zip(frame.columns, row)
                if pd.isna(value)]
        reasons.append(
            "前処理の後に欠損: " + ", ".join(lost[:5])
            + (" など" if len(lost) > 5 else "") if lost else ""
        )
    reasons = [a or b for a, b in zip(failed, reasons)]

    return x, reasons


def transform_rows(
    bundle: Bundle, frame: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """
    1 行ずつ前処理をかけ, 失敗した行を欠損と理由にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    frame: pd.DataFrame
        元の表

    Returns
    ----------
    result: pd.DataFrame
        学習に使う列の表 (失敗した行は欠損)
    failed: list[str]
        行ごとの失敗の理由 (成功した行は空の文字)
    """
    # 行ごとの前処理 (遅いので, 全体で失敗したときだけ使う)
    parts = []
    failed = []
    columns = bundle.dataset.features
    for index in range(len(frame)):
        try:
            part = bundle.dataset.transform(frame.iloc[[index]])
            failed.append("")
        except ValueError as error:
            part = pd.DataFrame(
                [[np.nan] * len(columns)], columns=columns,
                index=frame.index[[index]]
            )
            failed.append(f"前処理できません ({error})")
        parts.append(part)
    result = pd.concat(parts) if parts else pd.DataFrame(columns=columns)

    return result, failed


def items_of(
    bundle: Bundle, data: Data
) -> tuple[Optional[np.ndarray], list[str]]:
    """
    1 件ずつ中身の手順をかけて並べ, 行ごとの問題を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    data: Data
        読み込んだデータ

    Returns
    ----------
    items: Optional[np.ndarray]
        (件数, ...) の配列 (読めなかった行は 0. 1 件も読めなければ None)
    reasons: list[str]
        行ごとの予測できない理由 (予測できる行は空の文字)
    """
    # 読めない件と形の違う件は理由を付けて飛ばす
    arrays: list = []
    reasons = []
    shape = None
    for index in range(len(data.frame)):
        try:
            item = data.load_item(index)
            if isinstance(item, str) and not item.strip():
                arrays.append(None)
                reasons.append("文が空です")
                continue
            array = bundle.dataset.transform_item(item)
        except Exception as error:
            arrays.append(None)
            reasons.append(f"中身を読めません ({error})")
            continue
        if shape is not None and array.shape != shape:
            arrays.append(None)
            reasons.append(f"形が {array.shape} で, 他の件の {shape} と違います")
            continue
        shape = array.shape
        arrays.append(array)
        reasons.append("")
    if shape is None:
        return None, reasons
    first = next(a for a in arrays if a is not None)
    blank = np.zeros(shape, dtype=first.dtype)
    items = np.stack([blank if a is None else a for a in arrays])

    return items, reasons


def shape_inputs(bundle: Bundle, sources: dict) -> list:
    """
    中身の出どころごとの配列を, モデルの入力の順のテンソルにする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    sources: dict
        features と items の配列 (あるものだけ)

    Returns
    ----------
    tensors: list
        モデルの入力の順のテンソル

    Raises
    ----------
    ValueError
        データに無い出どころを使う入力がある場合
    """
    # 文の番号の並びは整数, 他は小数. 先頭に足す番号は文の並びだけ
    tensors = []
    for name, item in bundle.input_plan().items():
        if item["source"] not in sources:
            raise ValueError(
                f"入力 {name} の {item['source']} が {bundle.kind} のデータに"
                "ありません"
            )
        array = np.asarray(sources[item["source"]])
        integer = item["source"] == "items" and bundle.kind == "text"
        tensor = torch.from_numpy(
            np.array(array, dtype=np.int64 if integer else np.float32)
        )
        if item["prepend"]:
            if not integer:
                raise ValueError(
                    f"入力 {name} の prepend は文の番号の並びにだけ使えます"
                )
            head = torch.tensor(item["prepend"], dtype=torch.long)
            tensor = torch.cat([head.expand(len(tensor), -1), tensor], dim=1)
        tensors.append(tensor)

    return tensors


def run_model(
    bundle: Bundle, tensors: list, batch_size: int, device: str
) -> np.ndarray:
    """
    モデルを推論のモードで決まった件数ずつ動かす関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    tensors: list
        モデルの入力の順のテンソル
    batch_size: int
        1 回に渡す件数
    device: str
        使うデバイス

    Returns
    ----------
    outputs: np.ndarray
        使う出力を並べた配列

    Raises
    ----------
    ValueError
        出力が辞書で, 使う出力が決まらない場合
    """
    # 出力が辞書なら, 書いた名前か 1 つだけの出力
    model = bundle.model().to(device)
    chosen = bundle.info["model"].get("output")
    parts = []
    with torch.no_grad():
        for start in range(0, len(tensors[0]), batch_size):
            batch = [t[start:start + batch_size].to(device) for t in tensors]
            out = model(*batch)
            if isinstance(out, dict):
                if chosen is None and len(out) != 1:
                    raise ValueError(
                        f"モデルの出力が {', '.join(out)} の {len(out)} つ "
                        "あります. 使う出力を model.output に書いてください"
                    )
                if chosen is not None and chosen not in out:
                    raise ValueError(
                        f"出力 {chosen} がありません (出力: {', '.join(out)})"
                    )
                out = out[chosen] if chosen is not None else next(
                    iter(out.values())
                )
            parts.append(out.detach().float().cpu().numpy())
    model.to("cpu")
    outputs = np.concatenate(parts) if parts else np.zeros((0,))

    return outputs


def inverse_target(bundle: Bundle, values: np.ndarray) -> np.ndarray:
    """
    レシピで変換した目的の列の値を, 元の単位に戻す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    values: np.ndarray
        モデルの出力の値

    Returns
    ----------
    result: np.ndarray
        元の単位の値
    """
    # 目的の列を変換した手順を逆の順に戻す
    target = bundle.dataset.info.get("target")
    result = values.astype(np.float64)
    for spec in reversed(bundle.dataset.recipe.get("steps", [])):
        state = spec.get("state") or {}
        if spec["op"] in ("standardize", "minmax"):
            pair = (state.get("values") or {}).get(target)
            if pair is not None:
                center, scale = pair
                result = result * scale + center
        elif spec["op"] == "log":
            columns = spec.get("columns")
            columns = [columns] if isinstance(columns, str) else columns
            if columns and target in columns:
                result = np.expm1(result)

    return result


def decode(bundle: Bundle, outputs: np.ndarray) -> dict:
    """
    モデルの出力を, タスクに合わせて元の形の列にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    outputs: np.ndarray
        モデルの出力

    Returns
    ----------
    columns: dict
        prediction (と分類の confidence, p:<クラス>) の列

    Raises
    ----------
    ValueError
        出力の形がタスクに合わない場合
    """
    # 回帰と予測は 1 つの値
    count = len(outputs)
    if bundle.task != "classification":
        if outputs.ndim == 2 and outputs.shape[1] == 1:
            outputs = outputs[:, 0]
        if outputs.ndim != 1:
            raise ValueError(
                f"{bundle.task} の出力は (件数,) か (件数, 1) にしてください "
                f"(今は {tuple(outputs.shape)})"
            )
        scale = bundle.info["model"].get("target_scale")
        if scale:
            outputs = outputs * float(scale["scale"]) + float(scale["center"])
        columns = {"prediction": inverse_target(bundle, outputs)}
        return columns

    # 分類はクラスごとの確率 (2 クラスで 1 列なら 2 つ目のクラス)
    classes = bundle.classes
    kind = bundle.output_kind
    single = outputs.ndim == 1 or (
        outputs.ndim == 2 and outputs.shape[1] == 1
    )
    if single and len(classes) == 2:
        column = outputs.reshape(count).astype(np.float64)
        positive = 1 / (1 + np.exp(-column)) if kind == "logits" else column
        probability = np.stack([1 - positive, positive], axis=1)
    elif outputs.ndim == 2 and outputs.shape[1] == len(classes):
        probability = outputs.astype(np.float64)
        if kind == "logits":
            shifted = np.exp(probability - probability.max(axis=1,
                                                           keepdims=True))
            probability = shifted / shifted.sum(axis=1, keepdims=True)
    else:
        raise ValueError(
            f"分類の出力の形 {tuple(outputs.shape)} がクラスの数 "
            f"{len(classes)} と合いません"
        )
    best = probability.argmax(axis=1) if count else np.zeros(0, dtype=int)
    columns = {
        "prediction": [classes[i] for i in best],
        "confidence": probability[np.arange(count), best],
    }
    for index, name in enumerate(classes):
        columns[f"p:{name}"] = probability[:, index]

    return columns


def predict(
    bundle: Union[str, Path, Bundle],
    source: Union[str, Path, pd.DataFrame, Data],
    options: Optional[dict] = None, batch_size: int = 256,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    新しいデータを前処理して推論し, 行ごとの予測を返す関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ
    source: Union[str, Path, pd.DataFrame, Data]
        推論するデータ
    options: Optional[dict] = None
        読み込みの設定 (学習のときの設定に上書きする)
    batch_size: int = 256
        1 回に渡す件数
    device: str = "cpu"
        使うデバイス

    Returns
    ----------
    result: pd.DataFrame
        元の列と, status, reason, prediction (分類は confidence と
        p:<クラス> も) の列

    Raises
    ----------
    ValueError
        前処理できない場合と, 出力の形が合わない場合
    """
    # データの読み込みと, 種類に合わせた中身
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    data = read_source(bundle, source, options)
    count = len(data.frame)
    sources: dict = {}
    reasons = [""] * count
    plan = {item["source"] for item in bundle.input_plan().values()}
    if "features" in plan:
        sources["features"], found = features_of(bundle, data)
        reasons = [a or b for a, b in zip(reasons, found)]
    if "items" in plan:
        items, found = items_of(bundle, data)
        if items is not None:
            sources["items"] = items
        reasons = [a or b for a, b in zip(reasons, found)]

    # 予測できる行だけ推論
    keep = np.array([not r for r in reasons], dtype=bool)
    if keep.any():
        tensors = shape_inputs(
            bundle, {k: np.asarray(v)[keep] for k, v in sources.items()}
        )
        columns = decode(
            bundle, run_model(bundle, tensors, batch_size, device)
        )
    else:
        columns = decode(bundle, empty_output(bundle))

    # 元の列に予測の列を並べる (予測しなかった行は空). 予測の列と同じ
    # 名前の元の列は「(入力)」を付けて残す
    result = data.frame.copy().reset_index(drop=True)
    result.columns = [str(c) for c in result.columns]
    added = {"status", "reason", "decision", *columns}
    result = result.rename(
        columns={c: f"{c} (入力)" for c in result.columns if c in added}
    )
    result["status"] = np.where(keep, OK, SKIPPED)
    result["reason"] = reasons
    for name, values in columns.items():
        full = pd.Series([None] * count, dtype=object)
        full[keep] = list(values)
        result[name] = (
            full if name == "prediction" and bundle.task == "classification"
            else pd.to_numeric(full)
        )

    return result


def empty_output(bundle: Bundle) -> np.ndarray:
    """
    予測する行が無い場合の, 0 件の出力を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    outputs: np.ndarray
        0 件の出力
    """
    # 分類はクラスの数の列
    width = len(bundle.classes) if bundle.task == "classification" else 1
    outputs = np.zeros((0, width), dtype=np.float32)

    return outputs


def options_of(bundle: Bundle) -> dict:
    """
    学習のときの読み込みの名前と設定を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    reading: dict
        loader と options
    """
    # データセットの説明の source から
    recorded = bundle.dataset.info.get("source") or {}
    reading = {
        "loader": recorded.get("loader"),
        "options": dict(recorded.get("options") or {}),
    }

    return reading


def describe_bundle(bundle: Union[str, Path, Bundle]) -> dict:
    """
    推論の一式の説明を JSON にできる辞書で返す関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ

    Returns
    ----------
    description: dict
        名前, 種類, タスク, 目的の列, クラス, 学習に使う列, 読み込み,
        前処理の手順, モデルの入力, 実験, 結論の既定値, 注意
    """
    # 画面が一式を選ぶときに要ること
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    info = bundle.info
    dataset = bundle.dataset.info
    description: dict[str, Any] = {
        "path": str(bundle.path),
        "name": info.get("name"),
        "created_at": info.get("created_at"),
        "kind": bundle.kind,
        "task": bundle.task,
        "target": dataset.get("target"),
        "classes": bundle.classes,
        "features": [item["name"] for item in dataset.get("features", [])],
        "text_column": dataset.get("text_column"),
        "reading": options_of(bundle),
        "steps": [s["op"] for s in bundle.dataset.recipe.get("steps", [])],
        "item_steps": [
            s["op"] for s in bundle.dataset.recipe.get("item_steps", [])
        ],
        "model": {
            "name": bundle.spec.get("name"),
            "inputs": bundle.input_plan(),
            "output": info["model"].get("output"),
            "kind": bundle.output_kind,
        },
        "run": info.get("run"),
        "conclusion": info.get("conclusion") or {},
        "notes": list(info.get("notes") or []),
    }

    return description
