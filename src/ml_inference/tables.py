import json
import math
from pathlib import Path
from typing import Any, Union

import numpy as np

# 列の型と, 型ごとの欄 (name と type の他)
COLUMN_TYPES = {
    "number": ("mean", "std", "integer"),
    "category": ("categories",),
}

# 表の出力の列に使えない名前 (予測の表の件の番号, 種, 状態, 結論の列と,
# 他の出力の形の列)
RESERVED_COLUMNS = (
    "sample", "seed", "status", "reason", "decision", "output", "artifact",
    "tokens", "prediction", "confidence"
)
RESERVED_PREFIXES = ("p:", "q:")

# 種類の値に使える型 (JSON と YAML のまま書けるもの)
CATEGORY_TYPES = (str, int, float, bool)


def is_number(value: Any) -> bool:
    """
    有限の数 (真偽値を除く) かを確かめる関数

    Parameters
    ----------
    value: Any
        確かめる値

    Returns
    ----------
    result: bool
        有限の int か float なら True
    """
    # 真偽値と無限大と欠損を除く
    result = isinstance(value, (int, float)) and not isinstance(
        value, bool
    ) and math.isfinite(value)

    return result


def check_table_columns(columns: Any) -> list[str]:
    """
    表の出力の列の符号化 (output.columns) の形を確かめる関数

    Parameters
    ----------
    columns: Any
        列の順の符号化の並び

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 1 つ以上の辞書の並び
    if not isinstance(columns, list) or not columns or not all(
        isinstance(c, dict) for c in columns
    ):
        return ["output.columns は列の符号化 (辞書) の 1 つ以上の並びに"
                "してください"]
    errors = []
    names = [c.get("name") for c in columns]
    if not all(isinstance(n, str) and n for n in names):
        errors.append("output.columns の name は空でない文字にしてください")
        return errors
    if len(set(names)) != len(names):
        errors.append("output.columns の name が重なっています")
    reserved = [
        n for n in names
        if n in RESERVED_COLUMNS or n.startswith(RESERVED_PREFIXES)
        or n.endswith(" (入力)")
    ]
    if reserved:
        errors.append(
            f"output.columns の {', '.join(reserved)} は予測の表の列の名前"
            f"と重なります ({', '.join(RESERVED_COLUMNS)} と "
            f"{', '.join(RESERVED_PREFIXES)} で始まる名前は使えません. "
            "データセットの列の名前を変えてください)"
        )

    # 型ごとの欄
    for column in columns:
        name, kind = column["name"], column.get("type")
        if kind not in COLUMN_TYPES:
            errors.append(
                f"output.columns の {name} の type は "
                f"{', '.join(COLUMN_TYPES)} のどれかにしてください"
            )
            continue
        extra = [k for k in column
                 if k not in ("name", "type", *COLUMN_TYPES[kind])]
        if extra:
            errors.append(
                f"output.columns の {name} の {', '.join(map(str, extra))} "
                "は知らない欄です"
            )
        if kind == "number":
            std = column.get("std")
            if not is_number(column.get("mean")) or not is_number(std) or (
                std <= 0
            ):
                errors.append(
                    f"output.columns の {name} は mean (数) と std (正の数) "
                    "が要ります"
                )
            if not isinstance(column.get("integer"), bool):
                errors.append(
                    f"output.columns の {name} の integer は真偽値にして"
                    "ください"
                )
            continue
        categories = column.get("categories")
        if not isinstance(categories, list) or not categories or not all(
            isinstance(v, CATEGORY_TYPES) for v in categories
        ):
            errors.append(
                f"output.columns の {name} の categories は種類の値 (文字, "
                "数, 真偽値) の 1 つ以上の並びにしてください"
            )
        elif len({str(v) for v in categories}) != len(categories):
            errors.append(
                f"output.columns の {name} の categories が (文字にして) "
                "重なっています"
            )

    return errors


def table_width(columns: list[dict]) -> int:
    """
    符号化した行の幅を求める関数

    Parameters
    ----------
    columns: list[dict]
        列の順の符号化

    Returns
    ----------
    width: int
        数の列の数と, 種類の列の値の種類の数の和
    """
    # 数の列は 1, 種類の列は値の種類の数
    width = sum(1 if c["type"] == "number" else len(c["categories"])
                for c in columns)

    return width


def read_table_columns(value: Union[str, Path, list, dict]) -> list:
    """
    一式に写す列の符号化を, 並びかファイルから読む関数

    Parameters
    ----------
    value: Union[str, Path, list, dict]
        列の符号化の並び, columns を持つ辞書, そのどちらかの JSON の
        ファイル (ML-Quest の骨組みの table_columns.json など)

    Returns
    ----------
    columns: list
        列の順の符号化 (中身の確かめは check_table_columns)

    Raises
    ----------
    ValueError
        ファイルが無い場合と, JSON として読めない場合
    """
    # ファイルは JSON で読む
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"列の符号化のファイル {path} がありません")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"列の符号化のファイル {path} を JSON で読めません ({error})"
            ) from error

    # columns を持つ辞書は中の並び
    columns = value.get("columns") if isinstance(value, dict) else value

    return columns


def decode_table_rows(columns: list[dict], outputs: np.ndarray) -> dict:
    """
    符号化した行を, 列の名前ごとの値に戻す関数

    Parameters
    ----------
    columns: list[dict]
        列の順の符号化
    outputs: np.ndarray
        (件数, 幅) の符号化した行

    Returns
    ----------
    values: dict
        列の名前ごとの値の並び (数の列は値 × std + mean で, 整数の列は
        丸めた int. 数でない値は None. 種類の列はその列の区間で最も大きい
        値の種類)

    Raises
    ----------
    ValueError
        出力が (件数, 幅) でない場合
    """
    # 幅は列の符号化から
    width = table_width(columns)
    if outputs.ndim != 2 or outputs.shape[1] != width:
        raise ValueError(
            f"table の出力は (件数, {width}) にしてください (今は "
            f"{tuple(outputs.shape)})"
        )

    # 列の順に区間を切り出して戻す
    array = outputs.astype(np.float64)
    values: dict = {}
    start = 0
    for column in columns:
        if column["type"] == "number":
            raw = array[:, start] * column["std"] + column["mean"]
            values[column["name"]] = [
                (int(round(v)) if column["integer"] else float(v))
                if math.isfinite(v) else None for v in raw
            ]
            start += 1
            continue
        size = len(column["categories"])
        picked = array[:, start:start + size].argmax(axis=1)
        values[column["name"]] = [column["categories"][int(i)]
                                  for i in picked]
        start += size

    return values
