import contextlib
import csv
import datetime as dt
import os
from pathlib import Path
import threading
from typing import Any, Union

import numpy as np
import pandas as pd
import yaml

from ml_inference.result import read_result

# 人の確認の記録のファイル (結果のディレクトリの中) と列
REVIEWS_FILE = "reviews.csv"
REVIEW_COLUMNS = ("id", "judgment", "reviewer", "reviewed_at")

# 人の判断 (異常か正常か)
JUDGMENTS = ("anomaly", "normal")

# 異常の度合いの結果の summary に要る鍵
ANOMALY_SUMMARY_KEYS = ("id", "score")


def check_header(path: Path) -> None:
    """
    前からある人の確認の記録の列が決まりと同じかを確かめる関数

    Parameters
    ----------
    path: Path
        reviews.csv

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        列が違う場合
    """
    # 見出しの行だけの読み込み
    with open(path, encoding="utf-8-sig", newline="") as file:
        header = next(csv.reader(file), [])
    if tuple(header) != REVIEW_COLUMNS:
        raise ValueError(
            f"{path} の列は {', '.join(REVIEW_COLUMNS)} にしてください (今は"
            f" {', '.join(header) or '無し'})"
        )


def review_time(reviewed_at: Any) -> str:
    """
    確かめた時刻を時差付きの ISO 8601 の文字にする関数

    Parameters
    ----------
    reviewed_at: Any
        時差付きの日時か ISO 8601 の文字 (None なら今の時刻)

    Returns
    ----------
    text: str
        時差付きの ISO 8601 の文字

    Raises
    ----------
    ValueError
        日時にできない場合と, 時差の無い場合
    """
    # 省略は今の時刻 (手元の時差付き)
    if reviewed_at is None:
        text = dt.datetime.now().astimezone().isoformat()
        return text
    if isinstance(reviewed_at, str):
        try:
            reviewed_at = dt.datetime.fromisoformat(reviewed_at)
        except ValueError as error:
            raise ValueError(
                f"reviewed_at {reviewed_at} は ISO 8601 の日時にしてください"
            ) from error
    if not isinstance(reviewed_at, dt.datetime):
        raise ValueError("reviewed_at は日時か ISO 8601 の文字にしてください")
    if reviewed_at.utcoffset() is None:
        raise ValueError(
            "reviewed_at は時差の付いた日時にしてください (時差が無いと, "
            "どこの時刻か決められない)"
        )
    text = reviewed_at.isoformat()

    return text


def row_key(result: Path, row: Any) -> str:
    """
    確かめた行が結果にあるかを確かめ, 行の id の文字を返す関数

    Parameters
    ----------
    result: Path
        結果のディレクトリ
    row: Any
        行の id の値 (id の列が無い結果は行の位置)

    Returns
    ----------
    key: str
        記録に書く行の id の文字

    Raises
    ----------
    FileNotFoundError
        result.yaml が無い場合
    ValueError
        異常の度合いの結果でない場合と, 行が無い場合
    """
    # 異常の度合いの結果の summary の id の列
    meta_file = result / "result.yaml"
    if not meta_file.is_file():
        raise FileNotFoundError(f"{result} に result.yaml がありません")
    meta = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
    summary = (meta or {}).get("summary") if isinstance(meta, dict) else None
    if not isinstance(summary, dict) or not all(
        key in summary for key in ANOMALY_SUMMARY_KEYS
    ):
        raise ValueError(
            f"{result} は異常の度合いの結果ではありません (summary に id と "
            "score がありません)"
        )

    # 文字で比べる id の列の値 (列の無い結果は行の位置)
    frame = read_result(result)["frame"]
    column = summary["id"]
    if column is None:
        if isinstance(row, (bool, np.bool_)) or not isinstance(
            row, (int, np.integer)
        ) or not 0 <= int(row) < len(frame):
            raise ValueError(
                f"行の位置 {row} は 0 から {len(frame) - 1} の整数にして"
                "ください (行の id の列の無い結果)"
            )
        key = str(int(row))
        return key
    if column not in frame.columns:
        raise ValueError(f"予測の表に行の id の列 {column} がありません")
    key = str(row)
    if key not in set(frame[column].dropna().astype(str)):
        raise ValueError(f"予測の表に行の id {key} の行がありません")

    return key


def create_reviews(path: Path) -> None:
    """
    見出しの行だけの人の確認の記録を, 無ければ排他で作る関数

    Parameters
    ----------
    path: Path
        reviews.csv

    Returns
    ----------
    None
    """
    # 見出しを書き終えた一時のファイルの, 排他の置き (os.link は置く先が
    # あれば FileExistsError. 中身ごと一度に置くので, 同時に追記する側が
    # 見出しの無いファイルを見ることも無し)
    if path.is_file():
        return
    writing = path.with_name(f".{path.name}.{os.getpid()}."
                             f"{threading.get_ident()}.writing")
    with open(writing, "w", encoding="utf-8-sig", newline="") as file:
        csv.writer(file).writerow(REVIEW_COLUMNS)
    try:
        with contextlib.suppress(FileExistsError):
            os.link(writing, path)
    finally:
        writing.unlink()


def record_review(
    result: Union[str, Path], row: Any, judgment: str, reviewer: str,
    reviewed_at: Any = None
) -> dict:
    """
    人の確認の結果を 1 件, 結果のディレクトリの記録に足す関数

    Parameters
    ----------
    result: Union[str, Path]
        異常の度合いの結果のディレクトリ (write_result で書いたもの)
    row: Any
        行の id の値 (id の列が無い結果は行の位置)
    judgment: str
        anomaly (異常) か normal (正常)
    reviewer: str
        確かめた人の名前
    reviewed_at: Any = None
        確かめた時刻 (時差付きの日時か ISO 8601 の文字. 省略すると今)

    Returns
    ----------
    record: dict
        足した記録 (id, judgment, reviewer, reviewed_at)

    Raises
    ----------
    FileNotFoundError
        result.yaml が無い場合
    ValueError
        異常の度合いの結果でない場合, 行が無い場合, 判断, 確かめた人,
        時刻の形が違う場合, 前からある記録の列が違う場合
    """
    # 判断, 確かめた人, 時刻の形
    root = Path(result).expanduser()
    if judgment not in JUDGMENTS:
        raise ValueError(
            f"judgment は {' か '.join(JUDGMENTS)} にしてください (今は "
            f"{judgment})"
        )
    if not isinstance(reviewer, str) or not reviewer.strip() or any(
        mark in reviewer for mark in ("\n", "\r")
    ):
        raise ValueError("reviewer は空でなく改行の無い名前にしてください")
    when = review_time(reviewed_at)
    key = row_key(root, row)

    # 見出しの行だけのファイルを隣に書いて排他で置き (同時に最初の 1 件を
    # 書いても見出しは 1 行), 記録は後ろへの 1 行の追記
    path = root / REVIEWS_FILE
    record = {"id": key, "judgment": judgment,
              "reviewer": reviewer.strip(), "reviewed_at": when}
    create_reviews(path)
    check_header(path)
    with open(path, "a", encoding="utf-8", newline="") as file:
        csv.writer(file).writerow([record[name] for name in REVIEW_COLUMNS])

    return record


def read_reviews(result: Union[str, Path]) -> pd.DataFrame:
    """
    結果のディレクトリの人の確認の記録を読む関数

    Parameters
    ----------
    result: Union[str, Path]
        結果のディレクトリ

    Returns
    ----------
    reviews: pd.DataFrame
        書いた順の id, judgment, reviewer, reviewed_at の表 (全て文字.
        記録が無ければ 0 行)

    Raises
    ----------
    ValueError
        記録の列が違う場合
    """
    # 全ての列を文字で読み込み (行の id の先頭の 0 などを残すため)
    path = Path(result).expanduser() / REVIEWS_FILE
    if not path.is_file():
        reviews = pd.DataFrame({name: pd.Series(dtype=object)
                                for name in REVIEW_COLUMNS})
        return reviews
    check_header(path)
    reviews = pd.read_csv(path, dtype=str, encoding="utf-8-sig",
                          keep_default_na=False)
    reviews = reviews.astype(object)

    return reviews

