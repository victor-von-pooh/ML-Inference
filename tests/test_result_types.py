from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml_inference import read_result, write_result


def kept(values: pd.Series) -> list:
    """
    列の値を, 欠けを None にした並びで返す関数

    Parameters
    ----------
    values: pd.Series
        表の列

    Returns
    ----------
    values: list
        値の並び (欠けは None)
    """
    # 欠けの種類 (NaN, None, NA) の違いを除いて比べるため
    values = [None if pd.isna(v) else v for v in values.astype(object)]

    return values


def test_types_survive_round_trip(tmp_path: Path) -> None:
    """
    数や真偽値に見える文字の列と欠けのある整数の列が, 書き出して読み直して
    も同じ値になり, 値の変わる列だけの型が版 3 で残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 分類の予測のような表 (数に見えるクラスの名前) と, 入力の元の列
    frame = pd.DataFrame({
        "code": ["001", "1e3", None],
        "flag": ["True", "False", None],
        "word": ["NA", "null", None],
        "name": ["a", "b", None],
        "count": pd.Series([3, None, 5], dtype=object),
        "group": pd.array([0, None, 2], dtype="Int64"),
        "whole": [1, 2, 3],
        "mixed": ["a", 1, None],
        "ratio": [0.5, None, 1.0],
        "prediction": ["001", "002", None],
        "status": ["ok", "ok", "skipped"],
    })
    root = write_result({"frame": frame, "summary": {"rows": 3}},
                        tmp_path / "result")
    meta = yaml.safe_load((root / "result.yaml").read_text(encoding="utf-8"))
    assert meta["format_version"] == 3
    assert meta["column_types"] == {
        "code": "text", "flag": "text", "count": "integer",
        "group": "integer", "prediction": "text",
    }

    # 型を残した列と, 型を知らずに読んでも変わらない列は同じ値
    back = read_result(root)
    table = back["frame"]
    for name in ("code", "flag", "word", "name", "count", "group", "whole",
                 "ratio", "prediction", "status"):
        assert kept(table[name]) == kept(frame[name]), name
    assert type(table["code"].iloc[0]) is str
    assert str(table["count"].dtype) == "Int64"
    assert str(table["group"].dtype) == "Int64"
    assert back["column_types"] == meta["column_types"]


def test_versions_and_rewrite(tmp_path: Path) -> None:
    """
    型を残す列が無ければ版 1 のまま, 古い結果は今までどおり読め, 書き直す
    と型は書く表から決め直すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 読み直して変わらない表は型を残さず版 1
    frame = pd.DataFrame({"name": ["a", None], "value": [1.5, 2.0],
                          "status": ["ok", "ok"]})
    root = write_result({"frame": frame}, tmp_path / "plain")
    meta = yaml.safe_load((root / "result.yaml").read_text(encoding="utf-8"))
    assert meta["format_version"] == 1 and "column_types" not in meta

    # 型の記録の無い古い結果は, 空の欄だけを欠損にして読む
    (root / "predictions.csv").write_text(
        "code,word,status\n001,NA,ok\n,x,skipped\n", encoding="utf-8-sig"
    )
    table = read_result(root)["frame"]
    assert table["code"].iloc[0] == 1 and pd.isna(table["code"].iloc[1])
    assert table["word"].tolist() == ["NA", "x"]

    # 読んだ結果の型は持ち越さず, 書く表から決め直す
    typed = write_result(
        {"frame": pd.DataFrame({"code": ["001"], "status": ["ok"]})},
        tmp_path / "typed"
    )
    result = read_result(typed)
    result["frame"] = pd.DataFrame({"code": ["a"], "status": ["ok"]})
    again = write_result(result, tmp_path / "again")
    meta = yaml.safe_load((again / "result.yaml").read_text(encoding="utf-8"))
    assert meta["format_version"] == 1 and "column_types" not in meta


def test_types_with_artifacts(tmp_path: Path) -> None:
    """
    配列の出力の結果でも型が残り, 版 3 の結果の配列を読んで書き直せること
    を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 番号の列 (数に見える文字) を持つ配列の出力の結果
    frame = pd.DataFrame({
        "id": ["007", "008"], "status": ["ok", "skipped"],
        "output": [np.arange(3.0), None],
    })
    root = write_result({"frame": frame, "form": "vector"},
                        tmp_path / "result")
    back = read_result(root)
    assert back["format_version"] == 3 and back["form"] == "vector"
    assert back["column_types"] == {"id": "text"}
    assert back["frame"]["id"].tolist() == ["007", "008"]
    assert back["frame"]["artifact"].tolist() == ["artifacts/000000.npy", ""]

    # 別の場所に書き直しても配列と型は残る
    copied = write_result(back, tmp_path / "copied")
    again = read_result(copied)
    assert again["column_types"] == {"id": "text"}
    assert np.allclose(np.load(copied / "artifacts" / "000000.npy"),
                       np.arange(3.0))


def test_columns_without_types(tmp_path: Path) -> None:
    """
    Int64 に収まらない整数の列, 全てが欠けの列, 0 行の表は型を残さず, 読み
    直せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # Int64 に収まらない符号なしの整数と欠けの列は残さず, 読み直せる
    frame = pd.DataFrame({
        "big": pd.array([2 ** 63, None], dtype="UInt64"),
        "status": ["ok", "ok"],
    })
    root = write_result({"frame": frame}, tmp_path / "big")
    back = read_result(root)
    assert back["format_version"] == 1 and "column_types" not in back
    assert back["frame"]["status"].tolist() == ["ok", "ok"]

    # 全てが欠けの列 (整数の型, 文字) は残さない
    frame = pd.DataFrame({
        "empty": pd.array([None, None], dtype="Int64"),
        "blank": pd.Series([None, None], dtype=object),
        "status": ["ok", "skipped"],
    })
    root = write_result({"frame": frame}, tmp_path / "empty")
    back = read_result(root)
    assert back["format_version"] == 1 and "column_types" not in back
    assert back["frame"]["empty"].isna().all()

    # 0 行の表も版 1 のまま
    frame = pd.DataFrame({
        "code": pd.Series([], dtype=object),
        "group": pd.array([], dtype="Int64"),
        "status": pd.Series([], dtype=object),
    })
    root = write_result({"frame": frame}, tmp_path / "zero")
    back = read_result(root)
    assert back["format_version"] == 1 and "column_types" not in back
    assert len(back["frame"]) == 0
    assert list(back["frame"].columns) == ["code", "group", "status"]
