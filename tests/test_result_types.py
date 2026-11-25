from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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


def write_raw(root: Path, csv: str, meta: dict) -> Path:
    """
    予測の表と result.yaml を, そのまま書く関数

    Parameters
    ----------
    root: Path
        結果のディレクトリ
    csv: str
        predictions.csv の中身
    meta: dict
        result.yaml の中身

    Returns
    ----------
    root: Path
        結果のディレクトリ
    """
    # 版と列の型を決めて書くため (write_result は型を表から決める)
    root.mkdir(parents=True)
    (root / "predictions.csv").write_text(csv, encoding="utf-8-sig")
    (root / "result.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")

    return root


def test_given_types(tmp_path: Path) -> None:
    """
    呼び手の列の型が, override_types は column_types より強く,
    fallback_types は column_types に無い列だけに効くことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 3 の結果 (順位は欠けのある整数, 番号は数に見える文字)
    root = write_raw(
        tmp_path / "typed",
        "user,rank,code,count,status\n1,1,001,3,ok\n2,,002,,skipped\n",
        {"format_version": 3,
         "column_types": {"rank": "integer", "code": "text"}},
    )

    # 型を渡さないときは column_types だけで読む (省くのと None と空の
    # 辞書は同じ)
    plain = read_result(root)["frame"]
    assert plain["user"].tolist() == [1, 2]
    assert str(plain["rank"].dtype) == "Int64"
    assert plain["code"].tolist() == ["001", "002"]
    assert plain["count"].iloc[0] == 3.0 and pd.isna(plain["count"].iloc[1])
    for given in ({}, None):
        pd.testing.assert_frame_equal(
            read_result(root, given, given)["frame"], plain
        )

    # 強い型は column_types より先, 補いの型は column_types に無い列だけ
    table = read_result(
        root, override_types={"user": "text", "rank": "text"},
        fallback_types={"code": "integer", "count": "integer",
                        "missing": "text"},
    )["frame"]
    assert table["user"].tolist() == ["1", "2"]
    assert table["rank"].iloc[0] == "1" and pd.isna(table["rank"].iloc[1])
    assert table["code"].tolist() == ["001", "002"]
    assert str(table["count"].dtype) == "Int64"
    assert kept(table["count"]) == [3, None]
    assert "missing" not in table.columns

    # 強い型と補いの型が同じ列なら強い型
    table = read_result(root, override_types={"count": "text"},
                        fallback_types={"count": "integer"})["frame"]
    assert kept(table["count"]) == ["3", None]

    # 辞書でない対応と知らない型の名前は誤り
    with pytest.raises(ValueError, match="override_types"):
        read_result(root, override_types=["user"])
    with pytest.raises(ValueError, match="fallback_types"):
        read_result(root, fallback_types={"user": "float"})
    for kind in (["text"], {"text": 1}):
        with pytest.raises(ValueError, match="override_types"):
            read_result(root, override_types={"user": kind})


def test_given_types_old_versions(tmp_path: Path) -> None:
    """
    column_types の無い版 1 と 2 の結果では, 補いの型が全ての列に効き,
    配列の列の欠けは空の文字のままであることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 1 の結果 (群の番号は欠けがあると小数に見える)
    root = write_raw(
        tmp_path / "one", "group,name,status\n0,a,ok\n,b,skipped\n",
        {"format_version": 1},
    )
    plain = read_result(root)["frame"]
    assert plain["group"].dtype == float
    table = read_result(root, fallback_types={"group": "integer"})["frame"]
    assert str(table["group"].dtype) == "Int64"
    assert kept(table["group"]) == [0, None]
    assert table["name"].tolist() == ["a", "b"]

    # 版 2 の配列の結果も, 配列の列の欠けは型に依らず空の文字
    root = write_raw(
        tmp_path / "two",
        "id,artifact,status\n7,artifacts/000000.npy,ok\n8,,skipped\n",
        {"format_version": 2,
         "artifacts": {"directory": "artifacts", "column": "artifact",
                       "format": "npy", "count": 1, "form": "vector"}},
    )
    back = read_result(root, override_types={"artifact": "text"},
                       fallback_types={"id": "text"})
    assert back["form"] == "vector"
    assert back["frame"]["id"].tolist() == ["7", "8"]
    assert back["frame"]["artifact"].tolist() == ["artifacts/000000.npy", ""]


def test_given_types_text_output(tmp_path: Path) -> None:
    """
    文の出力の文と番号の型は補いの型より強く, 強い型より弱いこと, 予測した
    行の空の文は空の文字のままであることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 番号 1 つの並び, 空の文 (予測した行), 予測しなかった行
    root = write_raw(
        tmp_path / "text",
        "user,status,prediction,tokens\n"
        "1,ok,5,5\n2,ok,,\n3,skipped,,\n",
        {"format_version": 1},
    )
    plain = read_result(root)["frame"]
    assert kept(plain["prediction"]) == ["5", "", None]
    for given in ({}, None):
        table = read_result(root, given, given)["frame"]
        assert kept(table["prediction"]) == ["5", "", None]
        assert kept(table["tokens"]) == ["5", "", None]

    # 補いの型は文と番号に効かず, 他の列には効く
    table = read_result(
        root, fallback_types={"prediction": "integer", "user": "text"}
    )["frame"]
    pd.testing.assert_series_equal(table["prediction"], plain["prediction"])
    assert kept(table["tokens"]) == ["5", "", None]
    assert table["user"].tolist() == ["1", "2", "3"]

    # 強い型は文と番号にも効き, 文字でない型の列は空の文字にしない
    table = read_result(root, override_types={"tokens": "integer"})["frame"]
    assert str(table["tokens"].dtype) == "Int64"
    assert kept(table["tokens"]) == [5, None, None]
    assert kept(table["prediction"]) == ["5", "", None]
    table = read_result(root, override_types={"prediction": "integer"})
    table = table["frame"]
    assert str(table["prediction"].dtype) == "Int64"
    assert kept(table["prediction"]) == [5, None, None]
    assert kept(table["tokens"]) == ["5", "", None]


def test_text_without_tokens(tmp_path: Path) -> None:
    """
    tokens の列の無い古い文の結果は文の出力と見分けず, 呼び手が
    override_types で予測を文字で読めることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 番号 1 つの予測, 空の文 (予測した行), 予測しなかった行
    root = write_raw(tmp_path / "text",
                     "status,prediction\nok,5\nok,\nskipped,\n",
                     {"format_version": 1, "output_form": "text"})
    table = read_result(root)["frame"]
    assert kept(table["prediction"]) == [5.0, None, None]

    # 呼び手の強い型で文字にでき, 空の文は欠損のまま (戻すのは呼び手)
    table = read_result(root, override_types={"prediction": "text"})
    assert kept(table["frame"]["prediction"]) == ["5", None, None]
