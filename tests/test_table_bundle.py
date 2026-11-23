import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from conftest import save_weights
from ml_modelcontrol import load_example

import ml_inference.bundle as bundle_module
from ml_inference import (
    describe_bundle, infer, load_bundle, make_bundle, predict, read_result,
    write_result
)
from ml_inference.tables import decode_table_rows, table_width

# 符号化した行の幅が 16 (拡散の雑音の予測器の幅) になる列の符号化
COLUMNS = [
    {"name": "x", "type": "number", "mean": 10.0, "std": 2.0,
     "integer": False},
    {"name": "n", "type": "number", "mean": 3.0, "std": 1.5,
     "integer": True},
    {"name": "color", "type": "category",
     "categories": ["blue", "green", "red"]},
    {"name": "flag", "type": "category", "categories": [False, True]},
    {"name": "grade", "type": "category", "categories": list(range(1, 10))},
]

# 拡散のモデルの一式のつなぎ方と手順
SETTINGS = {
    "inputs": {"x": {"source": "state"}, "t": {"source": "time"}},
    "procedure": {"name": "ddim", "steps": 5, "seed": 3},
}


def read_info(path: Path) -> dict:
    """
    一式の inference.yaml を読む関数

    Parameters
    ----------
    path: Path
        一式のディレクトリ

    Returns
    ----------
    info: dict
        中身
    """
    # yaml のまま
    info = yaml.safe_load(
        (path / "inference.yaml").read_text(encoding="utf-8")
    )

    return info


def rewrite(path: Path, info: dict) -> None:
    """
    一式の inference.yaml を書き換える関数

    Parameters
    ----------
    path: Path
        一式のディレクトリ
    info: dict
        書く中身

    Returns
    ----------
    None
    """
    # そのまま書く
    (path / "inference.yaml").write_text(
        yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )


def table_bundle(tmp_path: Path, name: str, **options: Any) -> Path:
    """
    拡散のモデルの表の出力の雑音だけの一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    name: str
        一式の名前
    **options: Any
        make_bundle に渡す引数 (つなぎ方, 手順, 出力の形に重ねる)

    Returns
    ----------
    path: Path
        一式のディレクトリ
    """
    # 雑音の予測器の重みは名前に依らず同じ種で
    spec = load_example("diffusion_denoiser")
    weights = save_weights(spec, tmp_path / "w.pt")
    path = tmp_path / name
    settings = {
        **SETTINGS, "form": "table",
        "output_options": {"columns": COLUMNS}, **options,
    }
    make_bundle(path, spec, None, weights, **settings)

    return path


def test_table_bundle_returns_typed_rows(tmp_path: Path) -> None:
    """
    表の出力の雑音だけの一式が版 7 で列の符号化を写し, 件数と種から列の
    名前と型のある表を返し, ベクトルの一式の行を戻した値と同じになること
    を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 7 で, 列の符号化と 1 件の形 (符号化した行の幅) を持つ
    path = table_bundle(tmp_path, "table")
    info = read_info(path)
    assert info["format_version"] == 7
    assert info["dataset"] is None and info["target"] is None
    assert info["output"]["form"] == "table"
    assert info["output"]["shape"] == [16]
    assert info["output"]["columns"] == COLUMNS

    # 件の番号と種に, 列の符号化の順の列
    frame = predict(path, {"count": 6, "seed": 1})
    assert frame.columns.tolist() == [
        "sample", "seed", "status", "reason", "x", "n", "color", "flag",
        "grade"
    ]
    assert (frame["status"] == "ok").all()
    assert frame["x"].dtype == np.float64
    assert str(frame["n"].dtype) == "Int64"
    assert frame["color"].dtype == object
    assert set(frame["color"]) <= {"blue", "green", "red"}
    assert all(isinstance(v, bool) for v in frame["flag"])
    assert all(isinstance(v, int) and 1 <= v <= 9 for v in frame["grade"])

    # 同じ重みと種のベクトルの一式の行を, 列の符号化で戻した値と同じ
    vector = table_bundle(
        tmp_path, "vector", form="vector",
        output_options={"shape": [16]}
    )
    assert read_info(vector)["format_version"] == 5
    rows = predict(vector, {"count": 6, "seed": 1})
    encoded = np.stack(rows["output"].to_list())
    expected = decode_table_rows(COLUMNS, encoded)
    assert frame["x"].tolist() == pytest.approx(expected.pop("x"))
    for name, values in expected.items():
        assert frame[name].tolist() == values
    assert frame["n"].tolist() == [
        int(np.round(v * 1.5 + 3.0)) for v in encoded[:, 1]
    ]


def test_table_bundle_result_and_description(tmp_path: Path) -> None:
    """
    表の出力の一式を列の符号化のファイルから作り, 結論, 書き出し, 説明を
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # ML-Quest の骨組みの table_columns.json の形 ({"columns": [...]})
    file = tmp_path / "table_columns.json"
    file.write_text(json.dumps({"columns": COLUMNS}), encoding="utf-8")
    path = table_bundle(tmp_path, "table",
                        output_options={"columns": file})
    assert read_info(path)["output"]["columns"] == COLUMNS

    # 結論は件数だけ. 書き出しは版 1 の表で, 配列を書かない
    result = infer(path, {"count": 4, "seed": 0})
    assert result["form"] == "table"
    assert result["summary"]["rows"] == 4
    place = tmp_path / "result"
    write_result(result, place)
    assert not (place / "artifacts").exists()
    back = read_result(place)
    assert back["format_version"] == 1 and back["form"] is None
    assert back["frame"]["color"].tolist() == (
        result["frame"]["color"].tolist()
    )
    assert back["frame"]["n"].tolist() == result["frame"]["n"].tolist()

    # 説明の出力の形に列の符号化
    description = describe_bundle(load_bundle(path))
    assert description["kind"] == "noise"
    assert description["format_version"] == 7
    assert description["column_output"] is False
    assert description["output"]["columns"] == COLUMNS


def test_make_table_bundle_errors(
    tmp_path: Path, table_dataset: Path
) -> None:
    """
    表の出力の一式に使えない設定で, 作る前に止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_dataset: Path
        データセット (表の出力の形を使えない)

    Returns
    ----------
    None
    """
    # 列の符号化の無い場合, 形の誤り, 予測の表の列と重なる名前, 幅の違い
    number = {"type": "number", "mean": 0.0, "std": 1.0, "integer": False}
    cases = [
        ({}, "columns"),
        ({"columns": []}, "1 つ以上"),
        ({"columns": [{**number, "name": "seed"}]}, "重なります"),
        ({"columns": [{**number, "name": "a"}, {**number, "name": "a"}]},
         "重なっています"),
        ({"columns": [{**number, "name": "a", "std": 0.0}]}, "std"),
        ({"columns": [{"name": "a", "type": "date"}]}, "type"),
        ({"columns": [{"name": "a", "type": "category",
                       "categories": [1, "1"]}]}, "categories"),
        ({"columns": COLUMNS, "shape": [8]}, "幅"),
        ({"columns": tmp_path / "missing.json"}, "ありません"),
    ]
    for index, (options, message) in enumerate(cases):
        with pytest.raises(ValueError, match=message):
            table_bundle(tmp_path, f"b{index}", output_options=options)
        assert not (tmp_path / f"b{index}").exists()

    # 列の符号化は表の出力だけ, 表の出力は雑音だけの一式だけ
    with pytest.raises(ValueError, match="table だけ"):
        table_bundle(tmp_path, "vector", form="vector",
                     output_options={"shape": [16], "columns": COLUMNS})
    spec = load_example("diffusion_denoiser")
    weights = save_weights(spec, tmp_path / "w.pt")
    with pytest.raises(ValueError, match="データセット無し"):
        make_bundle(tmp_path / "data", spec, table_dataset, weights,
                    form="table", output_options={"columns": COLUMNS},
                    **SETTINGS)
    assert not (tmp_path / "data").exists()


def test_table_bundle_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    版 7 の決まりと, 版 6 までの読む側が版 7 の一式で止まることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        読める版を戻す

    Returns
    ----------
    None
    """
    # 表の出力は版 7 だけ, 版 7 は表の出力だけ, 列の符号化の誤り
    path = table_bundle(tmp_path, "table")
    original = read_info(path)
    rewrite(path, {**original, "format_version": 5})
    with pytest.raises(ValueError, match="版 7"):
        load_bundle(path)
    rewrite(path, {**original, "output": {"form": "vector",
                                          "shape": [16]}})
    with pytest.raises(ValueError, match="版 7"):
        load_bundle(path)
    rewrite(path, {**original, "output": {**original["output"],
                                          "shape": [15]}})
    with pytest.raises(ValueError, match="幅"):
        load_bundle(path)
    rewrite(path, {**original, "output": {**original["output"],
                                          "columns": "x"}})
    with pytest.raises(ValueError, match="columns"):
        load_bundle(path)
    rewrite(path, original)
    load_bundle(path)

    # 版 6 までしか知らない読む側は, 版 7 の一式を読めないと止まる
    monkeypatch.setattr(bundle_module, "BUNDLE_FORMAT_VERSION", 6)
    with pytest.raises(ValueError, match="format_version 7 .*読めません"):
        load_bundle(path)


def test_decode_table_rows() -> None:
    """
    符号化した行の戻し方 (標準化の逆, 整数の丸め, 区間の最も大きい種類,
    数でない値) を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 数の列 2 つと種類の列 1 つの幅 5
    columns = [
        {"name": "a", "type": "number", "mean": 1.0, "std": 2.0,
         "integer": False},
        {"name": "b", "type": "number", "mean": 0.0, "std": 1.0,
         "integer": True},
        {"name": "c", "type": "category", "categories": ["p", "q", "r"]},
    ]
    assert table_width(columns) == 5
    outputs = np.array([
        [0.5, 1.4, 0.1, 0.9, -1.0],
        [np.nan, -2.6, 2.0, 0.0, 1.0],
    ], dtype=np.float32)
    values = decode_table_rows(columns, outputs)
    assert values["a"][0] == pytest.approx(2.0) and values["a"][1] is None
    assert values["b"] == [1, -3]
    assert values["c"] == ["q", "p"]
    with pytest.raises(ValueError, match="件数, 5"):
        decode_table_rows(columns, outputs[:, :4])
