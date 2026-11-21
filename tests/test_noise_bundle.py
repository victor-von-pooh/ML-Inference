from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from conftest import linear_spec, save_weights
from ml_eda import load_dataset
from ml_modelcontrol import load_example

import ml_inference.bundle as bundle_module
from ml_inference import (
    describe_bundle, infer, load_bundle, make_bundle, predict, read_result,
    register_procedure, write_result
)
from ml_inference.procedures import PROCEDURES

# 拡散のモデルの一式のつなぎ方 (雑音の中身 x と時刻 t)
DIFFUSION = {
    "inputs": {"x": {"source": "state"}, "t": {"source": "time"}},
    "form": "vector",
    "output_options": {"shape": [16]},
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


def diffusion_bundle(tmp_path: Path, name: str, **options: Any) -> Path:
    """
    拡散のモデルの雑音だけの一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    name: str
        一式の名前
    **options: Any
        make_bundle に渡す引数 (つなぎ方と手順に重ねる)

    Returns
    ----------
    path: Path
        一式のディレクトリ
    """
    # 雑音の予測器 (x と時刻 t から 16 の値) を DDIM で
    spec = load_example("diffusion_denoiser")
    weights = save_weights(spec, tmp_path / f"{name}.pt")
    path = tmp_path / name
    settings = {
        **DIFFUSION,
        "procedure": {"name": "ddim", "steps": 5, "seed": 3},
        **options,
    }
    make_bundle(path, spec, None, weights, **settings)

    return path


def once(
    model: Any, state: torch.Tensor, steps: int,
    generator: torch.Generator
) -> torch.Tensor:
    """
    雑音をモデルに 1 回だけ通す自作の手順の関数

    Parameters
    ----------
    model: Any
        model(x) を返すもの
    state: torch.Tensor
        始まりの雑音
    steps: int
        繰り返しの回数 (使わない)
    generator: torch.Generator
        乱数の生成器 (使わない)

    Returns
    ----------
    x: torch.Tensor
        モデルの出力
    """
    # 時刻を渡さずに 1 回
    x = model(state)

    return x


def outputs_of(frame: pd.DataFrame) -> np.ndarray:
    """
    予測の表の output の列を 1 つの配列にする関数

    Parameters
    ----------
    frame: pd.DataFrame
        予測の表

    Returns
    ----------
    stacked: np.ndarray
        (件数, ...) の配列
    """
    # 件ごとの配列を重ねる
    stacked = np.stack(frame["output"].to_list())

    return stacked


def test_noise_bundle_generates_from_count_and_seed(
    tmp_path: Path
) -> None:
    """
    データセット無しで雑音だけの一式を作り, 件数と種で推論でき, 同じ件数
    と種なら同じ出力になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 5 で, データセットを持たない
    path = diffusion_bundle(tmp_path, "ddim")
    info = read_info(path)
    assert info["format_version"] == 5
    assert info["dataset"] is None
    assert not (path / "dataset").exists()
    assert info["role"] == "generate"
    assert info["target"] is None
    assert info["check"] is None and info["uncertainty"] is None

    # 件ごとの番号と種と出力
    first = predict(path, {"count": 5, "seed": 1})
    assert first["sample"].tolist() == [0, 1, 2, 3, 4]
    assert first["seed"].tolist() == [1] * 5
    assert (first["status"] == "ok").all()
    assert outputs_of(first).shape == (5, 16)

    # 同じ件数と種なら同じ, 種を変えると違う, 省くと手順の種
    again = predict(path, {"count": 5, "seed": 1})
    assert np.array_equal(outputs_of(first), outputs_of(again))
    other = predict(path, {"count": 5, "seed": 2})
    assert not np.allclose(outputs_of(first), outputs_of(other))
    default = predict(path, {"count": 5})
    assert default["seed"].tolist() == [3] * 5
    assert np.array_equal(
        outputs_of(default), outputs_of(predict(path, {"count": 5,
                                                       "seed": 3}))
    )

    # 1 回に渡す件数に依らず, 件数を増やしても前の件は同じ
    small = predict(path, {"count": 5, "seed": 1}, batch_size=2)
    assert np.allclose(outputs_of(first), outputs_of(small), atol=1e-6)
    longer = predict(path, {"count": 7, "seed": 1})
    assert np.allclose(
        outputs_of(first), outputs_of(longer)[:5], atol=1e-6
    )


def test_noise_bundle_result_and_description(tmp_path: Path) -> None:
    """
    雑音だけの一式の結論, 書き出し, 説明を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 結論は件数だけで, 入力は件数と種の文
    path = diffusion_bundle(tmp_path, "ddim")
    result = infer(path, {"count": 4, "seed": 0})
    assert result["source"] == "件数 4, 種 0"
    assert result["summary"]["rows"] == 4
    with pytest.raises(ValueError, match="使えません"):
        infer(path, {"count": 4}, rules={"min_confidence": 0.5})

    # 配列は artifacts/ に書き, 読み直せる
    place = tmp_path / "result"
    write_result(result, place)
    back = read_result(place)
    assert back["source"] == "件数 4, 種 0"
    assert len(list((place / "artifacts").glob("*.npy"))) == 4

    # 説明はデータセットの欄が空で, 種類は noise
    bundle = load_bundle(path)
    assert bundle.dataset.info == {} and bundle.dataset.recipe == {}
    description = describe_bundle(bundle)
    assert description["kind"] == "noise"
    assert description["task"] == ""
    assert description["target"] is None
    assert description["features"] == []
    assert description["format_version"] == 5
    assert description["procedure"]["name"] == "ddim"


def test_noise_bundle_requests(tmp_path: Path, table_dataset: Path) -> None:
    """
    件数と種の誤りと, データと件数の取り違えで止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_dataset: Path
        データセット (列の予測の一式を作る)

    Returns
    ----------
    None
    """
    # 件数は 1 以上の整数, 種は整数, 知らない項目は誤り
    path = diffusion_bundle(tmp_path, "ddim")
    cases = [
        ({"count": 0}, "count"),
        ({"count": True}, "count"),
        ({"count": 2.0}, "count"),
        ({"seed": 0}, "count"),
        ({"count": 2, "seed": "a"}, "seed"),
        ({"count": 2, "seed": 2 ** 64}, "seed"),
        ({"count": 2, "rows": 3}, "知らない項目"),
    ]
    for request, message in cases:
        with pytest.raises(ValueError, match=message):
            predict(path, request)
    assert len(predict(path, {"count": np.int64(2)})) == 2

    # 雑音だけの一式にデータ, 他の一式に件数と種は誤り
    with pytest.raises(ValueError, match="件数と種"):
        predict(path, pd.DataFrame({"a": [1]}))
    width = len(load_dataset(table_dataset).features)
    spec = linear_spec(width, 3)
    weights = save_weights(spec, tmp_path / "table.pt")
    make_bundle(tmp_path / "table", spec, table_dataset, weights)
    with pytest.raises(ValueError, match="雑音だけの一式"):
        predict(tmp_path / "table", {"count": 2})


def test_make_noise_bundle_errors(tmp_path: Path) -> None:
    """
    雑音だけの一式に使えない設定で, 作る前に止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 手順, 出力の形, 雑音から始まる手順, データの入力, 列の項目
    spec = load_example("diffusion_denoiser")
    weights = save_weights(spec, tmp_path / "w.pt")
    ddim = {"name": "ddim", "steps": 2, "seed": 0}
    cases = [
        ({**DIFFUSION}, "procedure"),
        ({**DIFFUSION, "form": None, "procedure": ddim}, "form"),
        ({**DIFFUSION, "procedure": {"name": "greedy", "steps": 2,
                                     "seed": 0}}, "雑音から始まる"),
        ({**DIFFUSION, "procedure": ddim,
          "inputs": {"x": {"source": "features"},
                     "t": {"source": "time"}}}, "source"),
        ({**DIFFUSION, "procedure": ddim, "target_scale": "train"},
         "target_scale"),
        ({**DIFFUSION, "procedure": ddim,
          "conclusion": {"min_confidence": 0.5}}, "結論"),
        ({**DIFFUSION, "procedure": ddim, "form": "class"}, "class"),
        ({**DIFFUSION, "procedure": ddim, "output_options": {}},
         "output.shape"),
    ]
    for index, (options, message) in enumerate(cases):
        with pytest.raises(ValueError, match=message):
            make_bundle(tmp_path / f"b{index}", spec, None, weights,
                        **options)
        assert not (tmp_path / f"b{index}").exists()


def test_noise_bundle_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    版 5 の決まりと, 版 4 までの読む側が版 5 の一式で止まることを確かめる
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
    # dataset が null の一式は版 5 だけ, 版 5 は items を持てない
    path = diffusion_bundle(tmp_path, "ddim")
    original = read_info(path)
    rewrite(path, {**original, "format_version": 4})
    with pytest.raises(ValueError, match="dataset"):
        load_bundle(path)
    rewrite(path, {**original, "items": {"input": "image"}})
    with pytest.raises(ValueError, match="items"):
        load_bundle(path)
    rewrite(path, {**original, "inputs": {
        "x": {"source": "items"}, "t": {"source": "time"}}})
    with pytest.raises(ValueError, match="データの入力"):
        load_bundle(path)
    rewrite(path, original)
    load_bundle(path)

    # 版 4 までしか知らない読む側は, 版 5 の一式を読めないと止まる
    monkeypatch.setattr(bundle_module, "BUNDLE_FORMAT_VERSION", 4)
    with pytest.raises(ValueError, match="format_version 5 .*読めません"):
        load_bundle(path)


@pytest.fixture
def once_procedure() -> Iterator[str]:
    """
    雑音を 1 回だけ通す自作の手順を登録し, 終わったら登録簿から除く関数

    Parameters
    ----------
    None

    Returns
    ----------
    name: Iterator[str]
        登録した手順の名前
    """
    # 他のテストに残さない
    register_procedure("noise_once", once, "noise", False)
    yield "noise_once"
    PROCEDURES.pop("noise_once", None)


def test_single_input_noise_bundle(
    tmp_path: Path, once_procedure: str
) -> None:
    """
    入力が 1 つのモデルで inputs を省いた雑音だけの一式が, 1 件の要素の
    少ない形でも 1 回に渡す件数に依らず同じ出力になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    once_procedure: str
        時刻を渡さない自作の手順の名前

    Returns
    ----------
    None
    """
    # 3 つの値の全結合
    spec = linear_spec(3, 3)
    weights = save_weights(spec, tmp_path / "w.pt")
    path = tmp_path / "once"
    make_bundle(
        path, spec, None, weights, form="vector",
        output_options={"shape": [3]},
        procedure={"name": once_procedure, "steps": 1, "seed": 0}
    )
    assert read_info(path)["inputs"] == {"x": {"source": "state"}}
    whole = outputs_of(predict(path, {"count": 7}))
    parts = outputs_of(predict(path, {"count": 7}, batch_size=3))
    assert whole.shape == (7, 3)
    assert np.allclose(whole, parts, atol=1e-6)

    # 読み込みの設定と, 正解, 確かめ, 外れ方を持つ版 5 は誤り
    with pytest.raises(ValueError, match="options"):
        predict(path, {"count": 2}, options={"sep": ","})
    original = read_info(path)
    for key in ("check", "uncertainty", "target"):
        rewrite(path, {**original, key: {"kind": "column"}})
        with pytest.raises(ValueError, match=key):
            load_bundle(path)
    with pytest.raises(ValueError, match="target"):
        make_bundle(
            tmp_path / "target", spec, None, weights, form="vector",
            output_options={"shape": [3]}, target={"kind": "image"},
            procedure={"name": once_procedure, "steps": 1, "seed": 0}
        )


def test_dataset_noise_procedure_ignores_batch_size(
    tmp_path: Path, table_dataset: Path, once_procedure: str
) -> None:
    """
    データセットの一式の雑音から始まる手順も, 1 件の要素が 16 の倍数で
    ない形で 1 回に渡す件数に依らず同じ出力になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_dataset: Path
        データセット (行の数を決めるだけで, 列は使わない)
    once_procedure: str
        時刻を渡さない自作の手順の名前

    Returns
    ----------
    None
    """
    # 行の数だけ雑音を引く
    spec = linear_spec(3, 3)
    weights = save_weights(spec, tmp_path / "w.pt")
    path = tmp_path / "rows"
    make_bundle(
        path, spec, table_dataset, weights, form="vector",
        inputs={"x": {"source": "state"}}, output_options={"shape": [3]},
        procedure={"name": once_procedure, "steps": 1, "seed": 0}
    )
    assert read_info(path)["format_version"] == 3
    frame = pd.DataFrame({"a": range(7)})
    whole = outputs_of(predict(path, frame))
    parts = outputs_of(predict(path, frame, batch_size=3))
    assert np.allclose(whole, parts, atol=1e-6)
