from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from conftest import linear_spec, save_weights
from ml_eda import export_dataset, load_data, load_dataset
from test_decision import regression_bundle
from test_noise_bundle import diffusion_bundle

from ml_inference import (
    backtest, explain, explain_rows, forecast, infer, load_bundle,
    make_bundle, predict, read_result, read_source, remeasure, write_result
)


def bundle_of(tmp_path: Path, dataset: Path, outputs: int) -> Path:
    """
    データセットの学習に使う列の数に合わせた全結合の 1 層の一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    dataset: Path
        データセット
    outputs: int
        出力の幅

    Returns
    ----------
    path: Path
        一式のディレクトリ
    """
    # 乱数の初期値のままの重み
    width = len(load_dataset(dataset).features)
    spec = linear_spec(width, outputs)
    weights = save_weights(spec, tmp_path / "weights.pt")
    path = tmp_path / "bundle"
    make_bundle(path, spec, dataset, weights)

    return path


def test_split_prediction_matches_model(
    tmp_path: Path, table_dataset: Path
) -> None:
    """
    分割を前処理をかけずに推論し, 分割の配列をモデルに通した確率と一致し,
    目的の列がクラスの名前に戻ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_dataset: Path
        データセット (分割のファイルは前処理の後の列)

    Returns
    ----------
    None
    """
    # 分割の名前で推論 (前処理を 2 度かけない)
    path = bundle_of(tmp_path, table_dataset, 3)
    source = {"dataset": str(table_dataset), "split": "test"}
    result = predict(path, source)
    dataset = load_dataset(table_dataset)
    x, y = dataset.arrays("test")
    assert len(result) == len(x) and (result["status"] == "ok").all()

    # 分割の配列をモデルに通した確率と同じ
    bundle = load_bundle(path)
    with torch.no_grad():
        expected = bundle.model()(torch.from_numpy(x)).softmax(dim=1)
    classes = bundle.classes
    got = result[[f"p:{c}" for c in classes]].to_numpy()
    assert np.allclose(got, expected.numpy(), atol=1e-6)

    # 目的の列はクラスの名前, 他の列は分割の列のまま
    assert list(result["label"]) == [classes[int(i)] for i in y]
    assert np.allclose(result["a"], dataset.frames["test"]["a"])

    # 印付きの Data を返す読み込みと, 印付きの Data の推論
    data = read_source(bundle, source)
    assert data.source["preprocessed"] is True
    assert data.source["split"] == "test"
    assert data.source["dataset"] == str(table_dataset.resolve())
    again = predict(bundle, data)
    assert np.allclose(
        again[[f"p:{c}" for c in classes]].to_numpy(), got
    )


def test_split_target_is_restored(
    tmp_path: Path, regression_dataset: Path
) -> None:
    """
    回帰の目的の列が, レシピの対数と標準化を戻した元の値になることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    regression_dataset: Path
        目的の列も対数と標準化で変換したデータセット

    Returns
    ----------
    None
    """
    # 元の y は 100 から 1000 の一様
    path = bundle_of(tmp_path, regression_dataset, 1)
    result = predict(path, {"dataset": regression_dataset, "split": "valid"})
    values = result["y"].to_numpy(float)
    assert np.all((values >= 100) & (values <= 1000))
    assert (result["status"] == "ok").all()


def test_split_result_and_rereading(tmp_path: Path) -> None:
    """
    分割の推論の結果に分割を残して版 5 で書き, 記録から作り直した辞書で
    外れ方の測り直しとたどり直しができることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 結果は入力の場所を空にし, 分割を残す
    path = regression_bundle(tmp_path)
    source = {"dataset": str(tmp_path / "ds"), "split": "test"}
    result = infer(path, source)
    assert result["source"] is None
    assert result["split"] == {
        "dataset": str((tmp_path / "ds").resolve()), "name": "test"
    }
    assert result["summary"]["rows"] == 100

    # 版 5 で書き, 読み直すと分割の記録が戻る
    root = write_result(result, tmp_path / "result")
    meta = yaml.safe_load((root / "result.yaml").read_text(encoding="utf-8"))
    assert meta["format_version"] == 5
    back = read_result(root)
    assert back["split"] == result["split"]

    # 分割でない結果は今までの版のまま
    plain = infer(path, pd.DataFrame({"x": [1.0, 2.0]}))
    assert "split" not in plain
    meta = yaml.safe_load(
        (write_result(plain, tmp_path / "plain") / "result.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert meta["format_version"] == 1

    # 印付きの Data を渡しても分割を残す
    data = read_source(load_bundle(path), source)
    assert infer(path, data)["split"] == result["split"]

    # 記録から作り直した辞書で測り直しとたどり直し (目的の列は元の値)
    again = {"dataset": back["split"]["dataset"],
             "split": back["split"]["name"]}
    measured = remeasure(path, again, write=False)
    assert measured["rows"] == 100 and measured["written"] is False
    decision = {"type": "quantity", "under_cost": 9, "over_cost": 1}
    traced = backtest(path, again, decision)
    assert traced["rows"] == 100 and "この方針" in traced["policies"]


def test_split_errors(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    分割を指す辞書の誤り, 違うデータセット, 前処理の前の行が要る処理を
    断ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_dataset: Path
        データセット
    table_csv: Path
        元の CSV

    Returns
    ----------
    None
    """
    # 辞書の形と分割の名前
    path = bundle_of(tmp_path, table_dataset, 3)
    dataset = str(table_dataset)
    with pytest.raises(ValueError, match="知らない項目"):
        predict(path, {"dataset": dataset, "split": "test", "x": 1})
    with pytest.raises(ValueError, match="split"):
        predict(path, {"dataset": dataset})
    with pytest.raises(ValueError, match="dataset"):
        predict(path, {"dataset": "", "split": "test"})
    with pytest.raises(ValueError, match="unlabeled の分割はありません"):
        predict(path, {"dataset": dataset, "split": "unlabeled"})
    with pytest.raises(ValueError, match="読めません"):
        predict(path, {"dataset": str(tmp_path / "none"), "split": "test"})
    with pytest.raises(ValueError, match="options"):
        predict(path, {"dataset": dataset, "split": "test"},
                options={"sep": ","})

    # 分け方の違うデータセットは当てはめた値が違う
    recipe = yaml.safe_load(
        (table_dataset / "recipe.yaml").read_text(encoding="utf-8")
    )
    recipe["split"]["seed"] = 1
    for step in recipe["steps"]:
        step.pop("state", None)
    other = tmp_path / "other"
    export_dataset(load_data(table_csv), recipe, other)
    with pytest.raises(ValueError, match="一式を作ったデータセットと違い"):
        predict(path, {"dataset": str(other), "split": "test"})

    # 前処理の前の行が要る処理
    source = {"dataset": dataset, "split": "test"}
    with pytest.raises(ValueError, match="列の効きは前処理の前の行"):
        explain(path, source)
    with pytest.raises(ValueError, match="行ごとの予測の理由は"):
        explain_rows(path, source)


def test_split_refused_for_steps_and_other_bundles(tmp_path: Path) -> None:
    """
    何段も先の予測と何段も先の外れ方と, 表と時系列のデータセットの一式で
    ない一式に分割を渡すと断ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 回帰の一式でも前処理の前の行が要る
    path = regression_bundle(tmp_path)
    source = {"dataset": str(tmp_path / "ds"), "split": "test"}
    with pytest.raises(ValueError, match="何段も先の予測は前処理の前の行"):
        forecast(path, source)
    with pytest.raises(ValueError, match="何段も先の外れ方は"):
        remeasure(path, source, write=False, horizon=2)
    with pytest.raises(ValueError, match="options"):
        predict(path, read_source(load_bundle(path), source),
                options={"sep": ","})

    # 雑音だけの一式
    (tmp_path / "noise").mkdir()
    noise = diffusion_bundle(tmp_path / "noise", "ddim")
    with pytest.raises(ValueError, match="表と時系列の版 1"):
        predict(noise, source)
