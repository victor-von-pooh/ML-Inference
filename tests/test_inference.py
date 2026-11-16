from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from conftest import linear_spec, save_weights
from ml_eda import export_dataset, load_data, load_dataset
from ml_inference import (
    check_rules, conclude, describe_bundle, infer, load_bundle, make_bundle,
    predict, write_result
)


def table_bundle(tmp_path: Path, dataset: Path, **options) -> Path:
    """
    表の分類の一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    dataset: Path
        データセット
    **options
        make_bundle に渡す引数

    Returns
    ----------
    path: Path
        一式のディレクトリ
    """
    # 学習に使う列の数に合わせた全結合の 1 層
    width = len(load_dataset(dataset).features)
    spec = linear_spec(width, 3)
    weights = save_weights(spec, tmp_path / "weights.pt")
    path = tmp_path / "bundle"
    make_bundle(path, spec, dataset, weights, **options)

    return path


def test_table_prediction_matches_model(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    表の予測が, 前処理とモデルを手で通した確率と一致することを確かめる
    関数

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
    # 一式で予測
    path = table_bundle(tmp_path, table_dataset)
    result = predict(path, table_csv)
    assert list(result.columns[:5]) == ["id", "a", "b", "color", "label"]
    assert (result["status"] == "ok").all()

    # 手で前処理してモデルに通した確率
    bundle = load_bundle(path)
    x = bundle.dataset.transform(pd.read_csv(table_csv)).to_numpy(
        dtype=np.float32
    )
    with torch.no_grad():
        expected = bundle.model()(torch.from_numpy(x)).softmax(dim=1)
    classes = bundle.classes
    got = result[[f"p:{c}" for c in classes]].to_numpy()
    assert np.allclose(got, expected.numpy(), atol=1e-6)
    assert list(result["prediction"]) == [
        classes[i] for i in expected.argmax(dim=1)
    ]
    assert np.allclose(result["confidence"], expected.max(dim=1).values)


def test_dataframe_source_and_describe(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    表を直接渡せることと, 一式の説明を確かめる関数

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
    # 目的の列の無い表でも予測できる
    path = table_bundle(tmp_path, table_dataset)
    frame = pd.read_csv(table_csv).drop(columns=["label"]).head(5)
    result = predict(path, frame)
    assert len(result) == 5 and (result["status"] == "ok").all()

    # 説明
    description = describe_bundle(path)
    assert description["task"] == "classification"
    assert description["classes"] == ["犬", "猫", "鳥"]
    assert description["model"]["inputs"] == {
        "x": {"source": "features", "prepend": []}
    }
    assert description["reading"]["loader"] == "csv"


def test_missing_column_is_explained(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    前処理に要る列が無い場合に, 列の名前を挙げた誤りになることを確かめる
    関数

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
    # 色の列を落とした表
    path = table_bundle(tmp_path, table_dataset)
    frame = pd.read_csv(table_csv).drop(columns=["color"])
    with pytest.raises(ValueError, match="color"):
        predict(path, frame)


def test_regression_restores_units(
    tmp_path: Path, regression_dataset: Path
) -> None:
    """
    回帰の予測が, 前処理の外とレシピの両方の変換を戻した値になることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    regression_dataset: Path
        目的の列を対数と標準化で変換したデータセット

    Returns
    ----------
    None
    """
    # 前処理の外の変換も書いた一式
    spec = linear_spec(1, 1)
    weights = save_weights(spec, tmp_path / "w.pt")
    path = tmp_path / "bundle"
    scale = {"center": 0.5, "scale": 2.0}
    make_bundle(path, spec, regression_dataset, weights, target_scale=scale)
    frame = pd.DataFrame({"shop": ["東", "西"], "x": [0.3, -1.2]})
    result = predict(path, frame)

    # 手で戻した値 (外の変換, 標準化, 対数の順)
    bundle = load_bundle(path)
    x = bundle.dataset.transform(frame).to_numpy(dtype=np.float32)
    with torch.no_grad():
        raw = bundle.model()(torch.from_numpy(x.copy()))[:, 0].double()
    raw = raw.numpy()
    steps = {s["op"]: s for s in bundle.dataset.recipe["steps"]}
    center, width = steps["standardize"]["state"]["values"]["y"]
    expected = np.expm1((raw * 2.0 + 0.5) * width + center)
    assert np.allclose(result["prediction"], expected)
    assert "confidence" not in result.columns


def test_rows_with_missing_features_are_skipped(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    前処理の後に欠損のある行を, 理由を付けて飛ばすことを確かめる関数

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
    # 1 行目の a を欠損に
    path = table_bundle(tmp_path, table_dataset)
    frame = pd.read_csv(table_csv).head(4)
    frame.loc[0, "a"] = np.nan
    result = predict(path, frame)
    assert list(result["status"]) == ["skipped", "ok", "ok", "ok"]
    assert "a" in result.loc[0, "reason"]
    assert pd.isna(result.loc[0, "prediction"])
    assert pd.isna(result.loc[0, "confidence"])


def test_text_with_prepended_token(tmp_path: Path) -> None:
    """
    文の番号の並びの先頭に印を足して推論できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 文字の語彙で長さ 6 の並びにするデータセット
    texts = ["良い品", "悪い品", "とても良い", "とても悪い"] * 5
    labels = ["良", "悪", "良", "悪"] * 5
    source = tmp_path / "reviews.csv"
    pd.DataFrame({"text": texts, "label": labels}).to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [],
        "item_steps": [{"op": "vocabulary", "unit": "char", "length": 6}],
        "split": {"method": "random", "valid": 0.2, "test": 0.0, "seed": 0},
    }
    dataset = tmp_path / "dataset"
    export_dataset(load_data(source, text_column="text"), recipe, dataset)
    vocabulary = next(
        s["state"]["vocabulary"]
        for s in load_dataset(dataset).recipe["item_steps"]
    )
    cls_id = len(vocabulary)

    # 埋め込みの平均と全結合 ([CLS] の分だけ語の数が多い)
    spec = {
        "format_version": 1,
        "inputs": {"ids": {"shape": [2, 7], "dtype": "int64"}},
        "nodes": {
            "bag": {
                "unit": "nn.EmbeddingBag",
                "params": {"num_embeddings": cls_id + 1, "embedding_dim": 4,
                           "mode": "mean"},
                "inputs": {"input": "ids"},
            },
            "head": {
                "unit": "nn.Linear",
                "params": {"in_features": 4, "out_features": 2},
                "inputs": "bag",
            },
        },
        "outputs": {"logits": "head"},
    }
    weights = save_weights(spec, tmp_path / "w.pt")
    path = tmp_path / "bundle"
    make_bundle(
        path, spec, dataset, weights,
        inputs={"ids": {"source": "items", "prepend": [cls_id]}}
    )

    # 表とファイルのどちらでも, 手で並べた入力と同じ確率
    frame = pd.DataFrame({"text": ["良い品", "未知の文字"]})
    result = predict(path, frame)
    bundle = load_bundle(path)
    ids = np.stack([bundle.dataset.transform_item(t) for t in frame["text"]])
    ids = np.concatenate([np.full((2, 1), cls_id), ids], axis=1)
    with torch.no_grad():
        expected = bundle.model()(torch.from_numpy(ids)).softmax(dim=1)
    got = result[[f"p:{c}" for c in bundle.classes]].to_numpy()
    assert np.allclose(got, expected.numpy(), atol=1e-6)
    assert len(predict(path, source)) == len(texts)


def test_make_bundle_rejects_mismatch(
    tmp_path: Path, table_dataset: Path
) -> None:
    """
    出力の幅がクラスの数と合わない一式は作らないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_dataset: Path
        データセット

    Returns
    ----------
    None
    """
    # 3 クラスに 4 つの出力
    width = len(load_dataset(table_dataset).features)
    spec = linear_spec(width, 4)
    weights = save_weights(spec, tmp_path / "w.pt")
    path = tmp_path / "bundle"
    with pytest.raises(ValueError, match="クラスの数"):
        make_bundle(path, spec, table_dataset, weights)
    assert not path.exists()
    assert not (tmp_path / ".bundle.writing").exists()

    # 入力の幅の違い
    spec = linear_spec(width + 1, 3)
    weights = save_weights(spec, tmp_path / "w2.pt")
    with pytest.raises(ValueError, match="見本の推論に失敗"):
        make_bundle(path, spec, table_dataset, weights)


def test_bundle_from_run_dir_and_overwrite(
    tmp_path: Path, table_dataset: Path
) -> None:
    """
    実験のディレクトリから重みと状態を取り, 置き換えの決まりを守ることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_dataset: Path
        データセット

    Returns
    ----------
    None
    """
    # ML-Found の実験のディレクトリの形
    width = len(load_dataset(table_dataset).features)
    spec = linear_spec(width, 3)
    run = tmp_path / "run"
    run.mkdir()
    save_weights(spec, run / "final_model.pt")
    (run / "run.json").write_text(
        '{"format_version": 1, "status": "completed", "monitor": "valid/acc",'
        ' "best_value": 0.9, "best_epoch": 3}', encoding="utf-8"
    )
    path = tmp_path / "bundle"
    info = make_bundle(path, spec, table_dataset, run, copy_weights=False)
    assert info["run"]["status"] == "completed"
    assert info["run"]["best_value"] == 0.9
    assert Path(info["model"]["weights"]).is_absolute()
    assert not (path / "final_model.pt").exists()

    # 置き換えは overwrite で, 一式でない場所は置き換えない
    with pytest.raises(FileExistsError):
        make_bundle(path, spec, table_dataset, run)
    make_bundle(path, spec, table_dataset, run, overwrite=True)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(FileExistsError, match="一式ではない"):
        make_bundle(other, spec, table_dataset, run, overwrite=True)


def test_load_bundle_checks(tmp_path: Path, table_dataset: Path) -> None:
    """
    一式の外を指すパス, 新しい版, 知らない入力を誤りにすることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_dataset: Path
        データセット

    Returns
    ----------
    None
    """
    # 正しい一式を書き換えて確かめる
    path = table_bundle(tmp_path, table_dataset)
    file = path / "inference.yaml"
    original = yaml.safe_load(file.read_text(encoding="utf-8"))

    def write(**changes: object) -> None:
        file.write_text(
            yaml.safe_dump({**original, **changes}, allow_unicode=True),
            encoding="utf-8"
        )

    write(dataset="../dataset")
    with pytest.raises(ValueError, match="一式の外"):
        load_bundle(path)
    write(format_version=2)
    with pytest.raises(ValueError, match="読めません"):
        load_bundle(path)
    write(inputs={"y": {"source": "features"}})
    with pytest.raises(ValueError, match="モデルの入力にありません"):
        load_bundle(path)
    write(extra=1)
    with pytest.raises(ValueError, match="知らない項目"):
        load_bundle(path)


def test_conclusion_rules(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    保留, 陽性のしきい値, まとめる列の結論を確かめる関数

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
    # 確信の下限より低い行は保留
    path = table_bundle(tmp_path, table_dataset)
    result = infer(path, table_csv, rules={"min_confidence": 0.5,
                                           "group_by": "color"})
    frame = result["frame"]
    low = frame["confidence"] < 0.5
    assert (frame.loc[low, "decision"] == "保留").all()
    assert (frame.loc[~low, "decision"] == frame.loc[~low, "prediction"]).all()
    summary = result["summary"]
    assert summary["rows"] == summary["predicted"] == 60
    assert sum(d["count"] for d in summary["decisions"]) == 60
    assert {g["group"] for g in summary["groups"]} == {"赤", "青"}
    assert result["sentences"][0] == "60 件のうち 60 件を予測"

    # 2 クラスの陽性のしきい値
    prediction = pd.DataFrame({
        "status": ["ok", "ok", "skipped"],
        "prediction": ["良", "悪", None],
        "confidence": [0.6, 0.7, None],
        "p:悪": [0.4, 0.7, None],
        "p:良": [0.6, 0.3, None],
    })
    rules = {"positive": "悪", "threshold": 0.3}
    decided = conclude(prediction, "classification", ["悪", "良"], rules)
    assert list(decided["frame"]["decision"]) == ["悪", "悪", "予測なし"]
    assert decided["summary"]["skipped"] == 1


def test_bands_and_rule_errors() -> None:
    """
    値の帯と, 結論の決め方の誤りを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 上限より小さい値はその帯
    prediction = pd.DataFrame({
        "status": ["ok", "ok", "ok", "skipped"],
        "prediction": [50.0, 100.0, 500.0, None],
    })
    bands = [{"label": "少", "upper": 100}, {"label": "中", "upper": 300},
             {"label": "多"}]
    result = conclude(prediction, "regression", [], {"bands": bands})
    assert list(result["frame"]["decision"]) == ["少", "中", "多", "予測なし"]
    assert result["summary"]["values"]["sum"] == 650.0

    # 誤りの一覧
    assert check_rules({"min_confidence": 2}, "classification", ["a", "b"])
    assert check_rules({"positive": "c", "threshold": 0.5},
                       "classification", ["a", "b"])
    assert check_rules({"positive": "a", "threshold": 0.5},
                       "classification", ["a", "b", "c"])
    assert check_rules({"bands": [{"label": "x", "upper": 5},
                                  {"label": "y", "upper": 3},
                                  {"label": "z"}]}, "regression", [])
    assert check_rules({"bands": bands}, "classification", ["a", "b"])
    assert check_rules({"unknown": 1}, "regression", [])
    assert check_rules({"group_by": "shop"}, "regression", [], ["x"])
    assert check_rules({"bands": bands, "group_by": "x"}, "regression", [],
                       ["x"]) == []
    with pytest.raises(ValueError, match="決め方の誤り"):
        conclude(prediction, "regression", [], {"min_confidence": 0.5})


def test_write_result(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    予測と結論をファイルに書き出せることを確かめる関数

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
    # 既定の決め方を持つ一式で推論し, 書き出す
    path = table_bundle(
        tmp_path, table_dataset, conclusion={"min_confidence": 0.4}
    )
    result = infer(path, table_csv)
    assert result["rules"] == {"min_confidence": 0.4}
    out = write_result(result, tmp_path / "result")
    table = pd.read_csv(out / "predictions.csv", encoding="utf-8-sig")
    assert len(table) == 60 and "decision" in table.columns
    meta = yaml.safe_load((out / "result.yaml").read_text(encoding="utf-8"))
    assert meta["summary"]["rows"] == 60
    assert meta["sentences"] == result["sentences"]

    # 既定の決め方は None で消せる
    result = infer(path, table_csv, rules={"min_confidence": None})
    assert result["rules"] == {}
