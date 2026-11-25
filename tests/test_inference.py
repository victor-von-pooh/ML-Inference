import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from conftest import linear_spec, save_weights
from ml_eda import export_dataset, load_data, load_dataset
from ml_modelcontrol import build_model
from torch import nn

from ml_inference import (
    check_rules, conclude, describe_bundle, infer, load_bundle, make_bundle,
    predict, write_result
)


def table_bundle(tmp_path: Path, dataset: Path, **options: Any) -> Path:
    """
    表の分類の一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    dataset: Path
        データセット
    **options: Any
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
    state = {"format_version": 1, "status": "completed",
             "monitor": "valid/acc", "best_value": 0.9, "best_epoch": 3}
    (run / "run.json").write_text(json.dumps(state), encoding="utf-8")
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


def test_relative_external_weights(
    tmp_path: Path, table_dataset: Path, table_csv: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    外部の重みを相対パスで持つ定義から, 定義と別の今のディレクトリで一式を
    作り, 一式を別の場所へ移しても推論できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_dataset: Path
        データセット
    table_csv: Path
        元の CSV
    monkeypatch: pytest.MonkeyPatch
        今のディレクトリの差し替え

    Returns
    ----------
    None
    """
    # 定義の隣の外部の重みを相対パスで読む定義
    width = len(load_dataset(table_dataset).features)
    spec = linear_spec(width, 3)
    spec["format_version"] = 2
    spec["nodes"]["head"]["weights"] = {"path": "head.pt"}
    defs = tmp_path / "defs"
    defs.mkdir()
    head = nn.Linear(width, 3)
    torch.save(head.state_dict(), defs / "head.pt")
    spec_file = defs / "model.yaml"
    spec_file.write_text(yaml.safe_dump(spec), encoding="utf-8")

    # 外部の重みを読んで学習した後の重み
    model = build_model(spec_file, seed=0)
    state = model.state_dict()
    state["nodes.head.bias"] = state["nodes.head.bias"] + 0.5
    weights = tmp_path / "trained.pt"
    torch.save(state, weights)
    model.load_state_dict(state)
    model.eval()

    # 定義と別の今のディレクトリで, ファイルと辞書の定義から一式づくり
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    make_bundle(tmp_path / "from_file", spec_file, table_dataset, weights)
    make_bundle(tmp_path / "from_dict", yaml.safe_load(
        spec_file.read_text(encoding="utf-8")
    ), table_dataset, weights)
    copied = yaml.safe_load(
        (tmp_path / "from_file" / "model_spec.yaml").read_text("utf-8")
    )
    assert copied["nodes"]["head"]["weights"] == {"path": "head.pt"}

    # 一式を移し, 外部の重みのファイルを消しても同じ予測
    moved = tmp_path / "moved"
    moved.mkdir()
    for name in ("from_file", "from_dict"):
        (tmp_path / name).rename(moved / name)
    (defs / "head.pt").unlink()
    x = load_bundle(moved / "from_file").dataset.transform(
        pd.read_csv(table_csv)
    ).to_numpy(dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).softmax(dim=1).numpy()
    for name in ("from_file", "from_dict"):
        result = predict(moved / name, table_csv)
        bundle = load_bundle(moved / name)
        got = result[[f"p:{c}" for c in bundle.classes]].to_numpy()
        assert np.allclose(got, expected, atol=1e-6)


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
        """
        正しい一式の inference.yaml の一部を変えて書く関数

        Parameters
        ----------
        **changes: object
            変える項目と値

        Returns
        ----------
        None
        """
        # 元の中身に変えた項目を重ねる
        file.write_text(
            yaml.safe_dump({**original, **changes}, allow_unicode=True),
            encoding="utf-8"
        )

    write(dataset="../dataset")
    with pytest.raises(ValueError, match="一式の外"):
        load_bundle(path)
    write(format_version=12)
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
    assert meta["format_version"] == 1

    # 既定の決め方は None で消せる
    result = infer(path, table_csv, rules={"min_confidence": None})
    assert result["rules"] == {}


def test_rows_that_fail_preprocessing_are_skipped(
    tmp_path: Path, regression_dataset: Path
) -> None:
    """
    一部の行の前処理の失敗で全体を止めずに, その行を飛ばすことを確かめる
    関数

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
    # 数にできない値の行
    spec = linear_spec(1, 1)
    weights = save_weights(spec, tmp_path / "w.pt")
    path = tmp_path / "bundle"
    make_bundle(path, spec, regression_dataset, weights)
    frame = pd.DataFrame({"shop": ["東", "西", "東"], "x": ["0.3", "abc", 1]})
    result = predict(path, frame)
    assert list(result["status"]) == ["ok", "skipped", "ok"]
    assert "前処理できません" in result.loc[1, "reason"]


def test_text_edge_cases(tmp_path: Path) -> None:
    """
    空の文を飛ばし, 予測の列と同じ名前の元の列を残すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 文字の語彙で長さ 4 の並び. 出力が 1 つの値の 2 クラス
    source = tmp_path / "t.csv"
    pd.DataFrame({
        "text": ["良い", "悪い", "とても良い", "とても悪い"] * 3,
        "label": ["良", "悪", "良", "悪"] * 3,
    }).to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [],
        "item_steps": [{"op": "vocabulary", "unit": "char", "length": 4}],
        "split": {"method": "random", "valid": 0.25, "seed": 0},
    }
    dataset = tmp_path / "dataset"
    export_dataset(load_data(source, text_column="text"), recipe, dataset)
    vocabulary = next(
        s["state"]["vocabulary"]
        for s in load_dataset(dataset).recipe["item_steps"]
    )
    spec = {
        "format_version": 1,
        "inputs": {"ids": {"shape": [2, 4], "dtype": "int64"}},
        "nodes": {
            "bag": {
                "unit": "nn.EmbeddingBag",
                "params": {"num_embeddings": len(vocabulary),
                           "embedding_dim": 3},
                "inputs": {"input": "ids"},
            },
            "head": {
                "unit": "nn.Linear",
                "params": {"in_features": 3, "out_features": 1},
                "inputs": "bag",
            },
            "flat": {"unit": "nn.Flatten", "params": {"start_dim": 0},
                     "inputs": "head"},
        },
        "outputs": {"logits": "flat"},
    }
    weights = save_weights(spec, tmp_path / "w.pt")
    path = tmp_path / "bundle"
    make_bundle(path, spec, dataset, weights)

    # 空の文と欠損の文は飛ばし, 元の status の列は名前を変えて残す
    frame = pd.DataFrame({
        "text": ["良い", "", None], "status": ["a", "b", "c"]
    })
    result = predict(path, frame)
    assert list(result["status"]) == ["ok", "skipped", "skipped"]
    assert list(result["status (入力)"]) == ["a", "b", "c"]
    assert "文が空" in result.loc[1, "reason"]
    p = result.loc[0, ["p:悪", "p:良"]].to_numpy(dtype=float)
    assert np.isclose(p.sum(), 1.0)


def test_rule_combinations(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    一緒に使えない決め方と, 陽性のクラスを消したときのしきい値を確かめる
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
    # 陽性のしきい値と確信の下限は一緒に使えない
    errors = check_rules(
        {"positive": "a", "threshold": 0.3, "min_confidence": 0.6},
        "classification", ["a", "b"]
    )
    assert any("一緒に使えません" in e for e in errors)

    # 一式の既定の positive を消すと threshold も消える
    path = table_bundle(tmp_path, table_dataset)
    file = path / "inference.yaml"
    info = yaml.safe_load(file.read_text(encoding="utf-8"))
    info["conclusion"] = {"positive": "犬", "threshold": 0.5}
    file.write_text(yaml.safe_dump(info, allow_unicode=True), "utf-8")
    result = infer(path, table_csv, rules={"positive": None})
    assert result["rules"] == {}

    # 分類に value は使えない
    info["conclusion"] = {}
    info["model"]["kind"] = "value"
    file.write_text(yaml.safe_dump(info, allow_unicode=True), "utf-8")
    with pytest.raises(ValueError, match="使えません"):
        load_bundle(path)


def test_rows_dropped_by_recipe(tmp_path: Path) -> None:
    """
    レシピの行を落とす手順で落ちる行を, 理由を付けて飛ばすことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 負の a の行を落とすレシピ
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"a": rng.normal(size=30), "y": rng.normal(size=30)})
    source = tmp_path / "d.csv"
    frame.to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "y", "task": "regression",
        "steps": [{"op": "filter_rows", "column": "a", "compare": ">=",
                   "value": 0}],
        "split": {"method": "random", "valid": 0.2, "seed": 0},
    }
    dataset = tmp_path / "dataset"
    export_dataset(load_data(source), recipe, dataset)
    spec = linear_spec(1, 1)
    weights = save_weights(spec, tmp_path / "w.pt")
    path = tmp_path / "bundle"
    make_bundle(path, spec, dataset, weights)

    # 負の行は理由付きで飛ばし, 他は予測
    result = predict(path, pd.DataFrame({"a": [-1.0, 0.5]}))
    assert list(result["status"]) == ["skipped", "ok"]
    assert "filter_rows" in result.loc[0, "reason"]


def test_check_against_run(tmp_path: Path, table_dataset: Path) -> None:
    """
    検証用の分割の評価を実験の記録と比べ, 合わなければ注意を出すことを
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
    # 検証用の分割の正解率を手で求める
    dataset = load_dataset(table_dataset)
    spec = linear_spec(len(dataset.features), 3)
    run = tmp_path / "run"
    run.mkdir()
    save_weights(spec, run / "final_model.pt")
    x, y = dataset.tensors("valid")
    model = build_model(spec)
    model.load_state_dict(torch.load(run / "final_model.pt"))
    with torch.no_grad():
        accuracy = float((model(x).argmax(dim=1) == y).float().mean())

    # 記録と同じなら合う, 違えば注意
    for recorded, expected in ((accuracy, True), (accuracy + 0.3, False)):
        state = {"format_version": 1, "status": "completed",
                 "monitor": "valid/acc", "best_value": recorded,
                 "final_weights": "best"}
        (run / "run.json").write_text(json.dumps(state), encoding="utf-8")
        info = make_bundle(tmp_path / "b", spec, table_dataset, run,
                           overwrite=True)
        assert info["check"]["metric"] == "acc"
        assert np.isclose(info["check"]["value"], accuracy)
        assert info["check"]["matches"] is expected
        assert any("合いません" in n for n in info["notes"]) is not expected
        assert describe_bundle(tmp_path / "b")["check"] == info["check"]

    # 監視した指標が違えば比べない
    state["monitor"] = "valid/loss"
    (run / "run.json").write_text(json.dumps(state), encoding="utf-8")
    info = make_bundle(tmp_path / "b", spec, table_dataset, run,
                       overwrite=True)
    assert info["check"]["matches"] is None


def test_holistic_fixes(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    既定値と渡した決め方の重ね方, numpy の数, 重みの無い一式を確かめる
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
    # 既定の確信の下限は, 渡した numpy の数の下限で置き換わり書き出せる
    path = table_bundle(
        tmp_path, table_dataset, conclusion={"min_confidence": 0.7}
    )
    result = infer(path, table_csv,
                   rules={"min_confidence": np.float64(0.5)})
    assert result["rules"] == {"min_confidence": 0.5}
    assert type(result["rules"]["min_confidence"]) is float
    write_result(result, tmp_path / "out")
    with pytest.raises(ValueError, match="batch_size"):
        predict(path, table_csv, batch_size=0)

    # 重みが無い一式は, 読むときに分かる
    (path / "final_model.pt").unlink()
    with pytest.raises(FileNotFoundError, match="重み"):
        load_bundle(path)


def test_train_scale_and_reading(
    tmp_path: Path, regression_dataset: Path
) -> None:
    """
    学習用の分割から目的の値の標準化を求めることと, 学習と違う形式の
    ファイルを読めることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    regression_dataset: Path
        回帰のデータセット

    Returns
    ----------
    None
    """
    # 学習用の目的の値の平均と不偏の標準偏差
    spec = linear_spec(1, 1)
    weights = save_weights(spec, tmp_path / "w.pt")
    info = make_bundle(tmp_path / "bundle", spec, regression_dataset,
                       weights, target_scale="train")
    _, y = load_dataset(regression_dataset).tensors("train")
    assert np.isclose(info["model"]["target_scale"]["center"],
                      y.float().mean().item())
    assert np.isclose(info["model"]["target_scale"]["scale"],
                      y.float().std().item())

    # csv で学習した一式に parquet, ディレクトリは分かる誤り
    frame = pd.DataFrame({"shop": ["東"], "x": [0.2]})
    frame.to_parquet(tmp_path / "new.parquet")
    assert predict(tmp_path / "bundle", tmp_path / "new.parquet")[
        "status"
    ].tolist() == ["ok"]
    with pytest.raises(ValueError, match="ディレクトリ"):
        predict(tmp_path / "bundle", tmp_path)
