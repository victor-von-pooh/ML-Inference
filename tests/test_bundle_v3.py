from pathlib import Path
import math
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from conftest import linear_spec, save_weights
import ml_modelcontrol
from ml_eda import export_dataset, load_data, load_dataset
from ml_modelcontrol import build_model, load_example, register_unit
from torch import nn

import ml_inference.bundle as bundle_module
from ml_inference import (
    artifact_path, decide, describe_bundle, infer, load_bundle, make_bundle,
    predict, read_result, write_result
)
from ml_inference.conclude import conclude
from ml_inference.predict import (
    DECODERS, finite_raw, join_parts, median_of
)
from ml_inference import procedure_options
from ml_inference.procedures import (
    get_procedure, make_generator, option_names, register_procedure,
    run_procedure, start_noise
)
from ml_inference.weights import node_items, state_hash

# 出どころの重みを作る乱数の種 (入れ替わった出どころを試すため変える)
SOURCE = {"seed": 1}

# 版 3 だけが持つ項目
ADDED = ("role", "output", "target", "procedure")


class SourceLinear(nn.Module):
    def __init__(
        self, in_features: int = 2, out_features: int = 3,
        revision: str = "main"
    ):
        # 親クラスのコンストラクタを呼び出す
        super(SourceLinear, self).__init__()

        # 出どころの重み (組み立てのたびに同じ値)
        self.fc = nn.Linear(in_features, out_features)
        generator = torch.Generator().manual_seed(SOURCE["seed"])
        with torch.no_grad():
            self.fc.weight.copy_(
                torch.randn(self.fc.weight.shape, generator=generator)
            )
            self.fc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        全結合の 1 層を通す関数

        Parameters
        ----------
        x: torch.Tensor
            入力

        Returns
        ----------
        out: torch.Tensor
            出力
        """
        # 全結合
        out = self.fc(x)

        return out


register_unit(SourceLinear, name="test.SourceLinear", overwrite=True)


def table_bundle(
    tmp_path: Path, dataset: Path, outputs: int = 3, name: str = "bundle",
    **options: Any
) -> Path:
    """
    表の一式を全結合の 1 層で作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    dataset: Path
        データセット
    outputs: int = 3
        出力の幅
    name: str = "bundle"
        一式のディレクトリの名前
    **options: Any
        make_bundle に渡す引数

    Returns
    ----------
    path: Path
        一式のディレクトリ
    """
    # 学習に使う列の数に合わせる
    width = len(load_dataset(dataset).features)
    spec = linear_spec(width, outputs)
    weights = save_weights(spec, tmp_path / f"{name}.pt")
    path = tmp_path / name
    make_bundle(path, spec, dataset, weights, **options)

    return path


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


def old_copy(path: Path, target: Path, version: int) -> Path:
    """
    版 3 の列の予測の一式から, 足した項目を除いた古い版の一式を作る関数

    Parameters
    ----------
    path: Path
        版 3 の一式
    target: Path
        作る場所
    version: int
        版 (1 か 2. 1 は uncertainty と decision も除く)

    Returns
    ----------
    target: Path
        古い版の一式
    """
    # 足した項目を除いて版を戻す
    shutil.copytree(path, target)
    info = read_info(path)
    old = {k: v for k, v in info.items() if k not in ADDED}
    old["model"] = {
        k: v for k, v in info["model"].items() if k != "pretrained"
    }
    old["format_version"] = version
    if version == 1:
        old.pop("uncertainty", None)
        old.pop("decision", None)
    rewrite(target, old)

    return target


def test_column_bundle_differs_only_by_added_fields(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    列の予測の版 3 の一式が版 2 と足した項目だけ違い, 版 1 と 2 の一式も
    同じに動くことを確かめる関数

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
    # 版 3 の足した項目 (受け持ち, 出力の形, 正解, 手順, 重みの置き方)
    path = table_bundle(
        tmp_path, table_dataset, conclusion={"min_confidence": 0.4}
    )
    info = read_info(path)
    assert info["format_version"] == 3
    assert info["role"] == "predict"
    assert info["output"] == {"form": "class"}
    assert info["target"] == {"kind": "column"}
    assert info["procedure"] is None
    assert info["model"]["pretrained"] == []
    expected = (
        "format_version", "name", "created_at", "role", "output", "target",
        "procedure", "model", "inputs", "dataset", "run", "conclusion",
        "decision", "notes", "check", "uncertainty"
    )
    assert tuple(info) == expected

    # 版 2 と版 1 の一式も読め, 予測と結論が同じ
    new = infer(path, table_csv)
    for version in (2, 1):
        old = old_copy(path, tmp_path / f"v{version}", version)
        bundle = load_bundle(old)
        assert bundle.role == "predict" and bundle.form == "class"
        result = infer(old, table_csv)
        pd.testing.assert_frame_equal(result["frame"], new["frame"])
        assert result["sentences"] == new["sentences"]
        description = describe_bundle(old)
        assert description["format_version"] == version
        assert description["target_info"] == {"kind": "column"}
        assert description["target"] == "label"
        assert description["column_output"] is True


def test_newer_bundle_stops(tmp_path: Path, table_dataset: Path) -> None:
    """
    知らない新しい版と, 版 3 の項目の誤りで止まることを確かめる関数

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
    # 版 16 は読めない. 版 15 は search が要る. 版 14 は demand が要る. 版 13 と 12 は効果の出力の
    # 形だけ, 版 11 は anomaly, 版 10 と 9 は group が要る. 版 8 は rank が
    # 要る. 版 7 は表の出力だけ. 版 6 は ready が要る. 版 5 は dataset を
    # 持てない. 版 4 は items が要る
    path = table_bundle(tmp_path, table_dataset)
    original = read_info(path)
    rewrite(path, {**original, "format_version": 16})
    with pytest.raises(ValueError, match="format_version 16 .*読めません"):
        load_bundle(path)
    rewrite(path, {**original, "format_version": 15})
    with pytest.raises(ValueError, match="版 15 の一式には search"):
        load_bundle(path)
    rewrite(path, {**original, "format_version": 14})
    with pytest.raises(ValueError, match="版 14 の一式には demand"):
        load_bundle(path)
    rewrite(path, {**original, "format_version": 13})
    with pytest.raises(ValueError, match="effect だけ"):
        load_bundle(path)
    rewrite(path, {**original, "format_version": 12})
    with pytest.raises(ValueError, match="effect だけ"):
        load_bundle(path)
    rewrite(path, {**original, "format_version": 11})
    with pytest.raises(ValueError, match="版 11 の一式には anomaly"):
        load_bundle(path)
    rewrite(path, {**original, "format_version": 10})
    with pytest.raises(ValueError, match="版 10 の一式には group"):
        load_bundle(path)
    rewrite(path, {**original, "format_version": 9})
    with pytest.raises(ValueError, match="group"):
        load_bundle(path)
    rewrite(path, {**original, "format_version": 8})
    with pytest.raises(ValueError, match="rank"):
        load_bundle(path)
    rewrite(path, {**original, "format_version": 7})
    with pytest.raises(ValueError, match="table だけ"):
        load_bundle(path)
    rewrite(path, {**original, "format_version": 6})
    with pytest.raises(ValueError, match="ready"):
        load_bundle(path)
    rewrite(path, {**original, "format_version": 5})
    with pytest.raises(ValueError, match="dataset を持てません"):
        load_bundle(path)
    rewrite(path, {**original, "format_version": 4})
    with pytest.raises(ValueError, match="items"):
        load_bundle(path)

    # 知らない形, 形に合わない出力の扱い方, タスクに合わない形
    cases = [
        ({"output": {"form": "hologram"}}, "output.form"),
        ({"role": "guess"}, "role は"),
        ({"output": {"form": "class"},
          "model": {**original["model"], "kind": "value"}}, "model.kind"),
        ({"output": {"form": "value"}}, "出力の形 value"),
        ({"output": {"form": "distribution", "quantiles": [0.9, 0.1]}},
         "quantiles"),
        ({"target": {"kind": "column", "origin": {"type": "guess"}}},
         "origin.type"),
        ({"procedure": {"name": "nothing", "steps": 1, "seed": 0}},
         "登録簿にありません"),
    ]
    for change, message in cases:
        rewrite(path, {**original, **change})
        with pytest.raises(ValueError, match=message):
            load_bundle(path)


def test_vector_bundle_and_artifacts(
    tmp_path: Path, table_csv: Path
) -> None:
    """
    正解の無いデータセットでベクトルの出力の一式を作り, 結果の配列を
    artifacts/ に書けることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_csv: Path
        元の CSV

    Returns
    ----------
    None
    """
    # 正解の無いデータセット (埋め込みのモデルの学習のような)
    recipe = {
        "format_version": 1, "target": None, "task": "none",
        "steps": [
            {"op": "drop_columns", "columns": ["id", "label"]},
            {"op": "one_hot", "columns": ["color"]},
            {"op": "standardize", "columns": "auto"},
        ],
        "split": {"method": "random", "valid": 0.2, "test": 0.0, "seed": 0},
    }
    dataset = tmp_path / "unlabeled"
    export_dataset(load_data(table_csv), recipe, dataset)
    with pytest.raises(ValueError, match="出力の形"):
        table_bundle(tmp_path, dataset, 4, name="no_form")
    path = table_bundle(tmp_path, dataset, 4, form="vector", role="group")
    info = read_info(path)
    assert info["output"] == {"form": "vector"}
    assert info["target"] is None
    assert info["check"] is None and info["uncertainty"] is None
    description = describe_bundle(path)
    assert description["column_output"] is False
    assert description["role"] == "group"

    # 予測は 1 行ごとの (次元,) の配列で, 手で通したものと同じ
    bundle = load_bundle(path)
    frame = pd.read_csv(table_csv)
    frame.loc[3, "a"] = np.nan
    result = predict(bundle, frame)
    assert result.loc[3, "status"] == "skipped"
    assert result.loc[3, "output"] is None
    x = bundle.dataset.transform(frame.iloc[[0]]).to_numpy(np.float32)
    with torch.no_grad():
        expected = bundle.model()(torch.from_numpy(x))[0].numpy()
    assert np.allclose(result.loc[0, "output"], expected, atol=1e-6)

    # 結論は件数だけ. 決め方と意思決定は使えない
    out = infer(bundle, frame)
    assert out["summary"] == {"rows": 60, "predicted": 59, "skipped": 1}
    assert out["sentences"] == ["60 件のうち 59 件を推論 (1 件は推論できず)"]
    with pytest.raises(ValueError, match="vector"):
        infer(bundle, frame, rules={"min_confidence": 0.5})
    with pytest.raises(ValueError, match="vector"):
        decide(bundle, result, {"type": "triage"})

    # 書き出しは artifacts/ に 1 行 1 つ, 表は相対パス, 結果の版 2
    root = write_result(out, tmp_path / "result")
    meta = yaml.safe_load((root / "result.yaml").read_text(encoding="utf-8"))
    assert meta["format_version"] == 2
    assert meta["artifacts"] == {
        "directory": "artifacts", "column": "artifact", "format": "npy",
        "count": 59, "form": "vector",
    }
    back = read_result(root)
    table = back["frame"]
    assert "output" not in table.columns
    assert table.loc[0, "artifact"] == "artifacts/000000.npy"
    assert table.loc[3, "artifact"] == ""
    loaded = np.load(artifact_path(root, table.loc[0, "artifact"]))
    assert np.allclose(loaded, expected, atol=1e-6)
    with pytest.raises(ValueError, match="外"):
        artifact_path(root, "../result.yaml")

    # 書き直すと前の配列は残らない. 列の予測の結果は版 1 のまま
    (root / "artifacts" / "999999.npy").write_bytes(b"")
    write_result(out, root)
    assert not (root / "artifacts" / "999999.npy").exists()
    meta["format_version"] = 5
    (root / "result.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="読めません"):
        read_result(root)


def test_column_result_stays_version_one(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    表に収まる結果は版 1 で書き, 正解の出どころが今の処理なら文を添える
    ことを確かめる関数

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
    # 今の処理の出力を正解にした一式
    origin = {"type": "process", "name": "手作業の判定", "settings": "v1"}
    path = table_bundle(
        tmp_path, table_dataset,
        target={"kind": "column", "origin": origin}
    )
    assert read_info(path)["target"]["origin"] == origin
    result = infer(path, table_csv)
    assert any("手作業の判定 の出力" in s for s in result["sentences"])
    root = write_result(result, tmp_path / "result")
    assert read_result(root)["format_version"] == 1
    assert not (root / "artifacts").exists()


def test_distribution_bundle(
    tmp_path: Path, regression_dataset: Path
) -> None:
    """
    分位点の出力を元の単位の列にし, 交差を直すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    regression_dataset: Path
        目的の列を変換した回帰のデータセット

    Returns
    ----------
    None
    """
    # 3 つの分位点を出す一式
    levels = [0.1, 0.5, 0.9]
    path = table_bundle(
        tmp_path, regression_dataset, 3, form="distribution",
        output_options={"quantiles": levels}
    )
    bundle = load_bundle(path)
    assert bundle.column_output
    frame = pd.DataFrame({"shop": ["東", "西"], "x": [0.1, -1.0],
                          "y": [100.0, 200.0]})
    result = predict(bundle, frame)
    values = result[["q:0.1", "q:0.5", "q:0.9"]].to_numpy()
    assert (np.diff(values, axis=1) >= 0).all()
    assert np.allclose(result["prediction"], values[:, 1])
    assert (values > 0).all()
    assert read_info(path)["check"]["metric"] == "mae"

    # 回帰のデータセットにクラスの形は使えない
    with pytest.raises(ValueError, match="出力の形 class"):
        table_bundle(
            tmp_path, regression_dataset, 3, name="bad", form="class"
        )


def test_array_decoders() -> None:
    """
    領域, 点群, 画像, 文の出力の読み方を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 領域: クラスの次元は最も大きいクラス, 1 チャンネルは 2 値
    logits = np.zeros((2, 3, 2, 2), dtype=np.float32)
    logits[0, 2, 0, 1] = 5.0
    stub = SimpleNamespace(form="mask", output_kind="logits")
    masks = DECODERS["mask"](stub, logits)["output"]
    assert masks[0][0, 1] == 2 and masks[0][0, 0] == 0
    single = np.array([[[0.2, -0.1]]], dtype=np.float32)
    assert DECODERS["mask"](stub, single)["output"][0].tolist() == [[1, 0]]
    stub = SimpleNamespace(form="mask", output_kind="probabilities")
    single = np.array([[[[0.5, 0.4]]]], dtype=np.float32)
    assert DECODERS["mask"](stub, single)["output"][0].tolist() == [[1, 0]]

    # 点群は (件数, 点, 次元) だけ, ベクトルは (件数, 次元) だけ
    stub = SimpleNamespace(form="point_cloud", output_kind="value")
    clouds = DECODERS["point_cloud"](stub, np.ones((2, 5, 3)))["output"]
    assert clouds[1].shape == (5, 3)
    with pytest.raises(ValueError, match="点の数"):
        DECODERS["point_cloud"](stub, np.ones((2, 3)))
    stub = SimpleNamespace(form="vector", output_kind="value")
    with pytest.raises(ValueError, match="次元"):
        DECODERS["vector"](stub, np.ones((2, 3, 4)))
    stub = SimpleNamespace(form="image", output_kind="value")
    images = DECODERS["image"](stub, np.ones((2, 3, 4, 4)))["output"]
    assert images[0].shape == (3, 4, 4)

    # 文は位置ごとに最も大きい番号を空白でつなぐ (語彙が無ければ文も番号)
    stub = SimpleNamespace(
        form="text", output_kind="logits", procedure=None,
        dataset=SimpleNamespace(recipe={})
    )
    tokens = np.zeros((1, 2, 4))
    tokens[0, 0, 3] = 1.0
    tokens[0, 1, 1] = 1.0
    decoded = DECODERS["text"](stub, tokens)
    assert decoded["tokens"] == ["3 1"]
    assert decoded["prediction"] == ["3 1"]


def test_generation_bundle_with_procedure(
    tmp_path: Path, table_dataset: Path
) -> None:
    """
    拡散のモデルの一式を繰り返しの手順で推論し, 同じ種なら同じ出力に
    なることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_dataset: Path
        データセット (行の数を決めるだけで, 列は使わない)

    Returns
    ----------
    None
    """
    # 雑音の予測器 (x と時刻 t から 16 の値)
    spec = load_example("diffusion_denoiser")
    weights = save_weights(spec, tmp_path / "denoiser.pt")
    options = {
        "inputs": {"x": {"source": "state"}, "t": {"source": "time"}},
        "form": "vector", "role": "generate",
        "output_options": {"shape": [16]},
    }
    with pytest.raises(ValueError, match="procedure"):
        make_bundle(tmp_path / "none", spec, table_dataset, weights,
                    **options)
    with pytest.raises(ValueError, match="output.shape"):
        make_bundle(
            tmp_path / "noshape", spec, table_dataset, weights,
            **{**options, "output_options": {}},
            procedure={"name": "ddim", "steps": 5, "seed": 0}
        )
    outputs = {}
    for name, seed in (("ddim", 0), ("euler", 0), ("ddim", 1)):
        path = tmp_path / f"{name}{seed}"
        make_bundle(
            path, spec, table_dataset, weights, **options,
            procedure={"name": name, "steps": 5, "seed": seed}
        )
        assert read_info(path)["procedure"]["name"] == name
        frame = pd.DataFrame({"a": range(7)})
        first = predict(path, frame, batch_size=3)
        again = predict(path, frame, batch_size=3)
        stacked = np.stack(first["output"].to_list())
        assert stacked.shape == (7, 16)
        assert np.array_equal(stacked, np.stack(again["output"].to_list()))
        outputs[(name, seed)] = stacked
    assert not np.allclose(outputs[("ddim", 0)], outputs[("ddim", 1)])
    assert not np.allclose(outputs[("ddim", 0)], outputs[("euler", 0)])

    # 時刻を渡さない手順に時刻の入力, 知らない設定は誤り
    path = tmp_path / "ddim0"
    original = read_info(path)
    rewrite(path, {**original, "procedure": {
        "name": "greedy", "steps": 2, "seed": 0}})
    with pytest.raises(ValueError, match="time"):
        load_bundle(path)
    rewrite(path, {**original, "procedure": {
        "name": "ddim", "steps": 2, "seed": 0, "options": {"speed": 1}}})
    with pytest.raises(ValueError, match="speed"):
        load_bundle(path)


def test_text_generation_bundle(tmp_path: Path) -> None:
    """
    文の番号を 1 つずつ足す一式を, 渡した文から始めて推論できることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 文字の語彙で長さ 4 の並びにするデータセット
    texts = ["良い品", "悪い品", "とても良い", "とても悪い"] * 5
    labels = ["良", "悪", "良", "悪"] * 5
    source = tmp_path / "reviews.csv"
    pd.DataFrame({"text": texts, "label": labels}).to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [],
        "item_steps": [{"op": "vocabulary", "unit": "char", "length": 4}],
        "split": {"method": "random", "valid": 0.2, "test": 0.0, "seed": 0},
    }
    dataset = tmp_path / "dataset"
    export_dataset(load_data(source, text_column="text"), recipe, dataset)
    size = 12

    # 番号ごとの埋め込みから次の番号の logits ((件数, 長さ, 語彙))
    spec = {
        "format_version": 1,
        "inputs": {"ids": {"shape": [2, 4], "dtype": "int64"}},
        "nodes": {
            "embed": {
                "unit": "nn.Embedding",
                "params": {"num_embeddings": size, "embedding_dim": 4},
                "inputs": "ids",
            },
            "head": {
                "unit": "nn.Linear",
                "params": {"in_features": 4, "out_features": size},
                "inputs": "embed",
            },
        },
        "outputs": {"logits": "head"},
    }
    weights = save_weights(spec, tmp_path / "lm.pt")
    results = {}
    for name in ("greedy", "sample"):
        path = tmp_path / name
        make_bundle(
            path, spec, dataset, weights, form="text", role="generate",
            inputs={"ids": {"source": "state"}},
            procedure={"name": name, "steps": 3, "seed": 0,
                       "options": {"temperature": 0.7} if name == "sample"
                       else {}}
        )
        frame = pd.DataFrame({"text": ["良い品", "悪い"]})
        first = predict(path, frame)
        lengths = [len(text.split()) for text in first["tokens"]]
        assert lengths == [7, 7]
        again = predict(path, frame)["tokens"].tolist()
        assert again == first["tokens"].tolist()
        results[name] = first
    # 渡した文の番号が先頭に残る
    bundle = load_bundle(tmp_path / "greedy")
    head = " ".join(
        str(int(v)) for v in bundle.dataset.transform_item("良い品")
    )
    assert results["greedy"].loc[0, "tokens"].startswith(head)

    # 文はレシピの文字の語彙で戻し, 渡した文で始まる
    assert results["greedy"].loc[0, "prediction"].startswith("良い品")
    assert results["greedy"].loc[1, "prediction"].startswith("悪い")


def pretrained_spec(width: int) -> dict:
    """
    出どころの重みを読む部品を持つ定義を返す関数

    Parameters
    ----------
    width: int
        入力の列の数

    Returns
    ----------
    spec: dict
        定義
    """
    # 出どころの全結合だけで 3 クラス
    spec = {
        "format_version": 1,
        "name": "sourced",
        "inputs": {"x": {"shape": [4, width]}},
        "nodes": {
            "base": {
                "unit": "test.SourceLinear",
                "params": {"in_features": width, "out_features": 3},
                "inputs": "x",
            }
        },
        "outputs": {"logits": "base"},
    }

    return spec


def license_info(redistributable: bool) -> dict:
    """
    pretrained_info の形の情報を返す関数

    Parameters
    ----------
    redistributable: bool
        再配布できるか

    Returns
    ----------
    info: dict
        情報
    """
    # 版の識別子とライセンス
    info = {
        "unit": "test.SourceLinear", "model": "demo", "hub_id": "org/demo",
        "revision": "main", "pinned_revision": "abc123",
        "license": "mit" if redistributable else "cc-by-nc-4.0",
        "status": "commercial" if redistributable else "noncommercial",
        "redistributable": redistributable, "source": "テストの情報",
    }

    return info


def test_pretrained_weights_placement(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    再配布できない学習済みの重みを一式に写さずに出どころから読み,
    hash を確かめることを確かめる関数

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
    # 学習で変えていない重み (出どころのまま)
    SOURCE["seed"] = 1
    width = len(load_dataset(table_dataset).features)
    spec = pretrained_spec(width)
    weights = tmp_path / "w.pt"
    torch.save(build_model(spec, seed=0).state_dict(), weights)

    # 写せない重みは除き, 版を固定し, hash を残す
    path = tmp_path / "sourced"
    make_bundle(path, spec, table_dataset, weights,
                pretrained={"base": license_info(False)})
    record = read_info(path)["model"]["pretrained"][0]
    assert record["place"] == "source" and record["keys"] == 2
    assert any("一式に写さず" in n and "abc123" in n
               for n in read_info(path)["notes"])
    assert record["pinned_revision"] == "abc123"
    assert torch.load(path / "final_model.pt", weights_only=True) == {}
    pinned = yaml.safe_load((path / "model_spec.yaml").read_text("utf-8"))
    assert pinned["nodes"]["base"]["params"]["revision"] == "abc123"
    full = tmp_path / "full"
    make_bundle(full, spec, table_dataset, weights,
                pretrained={"base": license_info(True)})
    assert read_info(full)["model"]["pretrained"][0]["place"] == "bundle"
    assert set(torch.load(full / "final_model.pt", weights_only=True)) == {
        "nodes.base.fc.weight", "nodes.base.fc.bias"
    }
    pd.testing.assert_frame_equal(
        predict(path, table_csv), predict(full, table_csv)
    )
    assert describe_bundle(path)["pretrained_warnings"] == []

    # 出どころの重みが入れ替わると警告 (推論は止めない)
    SOURCE["seed"] = 2
    try:
        bundle = load_bundle(path)
        with pytest.warns(UserWarning, match="一式を作ったときと違います"):
            predict(bundle, table_csv)
        assert len(describe_bundle(bundle)["pretrained_warnings"]) == 1
    finally:
        SOURCE["seed"] = 1

    # 学習で変えた写せない重みは一式に残し, 注意を書く
    state = torch.load(weights, weights_only=True)
    state["nodes.base.fc.bias"] = state["nodes.base.fc.bias"] + 1.0
    trained = tmp_path / "trained.pt"
    torch.save(state, trained)
    path = tmp_path / "trained"
    info = make_bundle(path, spec, table_dataset, trained,
                       pretrained={"base": license_info(False)})
    assert info["model"]["pretrained"][0]["place"] == "bundle"
    assert any("ライセンスを確かめて" in n for n in info["notes"])
    assert len(torch.load(path / "final_model.pt", weights_only=True)) == 2

    # 出どころを読めないとノードと版を添えた誤り
    broken = load_bundle(tmp_path / "sourced")
    original = bundle_module.build_model
    bundle_module.build_model = FailingBuild()
    try:
        with pytest.raises(ValueError, match="base .*abc123"):
            broken.model()
    finally:
        bundle_module.build_model = original

    # ライセンスか版を読めなかった重みは除かずに残し, 理由を書く
    unread = {**license_info(False), "license": None,
              "pinned_revision": None, "status": "unknown"}
    info = make_bundle(tmp_path / "unread", spec, table_dataset, weights,
                       pretrained={"base": unread})
    assert info["model"]["pretrained"][0]["place"] == "bundle"
    assert any("ライセンスか版を読めなかった" in n for n in info["notes"])

    # 定義に無いノードの情報は誤り
    with pytest.raises(ValueError, match="ノードにありません"):
        make_bundle(tmp_path / "bad", spec, table_dataset, weights,
                    pretrained={"other": license_info(False)})


def test_layered_pretrained_node(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    外部の重みを重ねた学習済みのノードを一式の重みに含め, そのノードを
    出どころから読む以前の一式を誤りにすることを確かめる関数

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
    # 出どころの部品の上に外部の重みを重ねる定義
    SOURCE["seed"] = 1
    width = len(load_dataset(table_dataset).features)
    spec = pretrained_spec(width)
    spec["format_version"] = 2
    spec["nodes"]["base"]["weights"] = {"path": "over.pt"}
    defs = tmp_path / "defs"
    defs.mkdir()
    over = nn.Linear(width, 3)
    torch.save({f"fc.{k}": v for k, v in over.state_dict().items()},
               defs / "over.pt")
    spec_file = defs / "model.yaml"
    spec_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    weights = tmp_path / "w.pt"
    torch.save(build_model(spec_file, seed=0).state_dict(), weights)

    # 写せない重みでも, 重ねたので出どころと違い一式の重みに含める
    path = tmp_path / "layered"
    info = make_bundle(path, spec_file, table_dataset, weights,
                       pretrained={"base": license_info(False)})
    assert info["model"]["pretrained"][0]["place"] == "bundle"
    assert any("外部の重み (weights) を重ねた" in n for n in info["notes"])
    assert len(torch.load(path / "final_model.pt", weights_only=True)) == 2
    model = build_model(spec_file, seed=0)
    model.eval()
    x = load_bundle(path).dataset.transform(
        pd.read_csv(table_csv)
    ).to_numpy(dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).softmax(dim=1).numpy()
    result = predict(path, table_csv)
    got = result[[f"p:{c}" for c in load_bundle(path).classes]].to_numpy()
    assert np.allclose(got, expected, atol=1e-6)

    # 重ねたノードを出どころから読む以前の一式 (記録の hash は重ねた重み)
    plain_spec = pretrained_spec(width)
    plain = tmp_path / "plain.pt"
    torch.save(build_model(plain_spec, seed=0).state_dict(), plain)
    old = tmp_path / "old"
    make_bundle(old, plain_spec, table_dataset, plain,
                pretrained={"base": license_info(False)})
    written = yaml.safe_load((old / "model_spec.yaml").read_text("utf-8"))
    written["format_version"] = 2
    written["nodes"]["base"]["weights"] = {"path": "over.pt"}
    (old / "model_spec.yaml").write_text(yaml.safe_dump(written), "utf-8")
    info = read_info(old)
    layered_state = torch.load(weights, weights_only=True)
    info["model"]["pretrained"][0]["sha256"] = state_hash(
        layered_state, sorted(layered_state)
    )
    rewrite(old, info)
    with pytest.raises(ValueError, match="一式を作り直して"):
        predict(old, table_csv)


def test_pretrained_info_is_read_lazily(
    tmp_path: Path, table_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    情報を渡さないと ML-ModelControl の pretrained_info で読むことを,
    ネットワークを使わずに確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_dataset: Path
        データセット
    monkeypatch: pytest.MonkeyPatch
        差し替え

    Returns
    ----------
    None
    """
    # 重みを読まない部品だけの定義は記録が空
    path = table_bundle(tmp_path, table_dataset)
    assert read_info(path)["model"]["pretrained"] == []

    # 読んだ情報で置き方を決める (読めない部品は unknown で写さない側)
    calls = []
    monkeypatch.setattr(
        ml_modelcontrol, "pretrained_info", FakeInfo(calls), raising=False
    )
    SOURCE["seed"] = 1
    width = len(load_dataset(table_dataset).features)
    spec = pretrained_spec(width)
    weights = tmp_path / "w.pt"
    torch.save(build_model(spec, seed=0).state_dict(), weights)
    info = make_bundle(tmp_path / "lazy", spec, table_dataset, weights)
    assert calls == ["test.SourceLinear"]
    record = info["model"]["pretrained"][0]
    assert record["status"] == "unknown" and record["place"] == "bundle"
    assert any("ライセンスか版を読めなかった" in n for n in info["notes"])


def nested_spec(width: int, inner: str = "nn.Linear") -> dict:
    """
    部分モデルの中の部分モデルに全結合を持つ定義を返す関数

    Parameters
    ----------
    width: int
        入力の列の数
    inner: str = "nn.Linear"
        中の部分モデルの 1 層目の部品の名前

    Returns
    ----------
    spec: dict
        定義 (本体の block が部分モデル outer を, その中の inner が部分
        モデル head を使う. 使わない部分モデル spare も持つ)
    """
    # 1 層目は渡した部品, 2 層目は全結合
    linear = {"unit": "nn.Linear", "inputs": "x",
              "params": {"in_features": width, "out_features": 3}}
    spec = {
        "format_version": 1,
        "name": "nested",
        "inputs": {"x": {"shape": [4, width]}},
        "nodes": {"block": {"unit": "sub.outer", "inputs": "x"}},
        "outputs": {"logits": "block"},
        "subgraphs": {
            "outer": {
                "inputs": {"x": {}},
                "nodes": {"inner": {"unit": "sub.head", "inputs": "x"}},
                "outputs": {"out": "inner"},
            },
            "head": {
                "inputs": {"x": {}},
                "nodes": {"fc": {**linear, "unit": inner}, "fc2": {
                    "unit": "nn.Linear", "inputs": "fc",
                    "params": {"in_features": 3, "out_features": 3},
                }},
                "outputs": {"out": "fc2"},
            },
            "spare": {
                "inputs": {"x": {}},
                "nodes": {"unused": dict(linear)},
                "outputs": {"out": "unused"},
            },
        },
    }

    return spec


def test_subgraph_pretrained_placement(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    部分モデルの中の再配布できない学習済みの重みを, 一番上のノードと同じく
    一式に写さずに出どころから読むことを確かめる関数

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
    # 部分モデルの中の部分モデルに出どころの部品
    SOURCE["seed"] = 1
    width = len(load_dataset(table_dataset).features)
    spec = nested_spec(width, "test.SourceLinear")
    weights = tmp_path / "w.pt"
    torch.save(build_model(spec, seed=0).state_dict(), weights)
    inner_keys = {"nodes.block.nodes.inner.nodes.fc.fc.weight",
                  "nodes.block.nodes.inner.nodes.fc.fc.bias"}

    # 写せない重みは除き, 部分モデルの定義の版を固定し, hash を残す
    path = tmp_path / "sourced"
    info = make_bundle(path, spec, table_dataset, weights,
                       pretrained={"block.inner.fc": license_info(False)})
    record = info["model"]["pretrained"][0]
    assert record["node"] == "block.inner.fc"
    assert record["place"] == "source" and record["keys"] == 2
    assert any("ノード block.inner.fc の学習済みの重み" in n
               and "一式に写さず" in n for n in info["notes"])
    kept = set(torch.load(path / "final_model.pt", weights_only=True))
    assert not kept & inner_keys and len(kept) == 2
    pinned = yaml.safe_load((path / "model_spec.yaml").read_text("utf-8"))
    head = pinned["subgraphs"]["head"]["nodes"]["fc"]
    assert head["params"]["revision"] == "abc123"
    full = tmp_path / "full"
    make_bundle(full, spec, table_dataset, weights,
                pretrained={"block.inner.fc": license_info(True)})
    assert read_info(full)["model"]["pretrained"][0]["place"] == "bundle"
    assert inner_keys <= set(
        torch.load(full / "final_model.pt", weights_only=True)
    )
    pd.testing.assert_frame_equal(
        predict(path, table_csv), predict(full, table_csv)
    )
    assert describe_bundle(path)["pretrained"][0]["node"] == "block.inner.fc"

    # 出どころの重みが入れ替わると警告
    SOURCE["seed"] = 2
    try:
        with pytest.warns(UserWarning, match="block.inner.fc の出どころ"):
            predict(load_bundle(path), table_csv)
    finally:
        SOURCE["seed"] = 1

    # 外のノードが外部の重みを重ねると, 中のノードも一式の重みに含める
    layered = nested_spec(width, "test.SourceLinear")
    layered["format_version"] = 2
    layered["nodes"]["block"]["weights"] = {"hub": "org/demo",
                                            "file": "w.pt"}
    info = make_bundle(tmp_path / "layered", layered, table_dataset, weights,
                       pretrained={"block.inner.fc": license_info(False)})
    assert info["model"]["pretrained"][0]["place"] == "bundle"
    assert any("block.inner.fc" in n and "外部の重み (weights) を重ねた" in n
               for n in info["notes"])
    assert inner_keys <= set(torch.load(
        tmp_path / "layered" / "final_model.pt", weights_only=True
    ))

    # 記録のノードが定義に無い一式は誤り
    info = read_info(path)
    info["model"]["pretrained"][0]["node"] = "block.fc"
    rewrite(path, info)
    with pytest.raises(ValueError, match="block.fc はモデルの定義"):
        load_bundle(path)

    # 定義に無い部分モデルの中のノードの情報は誤り
    with pytest.raises(ValueError, match="ノードにありません"):
        make_bundle(tmp_path / "bad", spec, table_dataset, weights,
                    pretrained={"block.fc": license_info(False)})


def test_subgraph_pretrained_info(
    tmp_path: Path, table_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    部分モデルを部品にしたノードを pretrained_info に渡さず, 部分モデルの
    中のノードの情報を読んで記録し, 情報を渡さなかった中の学習済みの部品と
    weights の hub から読む重みに注意を残すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    table_dataset: Path
        データセット
    monkeypatch: pytest.MonkeyPatch
        差し替え

    Returns
    ----------
    None
    """
    # 部分モデルの全結合 2 層だけのモデル
    width = len(load_dataset(table_dataset).features)
    spec = nested_spec(width)
    weights = tmp_path / "w.pt"
    torch.save(build_model(spec, seed=0).state_dict(), weights)

    # 学習済みの重みを読むノードは無く, ライセンスの注意も無い
    info = make_bundle(tmp_path / "nested", spec, table_dataset, weights)
    assert info["model"]["pretrained"] == []
    assert not any("ライセンス" in n for n in info["notes"])

    # 部分モデルの中のノードも読んで記録する (部分モデルを部品にした
    # ノードと, 使わない部分モデルは読まない)
    calls: list = []
    monkeypatch.setattr(
        ml_modelcontrol, "pretrained_info", LinearInfo(calls), raising=False
    )
    info = make_bundle(tmp_path / "inner", spec, table_dataset, weights)
    assert calls == [("nn.Linear", True), ("nn.Linear", True)]
    records = info["model"]["pretrained"]
    assert [r["node"] for r in records] == [
        "block.inner.fc", "block.inner.fc2"
    ]
    assert all(r["place"] == "source" for r in records)
    assert torch.load(tmp_path / "inner" / "final_model.pt",
                      weights_only=True) == {}
    assert not any("情報を渡されなかった" in n for n in info["notes"])

    # 情報を渡した場合は, 渡さなかった中の学習済みの部品をネットワークを
    # 使わずに見て, 写した注意を 1 文
    calls.clear()
    info = make_bundle(tmp_path / "given", spec, table_dataset, weights,
                       pretrained={"block.inner.fc": license_info(True)})
    assert calls == [("nn.Linear", False)]
    assert [r["node"] for r in info["model"]["pretrained"]] == [
        "block.inner.fc"
    ]
    inner = [n for n in info["notes"] if "情報を渡されなかった" in n]
    assert len(inner) == 1 and "block.inner.fc2 " in inner[0]

    # 重みを写さない一式には注意を出さない
    info = make_bundle(tmp_path / "unwritten", spec, table_dataset, weights,
                       copy_weights=False, pretrained={})
    assert not any("ライセンス" in n for n in info["notes"])

    # 本体の学習済みのノードが weights の hub を重ねる場合は置き方の 1 文
    sourced = pretrained_spec(width)
    sourced["format_version"] = 2
    sourced["nodes"]["base"]["weights"] = {"hub": "org/demo", "file": "w.pt"}
    sourced_weights = tmp_path / "sourced.pt"
    torch.save(build_model(sourced, seed=0, load_weights=False).state_dict(),
               sourced_weights)
    info = make_bundle(tmp_path / "layered", sourced, table_dataset,
                       sourced_weights,
                       pretrained={"base": license_info(False)})
    noted = [n for n in info["notes"] if "ライセンス" in n]
    assert len(noted) == 1 and "外部の重み (weights) を重ねた" in noted[0]
    monkeypatch.undo()

    # weights の hub から読む外部の重みは注意, path は注意しない
    flat = linear_spec(width, 3)
    flat["format_version"] = 2
    flat["nodes"]["head"]["weights"] = {"hub": "org/demo", "file": "w.pt"}
    plain = save_weights(linear_spec(width, 3), tmp_path / "plain.pt")
    info = make_bundle(tmp_path / "hub", flat, table_dataset, plain)
    external = [n for n in info["notes"] if "hub か unit" in n]
    assert len(external) == 1 and "ノード head " in external[0]
    flat["nodes"]["head"]["weights"] = {"path": "w.pt"}
    info = make_bundle(tmp_path / "path", flat, table_dataset, plain)
    assert not any("ライセンス" in n for n in info["notes"])


def test_shared_subgraph_pretrained(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    同じ部分モデルを使うノードごとに記録し, 違う情報を渡すと誤りにする
    ことと, 部分モデルの中を記録しない前の一式を読めることを確かめる関数

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
    # 全結合の後に, 同じ部分モデルを 2 つのノードで続けて使う定義
    SOURCE["seed"] = 1
    width = len(load_dataset(table_dataset).features)
    spec = nested_spec(3, "test.SourceLinear")
    del spec["subgraphs"]["spare"]
    spec["nodes"] = {
        "pre": {"unit": "nn.Linear", "inputs": "x",
                "params": {"in_features": width, "out_features": 3}},
        "block1": {"unit": "sub.outer", "inputs": "pre"},
        "block2": {"unit": "sub.outer", "inputs": "block1"},
    }
    spec["inputs"] = {"x": {"shape": [4, width]}}
    spec["outputs"] = {"logits": "block2"}
    weights = tmp_path / "w.pt"
    torch.save(build_model(spec, seed=0).state_dict(), weights)

    # ノードごとに記録し, どちらも出どころから読む
    given = {"block1.inner.fc": license_info(False),
             "block2.inner.fc": license_info(False)}
    path = tmp_path / "shared"
    info = make_bundle(path, spec, table_dataset, weights, pretrained=given)
    records = info["model"]["pretrained"]
    assert [r["node"] for r in records] == [
        "block1.inner.fc", "block2.inner.fc"
    ]
    assert all(r["place"] == "source" and r["keys"] == 2 for r in records)
    kept = set(torch.load(path / "final_model.pt", weights_only=True))
    assert not any(".inner.nodes.fc." in k for k in kept)
    full = tmp_path / "full"
    make_bundle(full, spec, table_dataset, weights, pretrained={
        node: license_info(True) for node in given
    })
    pd.testing.assert_frame_equal(
        predict(path, table_csv), predict(full, table_csv)
    )

    # 同じ部分モデルの同じノードに違う情報は誤り
    with pytest.raises(ValueError, match="block1.inner.fc と block2.inner.fc"):
        make_bundle(tmp_path / "differ", spec, table_dataset, weights,
                    pretrained={**given,
                                "block2.inner.fc": license_info(True)})
    with pytest.raises(ValueError, match="同じ部分モデルの同じノード"):
        make_bundle(tmp_path / "revision", spec, table_dataset, weights,
                    pretrained={**given, "block2.inner.fc": {
                        **license_info(False), "pinned_revision": "def456"
                    }})

    # 部分モデルの中を記録しない前の一式 (一番上のノードだけの記録と,
    # 全ての重み) もそのまま読む
    old = tmp_path / "old"
    make_bundle(old, spec, table_dataset, weights,
                pretrained={"block1.inner.fc": license_info(True)})
    info = read_info(old)
    info["model"]["pretrained"] = []
    rewrite(old, info)
    pd.testing.assert_frame_equal(
        predict(old, table_csv), predict(full, table_csv)
    )
    assert describe_bundle(old)["pretrained_warnings"] == []
    top = tmp_path / "top"
    make_bundle(top, pretrained_spec(width), table_dataset,
                save_weights(pretrained_spec(width), tmp_path / "top.pt"),
                pretrained={"base": license_info(False)})
    assert read_info(top)["model"]["pretrained"][0]["node"] == "base"
    assert len(predict(top, table_csv)) > 0


def test_cyclic_subgraphs_stop() -> None:
    """
    外側の部分モデルを中に含む循環を辿らずに止まることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # a が b を, b が a と自分を使う定義
    spec = {
        "nodes": {"top": {"unit": "sub.a"}},
        "subgraphs": {
            "a": {"nodes": {"b": {"unit": "sub.b"}}},
            "b": {"nodes": {"back": {"unit": "sub.a"},
                            "own": {"unit": "sub.b"},
                            "fc": {"unit": "nn.Linear"}}},
        },
    }
    assert list(node_items(spec)) == [
        "top", "top.b", "top.b.back", "top.b.own", "top.b.fc"
    ]


class LinearInfo:
    def __init__(self, calls: list):
        # 呼ばれた部品の名前とネットワークを使うかの記録を保存
        self.calls = calls

    def __call__(
        self, unit: str, params: dict, fetch: bool = True
    ) -> Any:
        """
        全結合を学習済みの部品とみなす pretrained_info の代わりの関数

        Parameters
        ----------
        unit: str
            部品の名前
        params: dict
            引数
        fetch: bool = True
            ネットワークを使うか

        Returns
        ----------
        info: Any
            全結合は再配布できない情報, 他は None
        """
        # 全結合だけ学習済み
        self.calls.append((unit, fetch))
        info = license_info(False) if unit == "nn.Linear" else None

        return info


class FakeInfo:
    def __init__(self, calls: list):
        # 呼ばれた部品の名前の記録を保存
        self.calls = calls

    def __call__(self, unit: str, params: dict) -> Any:
        """
        ネットワークを使わない pretrained_info の代わりの関数

        Parameters
        ----------
        unit: str
            部品の名前
        params: dict
            引数

        Returns
        ----------
        info: Any
            出どころの部品は読めなかった情報, 他は None
        """
        # 読めなかったものとして例外
        self.calls.append(unit)
        if unit != "test.SourceLinear":
            return None
        raise OSError("ネットワークを使わない")


def test_input_columns_named_like_outputs(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    入力の output と artifact の列が結果の版と配列の書き出しを変えない
    ことを確かめる関数

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
    # 列の出力の一式は版 1 のまま, 入力の列は (入力) 付きの名前
    frame = pd.read_csv(table_csv)
    frame["output"] = 1.0
    frame["artifact"] = "x"
    path = table_bundle(tmp_path, table_dataset)
    root = write_result(infer(path, frame), tmp_path / "column")
    back = read_result(root)
    assert back["format_version"] == 1 and back["form"] is None
    assert "output (入力)" in back["frame"].columns
    assert not (root / "artifacts").exists()

    # 配列の出力の一式でも, 入力の artifact の列と並べて書ける
    vector = table_bundle(tmp_path, table_dataset, 4, name="vec",
                          form="vector")
    root = write_result(infer(vector, frame), tmp_path / "vector")
    table = read_result(root)["frame"]
    assert table.loc[0, "artifact"] == "artifacts/000000.npy"
    assert (table["artifact (入力)"] == "x").all()


def test_procedures_registry() -> None:
    """
    手順の登録簿の関数の計算, 決まった出力, 終わりの番号を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # Euler 法: 一定の速度なら始まりに速度を足したもの
    start = torch.zeros(2, 3)
    out = run_procedure("euler", ConstantModel(2.0), start, 4,
                        make_generator(0))
    assert torch.allclose(out, torch.full((2, 3), 2.0))

    # DDIM: 雑音の予測が 0 なら, 最初の時刻の比で大きくしたもの
    noise = start_noise((2, 3), make_generator(0))
    out = run_procedure("ddim", ConstantModel(0.0), noise, 10,
                        make_generator(0))
    alphas = torch.cumprod(1 - torch.linspace(1e-4, 0.02, 1000,
                                              dtype=torch.float64), 0)
    assert torch.allclose(out, noise / float(alphas[999]) ** 0.5, atol=1e-4)

    # 同じ種なら同じ出力, eta > 0 は種で変わる
    first = run_procedure("ddim", ConstantModel(0.1), noise, 5,
                          make_generator(3), eta=1.0)
    again = run_procedure("ddim", ConstantModel(0.1), noise, 5,
                          make_generator(3), eta=1.0)
    other = run_procedure("ddim", ConstantModel(0.1), noise, 5,
                          make_generator(4), eta=1.0)
    assert torch.equal(first, again) and not torch.equal(first, other)

    # 文の生成: 次の番号は 1 つ前の番号 + 1, 終わりの番号で止める
    ids = torch.tensor([[0], [2]])
    out = run_procedure("greedy", NextToken(5), ids, 6, make_generator(0),
                        eos=3)
    assert out.tolist() == [[0, 1, 2, 3], [2, 3, 3, 3]]
    drawn = run_procedure("sample", NextToken(5), ids, 4, make_generator(1),
                          temperature=0.5)
    again = run_procedure("sample", NextToken(5), ids, 4, make_generator(1),
                          temperature=0.5)
    assert torch.equal(drawn, again) and drawn.shape == (2, 5)
    with pytest.raises(ValueError, match="temperature"):
        run_procedure("sample", NextToken(5), ids, 2, make_generator(0),
                      temperature=0.0)

    # 組み込みの名前は登録し直せない. 自作の手順は名前で呼べる
    with pytest.raises(ValueError, match="既に"):
        register_procedure("ddim", euler_twice, "noise", True)
    register_procedure("twice", euler_twice, "noise", True,
                       overwrite=True)
    out = run_procedure("twice", ConstantModel(1.0), start, 2,
                        make_generator(0))
    assert torch.allclose(out, torch.full((2, 3), 2.0))
    with pytest.raises(KeyError, match="登録簿"):
        get_procedure("beam")


def test_text_generation_pads_batches(tmp_path: Path) -> None:
    """
    終わりの番号で件数ずつ長さが違っても, 並べて返すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 長さの違う件数ずつの出力を, 終わりの番号で埋めて並べる
    parts = [np.array([[1, 2, 3]]), np.array([[4, 5]])]
    joined = join_parts(parts, 9)
    assert joined.tolist() == [[1, 2, 3], [4, 5, 9]]
    assert finite_raw(np.array([[1.0, np.nan], [0.0, 1.0]])).tolist() == [
        False, True
    ]
    assert finite_raw(np.array([[1, 2]])).tolist() == [True]


class ConstantModel:
    def __init__(self, value: float):
        # 返す値を保存
        self.value = value

    def __call__(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        どの入力にも同じ値を返す関数

        Parameters
        ----------
        x: torch.Tensor
            中身
        t: torch.Tensor
            時刻

        Returns
        ----------
        out: torch.Tensor
            中身と同じ形の一定の値
        """
        # 一定の値
        out = torch.full_like(x, self.value)

        return out


class NextToken:
    def __init__(self, size: int):
        # 語彙の数を保存
        self.size = size

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        """
        最後の番号 + 1 が最も大きい logits を返す関数

        Parameters
        ----------
        ids: torch.Tensor
            (件数, 長さ) の番号の並び

        Returns
        ----------
        logits: torch.Tensor
            (件数, 長さ, 語彙) の logits
        """
        # 最後の位置だけ次の番号を大きく
        logits = torch.zeros(len(ids), ids.shape[1], self.size)
        chosen = (ids[:, -1] + 1) % self.size
        logits[torch.arange(len(ids)), -1, chosen] = 5.0

        return logits


def euler_twice(
    model: Any, state: torch.Tensor, steps: int,
    generator: torch.Generator
) -> torch.Tensor:
    """
    自作の手順の例として, Euler 法を 2 倍の刻みで進める関数

    Parameters
    ----------
    model: Any
        model(x, t) で速度を返すもの
    state: torch.Tensor
        始まりの中身
    steps: int
        繰り返しの回数
    generator: torch.Generator
        乱数の生成器

    Returns
    ----------
    x: torch.Tensor
        時刻 2 の中身
    """
    # 組み込みの Euler 法の終わりの時刻を 2 に
    x = run_procedure("euler", model, state, steps, generator, t_end=2.0)

    return x


def test_result_round_trip_and_safety(
    tmp_path: Path, table_dataset: Path, table_csv: Path
) -> None:
    """
    配列の結果を読んで書き直しても配列が残り, 結果のものでない
    artifacts/ を消さないことを確かめる関数

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
    # 配列の結果を書き, 読んで同じ場所と別の場所に書き直す
    vector = table_bundle(tmp_path, table_dataset, 4, name="vec",
                          form="vector")
    result = infer(vector, table_csv)
    root = write_result(result, tmp_path / "first")
    first = np.load(root / "artifacts" / "000005.npy")
    back = read_result(root)
    assert back["form"] == "vector"
    write_result(back, root)
    again = read_result(root)
    assert again["artifacts"]["count"] == 60 and again["form"] == "vector"
    assert np.array_equal(np.load(root / "artifacts" / "000005.npy"), first)
    other = write_result(read_result(root), tmp_path / "second")
    assert np.array_equal(
        np.load(other / "artifacts" / "000005.npy"), first
    )
    assert "directory" not in yaml.safe_load(
        (other / "result.yaml").read_text(encoding="utf-8")
    )

    # 結果のものでない artifacts/ は消さずに止まる
    user = tmp_path / "user"
    (user / "artifacts").mkdir(parents=True)
    (user / "artifacts" / "mine.txt").write_text("大事", encoding="utf-8")
    with pytest.raises(FileExistsError, match="消しません"):
        write_result(result, user)
    assert (user / "artifacts" / "mine.txt").exists()

    # 列の結果は artifacts/ に触れない
    column = table_bundle(tmp_path, table_dataset, name="col")
    write_result(infer(column, table_csv), user)
    assert (user / "artifacts" / "mine.txt").exists()

    # 配列の表に結論の決め方は使えない (はっきりした誤り)
    with pytest.raises(ValueError, match="prediction の列がありません"):
        conclude(result["frame"], "classification", ["犬", "猫", "鳥"])


def test_median_and_option_checks(
    tmp_path: Path, regression_dataset: Path, table_dataset: Path
) -> None:
    """
    0.5 の無い分位点の真ん中の補いと, 手順の設定の確かめを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    regression_dataset: Path
        回帰のデータセット
    table_dataset: Path
        データセット

    Returns
    ----------
    None
    """
    # 0.25 と 0.75 の間を線形に補う
    values = np.array([[1.0, 3.0], [2.0, 6.0]])
    assert median_of([0.25, 0.75], values).tolist() == [2.0, 4.0]
    assert median_of([0.1, 0.5, 0.9], np.array([[1.0, 2.0, 9.0]])).tolist() \
        == [2.0]

    # 設定の無い手順は設定を受け取らず, **kwargs の手順は何でも受け取る
    register_procedure("plain", plain_steps, "noise", True, overwrite=True)
    register_procedure("loose", loose_steps, "noise", True, overwrite=True)
    assert option_names("plain") == [] and option_names("loose") is None
    spec = load_example("diffusion_denoiser")
    weights = save_weights(spec, tmp_path / "denoiser.pt")
    options = {
        "inputs": {"x": {"source": "state"}, "t": {"source": "time"}},
        "form": "vector", "output_options": {"shape": [16]},
    }
    with pytest.raises(ValueError, match="設定の無い手順"):
        make_bundle(tmp_path / "plain", spec, table_dataset, weights,
                    **options, procedure={"name": "plain", "steps": 2,
                                          "seed": 0, "options": {"a": 1}})
    make_bundle(tmp_path / "loose", spec, table_dataset, weights, **options,
                procedure={"name": "loose", "steps": 2, "seed": 0,
                           "options": {"a": 1}})

    # train_steps は整数だけ
    with pytest.raises(ValueError, match="train_steps は整数"):
        make_bundle(tmp_path / "float", spec, table_dataset, weights,
                    **options, procedure={"name": "ddim", "steps": 2,
                                          "seed": 0,
                                          "options": {"train_steps": 1000.0}})
    with pytest.raises(ValueError, match="train_steps は整数"):
        run_procedure("ddim", ConstantModel(0.0), torch.zeros(1, 2), 2,
                      make_generator(0), train_steps=1000.0)


def plain_steps(
    model: Any, state: torch.Tensor, steps: int,
    generator: torch.Generator
) -> torch.Tensor:
    """
    設定の無い自作の手順の例として, 中身をそのまま返す関数

    Parameters
    ----------
    model: Any
        モデル
    state: torch.Tensor
        始まりの中身
    steps: int
        回数
    generator: torch.Generator
        乱数の生成器

    Returns
    ----------
    state: torch.Tensor
        始まりの中身
    """
    # そのまま
    return state


def loose_steps(
    model: Any, state: torch.Tensor, steps: int,
    generator: torch.Generator, **options: Any
) -> torch.Tensor:
    """
    何でも設定を受け取る自作の手順の例として, 中身をそのまま返す関数

    Parameters
    ----------
    model: Any
        モデル
    state: torch.Tensor
        始まりの中身
    steps: int
        回数
    generator: torch.Generator
        乱数の生成器
    **options: Any
        設定 (使わない)

    Returns
    ----------
    state: torch.Tensor
        始まりの中身
    """
    # そのまま
    return state


class FailingBuild:
    def __call__(
        self, spec: dict, seed: int = 0, load_weights: bool = True
    ) -> Any:
        """
        ネットワークの無いときのように組み立てに失敗する関数

        Parameters
        ----------
        spec: dict
            定義
        seed: int = 0
            乱数の種
        load_weights: bool = True
            外部の重みを読むか

        Returns
        ----------
        model: Any
            返さない (常に誤り)
        """
        # 出どころを読めない誤り
        raise OSError("出どころにつながりません")


def odd_steps(
    model: Any, state: torch.Tensor, steps: int,
    generator: torch.Generator, scale: float = math.inf,
    label: str = "a", flag: bool = True, limit: Any = None,
    table: Any = (1, 2), count: int = 3, need: Any = 0.0
) -> torch.Tensor:
    """
    既定値の種類の違う設定を持つ自作の手順の例として, 中身をそのまま
    返す関数

    Parameters
    ----------
    model: Any
        モデル
    state: torch.Tensor
        始まりの中身
    steps: int
        回数
    generator: torch.Generator
        乱数の生成器
    scale: float = math.inf
        JSON にできない既定値の設定
    label: str = "a"
        文字の既定値の設定
    flag: bool = True
        真偽値の既定値の設定
    limit: Any = None
        既定値が None の設定
    table: Any = (1, 2)
        並びの既定値の設定
    count: int = 3
        整数の既定値の設定
    need: Any = 0.0
        小数の既定値の設定

    Returns
    ----------
    state: torch.Tensor
        始まりの中身
    """
    # そのまま
    return state


def test_procedure_options() -> None:
    """
    公開の procedure_options が, 手順の設定の名前と既定値を返すことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 組み込みの手順は 5 つ目からの引数と既定値 (None の既定値は除く)
    assert procedure_options("ddim") == {
        "names": ["train_steps", "beta_start", "beta_end", "eta"],
        "defaults": {"train_steps": 1000, "beta_start": 1e-4,
                     "beta_end": 0.02, "eta": 0.0},
    }
    assert procedure_options("greedy") == {"names": ["eos"], "defaults": {}}
    assert procedure_options("sample")["defaults"] == {"temperature": 1.0}

    # 設定の無い手順は空, **kwargs の手順は名前を決めない
    register_procedure("plain", plain_steps, "noise", True, overwrite=True)
    register_procedure("loose", loose_steps, "noise", True, overwrite=True)
    assert procedure_options("plain") == {"names": [], "defaults": {}}
    assert procedure_options("loose") == {"names": None, "defaults": {}}

    # 既定値は有限の数, 真偽値, 文字だけ
    register_procedure("odd", odd_steps, "noise", True, overwrite=True)
    found = procedure_options("odd")
    assert found["names"] == ["scale", "label", "flag", "limit", "table",
                              "count", "need"]
    assert found["defaults"] == {"label": "a", "flag": True, "count": 3,
                                 "need": 0.0}

    # 引数を読めない関数は何でも受け取る. 無い名前は登録簿の誤り
    register_procedure("opaque", max, "noise", True, overwrite=True)
    assert procedure_options("opaque") == {"names": None, "defaults": {}}
    assert option_names("opaque") is None
    with pytest.raises(KeyError, match="登録簿"):
        procedure_options("beam")
