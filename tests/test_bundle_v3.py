from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from conftest import linear_spec, save_weights
from ml_eda import export_dataset, load_data, load_dataset
from ml_modelcontrol import build_model, load_example, register_unit
from torch import nn

from ml_inference import (
    artifact_path, decide, describe_bundle, infer, load_bundle, make_bundle,
    predict, read_result, write_result
)
from ml_inference.predict import DECODERS

# 出どころの重みを作る乱数の種 (入れ替わった出どころを試すため変える)
SOURCE = {"seed": 1}

# 版 3 で足した項目
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
    import shutil

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
        assert description["target"] == {"kind": "column"}
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
    # 版 4 は読めない
    path = table_bundle(tmp_path, table_dataset)
    original = read_info(path)
    rewrite(path, {**original, "format_version": 4})
    with pytest.raises(ValueError, match="format_version 4 .*読めません"):
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
    meta["format_version"] = 3
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

    # 文は位置ごとに最も大きい番号を空白でつなぐ
    stub = SimpleNamespace(form="text", output_kind="logits")
    tokens = np.zeros((1, 2, 4))
    tokens[0, 0, 3] = 1.0
    tokens[0, 1, 1] = 1.0
    assert DECODERS["text"](stub, tokens)["prediction"] == ["3 1"]


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
        lengths = [len(text.split()) for text in first["prediction"]]
        assert lengths == [7, 7]
        again = predict(path, frame)["prediction"].tolist()
        assert again == first["prediction"].tolist()
        results[name] = first
    # 渡した文の番号が先頭に残る
    bundle = load_bundle(tmp_path / "greedy")
    head = " ".join(
        str(int(v)) for v in bundle.dataset.transform_item("良い品")
    )
    assert results["greedy"].loc[0, "prediction"].startswith(head)


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
        "license": "mit" if redistributable else None,
        "status": "commercial" if redistributable else "unknown",
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

    # 定義に無いノードの情報は誤り
    with pytest.raises(ValueError, match="ノードにありません"):
        make_bundle(tmp_path / "bad", spec, table_dataset, weights,
                    pretrained={"other": license_info(False)})


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
    import ml_modelcontrol

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
    assert record["status"] == "unknown" and record["place"] == "source"


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
