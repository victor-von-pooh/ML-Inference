import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from conftest import save_weights
from ml_eda import export_dataset, load_data

from ml_inference import (
    load_bundle, make_bundle, predict, read_result, write_result
)
from ml_inference.texts import ids_to_text, text_vocabulary


def test_ids_to_text_by_unit() -> None:
    """
    語, 文字, 部分語の分け方ごとに, 番号の並びを文に戻せることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 語は文字と数字の並びどうしの間だけに空白, 埋め草は除く
    words = {
        "vocabulary": ["<pad>", "<unk>", "good", "item", "!", "品", "悪"],
        "unit": "word",
    }
    assert ids_to_text([2, 3, 4, 0, 0], words, [], None) == "good item!"
    assert ids_to_text([6, 5, 2, 1, 3], words, [], None) == (
        "悪品good <unk> item"
    )

    # 先頭に足した番号は除き, 足した番号の中の終わりの番号で切る. 渡した文
    # (長さ 2) の中の終わりの番号と埋め草では切らない. 語彙の外は番号
    prompt = {**words, "length": 2}
    assert ids_to_text([99, 2, 3, 4, 2], prompt, [99], 4) == "good item"
    assert ids_to_text([99, 2, 3, 2, 4, 3], prompt, [99], 4) == (
        "good item good"
    )
    assert ids_to_text([2, 4, 2, 4, 3], prompt, [], 4) == "good! good"
    assert ids_to_text([2, 0, 3, 0, 0], prompt, [], 0) == "good item"
    assert ids_to_text([2, 50], words, [], None) == "good <50>"

    # 区切りの記号の後ろと開き括弧の前の空白は戻す
    marks = {
        "vocabulary": ["<pad>", "<unk>", "hello", ",", "world", ".", "this",
                       "(", "a", ")", "!"],
        "unit": "word",
    }
    assert ids_to_text([2, 3, 4, 5, 6, 7, 8, 9, 10], marks, [], None) == (
        "hello, world. this (a)!"
    )

    # 数字に挟まれた区切りの記号の後ろには空白を入れない
    numbers = {
        "vocabulary": ["<pad>", "<unk>", "3", ".", "5", "10", ":", "30",
                       "1", ",", "000", "at", "e", "g"],
        "unit": "word",
    }
    assert ids_to_text([2, 3, 4], numbers, [], None) == "3.5"
    assert ids_to_text([11, 5, 6, 7], numbers, [], None) == "at 10:30"
    assert ids_to_text([8, 9, 10, 3, 11], numbers, [], None) == "1,000. at"
    assert ids_to_text([12, 3, 13, 3], numbers, [], None) == "e. g."

    # 文字はそのままつなぐ
    chars = {"vocabulary": ["<pad>", "<unk>", "良", "い"], "unit": "char"}
    assert ids_to_text([2, 3, 0], chars, [], None) == "良い"

    # 部分語は空白の印を空白に
    pieces = {
        "vocabulary": ["<pad>", "<unk>", "\u2581go", "od", "\u2581item",
                       "!"],
        "unit": "subword",
    }
    assert ids_to_text([2, 3, 4, 5], pieces, [], None) == "good item!"


def test_text_vocabulary_from_recipe() -> None:
    """
    レシピの最後の当てはめた vocabulary の手順を読むことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 当てはめていない手順と語彙の無いレシピは None
    assert text_vocabulary({}) is None
    assert text_vocabulary(
        {"item_steps": [{"op": "vocabulary", "unit": "char"}]}
    ) is None

    # 最後の手順だけを見る (最後に語彙が無ければ前の手順の語彙も使わない)
    fitted = {"op": "vocabulary", "unit": "char", "length": 8,
              "state": {"vocabulary": ["<pad>", "<unk>", "a"]}}
    recipe = {"item_steps": [
        fitted,
        {"op": "vocabulary", "state": {"vocabulary": ["<pad>", "<unk>"]}},
    ]}
    assert text_vocabulary(recipe) == {
        "vocabulary": ["<pad>", "<unk>"], "unit": "word", "length": 128
    }
    assert text_vocabulary({"item_steps": [
        fitted, {"op": "vocabulary", "unit": "char"}
    ]}) is None

    # 部分語の並びの無い古いレシピは区切りの JSON の語彙から
    serialized = json.dumps({"model": {"vocab": {
        "<pad>": 0, "\u2581ab": 2, "<unk>": 1
    }}})
    old = {"item_steps": [{"op": "vocabulary", "unit": "subword",
                           "state": {"tokenizer": serialized}}]}
    assert text_vocabulary(old)["vocabulary"] == [
        "<pad>", "<unk>", "\u2581ab"
    ]
    broken = {"item_steps": [{"op": "vocabulary", "unit": "subword",
                              "state": {"tokenizer": "{"}}]}
    assert text_vocabulary(broken) is None


@pytest.mark.parametrize("unit", ["word", "subword"])
def test_text_bundle_prediction_is_text(tmp_path: Path, unit: str) -> None:
    """
    文のデータセットの一式の予測が, レシピの語彙で戻した文になることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    unit: str
        語彙の分け方

    Returns
    ----------
    None
    """
    # 部分語は tokenizers が要る
    if unit == "subword":
        pytest.importorskip("tokenizers")
    texts = ["good item", "bad item", "very good", "very bad"] * 5
    labels = ["g", "b", "g", "b"] * 5
    source = tmp_path / "reviews.csv"
    pd.DataFrame({"text": texts, "label": labels}).to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [],
        "item_steps": [{"op": "vocabulary", "unit": unit, "length": 8,
                        "max_tokens": 40}],
        "split": {"method": "random", "valid": 0.2, "test": 0.0, "seed": 0},
    }
    dataset = tmp_path / "dataset"
    export_dataset(load_data(source, text_column="text"), recipe, dataset)

    # 入力の番号をそのまま返す (位置ごとに入力の番号が最も大きい logits)
    size = 64
    spec = {
        "format_version": 1,
        "inputs": {"ids": {"shape": [2, 8], "dtype": "int64"}},
        "nodes": {
            "head": {
                "unit": "nn.Embedding",
                "params": {"num_embeddings": size, "embedding_dim": size},
                "inputs": "ids",
            },
        },
        "outputs": {"logits": "head"},
    }
    weights = save_weights(spec, tmp_path / "echo.pt")
    state = torch.load(weights, weights_only=True)
    state["nodes.head.weight"] = torch.eye(size)
    torch.save(state, weights)
    path = tmp_path / "bundle"
    make_bundle(path, spec, dataset, weights, form="text", role="generate")

    # 文は語彙で戻し, 番号は tokens に残る
    result = predict(path, pd.DataFrame({"text": ["good item", "very"]}))
    assert result["prediction"].tolist() == ["good item", "very"]
    bundle = load_bundle(path)
    ids = [int(v) for v in bundle.dataset.transform_item("good item")]
    assert result.loc[0, "tokens"] == " ".join(str(v) for v in ids)


def test_empty_text_survives_result(tmp_path: Path) -> None:
    """
    戻すと空になる文の行が, 書き出して読み直しても予測しなかった行の欠損
    と分かれることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 予測した空の文, 番号 1 つの文, 予測しなかった行
    frame = pd.DataFrame({
        "text": ["a", "b", ""],
        "status": ["ok", "ok", "skipped"],
        "reason": ["", "", "文が空"],
        "prediction": ["", "3", None],
        "tokens": ["0 0", "3", None],
        "decision": ["", "", "予測なし"],
    })
    result = {"bundle": {"name": "t"}, "frame": frame,
              "summary": {"rows": 3}, "sentences": [], "rules": {}}
    write_result(result, tmp_path / "result")
    back = read_result(tmp_path / "result")["frame"]
    assert back.loc[0, "prediction"] == ""
    assert back.loc[1, "prediction"] == "3"
    assert back.loc[1, "tokens"] == "3"
    assert pd.isna(back.loc[2, "prediction"])
    assert pd.isna(back.loc[2, "tokens"])
