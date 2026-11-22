from pathlib import Path

import numpy as np
import pandas as pd

import ml_inference
from ml_inference import (
    RULE_KEYS, conclude, conclude_outputs, load_bundle, make_sample_bundle,
    origin_sentence, read_source, summary_sentences, target_of
)

# 画面 (ML-Provide) が使うために公開する名前
SCREEN_NAMES = {
    "RULE_KEYS", "backproject", "conclude_outputs", "origin_sentence",
    "read_source", "summary_sentences", "target_of",
}


def test_screen_names_are_public() -> None:
    """
    画面が使う名前が公開の一覧にあり, 読めることを確かめる関数

    Returns
    ----------
    None
    """
    assert SCREEN_NAMES <= set(ml_inference.__all__)
    for name in SCREEN_NAMES:
        assert getattr(ml_inference, name) is not None
    assert "group_by" in RULE_KEYS


def test_summary_sentences_and_outputs() -> None:
    """
    まとめの文が conclude の文と同じで, 件数だけの結論が数えることを
    確かめる関数

    Returns
    ----------
    None
    """
    # 3 行のうち 2 行を予測した分類の表
    frame = pd.DataFrame({
        "status": ["ok", "ok", "skipped"], "reason": ["", "", "欠損"],
        "prediction": ["上", "下", np.nan],
        "confidence": [0.9, 0.8, np.nan],
    })
    conclusion = conclude(frame, "classification", ["上", "下"])
    assert summary_sentences(
        conclusion["summary"], "classification"
    ) == conclusion["sentences"]

    # 列の出力でない表は件数だけ
    counted = conclude_outputs(frame.drop(columns=["prediction"]))
    assert counted["summary"] == {"rows": 3, "predicted": 2, "skipped": 1}


def test_origin_sentence() -> None:
    """
    正解が処理の出力の場合だけ文を返すことを確かめる関数

    Returns
    ----------
    None
    """
    process = {"kind": "mask", "origin": {"type": "process", "name": "SAM"}}
    assert "SAM" in origin_sentence(process)
    assert origin_sentence({"kind": "column"}) is None
    assert origin_sentence(None) is None


def test_read_source_and_target(tmp_path: Path) -> None:
    """
    見本の一式で, 学習のときと同じ読み込みと正解の種類を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 表の分類の見本の一式
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    source = tmp_path / "class.csv"
    pd.DataFrame({"x": x, "y": np.where(x > 0, "上", "下")}).to_csv(
        source, index=False
    )
    recipe = {"target": "y", "task": "classification", "steps": [],
              "split": {"method": "random", "valid": 0.2, "seed": 0}}
    make_sample_bundle(tmp_path / "bundle", source, recipe, steps=20)
    bundle = load_bundle(tmp_path / "bundle")

    # 場所は学習のときの読み込みで読み, 表はそのまま
    data = read_source(bundle, source)
    assert len(data.frame) == 100
    assert list(read_source(bundle, data.frame.head(3)).frame.columns) == (
        list(data.frame.columns)
    )
    assert target_of(bundle) == {"kind": "column"}

    # 版 3 の正解は写しで, 書き換えても一式は変わらない
    bundle.info["target"] = {"kind": "mask",
                             "origin": {"type": "process", "name": "SAM"}}
    target = target_of(bundle)
    target["origin"]["name"] = "別の処理"
    assert bundle.info["target"]["origin"]["name"] == "SAM"
