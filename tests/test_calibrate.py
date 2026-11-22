from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from conftest import linear_spec
from ml_eda import export_dataset, load_data
from ml_modelcontrol import build_model

from ml_inference import (
    decide, describe_bundle, load_bundle, make_bundle, predict
)
from ml_inference.calibrate import (
    calibrated, fit_calibration, fit_isotonic, fit_platt
)


def write_classes(
    tmp_path: Path, classes: int, scale: float, split: dict
) -> tuple[Path, dict, Path]:
    """
    確率が当たりすぎる (logit が scale 倍) 分類のデータセットと重みを書く
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    classes: int
        クラスの数 (2 か 3)
    scale: float
        本当の logit に対するモデルの logit の倍率
    split: dict
        分け方

    Returns
    ----------
    written: tuple[Path, dict, Path]
        データセット, モデルの定義, 重み
    """
    # 本当の logit はクラスごとに (クラスの番号 - 真ん中) × 2x
    rng = np.random.default_rng(3)
    x = rng.normal(size=3000)
    slopes = 2.0 * (np.arange(classes) - (classes - 1) / 2)
    logits = x[:, None] * slopes[None, :]
    truth = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    names = np.array(["no", "yes"] if classes == 2 else ["a", "b", "c"])
    drawn = (rng.uniform(size=(3000, 1)) > truth.cumsum(axis=1)).sum(axis=1)
    label = names[np.minimum(drawn, classes - 1)]
    source = tmp_path / "c.csv"
    pd.DataFrame({"x": x, "label": label}).to_csv(source, index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [], "split": split,
    }
    dataset = tmp_path / "ds"
    export_dataset(load_data(source), recipe, dataset)

    # 当たりすぎる重み (クラスの名前の順に同じ傾き)
    order = np.argsort(names)
    spec = linear_spec(1, classes)
    model = build_model(spec)
    with torch.no_grad():
        model.nodes["head"].weight.copy_(torch.tensor(
            (scale * slopes[order])[:, None], dtype=torch.float32
        ))
        model.nodes["head"].bias.zero_()
    weights = tmp_path / "w.pt"
    torch.save(model.state_dict(), weights)
    written = (dataset, spec, weights)

    return written


# 検証用とテスト用を持つ分け方
SPLIT = {"method": "random", "valid": 0.4, "test": 0.3, "seed": 0}


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_binary_calibration(tmp_path: Path, method: str) -> None:
    """
    2 クラスの当たりすぎる確率を補正すると, テスト用の分割のずれが下がり,
    意思決定の文が補正を述べることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    method: str
        補正の選び方

    Returns
    ----------
    None
    """
    # 補正の前と後の一式
    dataset, spec, weights = write_classes(tmp_path, 2, 3.0, SPLIT)
    plain = make_bundle(tmp_path / "plain", spec, dataset, weights)
    info = make_bundle(tmp_path / "fixed", spec, dataset, weights,
                       calibrate=method)
    fitted = info["calibrate"]
    assert fitted["method"] == method and fitted["split"] == "valid"
    assert list(fitted["classes"]) == ["yes"]
    assert fitted["before"]["test_ece"] == pytest.approx(
        plain["uncertainty"]["test_ece"]
    )
    assert info["uncertainty"]["test_ece"] < (
        fitted["before"]["test_ece"] - 0.03
    )
    assert "calibrate" not in plain
    assert fitted["after"]["test_ece"] == pytest.approx(
        info["uncertainty"]["test_ece"]
    )

    # 読み直した一式の予測も補正した確率 (端の確率が真ん中に寄る)
    frame = pd.DataFrame({"x": [-2.0, 0.0, 2.0]})
    before = predict(tmp_path / "plain", frame)["p:yes"].to_numpy()
    after = predict(tmp_path / "fixed", frame)["p:yes"].to_numpy()
    assert after[0] > before[0] and after[2] < before[2]
    assert np.all(np.diff(after) >= 0)
    described = describe_bundle(tmp_path / "fixed")["calibrate"]
    assert described["method"] == method
    assert described["before"]["test_ece"] == fitted["before"]["test_ece"]
    assert describe_bundle(tmp_path / "plain")["calibrate"] is None

    # 意思決定の文
    triage = {"type": "triage", "positive": "yes", "miss_cost": 9,
              "false_alarm_cost": 1}
    notes = decide(tmp_path / "fixed", predict(tmp_path / "fixed", frame),
                   triage)["notes"]
    name = "Platt 法" if method == "platt" else "isotonic 回帰"
    assert any(
        name in note and "ECE" in note and "補正の前" in note
        and "補正の後" in note for note in notes
    )
    notes = decide(tmp_path / "plain", predict(tmp_path / "plain", frame),
                   triage)["notes"]
    assert not any("補正" in note for note in notes)


def test_multiclass_calibration(tmp_path: Path) -> None:
    """
    3 クラスはクラスごとに補正して和を 1 にし, テスト用の分割のずれが下がる
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # クラスごとの当てはめた値
    dataset, spec, weights = write_classes(tmp_path, 3, 3.0, SPLIT)
    info = make_bundle(tmp_path / "fixed", spec, dataset, weights,
                       calibrate="platt")
    assert sorted(info["calibrate"]["classes"]) == ["a", "b", "c"]
    assert info["uncertainty"]["test_ece"] < (
        info["calibrate"]["before"]["test_ece"]
    )
    table = predict(tmp_path / "fixed", pd.DataFrame({"x": [-1.5, 0.3]}))
    total = table[["p:a", "p:b", "p:c"]].sum(axis=1).to_numpy()
    assert total == pytest.approx(1.0)
    assert list(table["confidence"]) == pytest.approx(
        list(table[["p:a", "p:b", "p:c"]].max(axis=1))
    )


def test_calibration_errors(tmp_path: Path) -> None:
    """
    知らない選び方, 検証用の分割の無いデータセット, 分類でない一式, 崩れた
    calibrate で止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 選び方と, 検証用の分割
    dataset, spec, weights = write_classes(tmp_path, 2, 3.0, SPLIT)
    with pytest.raises(ValueError, match="calibrate は"):
        make_bundle(tmp_path / "x", spec, dataset, weights,
                    calibrate="beta")
    alone = tmp_path / "alone"
    alone.mkdir()
    no_valid, _, _ = write_classes(
        alone, 2, 3.0, {"method": "random", "valid": 0.0, "test": 0.3,
                        "seed": 0}
    )
    with pytest.raises(ValueError, match="検証用の分割"):
        make_bundle(tmp_path / "x", spec, no_valid, weights,
                    calibrate="platt")
    assert not (tmp_path / "x").exists()

    # 回帰の一式
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.normal(size=200)})
    frame["y"] = 2 * frame["x"]
    frame.to_csv(tmp_path / "r.csv", index=False)
    recipe = {"format_version": 1, "target": "y", "task": "regression",
              "steps": [], "split": SPLIT}
    export_dataset(load_data(tmp_path / "r.csv"), recipe, tmp_path / "rd")
    one = linear_spec(1, 1)
    torch.save(build_model(one).state_dict(), tmp_path / "r.pt")
    with pytest.raises(ValueError, match="分類の一式"):
        make_bundle(tmp_path / "x", one, tmp_path / "rd",
                    tmp_path / "r.pt", calibrate="platt")

    # 崩れた calibrate の一式は読まない
    make_bundle(tmp_path / "fixed", spec, dataset, weights,
                calibrate="isotonic")
    file = tmp_path / "fixed" / "inference.yaml"
    original = yaml.safe_load(file.read_text(encoding="utf-8"))
    for change, message in (
        ({"method": "beta"}, "calibrate.method"),
        ({"classes": {"no": {"x": [0.5], "y": [0.5]}}}, "calibrate.classes"),
        ({"classes": {"yes": {"x": [0.1, 0.9], "y": [0.5]}}}, "同じ長さ"),
        ({"extra": 1}, "知らない欄"),
    ):
        broken = {**original, "calibrate": {**original["calibrate"],
                                            **change}}
        file.write_text(yaml.safe_dump(broken, allow_unicode=True),
                        encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_bundle(tmp_path / "fixed")


def test_fitting() -> None:
    """
    Platt が当たりすぎる logit の倍率を戻し, isotonic が順を保つことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 本当の logit の 3 倍の確率
    rng = np.random.default_rng(0)
    z = rng.normal(size=20000)
    hit = rng.uniform(size=20000) < 1 / (1 + np.exp(-z))
    params = fit_platt(1 / (1 + np.exp(-3 * z)), hit)
    assert params["a"] == pytest.approx(1 / 3, abs=0.03)
    assert params["b"] == pytest.approx(0.0, abs=0.05)

    # 階段は小さい順に下がらない
    fitted = fit_isotonic(1 / (1 + np.exp(-3 * z)), hit)
    assert np.all(np.diff(fitted["x"]) >= 0)
    assert np.all(np.diff(fitted["y"]) >= 0)
    assert 0 <= min(fitted["y"]) and max(fitted["y"]) <= 1


def test_isotonic_ties() -> None:
    """
    同じ確率の行が多い場合も, isotonic 回帰の結果が行の順に依らず, 同じ
    確率に 1 つの値を当てることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 3 つの確率だけの行 (同じ確率に当たりと外れが混ざる)
    rng = np.random.default_rng(0)
    probability = rng.choice([0.2, 0.5, 0.8], size=3000)
    hit = rng.uniform(size=3000) < probability
    fitted = fit_isotonic(probability, hit)
    assert fitted["x"] == [0.2, 0.5, 0.8]
    for value in (0.2, 0.5, 0.8):
        inside = probability == value
        got = calibrated(fitted, "isotonic", np.array([value]))[0]
        assert got == pytest.approx(hit[inside].mean())

    # 行を並べ替えても同じ
    order = rng.permutation(3000)
    again = fit_isotonic(probability[order], hit[order])
    assert again == fitted


def test_fit_refuses() -> None:
    """
    出ないクラスがある場合と, Platt 法の傾きが 0 以下になる場合に止まる
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 クラスで 1 クラスだけ, 3 クラスで 1 つ出ないクラス
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200)
    binary = np.stack([1 - p, p], axis=1)
    with pytest.raises(ValueError, match="クラス no の行が無い"):
        fit_calibration("platt", ["no", "yes"], binary, np.ones(200))
    three = rng.dirichlet(np.ones(3), size=200)
    with pytest.raises(ValueError, match="クラス c の行が無い"):
        fit_calibration("isotonic", ["a", "b", "c"], three,
                        rng.integers(0, 2, size=200))

    # 確率と当たりが逆向き
    hit = rng.uniform(size=200) < 1 - p
    with pytest.raises(ValueError, match="傾き a が 0 以下"):
        fit_calibration("platt", ["no", "yes"], binary, hit.astype(int))
    fit_calibration("isotonic", ["no", "yes"], binary, hit.astype(int))
