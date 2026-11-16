from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_decision import classifier_bundle, regression_bundle

from ml_inference import (
    decide, describe_bundle, load_bundle, predict, remeasure,
    restore_uncertainty
)


def test_remeasure_regression(tmp_path: Path) -> None:
    """
    外れ方の広がった実績で測り直すと, 前の記録の当たり具合を返し, 一式の
    外れ方と量の判断が変わることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 誤りが比で ±40 % に広がった実績
    path = regression_bundle(tmp_path)
    rng = np.random.default_rng(5)
    x = rng.uniform(1, 10, 300)
    recent = pd.DataFrame({"x": x, "y": 10 * x * rng.uniform(0.6, 1.4, 300)})
    bundle = load_bundle(path)
    decision = {"type": "quantity", "under_cost": 9, "over_cost": 1}
    before = decide(bundle, predict(bundle, recent), decision)

    # 書き戻さない場合の一式 (変わらない)
    checked = remeasure(path, recent, write=False)
    assert checked["rows"] == 300 and checked["written"] is False
    assert checked["before"]["coverage"]["80"] < 0.75
    assert "外れ方を小さく見積もっていた" in checked["sentences"][0]
    assert load_bundle(path).info["uncertainty"]["split"] == "valid"

    # 書き戻した後の実績の分割の記録と, 9 : 1 の量の増え
    result = remeasure(path, recent)
    saved = load_bundle(path).info["uncertainty"]
    assert saved["split"] == "recent" and saved["rows"] == 300
    assert saved["previous"]["split"] == "valid"
    assert "test_below" not in saved and "coverage" not in saved
    assert result["uncertainty"]["kind"] == saved["kind"]
    bundle = load_bundle(path)
    after = decide(bundle, predict(bundle, recent), decision)
    assert after["frame"]["level"].sum() > before["frame"]["level"].sum()
    assert any("300 行で測り直した" in note for note in after["notes"])
    summary = describe_bundle(path)["uncertainty"]
    assert summary["split"] == "recent" and summary["rows"] == 300
    assert summary["measured_at"] and "absolute" not in summary
    assert summary["restorable"] is True

    # 2 回測り直しても, 戻すのは一式を作ったときの記録
    first = load_bundle(path).info["uncertainty"]["original"]
    remeasure(path, recent.head(100))
    assert load_bundle(path).info["uncertainty"]["original"] == first
    restored = restore_uncertainty(path)
    assert restored["split"] == "valid" and "test_below" in restored
    assert load_bundle(path).info["uncertainty"] == first
    with pytest.raises(ValueError, match="戻せる記録"):
        restore_uncertainty(path)


def test_remeasure_count_and_errors(tmp_path: Path) -> None:
    """
    数える行の印と, 測れない場合の誤りを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 印のある行と目的の値の分かる行だけの測り直し
    path = regression_bundle(tmp_path)
    x = np.linspace(1, 10, 50)
    recent = pd.DataFrame({"x": x, "y": 10 * x})
    recent.loc[:4, "y"] = np.nan
    count = np.arange(50) >= 10
    result = remeasure(path, recent, count=count, write=False)
    assert result["rows"] == 40

    # 少なすぎる行と, 目的の列の無いデータ
    with pytest.raises(ValueError, match="20 行以上"):
        remeasure(path, recent.head(15), write=False)
    with pytest.raises(ValueError, match="目的の列"):
        remeasure(path, recent.drop(columns="y"), write=False)
    with pytest.raises(ValueError, match="印の数"):
        remeasure(path, recent, count=count[:10], write=False)

    # 比の形の記録では, 予測が 0 以下の行を確かめから除く
    negative = pd.DataFrame({"x": np.linspace(-5, 5, 60),
                             "y": np.linspace(-50, 50, 60)})
    checked = remeasure(path, negative, write=False)
    assert checked["before"]["excluded"] == 30
    assert any("0 以下の 30 行" in s for s in checked["sentences"])


def test_remeasure_classification(tmp_path: Path) -> None:
    """
    分類で, 確率の当たり具合を測り直し, 前の記録のずれと比べることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 当たりの確率が逆向きになった実績 (確率のずれが大きい)
    path = classifier_bundle(tmp_path)
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    label = np.where(rng.uniform(size=400) < 1 / (1 + np.exp(2 * x)),
                     "yes", "no")
    result = remeasure(path, pd.DataFrame({"x": x, "label": label}))
    saved = load_bundle(path).info["uncertainty"]
    assert saved["split"] == "recent" and saved["calibration_of"] == "yes"
    assert saved["calibration"]["ece"] > 0.2
    assert result["before"]["ece"] < saved["calibration"]["ece"]
    assert "記録したときは" in result["sentences"][0]
    assert result["before"]["dropped"] == 0
    bundle = load_bundle(path)
    triage = {"type": "triage", "positive": "yes", "miss_cost": 3,
              "false_alarm_cost": 1}
    notes = decide(bundle, predict(bundle, pd.DataFrame({"x": x})),
                   triage)["notes"]
    assert any("400 行で測り直した値" in note for note in notes)
