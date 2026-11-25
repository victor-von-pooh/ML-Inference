import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from ml_eda import Data, describe_groups

from ml_inference import (
    READY_LIBRARIES, describe_bundle, group_representatives, group_summary,
    infer, load_bundle, make_group_bundle, make_ready_bundle, predict,
    read_result, write_result
)
from test_ready import LICENSE, SOURCE, tiny_weights

# 写真の名前, 色 (赤か青), 大きさ (幅, 高さ)
PHOTOS = [
    ("a_red", 0, (20, 12)), ("b_blue", 2, (12, 12)), ("c_red", 0, (16, 16)),
    ("d_blue", 2, (30, 10)), ("e_red", 0, (14, 22)), ("f_blue", 2, (18, 18)),
]

# 骨組みが書く群の安定性の行
STABILITY = {
    "k": 2, "method": "seed", "repeats": 5, "ari": 1.0,
    "ari_baseline": 0.8, "ari_diff": 0.2, "ari_min": 1.0,
    "ari_min_baseline": 0.7, "ari_min_diff": 0.3, "silhouette": 0.9,
    "silhouette_baseline": 0.5, "silhouette_diff": 0.4,
}


def write_photos(folder: Path, broken: bool = True) -> Path:
    """
    赤と青の写真 (大きさは別々) と読めないファイルを書く関数

    Parameters
    ----------
    folder: Path
        書くディレクトリ
    broken: bool = True
        読めないファイルも書くか

    Returns
    ----------
    folder: Path
        書いたディレクトリ
    """
    # 色の濃さを少しずつ変える
    image = pytest.importorskip("PIL.Image")
    folder.mkdir()
    for index, (name, channel, size) in enumerate(PHOTOS):
        array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        array[:, :, channel] = 200 + 10 * index // 2
        image.fromarray(array).save(folder / f"{name}.png")
    if broken:
        (folder / "g_broken.png").write_bytes(b"not an image")

    return folder


def write_artifacts(root: Path, photos: Path, monkeypatch) -> Path:
    """
    骨組みと同じ手順で, 既製の埋め込みの群分けの成果物を書く関数

    埋め込みの一式で写真を埋め込み, 赤と青の平均を群の中心にする

    Parameters
    ----------
    root: Path
        一時ディレクトリ
    photos: Path
        写真のディレクトリ
    monkeypatch: pytest.MonkeyPatch
        出どころの差し替え

    Returns
    ----------
    folder: Path
        成果物のディレクトリ
    """
    # 埋め込みの一式で, 読めた写真だけを埋め込む
    monkeypatch.setitem(READY_LIBRARIES, "test", tiny_weights)
    embedder = root / "embedder"
    info = make_ready_bundle(embedder, "tiny", "V1", library="test",
                             license=LICENSE, form="vector")
    frame = predict(embedder, photos)
    frame = frame[frame["status"] == "ok"].reset_index(drop=True)
    points = np.stack(frame["output"].to_list())
    labels = (points[:, 2] > points[:, 0]).astype(int)
    centers = np.stack([points[labels == n].mean(0) for n in range(2)])

    # 群の中心, 割り当て, 群の安定性と群の要約 (ML-EDA の群の要約)
    folder = root / "grouping"
    (folder / "evaluations").mkdir(parents=True)
    sha256 = info["model"]["pretrained"][0]["sha256"]
    (folder / "group_centers.json").write_text(json.dumps({
        "k": 2,
        "ready": {"library": "test", "model": "tiny", "weights": "V1",
                  "sha256": sha256},
        "centers": centers.tolist(),
    }), encoding="utf-8")
    pd.DataFrame({"path": frame["path"], "group": labels,
                  "split": "train"}).to_csv(
        folder / "group_assignments.csv", index=False
    )
    summary = describe_groups(
        Data(kind="table", frame=frame[["width", "height"]]),
        columns=["width", "height"], labels=labels.tolist()
    )["groups"]
    for row in summary[:-1]:
        row["stability"] = 1.0
    (folder / "evaluations" / "group_stability.json").write_text(
        json.dumps({"rows": [STABILITY], "epoch": None,
                    "baseline": "画素をそのまま k-means で分けた群"}),
        encoding="utf-8"
    )
    (folder / "evaluations" / "group_summary.json").write_text(
        json.dumps({"rows": summary}), encoding="utf-8"
    )

    return folder


@pytest.fixture
def ready_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """
    写真の成果物から既製の埋め込みの群分けの一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        出どころの差し替え

    Returns
    ----------
    found: dict
        bundle (一式), artifacts (成果物), photos (写真), info
    """
    # 群に分けた写真を渡して割り当て直す
    photos = write_photos(tmp_path / "photos")
    artifacts = write_artifacts(tmp_path, photos, monkeypatch)
    bundle = tmp_path / "bundle"
    info = make_group_bundle(bundle, artifacts, data=photos, license=LICENSE)
    found = {"bundle": bundle, "artifacts": artifacts, "photos": photos,
             "info": info}

    return found


def test_make_ready_group_bundle(ready_group: dict) -> None:
    """
    版 10 の一式が, 既製のモデルの出どころと群の中心と骨組みの群の安定性
    を持つことを確かめる関数

    Parameters
    ----------
    ready_group: dict
        既製の埋め込みの群分けの一式

    Returns
    ----------
    None
    """
    # 版 10. 重みは出どころから読み, 群の中心と要約は成果物の写し
    info = ready_group["info"]
    assert info["format_version"] == 10 and info["role"] == "group"
    assert info["output"] == {"form": "cluster"}
    assert info["model"]["pretrained"][0]["place"] == "source"
    assert info["ready"]["model"] == "tiny" and info["run"] is None
    group = info["group"]
    assert group["k"] == 2 and group["width"] == 3
    assert group["columns"] == ["width", "height"]
    assert group["features"] is None and group["id"] == "path"
    assert group["rows"] == 6
    assert group["check"] == {"rows": 6, "agreement": 1.0}
    assert group["stability"]["ari"] == 1.0
    assert group["stability"]["baseline"] == "画素をそのまま k-means で分けた群"
    assert [row["group"] for row in group["summary"]] == [0, 1, "全体"]

    # 説明は群分けの一式で, 出どころと置き場も返す
    description = describe_bundle(ready_group["bundle"])
    assert description["kind"] == "group" and description["task"] == ""
    assert description["group"]["k"] == 2
    assert description["ready"]["weights_file"].endswith("tiny-0123abcd.pth")


def test_predict_ready_group(ready_group: dict, tmp_path: Path) -> None:
    """
    新しい写真が見た目の近い群に入り, 群の要約と代表の写真が返ることを
    確かめる関数

    Parameters
    ----------
    ready_group: dict
        既製の埋め込みの群分けの一式
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 赤は群 0, 青は群 1, 読めない写真は飛ばす
    bundle = ready_group["bundle"]
    frame = predict(bundle, ready_group["photos"])
    rows = frame.set_index("path")
    assert list(rows.loc[["a_red.png", "c_red.png", "e_red.png"],
                         "group"]) == [0, 0, 0]
    assert list(rows.loc[["b_blue.png", "d_blue.png", "f_blue.png"],
                         "group"]) == [1, 1, 1]
    assert rows.loc["g_broken.png", "status"] == "skipped"
    assert "画像を読めません" in rows.loc["g_broken.png", "reason"]
    assert (frame["distance"].dropna() >= 0).all()
    with pytest.raises(ValueError, match="options"):
        predict(bundle, ready_group["photos"], options={"x": 1})

    # 群の要約は写真の幅と高さの平均, 代表の写真は場所も持つ
    summary = group_summary(bundle, frame)
    assert [row["rows"] for row in summary] == [3, 3, 6]
    assert summary[0]["stability"] == 1.0
    assert summary[0]["width"] == pytest.approx((20 + 16 + 14) / 3)
    groups = group_representatives(bundle, frame, count=2)
    first = groups[0]["rows"][0]
    assert set(first["values"]) == {"path", "width", "height"}
    assert first["values"]["path"].endswith("red.png")
    assert len(groups[1]["rows"]) == 2

    # 結論と書き出し. 読み直した表から同じ要約と代表の写真 (距離は文字に
    # した小数の桁の分だけ違ってよい)
    result = infer(bundle, ready_group["photos"])
    assert result["summary"]["predicted"] == 6
    assert result["sentences"][0] == "7 行のうち 6 行を 2 個の群に割り当てた"\
        " (1 行は割り当てられず)"
    out = tmp_path / "result"
    write_result(result, out)
    again = read_result(out)["frame"]
    assert group_summary(bundle, again) == summary
    found = group_representatives(bundle, again, count=2)
    assert [[(r["row"], r["values"]) for r in g["rows"]] for g in found] == [
        [(r["row"], r["values"]) for r in g["rows"]] for g in groups
    ]
    assert [r["distance"] for g in found for r in g["rows"]] == pytest.approx(
        [r["distance"] for g in groups for r in g["rows"]]
    )


def test_ready_group_errors(
    ready_group: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    重みと成果物と一式の食い違いで止まることを確かめる関数

    Parameters
    ----------
    ready_group: dict
        既製の埋め込みの群分けの一式
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        出どころの差し替え

    Returns
    ----------
    None
    """
    # 渡せない引数
    artifacts = ready_group["artifacts"]
    with pytest.raises(ValueError, match="options"):
        make_group_bundle(tmp_path / "x", artifacts, options={"a": 1},
                          license=LICENSE)
    with pytest.raises(ValueError, match="pretrained"):
        make_group_bundle(tmp_path / "x", artifacts, pretrained={},
                          license=LICENSE)

    # 割り当ての表の行を見分ける列が写真の場所でない
    file = artifacts / "group_assignments.csv"
    table = pd.read_csv(file)
    table.rename(columns={"path": "row"}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="path"):
        make_group_bundle(tmp_path / "x", artifacts, license=LICENSE)
    table.to_csv(file, index=False)

    # 群の中心に列の符号化と既製のモデルの両方
    centers_file = artifacts / "group_centers.json"
    centers = json.loads(centers_file.read_text("utf-8"))
    centers_file.write_text(json.dumps({**centers, "encoding": {}}),
                            encoding="utf-8")
    with pytest.raises(ValueError, match="どちらか 1 つ"):
        make_group_bundle(tmp_path / "x", artifacts, license=LICENSE)
    centers_file.write_text(json.dumps(centers), encoding="utf-8")

    # 一式の形の誤り (行を見分ける列, 群の要約の列, ready が無い)
    bundle = ready_group["bundle"]
    file = bundle / "inference.yaml"
    info = yaml.safe_load(file.read_text("utf-8"))
    cases: list[tuple[dict, str]] = [
        ({**info, "group": {**info["group"], "id": "name"}}, "group.id"),
        ({**info, "group": {**info["group"], "columns": ["width"]}},
         "group.columns"),
        ({**info, "ready": None}, "ready"),
        ({**info, "group": {**info["group"], "features": 3}},
         "group.features"),
    ]
    for changed, message in cases:
        file.write_text(yaml.safe_dump(changed, allow_unicode=True),
                        encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_bundle(bundle)
    file.write_text(yaml.safe_dump(info, allow_unicode=True),
                    encoding="utf-8")

    # 一式の群の中心が別のモデルのもの
    inner = bundle / "group_centers.json"
    inner.write_text(json.dumps({**centers, "ready": {
        **centers["ready"], "model": "other"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="群の中心を作った既製のモデル"):
        load_bundle(bundle)
    inner.write_text(json.dumps(centers), encoding="utf-8")

    # 出どころの重みが入れ替わると, 推論も一式づくりも止まる
    monkeypatch.setitem(SOURCE, "scale", 2.0)
    with pytest.raises(ValueError, match="hash が一式と違います"):
        predict(bundle, ready_group["photos"])
    with pytest.raises(ValueError, match="群の中心を作った重みの hash"):
        make_group_bundle(tmp_path / "x", artifacts, license=LICENSE)
    assert not (tmp_path / "x").exists()


def test_ready_group_check_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    群に分けた写真と違う写真を渡すと, 比べずに注意を残すことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        出どころの差し替え

    Returns
    ----------
    None
    """
    # 1 枚少ない写真
    photos = write_photos(tmp_path / "photos")
    artifacts = write_artifacts(tmp_path, photos, monkeypatch)
    (photos / "f_blue.png").unlink()
    info = make_group_bundle(tmp_path / "bundle", artifacts, data=photos,
                             license=LICENSE)
    assert info["group"]["check"] is None
    assert any("群に分けた写真の行の数 5" in n for n in info["notes"])
