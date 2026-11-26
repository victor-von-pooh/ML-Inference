import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from ml_inference import (
    READY_LIBRARIES, decide, describe_bundle, infer, load_bundle,
    make_ready_bundle, make_search_bundle, predict, read_result, write_result
)
from test_ready import LICENSE, SOURCE, tiny_weights

# 索引の写真の種類, 名前, 色 (赤, 緑, 青の濃さ)
PHOTOS = [
    ("red", "r1", (220, 20, 20)), ("red", "r2", (200, 40, 30)),
    ("red", "r3", (180, 30, 50)), ("blue", "b1", (20, 20, 220)),
    ("blue", "b2", (40, 30, 200)), ("green", "g1", (20, 210, 30)),
]

# 骨組みが書く比べる相手の名前と評価の行
BASELINE = "写真を縮めた画素をそのまま余弦の近さで並べたもの"
EVALUATION = {
    "k": 2, "queries": 4, "recall_at_k": 0.75, "ndcg_at_k": 0.7,
    "recall_at_k_baseline": 0.5, "ndcg_at_k_baseline": 0.45,
    "recall_at_k_diff": 0.25, "ndcg_at_k_diff": 0.25,
}


def write_image(path: Path, color: tuple) -> None:
    """
    1 色の写真を書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    color: tuple
        赤, 緑, 青の濃さ

    Returns
    ----------
    None
    """
    # 8 × 8 の 1 色
    image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    array[:, :] = color
    image.fromarray(array).save(path)


def write_artifacts(
    root: Path, monkeypatch: pytest.MonkeyPatch, labeled: bool = True
) -> dict:
    """
    骨組みと同じ手順で, 似た写真を探す成果物を書く関数

    Parameters
    ----------
    root: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        出どころの差し替え
    labeled: bool = True
        写真を種類のサブディレクトリに置くか

    Returns
    ----------
    found: dict
        photos (索引の写真), artifacts (成果物), sha256 (重みの hash)
    """
    # 索引の写真と, 埋め込みの一式での埋め込み
    monkeypatch.setitem(READY_LIBRARIES, "test", tiny_weights)
    photos = root / "photos"
    for label, name, color in PHOTOS:
        write_image(photos / (f"{label}/{name}.png" if labeled
                              else f"{name}.png"), color)
    embedder = root / "embedder"
    info = make_ready_bundle(embedder, "tiny", "V1", library="test",
                             license=LICENSE, form="vector")
    frame = predict(embedder, photos)
    points = np.stack(frame["output"].to_list()).astype(np.float32)

    # 索引の説明, 埋め込み, 評価 (種類がある場合だけ)
    artifacts = root / "artifacts"
    (artifacts / "evaluations").mkdir(parents=True)
    sha256 = info["model"]["pretrained"][0]["sha256"]
    items = [{"path": row["path"],
              "label": row["path"].split("/")[0] if labeled else None,
              "sha256": row["digest"]} for _, row in frame.iterrows()]
    (artifacts / "search_index.json").write_text(json.dumps({
        "ready": {"library": "test", "model": "tiny", "weights": "V1",
                  "sha256": sha256},
        "width": 3, "k": 2, "max_latency_ms": 10000.0,
        "root": str(photos), "items": items,
    }), encoding="utf-8")
    np.save(artifacts / "search_embeddings.npy", points)
    if labeled:
        (artifacts / "evaluations" / "search_valid.json").write_text(
            json.dumps({"rows": [EVALUATION], "baseline": BASELINE}),
            encoding="utf-8"
        )
    found = {"photos": photos, "artifacts": artifacts, "sha256": sha256}

    return found


@pytest.fixture
def search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """
    成果物から検索の一式を作り, 問いの写真を書く関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        出どころの差し替え

    Returns
    ----------
    found: dict
        bundle, artifacts, photos, queries, info
    """
    # 問いは赤い写真, 索引の写真と同じ写真, 読めない写真
    found = write_artifacts(tmp_path, monkeypatch)
    queries = tmp_path / "queries"
    write_image(queries / "q_red.png", (210, 30, 25))
    shutil.copy(found["photos"] / "blue" / "b1.png", queries / "q_same.png")
    (queries / "q_broken.png").write_bytes(b"not an image")
    bundle = tmp_path / "bundle"
    info = make_search_bundle(bundle, found["artifacts"], license=LICENSE)
    found.update(bundle=bundle, queries=queries, info=info)

    return found


def test_make_search_bundle(search: dict) -> None:
    """
    版 15 の一式が, 既製のモデルの出どころと索引と骨組みの評価を持つことを
    確かめる関数

    Parameters
    ----------
    search: dict
        検索の一式

    Returns
    ----------
    None
    """
    # 版 15. 重みは出どころから読み, 索引と評価は成果物の写し
    info = search["info"]
    assert info["format_version"] == 15 and info["role"] == "rank"
    assert info["output"] == {"form": "ranking"}
    assert info["model"]["pretrained"][0]["place"] == "source"
    assert info["decision"] == {"type": "search", "k": 2}
    section = info["search"]
    assert section["items"] == 6 and section["width"] == 3
    assert section["k"] == 2 and section["max_latency_ms"] == 10000.0
    assert section["labeled"] is True
    assert section["root"] == str(search["photos"])
    assert section["evaluation"]["recall_at_k"] == 0.75
    assert section["evaluation"]["baseline"] == BASELINE
    assert (search["bundle"] / "search_embeddings.npy").is_file()

    # 説明 (ファイルの名前を除いた写し)
    description = describe_bundle(search["bundle"])
    assert description["kind"] == "search" and description["task"] == ""
    assert description["search"]["items"] == 6
    assert "index" not in description["search"]
    assert description["ready"]["model"] == "tiny"
    assert description["rank"] is None and description["group"] is None


def test_predict_search(search: dict) -> None:
    """
    問いの写真ごとに, 近い順の上位 K 件と待ち時間が返ることを確かめる関数

    Parameters
    ----------
    search: dict
        検索の一式

    Returns
    ----------
    None
    """
    # 赤い問いは赤い写真が上位, 同じ写真は除き, 読めない写真は飛ばす
    frame = predict(search["bundle"], search["queries"])
    assert frame.attrs["k"] == 2
    red = frame[frame["path"] == "q_red.png"]
    assert list(red["rank"]) == [1, 2]
    assert list(red["label"]) == ["red", "red"]
    assert red["similarity"].is_monotonic_decreasing
    assert (red["similarity"] <= 1.0 + 1e-9).all()
    assert red["latency_ms"].nunique() == 1 and red["latency_ms"].iloc[0] > 0
    same = frame[frame["path"] == "q_same.png"]
    assert "blue/b1.png" not in list(same["item"])
    assert list(same["item"])[0] == "blue/b2.png"
    broken = frame[frame["path"] == "q_broken.png"]
    assert len(broken) == 1 and broken["status"].iloc[0] == "skipped"
    assert "画像を読めません" in broken["reason"].iloc[0]
    assert pd.isna(broken["rank"].iloc[0])

    # 件数を変える, 索引より多い件数, 誤り
    more = predict(search["bundle"], search["queries"], k=4)
    assert len(more[more["path"] == "q_red.png"]) == 4
    every = predict(search["bundle"], search["queries"], k=50)
    assert len(every[every["path"] == "q_red.png"]) == 6
    assert len(every[every["path"] == "q_same.png"]) == 5
    with pytest.raises(ValueError, match="k は 1 以上"):
        predict(search["bundle"], search["queries"], k=0)
    with pytest.raises(ValueError, match="options"):
        predict(search["bundle"], search["queries"], options={"a": 1})
    with pytest.raises(ValueError, match="seen"):
        predict(search["bundle"], search["queries"], seen=pd.DataFrame())


def test_infer_and_decide_search(search: dict, tmp_path: Path) -> None:
    """
    結論, 既定の検索の判断, 書き出しと読み直しを確かめる関数

    Parameters
    ----------
    search: dict
        検索の一式

    Returns
    ----------
    None
    """
    # 既定の判断 (k = 2) で並べ, 件数と待ち時間と評価の文
    result = infer(search["bundle"], search["queries"])
    summary = result["summary"]
    assert summary["rows"] == 3 and summary["predicted"] == 2
    assert summary["skipped"] == 1 and summary["k"] == 2
    assert summary["over_latency"] == 0
    assert summary["latency"]["max"] >= summary["latency"]["median"] > 0
    assert result["sentences"][0] == (
        "問いの写真 3 枚のうち 2 枚に, 索引の写真 6 枚から近い順に上位 2 件"
        "を並べた (1 枚は並べられず)"
    )
    assert any("Recall@K 0.750" in s and BASELINE in s
               for s in result["sentences"])
    decided = result["decision"]
    assert decided["summary"]["queries"] == 2
    assert decided["summary"]["returned"] == 4
    assert any("見た目が近い" in s for s in decided["notes"])
    assert result["frame"]["action"].dropna().all()

    # 判断の k で並べ直す, 渡した k と違えば誤り, 空の辞書で決めない
    wide = infer(search["bundle"], search["queries"],
                 decision={"type": "search", "k": 3})
    assert wide["summary"]["k"] == 3 and wide["decision"]["summary"][
        "returned"] == 6
    with pytest.raises(ValueError, match="検索の判断の k"):
        infer(search["bundle"], search["queries"], k=4)
    plain = infer(search["bundle"], search["queries"], decision={})
    assert "decision" not in plain and "action" not in plain["frame"]

    # 表から少ない件数を決める, 多い件数と知らない型は誤り
    frame = predict(search["bundle"], search["queries"], k=3)
    fewer = decide(search["bundle"], frame, {"type": "search", "k": 1})
    red = fewer["frame"][fewer["frame"]["path"] == "q_red.png"]
    assert list(red["action"]) == [True, False, False]
    with pytest.raises(ValueError, match="並べた件数"):
        decide(search["bundle"], frame, {"type": "search", "k": 5})
    with pytest.raises(ValueError, match="type: search"):
        decide(search["bundle"], frame, {"type": "order", "k": 1})
    with pytest.raises(ValueError, match="書けない項目"):
        decide(search["bundle"], frame, {"type": "search", "k": 1, "x": 1})
    with pytest.raises(ValueError, match="検索の判断"):
        infer(search["bundle"], search["queries"], rules={"a": 1})

    # 書き出しと読み直し (順位は欠損を持てる整数)
    out = tmp_path / "result"
    write_result(result, out)
    again = read_result(out)["frame"]
    assert list(again["rank"].dropna()) == list(result["frame"]["rank"]
                                                .dropna())
    assert list(again["item"].dropna()) == list(result["frame"]["item"]
                                                .dropna())


def test_search_over_latency(search: dict) -> None:
    """
    待ち時間の上限を超えた問いを数えて知らせることを確かめる関数

    Parameters
    ----------
    search: dict
        検索の一式

    Returns
    ----------
    None
    """
    # 上限をとても小さくした一式
    bundle = search["bundle"]
    file = bundle / "inference.yaml"
    info = yaml.safe_load(file.read_text("utf-8"))
    info["search"]["max_latency_ms"] = 1e-9
    file.write_text(yaml.safe_dump(info, allow_unicode=True),
                    encoding="utf-8")
    result = infer(bundle, search["queries"])
    assert result["summary"]["over_latency"] == 2
    assert any("上限 1e-09 ms を超えた問いは 2 枚" in s
               for s in result["sentences"])
    assert any("動かす場所を確かめる" in s
               for s in result["decision"]["sentences"])


def test_search_without_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    種類の無い索引は評価を持たず, 評価が無い旨の文を返すことを確かめる関数

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
    # 種類の無い写真の成果物
    found = write_artifacts(tmp_path, monkeypatch, labeled=False)
    info = make_search_bundle(tmp_path / "bundle", found["artifacts"],
                              license=LICENSE)
    assert info["search"]["labeled"] is False
    assert info["search"]["evaluation"] is None
    result = infer(tmp_path / "bundle", found["photos"] / "r1.png")
    assert any("種類が無いので評価は無い" in s for s in result["sentences"])
    assert result["summary"]["predicted"] == 1
    assert "r1.png" not in list(result["frame"]["item"])

    # 種類の無い索引に評価があれば作らない
    (found["artifacts"] / "evaluations" / "search_valid.json").write_text(
        json.dumps({"rows": [EVALUATION]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="種類の無い索引に評価"):
        make_search_bundle(tmp_path / "x", found["artifacts"],
                           license=LICENSE)


def test_search_errors(
    search: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    成果物と一式の食い違いと, 重みの入れ替わりで止まることを確かめる関数

    Parameters
    ----------
    search: dict
        検索の一式
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        出どころの差し替え

    Returns
    ----------
    None
    """
    # 成果物の誤り (索引の形, 埋め込みの形, 評価の k)
    artifacts = search["artifacts"]
    index_file = artifacts / "search_index.json"
    index = json.loads(index_file.read_text("utf-8"))
    cases: list[tuple[dict, str]] = [
        ({**index, "items": []}, "1 枚以上"),
        ({**index, "items": [*index["items"], index["items"][0]]},
         "重なっています"),
        ({**index, "items": [{**index["items"][0], "label": None},
                             *index["items"][1:]]}, "全ての写真"),
        ({**index, "k": 0}, "k は 1 以上"),
        ({**index, "max_latency_ms": -1}, "max_latency_ms"),
        ({**index, "extra": 1}, "知らない項目"),
        ({**index, "ready": {**index["ready"], "sha256": "x"}}, "sha256"),
    ]
    for changed, message in cases:
        index_file.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            make_search_bundle(tmp_path / "x", artifacts, license=LICENSE)
    index_file.write_text(json.dumps({**index, "width": 4}),
                          encoding="utf-8")
    with pytest.raises(ValueError, match="形"):
        make_search_bundle(tmp_path / "x", artifacts, license=LICENSE)
    index_file.write_text(json.dumps(index), encoding="utf-8")
    evaluation_file = artifacts / "evaluations" / "search_valid.json"
    evaluation_file.write_text(json.dumps(
        {"rows": [{**EVALUATION, "k": 3}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="索引の k"):
        make_search_bundle(tmp_path / "x", artifacts, license=LICENSE)
    evaluation_file.unlink()
    with pytest.raises(FileNotFoundError):
        make_search_bundle(tmp_path / "x", artifacts, license=LICENSE)
    evaluation_file.write_text(json.dumps(
        {"rows": [EVALUATION], "baseline": BASELINE}), encoding="utf-8")
    assert not (tmp_path / "x").exists()

    # 一式の形の誤り
    bundle = search["bundle"]
    file = bundle / "inference.yaml"
    info = yaml.safe_load(file.read_text("utf-8"))
    cases = [
        ({**info, "ready": None}, "ready"),
        ({**info, "format_version": 8}, "search を持つ一式は版 15"),
        ({**info, "search": {**info["search"], "k": 0}}, "search.k"),
        ({**info, "search": {**info["search"], "evaluation": None}},
         "search.evaluation"),
        ({**info, "decision": {"type": "order"}}, "検索の判断"),
        ({**info, "rank": {"k": 1}}, "rank"),
        ({**info, "search": {**info["search"], "items": 5}}, "写真の数と幅"),
    ]
    for changed, message in cases:
        file.write_text(yaml.safe_dump(changed, allow_unicode=True),
                        encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_bundle(bundle)
    file.write_text(yaml.safe_dump(info, allow_unicode=True),
                    encoding="utf-8")

    # 一式の索引が別のモデルのもの
    inner = bundle / "search_index.json"
    inner.write_text(json.dumps({**index, "ready": {
        **index["ready"], "model": "other"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="索引を作った既製のモデル"):
        load_bundle(bundle)
    inner.write_text(json.dumps(index), encoding="utf-8")

    # 出どころの重みの入れ替わりで止まる推論と一式づくり
    monkeypatch.setitem(SOURCE, "scale", 2.0)
    with pytest.raises(ValueError, match="hash が一式と違います"):
        predict(bundle, search["queries"])
    with pytest.raises(ValueError, match="索引を作った重みの hash"):
        make_search_bundle(tmp_path / "x", artifacts, license=LICENSE)
    assert not (tmp_path / "x").exists()
