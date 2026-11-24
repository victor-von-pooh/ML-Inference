import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from ml_modelcontrol import build_model, load_example

import ml_inference.bundle as bundle_module
from ml_inference import (
    describe_bundle, infer, load_bundle, make_bundle, make_rank_bundle,
    predict, rank_metrics, read_result, write_result
)

# 番号の表の利用者と品物 (文字と数を混ぜる)
USERS = ["u0", "u1", "u2", 103, "u4"]
ITEMS = [f"i{n}" for n in range(7)]


def two_tower_spec(users: int, items: int) -> dict:
    """
    利用者と品物の数に合わせた 2 塔のモデルの定義を返す関数

    Parameters
    ----------
    users: int
        利用者の数
    items: int
        品物の数

    Returns
    ----------
    spec: dict
        ML-ModelControl の同梱の例 two_tower_retrieval の大きさを変えたもの
    """
    # 入力の形と 1 段目の全結合の入力の幅
    spec = load_example("two_tower_retrieval")
    spec["inputs"]["user"]["shape"] = [4, users]
    spec["inputs"]["item"]["shape"] = [4, items]
    spec["nodes"]["user_fc1"]["params"]["in_features"] = users
    spec["nodes"]["item_fc1"]["params"]["in_features"] = items

    return spec


def write_run(root: Path, spec: dict, seed: int = 0) -> Path:
    """
    並べるの骨組みと同じ 3 つのファイルを持つ実験のディレクトリを作る関数

    Parameters
    ----------
    root: Path
        実験のディレクトリ
    spec: dict
        モデルの定義
    seed: int = 0
        重みの種

    Returns
    ----------
    root: Path
        実験のディレクトリ
    """
    # 重み, 定義, 番号の表, 実験の状態
    root.mkdir(parents=True)
    model = build_model(spec, seed=seed, load_weights=False)
    torch.save(model.state_dict(), root / "final_model.pt")
    (root / "model_spec.yaml").write_text(
        yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8"
    )
    (root / "rank_index.json").write_text(
        json.dumps({"users": USERS, "items": ITEMS}), encoding="utf-8"
    )
    (root / "run.json").write_text(
        json.dumps({"format_version": 1, "status": "completed"}),
        encoding="utf-8"
    )

    return root


def expected_scores(run: Path) -> np.ndarray:
    """
    実験の重みのモデルの, 全ての利用者と品物の点数を返す関数

    Parameters
    ----------
    run: Path
        実験のディレクトリ

    Returns
    ----------
    scores: np.ndarray
        (利用者の数, 品物の数) の score の出力
    """
    # 全ての利用者と全ての品物のワンホットを一度に渡す
    spec = yaml.safe_load((run / "model_spec.yaml").read_text())
    model = build_model(spec, seed=0, load_weights=False)
    model.load_state_dict(torch.load(run / "final_model.pt"))
    model.eval()
    with torch.no_grad():
        out = model(user=torch.eye(len(USERS)), item=torch.eye(len(ITEMS)))
    scores = out["score"].double().numpy()

    return scores


@pytest.fixture
def rank_run(tmp_path: Path) -> Path:
    """
    並べるの実験のディレクトリを作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    run: Path
        実験のディレクトリ
    """
    # 利用者 5 人と品物 7 つの 2 塔
    run = write_run(tmp_path / "run", two_tower_spec(len(USERS), len(ITEMS)))

    return run


def test_make_rank_bundle(tmp_path: Path, rank_run: Path) -> None:
    """
    実験から並べるの一式を作り, 索引と倍率と説明を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    rank_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 版 8 で, 番号の表と品物の埋め込みを一式に持つ
    info = make_rank_bundle(tmp_path / "bundle", rank_run, k=3)
    root = tmp_path / "bundle"
    assert info["format_version"] == 8
    assert info["role"] == "rank" and info["output"] == {"form": "ranking"}
    assert info["dataset"] is None and info["run"]["status"] == "completed"
    rank = info["rank"]
    assert (rank["users"], rank["items"], rank["width"], rank["k"]) == (
        5, 7, 16, 3
    )
    assert rank["columns"] == {"user": "user", "item": "item",
                               "reaction": None}
    assert rank["scale"] == pytest.approx(20.0, rel=1e-5)
    embeddings = np.load(root / "item_embeddings.npy")
    assert embeddings.shape == (7, 16) and embeddings.dtype == np.float32
    assert json.loads((root / "rank_index.json").read_text()) == {
        "users": USERS, "items": ITEMS
    }

    # 説明
    description = describe_bundle(root)
    assert description["kind"] == "rank"
    assert description["format_version"] == 8
    assert description["rank"]["k"] == 3
    assert description["column_output"] is False


def test_predict_rank(tmp_path: Path, rank_run: Path) -> None:
    """
    利用者ごとの上位 K 件と点数がモデルの点数の順と同じで, 知らない利用者
    を理由付きで断ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    rank_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 知っている利用者 (数の利用者は文字でも引ける), 知らない利用者, 空
    make_rank_bundle(tmp_path / "bundle", rank_run, k=3)
    frame = predict(tmp_path / "bundle", ["u2", "103", "x", None],
                    batch_size=1)
    scores = expected_scores(rank_run)
    for number, user in ((2, "u2"), (3, "103")):
        rows = frame[frame["user"] == user]
        order = np.argsort(-scores[number], kind="stable")[:3]
        assert list(rows["item"]) == [ITEMS[i] for i in order]
        assert list(rows["rank"]) == [1, 2, 3]
        assert np.allclose(rows["score"], scores[number][order], atol=1e-4)
        assert set(rows["status"]) == {"ok"}
    skipped = frame[frame["status"] == "skipped"]
    assert len(skipped) == 2
    assert "番号の表に無い" in skipped["reason"].iloc[0]
    assert skipped["reason"].iloc[1] == "利用者が空です"
    assert skipped["rank"].isna().all() and skipped["score"].isna().all()
    assert list(frame.columns) == ["user", "status", "reason", "rank",
                                   "item", "score"]

    # 引数の k は一式の既定に勝ち, 品物の数より大きければ全ての品物
    assert len(predict(tmp_path / "bundle", ["u0"], k=10)) == 7
    with pytest.raises(ValueError, match="k は 1 以上"):
        predict(tmp_path / "bundle", ["u0"], k=0)


def test_exclude_seen(tmp_path: Path, rank_run: Path) -> None:
    """
    前に反応した品物の表を渡したときだけ, その品物を除いて並べることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    rank_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 問題の設定の列の名前と反応の強さ (0 以下は反応でない)
    make_rank_bundle(tmp_path / "bundle", rank_run, k=7,
                     columns={"user": "member", "item": "sku",
                              "reaction": "qty"})
    scores = expected_scores(rank_run)
    order = [ITEMS[i] for i in np.argsort(-scores[0], kind="stable")]
    seen = pd.DataFrame({"member": ["u0", "u0", "u0", "u1"],
                         "sku": [order[0], order[1], "知らない", order[2]],
                         "qty": [1, 0, 2, 1]})
    users = pd.DataFrame({"member": ["u0"]})
    kept = predict(tmp_path / "bundle", users)
    assert list(kept["item"]) == order
    dropped = predict(tmp_path / "bundle", users, seen=seen)
    assert list(dropped["item"]) == order[1:]

    # ファイルでも渡せ, 列が無ければ誤り
    seen.to_csv(tmp_path / "seen.csv", index=False)
    assert list(predict(tmp_path / "bundle", users,
                        seen=tmp_path / "seen.csv")["item"]) == order[1:]
    with pytest.raises(ValueError, match="列 qty"):
        predict(tmp_path / "bundle", users, seen=seen.drop(columns="qty"))

    # 全ての品物を除くと理由付きで断る
    every = pd.DataFrame({"member": "u0", "sku": ITEMS, "qty": 1})
    nothing = predict(tmp_path / "bundle", users, seen=every)
    assert list(nothing["status"]) == ["skipped"]
    assert "残りません" in nothing["reason"].iloc[0]


def test_infer_and_result(tmp_path: Path, rank_run: Path) -> None:
    """
    並べるの結論が利用者の人数を数え, 結果の版 1 で書いて読めることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    rank_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 2 人を並べ, 1 人は知らない
    make_rank_bundle(tmp_path / "bundle", rank_run, k=2)
    seen = pd.DataFrame({"user": ["u0"], "item": ["i0"]})
    result = infer(tmp_path / "bundle", ["u0", "u1", "x"], seen=seen)
    assert result["form"] == "ranking"
    assert result["source"] == "利用者 3 人"
    assert result["summary"] == {"rows": 3, "predicted": 2, "skipped": 1,
                                 "k": 2, "excluded": True}
    assert "前に反応した品物は除いて並べた" in result["sentences"]
    assert list(result["frame"]["decision"]) == ["", "", "", "",
                                                 "予測なし"]

    # 書き出しは表だけなので結果の版 1
    root = write_result(result, tmp_path / "result")
    meta = yaml.safe_load((root / "result.yaml").read_text())
    assert meta["format_version"] == 1
    assert not (root / "artifacts").exists()
    back = read_result(root)
    assert list(back["frame"]["item"].dropna()) == list(
        result["frame"]["item"].dropna()
    )

    # 他の一式の引数は使えない
    with pytest.raises(ValueError, match="結論の決め方"):
        infer(tmp_path / "bundle", ["u0"], rules={"min_confidence": 0.5})


def test_rank_metrics(tmp_path: Path, rank_run: Path) -> None:
    """
    Recall@K と NDCG が骨組みと同じ数え方になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    rank_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # u0 は上位 2 件の 2 位だけが正解 (正解は 2 つ), u1 は正解が無い
    make_rank_bundle(tmp_path / "bundle", rank_run, k=2)
    frame = predict(tmp_path / "bundle", ["u0", "u1", "u0"])
    top = list(frame[frame["user"] == "u0"]["item"])[:2]
    other = [i for i in ITEMS if i not in top][0]
    truth = pd.DataFrame({"user": ["u0", "u0", "u9"],
                          "item": [top[1], other, "i0"]})
    metrics = rank_metrics(tmp_path / "bundle", frame, truth)
    ideal = 1 + 1 / math.log2(3)
    assert metrics == {"users": 1, "k": 2, "recall_at_k": 0.5,
                       "ndcg": pytest.approx((1 / math.log2(3)) / ideal)}

    # 前に反応した品物は正解から除く
    seen = pd.DataFrame({"user": ["u0"], "item": [other]})
    metrics = rank_metrics(tmp_path / "bundle", frame, truth, seen=seen)
    assert metrics["recall_at_k"] == 1.0

    # 数える利用者がいなければ None
    empty = rank_metrics(tmp_path / "bundle", frame,
                         pd.DataFrame({"user": ["u1"], "item": ["x"]}))
    assert empty == {"users": 0, "k": 2, "recall_at_k": None, "ndcg": None}


def test_contract_errors(tmp_path: Path, rank_run: Path) -> None:
    """
    ワンホットの入力の約束に合わない定義と番号の表を, 作る前に断ることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    rank_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 番号の表と入力の幅が違う
    spec = two_tower_spec(len(USERS), len(ITEMS))
    wide = two_tower_spec(len(USERS) + 1, len(ITEMS))
    with pytest.raises(ValueError, match="番号の表の数 5"):
        make_rank_bundle(tmp_path / "a", rank_run, spec=wide)

    # 埋め込みの出力が無い
    plain = two_tower_spec(len(USERS), len(ITEMS))
    del plain["outputs"]["item_embedding"]
    with pytest.raises(ValueError, match="item_embedding が要ります"):
        make_rank_bundle(tmp_path / "b", rank_run, spec=plain)

    # 点数が内積の定数倍でない (sigmoid を通す)
    bent = two_tower_spec(len(USERS), len(ITEMS))
    bent["nodes"]["bend"] = {"unit": "nn.Sigmoid", "inputs": "score"}
    bent["outputs"]["score"] = "bend"
    with pytest.raises(ValueError, match="定数倍"):
        make_rank_bundle(tmp_path / "c", rank_run, spec=bent)

    # 重みのファイルだけでは定義と番号の表が要り, 仮のモデルの実験は断る
    with pytest.raises(ValueError, match="番号の表"):
        make_rank_bundle(tmp_path / "d", rank_run / "final_model.pt",
                         spec=spec)
    (rank_run / "model_spec.yaml").unlink()
    with pytest.raises(ValueError, match="仮のモデル"):
        make_rank_bundle(tmp_path / "e", rank_run)

    # 作れなかったときは何も残さない
    assert not any(p.name.startswith((".a", ".b", ".c"))
                   for p in tmp_path.iterdir())


def test_index_errors(tmp_path: Path, rank_run: Path) -> None:
    """
    番号の表の形の誤りを断ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    rank_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 文字にすると重なる値
    (rank_run / "rank_index.json").write_text(
        json.dumps({"users": [1, "1", 2, 3, 4], "items": ITEMS})
    )
    with pytest.raises(ValueError, match="同じになる値"):
        make_rank_bundle(tmp_path / "a", rank_run)

    # 無い
    (rank_run / "rank_index.json").unlink()
    with pytest.raises(FileNotFoundError, match="番号の表"):
        make_rank_bundle(tmp_path / "b", rank_run)


def test_rank_bundle_format(
    tmp_path: Path, rank_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    版 8 の決まり (rank と出力の形 ranking の組, 版 7 までの読む側) を
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    rank_run: Path
        実験のディレクトリ
    monkeypatch: pytest.MonkeyPatch
        版を差し替える

    Returns
    ----------
    None
    """
    # 版 7 までしか知らない読む側は止まる
    root = tmp_path / "bundle"
    make_rank_bundle(root, rank_run)
    monkeypatch.setattr(bundle_module, "BUNDLE_FORMAT_VERSION", 7)
    with pytest.raises(ValueError, match="format_version 8 の一式"):
        load_bundle(root)
    monkeypatch.undo()

    # rank と版と出力の形の食い違い, 索引の大きさの違い
    info = yaml.safe_load((root / "inference.yaml").read_text())
    for change, message in (
        ({"format_version": 7}, "rank を持つ一式は版 8"),
        ({"output": {"form": "vector"}}, "出力の形 ranking"),
        ({"decision": {"type": "triage"}}, "null にしてください"),
        ({"rank": {**info["rank"], "k": 0}}, "rank.k"),
    ):
        (root / "inference.yaml").write_text(
            yaml.safe_dump({**info, **change}, allow_unicode=True)
        )
        with pytest.raises(ValueError, match=message):
            load_bundle(root)
    (root / "inference.yaml").write_text(
        yaml.safe_dump(info, allow_unicode=True)
    )
    np.save(root / "item_embeddings.npy", np.zeros((3, 16), np.float32))
    with pytest.raises(ValueError, match="索引の形"):
        load_bundle(root)

    # 並べるの出力の形は make_bundle では作らず, 他の一式に k は渡せない
    with pytest.raises(ValueError, match="make_rank_bundle"):
        make_bundle(tmp_path / "x", two_tower_spec(5, 7), None,
                    rank_run / "final_model.pt", form="ranking",
                    procedure={"name": "ddim", "steps": 1, "seed": 0})
