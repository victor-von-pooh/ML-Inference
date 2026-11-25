import datetime as dt
import json
from pathlib import Path
import shutil
import threading

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from ml_eda import export_dataset, load_data, load_dataset
from ml_modelcontrol import build_model, load_example

import ml_inference.bundle as bundle_module
from ml_inference.anomaly import read_scorer
from ml_inference import (
    backtest, check_decision, decide, describe_bundle, infer, load_bundle,
    make_anomaly_bundle, predict, read_result, read_reviews, record_review,
    write_result
)

# 正常の行と異常の行の数, 人が 1 回に確かめる件数の上限
NORMAL = 60
ANOMALIES = 4
CAPACITY = 3

# 度合いに使う列
FEATURES = ["temp", "vibration", "current"]

# ML-Quest の異常の検知の骨組みの write_results が書いた成果物
SKELETON_OUTPUT = Path(__file__).parent / "anomaly_skeleton"


def sensor_rows() -> pd.DataFrame:
    """
    正常の行と外れた少しの異常の行の, 合成のセンサの表を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    frame: pd.DataFrame
        reading_id, temp, vibration, current, is_anomaly の表 (異常の行
        にだけ印)
    """
    # 3 つの列が連れ立って動く正常の行と, 振動だけ外れた異常の行
    rng = np.random.default_rng(0)
    rows = []
    for number in range(NORMAL + ANOMALIES):
        base = rng.normal(0, 1)
        broken = number >= NORMAL
        rows.append({
            "reading_id": f"r{number:03d}",
            "temp": 50 + 5 * base + rng.normal(0, 0.3),
            "vibration": 2 + base + rng.normal(0, 0.1)
            + (6 if broken else 0),
            "current": 10 - 2 * base + rng.normal(0, 0.2),
            "is_anomaly": 1 if broken else None,
        })
    frame = pd.DataFrame(rows)

    return frame


def autoencoder_spec() -> dict:
    """
    列の数に合わせた自己符号化器の定義を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    spec: dict
        ML-ModelControl の同梱の例 tabular_autoencoder の幅を変えたもの
    """
    # 入力の幅と最後の全結合の出力の幅
    spec = load_example("tabular_autoencoder")
    width = len(FEATURES)
    spec["inputs"]["x"]["shape"] = [8, width]
    spec["nodes"]["enc1"]["params"]["in_features"] = width
    spec["nodes"]["dec2"]["params"]["out_features"] = width

    return spec


def skeleton_parts(
    score: str, x: np.ndarray, model: torch.nn.Module, distance: dict
) -> np.ndarray:
    """
    骨組みと同じ式で, 行ごと, 列ごとの度合いの分を求める関数

    Parameters
    ----------
    score: str
        autoencoder か distance
    x: np.ndarray
        (行の数, 列の数) の float32
    model: torch.nn.Module
        自己符号化器 (距離は None)
    distance: dict
        center と precision (自己符号化器は None)

    Returns
    ----------
    parts: np.ndarray
        (行の数, 列の数) の float64
    """
    # 自己符号化器は float32 の二乗誤差, 距離は float64 の積
    if score == "autoencoder":
        with torch.no_grad():
            tensor = torch.tensor(x)
            recon = model(x=tensor)
            recon = recon["recon"] if isinstance(recon, dict) else recon
            parts = ((recon - tensor) ** 2).double().numpy()
        return parts
    diff = x.astype(np.float64) - np.asarray(distance["center"])
    parts = diff * (diff @ np.asarray(distance["precision"]))

    return parts


def write_anomaly_run(
    root: Path, raw: Path, score: str, spec: bool = True
) -> Path:
    """
    異常の検知の骨組みと同じ成果物を持つ実験のディレクトリを作る関数

    Parameters
    ----------
    root: Path
        作業のディレクトリ (データセットと実験を置く)
    raw: Path
        元の表の CSV
    score: str
        autoencoder か distance
    spec: bool = True
        自己符号化器の定義を実験に置くか (置かなければ仮のモデルの実験)

    Returns
    ----------
    run: Path
        実験のディレクトリ
    """
    # 異常の検知の表を ML-EDA で書き出し, 分割ごとの行の特徴
    data_dir = root / "dataset"
    recipe = {"task": "anomaly", "target": "is_anomaly", "id": "reading_id",
              "steps": [{"op": "standardize", "columns": "auto"}],
              "split": {"method": "random", "valid": 0.2, "test": 0.2,
                        "seed": 0, "unlabeled": "normal"}}
    export_dataset(load_data(raw), recipe, data_dir)
    dataset = load_dataset(data_dir)
    run = root / "run"
    run.mkdir()
    splits = {}
    for split in dataset.frames:
        x, y = dataset.arrays(split)
        splits[split] = {
            "x": np.array(x, dtype=np.float32),
            "y": np.zeros(len(x), dtype=np.int64) if y is None
            else np.asarray(y, dtype=np.int64),
            "ids": dataset.frames[split]["reading_id"].to_numpy(),
        }

    # 度合いの決め方 (自己符号化器は学習しない重み, 距離は中心と精度行列)
    model, distance = None, None
    if score == "autoencoder":
        model = build_model(autoencoder_spec(), seed=0, load_weights=False)
        model.eval()
        torch.save(model.state_dict(), run / "final_model.pt")
        if spec:
            (run / "model_spec.yaml").write_text(
                yaml.safe_dump(autoencoder_spec(), allow_unicode=True),
                encoding="utf-8"
            )
    else:
        points = splits["train"]["x"].astype(np.float64)
        diff = points - points.mean(0)
        covariance = diff.T @ diff / (len(points) - 1) + np.eye(3) * 0.001
        distance = {"center": points.mean(0).tolist(),
                    "precision": np.linalg.inv(covariance).tolist(),
                    "ridge": 0.001}

    # 行ごとの度合い, 度合いの決め方, レシピの写し, 評価
    tables = []
    for split, part in splits.items():
        found = skeleton_parts(score, part["x"], model, distance)
        part["scores"] = found.mean(1)
        tables.append(pd.DataFrame({
            "reading_id": part["ids"], "split": split,
            "score": part["scores"], "anomaly": part["y"],
        }))
    pd.concat(tables).to_csv(run / "anomaly_scores.csv", index=False)
    train = splits["train"]["scores"]
    normal = {"rows": len(train), "mean": float(train.mean())}
    for point in (50, 90, 95, 99):
        normal[f"p{point}"] = float(np.percentile(train, point))
    normal["max"] = float(train.max())
    scorer = {
        "format_version": 1, "score": score,
        "recipe": "anomaly_recipe.yaml", "scores": "anomaly_scores.csv",
        "features": FEATURES, "id": "reading_id", "capacity": CAPACITY,
        "k": CAPACITY, "normal": normal,
        "model": {"input": "x", "output": "recon"} if model is not None
        else None,
        "distance": distance,
    }
    (run / "anomaly_scorer.json").write_text(json.dumps(scorer, indent=2),
                                             encoding="utf-8")
    (run / "anomaly_recipe.yaml").write_text(
        (data_dir / "recipe.yaml").read_text(encoding="utf-8"),
        encoding="utf-8"
    )
    (run / "evaluations").mkdir()
    (run / "evaluations" / "anomaly_valid.json").write_text(json.dumps({
        "name": "anomaly", "split": "valid", "kind": "table",
        "rows": [{"rows": 13, "anomalies": 1, "k": CAPACITY,
                  "precision_at_k": 0.333, "average_precision": 0.9}],
    }), encoding="utf-8")

    return run


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    """
    合成のセンサの表を CSV に書く関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    path: Path
        CSV のファイル
    """
    # 印の無い行は空欄
    path = tmp_path / "readings.csv"
    sensor_rows().to_csv(path, index=False)

    return path


@pytest.mark.parametrize("score", ["autoencoder", "distance"])
def test_anomaly_bundle(tmp_path: Path, raw: Path, score: str) -> None:
    """
    異常の度合いの一式の度合いと振り分けを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    raw: Path
        元の表の CSV
    score: str
        度合いの決め方

    Returns
    ----------
    None
    """
    # 一式を作り, 元の表の度合いを実験の度合いと比べる
    run = write_anomaly_run(tmp_path / "work", raw, score)
    info = make_anomaly_bundle(tmp_path / "bundle", run, data=raw)
    assert info["format_version"] == 11
    assert (info["role"], info["output"]) == ("predict", {"form": "value"})
    assert info["decision"] == {"type": "triage", "capacity": CAPACITY}
    anomaly = info["anomaly"]
    assert anomaly["score"] == score and anomaly["features"] == FEATURES
    assert anomaly["check"]["rows"] == NORMAL + ANOMALIES
    assert anomaly["check"]["agreement"] == 1.0
    assert anomaly["check"]["max_diff"] < 1e-6
    assert anomaly["evaluations"]["valid"]["k"] == CAPACITY
    assert anomaly["rows"] == NORMAL + ANOMALIES
    if score == "distance":
        assert info["model"] is None
        assert not (tmp_path / "bundle" / "final_model.pt").exists()
    else:
        assert info["model"]["output"] == "recon"
    assert (tmp_path / "bundle" / "anomaly_scorer.json").is_file()
    assert (tmp_path / "bundle" / "anomaly_recipe.yaml").is_file()

    # 説明
    bundle = load_bundle(tmp_path / "bundle")
    found = describe_bundle(bundle)
    assert (found["kind"], found["task"]) == ("anomaly", "")
    assert found["features"] == FEATURES and found["steps"] == ["standardize"]
    assert found["anomaly"]["capacity"] == CAPACITY
    assert found["group"] is None and not found["column_output"]

    # 新しい行の度合い (前処理で欠けた行は飛ばした行)
    rows = sensor_rows().drop(columns=["is_anomaly"])
    rows.loc[1, "temp"] = None
    frame = predict(bundle, rows)
    assert list(frame.columns[-6:]) == ["status", "reason", "score", "rank",
                                        "column", "column_share"]
    assert frame.loc[1, "status"] == "skipped"
    assert "前処理の後に欠損: temp" in frame.loc[1, "reason"]
    ok = frame[frame["status"] == "ok"]
    assert sorted(ok["rank"].tolist()) == list(range(1, len(ok) + 1))
    assert set(ok["column"]) <= set(FEATURES)
    if score == "distance":
        top = ok.sort_values("rank").head(ANOMALIES)
        assert set(top["reading_id"]) == {
            f"r{n:03d}" for n in range(NORMAL, NORMAL + ANOMALIES)
        }
        assert set(top["column"]) == {"vibration"}

    # 結論と既定の振り分け (上位から上限まで)
    result = infer(bundle, rows)
    summary = result["summary"]
    assert summary["predicted"] == NORMAL + ANOMALIES - 1
    assert summary["skipped"] == 1 and summary["id"] == "reading_id"
    assert len(summary["top"]) == CAPACITY
    assert summary["top"][0]["row"] == int(ok["rank"].idxmin())
    assert "異常の度合い" in result["sentences"][0]
    assert "検証用の上位 3 件の適合率" in result["sentences"][-1]
    acted = result["frame"]["action"]
    assert int(acted.sum()) == CAPACITY
    assert pd.isna(acted.iloc[1])
    chosen = result["frame"][acted.fillna(False).astype(bool)]
    assert sorted(chosen["rank"].tolist()) == [1, 2, 3]
    decided = result["decision"]
    assert decided["summary"]["acted"] == CAPACITY
    assert decided["summary"]["lowest"] == pytest.approx(
        chosen["score"].min()
    )
    assert "度合いの高い順に 3 件を人の確認に回す" in decided["sentences"][0]
    assert any("確率ではない" in note for note in decided["notes"])

    # 上限より少ない行は全ての行, 空の辞書は振り分け無し
    few = infer(bundle, rows.head(2), decision={"type": "triage",
                                                "capacity": 10})
    assert few["decision"]["summary"]["acted"] == 1
    plain = infer(bundle, rows, decision={})
    assert "decision" not in plain and "action" not in plain["frame"]

    # 書き出しと読み直し
    write_result(result, tmp_path / "result")
    back = read_result(tmp_path / "result")
    assert back["summary"]["id"] == "reading_id"
    assert back["frame"]["score"].tolist() == pytest.approx(
        result["frame"]["score"].tolist(), nan_ok=True
    )


def test_anomaly_decision(tmp_path: Path, raw: Path) -> None:
    """
    振り分けの決め方の誤りと使えない機能を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    raw: Path
        元の表の CSV

    Returns
    ----------
    None
    """
    # 書ける項目は type と capacity だけ
    run = write_anomaly_run(tmp_path / "work", raw, "distance")
    make_anomaly_bundle(tmp_path / "bundle", run)
    bundle = load_bundle(tmp_path / "bundle")
    assert check_decision({"type": "triage", "capacity": 5}, bundle) == []
    errors = check_decision({"type": "triage", "capacity": 0,
                             "positive": "1"}, bundle)
    assert any("positive" in e for e in errors)
    assert any("capacity" in e for e in errors)
    assert check_decision({"type": "select", "capacity": 2}, bundle)
    assert check_decision({"type": "triage", "capacity": True}, bundle)
    frame = predict(bundle, raw)
    with pytest.raises(ValueError, match="書けない項目"):
        decide(bundle, frame, {"type": "triage", "capacity": 2,
                               "miss_cost": 10})
    found = decide(bundle, frame, {"type": "triage", "capacity": 2})
    assert int(found["frame"]["action"].sum()) == 2

    # 結論の決め方とたどり直しは誤り
    with pytest.raises(ValueError, match="異常の度合いの一式には"):
        infer(bundle, raw, rules={"min_confidence": 0.5})
    with pytest.raises(ValueError, match="異常の度合いの一式には"):
        backtest(bundle, raw, {"type": "triage", "capacity": 2})


def test_anomaly_reviews(tmp_path: Path, raw: Path) -> None:
    """
    人の確認の結果を 1 件ずつ結果の隣に足し, 読み直せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    raw: Path
        元の表の CSV

    Returns
    ----------
    None
    """
    # 結果の書き出しと, 2 件の記録の追記
    run = write_anomaly_run(tmp_path / "work", raw, "distance")
    make_anomaly_bundle(tmp_path / "bundle", run)
    result = infer(tmp_path / "bundle", raw)
    place = write_result(result, tmp_path / "result")
    assert len(read_reviews(place)) == 0
    when = dt.datetime(2026, 11, 26, 9, 30,
                       tzinfo=dt.timezone(dt.timedelta(hours=9)))
    first = record_review(place, "r060", "anomaly", "山田", when)
    assert first == {"id": "r060", "judgment": "anomaly", "reviewer": "山田",
                     "reviewed_at": "2026-11-26T09:30:00+09:00"}
    second = record_review(place, "r001", "normal", " 佐藤 ")
    assert second["reviewer"] == "佐藤"
    assert dt.datetime.fromisoformat(
        second["reviewed_at"]
    ).utcoffset() is not None

    # 書いた順の文字の読み直しと, 同じ場所に書き直した後の記録
    reviews = read_reviews(place)
    assert list(reviews.columns) == ["id", "judgment", "reviewer",
                                     "reviewed_at"]
    assert reviews["id"].tolist() == ["r060", "r001"]
    assert reviews["judgment"].tolist() == ["anomaly", "normal"]
    write_result(read_result(place), place)
    assert len(read_reviews(place)) == 2
    record_review(place, "r060", "normal", "山田",
                  "2026-11-26T10:00:00+09:00")
    assert read_reviews(place)["id"].tolist() == ["r060", "r001", "r060"]

    # 形の誤り
    with pytest.raises(ValueError, match="judgment"):
        record_review(place, "r060", "異常", "山田")
    with pytest.raises(ValueError, match="reviewer"):
        record_review(place, "r060", "anomaly", "  ")
    with pytest.raises(ValueError, match="時差"):
        record_review(place, "r060", "anomaly", "山田",
                      "2026-11-26T10:00:00")
    with pytest.raises(ValueError, match="r999"):
        record_review(place, "r999", "anomaly", "山田")
    with pytest.raises(FileNotFoundError):
        record_review(tmp_path / "nothing", "r060", "anomaly", "山田")

    # 行の id の列の無い結果は行の位置
    rows = pd.read_csv(raw).drop(columns=["reading_id"])
    other = write_result(infer(tmp_path / "bundle", rows),
                         tmp_path / "other")
    assert record_review(other, 3, "normal", "山田")["id"] == "3"
    with pytest.raises(ValueError, match="行の位置"):
        record_review(other, len(rows), "normal", "山田")

    # 異常の度合いの結果でない結果と, 列の違う記録
    meta = yaml.safe_load((other / "result.yaml").read_text("utf-8"))
    meta["summary"].pop("score")
    (other / "result.yaml").write_text(yaml.safe_dump(meta,
                                                      allow_unicode=True),
                                       encoding="utf-8")
    with pytest.raises(ValueError, match="異常の度合いの結果ではありません"):
        record_review(other, 3, "normal", "山田")
    (place / "reviews.csv").write_text("id,who\n", encoding="utf-8")
    with pytest.raises(ValueError, match="列は"):
        read_reviews(place)


def test_anomaly_sources(
    tmp_path: Path, raw: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    一式にできない実験と成果物, 古い読む側を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    raw: Path
        元の表の CSV
    monkeypatch: pytest.MonkeyPatch
        読める版を古くする

    Returns
    ----------
    None
    """
    # 仮のモデルで学習した実験 (定義無し)
    placeholder = write_anomaly_run(tmp_path / "a", raw, "autoencoder",
                                    spec=False)
    with pytest.raises(ValueError, match="PlaceholderModel"):
        make_anomaly_bundle(tmp_path / "x", placeholder)

    # 成果物の版と形の誤り
    run = write_anomaly_run(tmp_path / "b", raw, "distance")
    scorer = json.loads((run / "anomaly_scorer.json").read_text("utf-8"))
    for change, words in (({"format_version": 2}, "版 2"),
                          ({"capacity": 0}, "capacity"),
                          ({"model": {"input": "x", "output": "recon"}},
                           "model"),
                          ({"distance": {"center": [0.0], "precision": [[1]],
                                         "ridge": 0.0}}, "distance")):
        (run / "anomaly_scorer.json").write_text(
            json.dumps({**scorer, **change}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match=words):
            make_anomaly_bundle(tmp_path / "x", run)
    (run / "anomaly_scorer.json").write_text(json.dumps(scorer),
                                             encoding="utf-8")
    assert not (tmp_path / "x").exists()

    # 行の id の列の無い元の表の比べ無し
    rows = pd.read_csv(raw).drop(columns=["reading_id"])
    info = make_anomaly_bundle(tmp_path / "bundle", run, data=rows)
    assert info["anomaly"]["check"] is None
    assert any("比べません" in note for note in info["notes"])

    # 一部の行だけの元の表の, 比べた行の数の知らせ
    info = make_anomaly_bundle(tmp_path / "bundle", run,
                               data=pd.read_csv(raw).head(10),
                               overwrite=True)
    assert info["anomaly"]["check"]["rows"] == 10
    assert any("比べたのは" in note for note in info["notes"])

    # 版 10 までしか知らない読む側と, 食い違う一式
    monkeypatch.setattr(bundle_module, "BUNDLE_FORMAT_VERSION", 10)
    with pytest.raises(ValueError, match="format_version 11 の一式"):
        load_bundle(tmp_path / "bundle")
    monkeypatch.undo()
    path = tmp_path / "bundle" / "inference.yaml"
    info = yaml.safe_load(path.read_text("utf-8"))
    for change, words in (({"format_version": 10}, "anomaly を持つ一式"),
                          ({"output": {"form": "class"}}, "value")):
        path.write_text(yaml.safe_dump({**info, **change},
                                       allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match=words):
            load_bundle(tmp_path / "bundle")
    broken = {**info, "anomaly": {**info["anomaly"], "id": "other"}}
    path.write_text(yaml.safe_dump(broken, allow_unicode=True),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="id"):
        load_bundle(tmp_path / "bundle")


def test_skeleton_output(tmp_path: Path) -> None:
    """
    骨組みが実際に書いた成果物から一式を作れることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 骨組みの write_results で書いた距離に基づく度合いの成果物 (末尾の
    # 改行も骨組みの書き出しのまま)
    run = tmp_path / "run"
    shutil.copytree(SKELETON_OUTPUT, run)
    scorer = read_scorer(run / "anomaly_scorer.json")
    assert (scorer["score"], scorer["features"]) == ("distance",
                                                     ["temp", "vibration"])
    assert (run / "anomaly_scorer.json").read_text("utf-8").endswith("}\n")

    # 元の行の度合いが骨組みの度合いと同じで, 外れた行が上位
    rows = pd.DataFrame({
        "reading_id": [f"r{n:02d}" for n in range(9)],
        "temp": [0, 1, -1, 0, 0.5, -0.5, 0.2, 4, -0.1],
        "vibration": [1, 0, 0, -1, 0.5, -0.5, 0.1, 4, 0.3],
    })
    info = make_anomaly_bundle(tmp_path / "bundle", run, data=rows)
    assert info["anomaly"]["check"]["rows"] == 9
    assert info["anomaly"]["check"]["agreement"] == 1.0
    frame = predict(tmp_path / "bundle", rows)
    assert frame.loc[frame["rank"] == 1, "reading_id"].tolist() == ["r07"]


def test_reviews_at_once(tmp_path: Path, raw: Path) -> None:
    """
    同時に最初の記録を書いても見出しが 1 行だけになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    raw: Path
        元の表の CSV

    Returns
    ----------
    None
    """
    # 記録の無い結果に, 8 つの thread で同時に 1 件ずつ
    run = write_anomaly_run(tmp_path / "work", raw, "distance")
    make_anomaly_bundle(tmp_path / "bundle", run)
    place = write_result(infer(tmp_path / "bundle", raw),
                         tmp_path / "result")
    start = threading.Barrier(8)

    def write(number: int) -> None:
        """
        揃ってから 1 件の記録を足す関数

        Parameters
        ----------
        number: int
            行の番号

        Returns
        ----------
        None
        """
        # 全ての thread が揃ってからの追記
        start.wait()
        record_review(place, f"r{number:03d}", "normal", "山田")

    workers = [threading.Thread(target=write, args=(n,)) for n in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    # 見出しは 1 行, 記録は 8 件
    lines = (place / "reviews.csv").read_text("utf-8-sig").splitlines()
    assert lines.count("id,judgment,reviewer,reviewed_at") == 1
    assert sorted(read_reviews(place)["id"]) == [f"r{n:03d}"
                                                 for n in range(8)]
    assert not list(place.glob(".reviews.csv.*"))
