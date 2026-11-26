import json
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from ml_eda import Data, describe_groups
from ml_modelcontrol import build_model, load_example

import ml_inference.bundle as bundle_module
from ml_inference.grouping import far_marks
from ml_inference import (
    describe_bundle, group_far, group_representatives, group_summary,
    infer, load_bundle, make_bundle, make_group_bundle, predict,
    read_result, write_result
)

# 群の数と, 群ごとの行の数
GROUPS = 3
ROWS = 20

# ML-Quest のまとめるの骨組みの比べる相手の名前
BASELINE = "列の特徴をそのまま k-means で分けた群"


def customer_rows(count: int = ROWS) -> pd.DataFrame:
    """
    3 つのはっきりした群のある, 合成の顧客の表を作る関数

    Parameters
    ----------
    count: int = ROWS
        群ごとの行の数

    Returns
    ----------
    frame: pd.DataFrame
        customer, spend, visits, region の表 (欠けた値を 1 つ含む)
    """
    # 群ごとに使う額, 来た回数, 地域が違う (ML-Quest の骨組みのテストと同じ形)
    rng = np.random.default_rng(0)
    rows = []
    for group in range(GROUPS):
        for number in range(count):
            rows.append({
                "customer": f"c{group}_{number}",
                "spend": float(rng.normal(1000 * (group + 1), 50)),
                "visits": int(rng.poisson(2 + 6 * group)),
                "region": ("north", "south", "east")[group],
            })
    rows[0]["spend"] = None
    frame = pd.DataFrame(rows)

    return frame


def skeleton_encoding(frame: pd.DataFrame) -> dict:
    """
    骨組みの make_encoding と同じ列の符号化を作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        群を作る列

    Returns
    ----------
    encoding: dict
        numeric (mean と std) と categories (初めて出た順の値)
    """
    # 数の列は母の標準偏差, 種類の列は文字の値
    numeric, categories = {}, {}
    for name in frame.columns:
        column = frame[name]
        if pd.api.types.is_numeric_dtype(column):
            values = column.astype(float)
            std = float(values.std(ddof=0))
            numeric[name] = {"mean": float(values.mean()),
                             "std": std if std > 0 else 1.0}
        else:
            categories[name] = list(dict.fromkeys(
                column.map(lambda v: "(欠損)" if pd.isna(v) else str(v))
            ))
    encoding = {"numeric": numeric, "categories": categories}

    return encoding


def skeleton_features(frame: pd.DataFrame, encoding: dict) -> np.ndarray:
    """
    骨組みの encode_rows と同じ式で行を符号化する関数

    Parameters
    ----------
    frame: pd.DataFrame
        群を作る列
    encoding: dict
        列の符号化

    Returns
    ----------
    features: np.ndarray
        (行の数, 幅) の float32
    """
    # 数の列は標準化して欠けは 0, 種類の列は値ごとの 0 と 1
    parts = []
    for name, stats in encoding["numeric"].items():
        values = (frame[name].astype(float) - stats["mean"]) / stats["std"]
        parts.append(values.fillna(0.0).to_numpy()[:, None])
    for name, values in encoding["categories"].items():
        text = frame[name].map(
            lambda v: "(欠損)" if pd.isna(v) else str(v)
        ).to_numpy()
        parts.append(np.stack([text == value for value in values], 1))
    features = np.concatenate(parts, 1).astype(np.float32)

    return features


def encoder_spec(width: int) -> dict:
    """
    符号化した行の幅に合わせた対照学習の符号化器の定義を返す関数

    Parameters
    ----------
    width: int
        符号化した行の幅

    Returns
    ----------
    spec: dict
        ML-ModelControl の同梱の例 contrastive_encoder の大きさを変えたもの
    """
    # 入力の形と 1 段目の全結合の入力の幅
    spec = load_example("contrastive_encoder")
    spec["inputs"]["x"]["shape"] = [8, width]
    spec["nodes"]["fc1"]["params"]["in_features"] = width

    return spec


def embeddings(spec: dict, state: dict, features: np.ndarray) -> np.ndarray:
    """
    骨組みの embed と同じく, 行の特徴の埋め込みを求める関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    state: dict
        重み
    features: np.ndarray
        (行の数, 幅) の float32

    Returns
    ----------
    points: np.ndarray
        (行の数, 埋め込みの幅) の float64
    """
    # 定義を組み立てて重みを読み, 埋め込みの出力
    model = build_model(spec, seed=0, load_weights=False)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = model(x=torch.from_numpy(features))
    points = out["embedding"].double().numpy()

    return points


def nearest(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    骨組みの assign と同じく, 最も近い群の中心の番号を返す関数

    Parameters
    ----------
    points: np.ndarray
        (件数, 幅) の点
    centers: np.ndarray
        (群の数, 幅) の群の中心

    Returns
    ----------
    labels: np.ndarray
        (件数,) の群の番号
    """
    # 二乗距離を内積で求める
    distance = ((points ** 2).sum(1)[:, None] - 2 * points @ centers.T
                + (centers ** 2).sum(1)[None, :])
    labels = distance.argmin(1)

    return labels


def write_group_run(root: Path, frame: pd.DataFrame) -> Path:
    """
    まとめるの骨組みと同じ成果物を持つ実験のディレクトリを作る関数

    Parameters
    ----------
    root: Path
        実験のディレクトリ
    frame: pd.DataFrame
        学習の表 (customer, spend, visits, region)

    Returns
    ----------
    root: Path
        実験のディレクトリ
    """
    # 重みと定義と実験の状態
    root.mkdir(parents=True)
    columns = frame[["spend", "visits", "region"]]
    encoding = skeleton_encoding(columns)
    features = skeleton_features(columns, encoding)
    spec = encoder_spec(features.shape[1])
    model = build_model(spec, seed=0, load_weights=False)
    torch.save(model.state_dict(), root / "final_model.pt")
    (root / "model_spec.yaml").write_text(
        yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8"
    )
    (root / "run.json").write_text(
        json.dumps({"format_version": 1, "status": "completed"}),
        encoding="utf-8"
    )

    # 群の中心は地域ごとの埋め込みの平均, 割り当ては最も近い中心
    points = embeddings(spec, model.state_dict(), features)
    truth = frame["region"].map({"north": 0, "south": 1, "east": 2})
    centers = np.stack([points[truth == g].mean(0) for g in range(GROUPS)])
    labels = nearest(points, centers)
    split = np.where(np.arange(len(frame)) % 5 == 0, "valid", "train")
    pd.DataFrame({"customer": frame["customer"], "group": labels,
                  "split": split}).to_csv(root / "group_assignments.csv",
                                          index=False)
    (root / "group_centers.json").write_text(json.dumps(
        {"k": GROUPS, "encoding": encoding, "centers": centers.tolist()},
        ensure_ascii=False, indent=2
    ), encoding="utf-8")

    # 群の安定性と群の要約 (ML-Found の save_evaluation の形)
    evaluations = root / "evaluations"
    evaluations.mkdir()
    row = {"k": GROUPS, "method": "seed", "repeats": 5}
    for name, value, base in (("ari", 0.97, 0.99), ("ari_min", 0.95, 0.98),
                              ("silhouette", 0.62, 0.55)):
        row.update({name: value, f"{name}_baseline": base,
                    f"{name}_diff": value - base})
    (evaluations / "group_stability.json").write_text(json.dumps({
        "name": "group_stability", "split": "all", "kind": "table",
        "x": None, "columns": list(row), "rows": [row],
        "description": "群の安定性", "baseline": BASELINE, "guard": False,
        "better": {}, "epoch": 2, "created_at": "2026-11-25T00:00:00+09:00"
    }, ensure_ascii=False), encoding="utf-8")
    found = describe_groups(Data(kind="table", frame=columns),
                            columns=list(columns.columns),
                            labels=list(labels))
    rows = found["groups"]
    for item in rows[:-1]:
        item["stability"] = 0.9 + 0.01 * item["group"]
    (evaluations / "group_summary.json").write_text(json.dumps({
        "name": "group_summary", "split": "all", "kind": "table",
        "x": None, "columns": list(rows[0]), "rows": rows,
        "description": "群の要約", "baseline": None, "guard": False,
        "better": {"stability": "higher"}, "epoch": 2,
        "created_at": "2026-11-25T00:00:00+09:00"
    }, ensure_ascii=False), encoding="utf-8")

    return root


def broken_run(
    run: Path, root: Path, name: str, edit: Callable[[Path], None]
) -> Path:
    """
    実験のディレクトリを写し, 1 つのファイルを書き換える関数

    Parameters
    ----------
    run: Path
        元の実験のディレクトリ
    root: Path
        写す先
    name: str
        書き換えるファイル (実験からの相対パス)
    edit: Callable[[Path], None]
        ファイルを書き換える関数

    Returns
    ----------
    root: Path
        写した実験のディレクトリ
    """
    # 写してから書き換える
    shutil.copytree(run, root)
    edit(root / name)

    return root


def rewrite(change: Callable[[dict], None]) -> Callable[[Path], None]:
    """
    JSON のファイルを読んで辞書を書き換えて書く関数を返す関数

    Parameters
    ----------
    change: Callable[[dict], None]
        辞書を書き換える関数

    Returns
    ----------
    edit: Callable[[Path], None]
        ファイルを書き換える関数
    """
    # 読んで書き換えて書く
    def edit(file: Path) -> None:
        data = json.loads(file.read_text(encoding="utf-8"))
        change(data)
        file.write_text(json.dumps(data, ensure_ascii=False),
                        encoding="utf-8")

    return edit


def widen(file: Path) -> None:
    """
    モデルの定義の入力の幅を符号化した行の幅と違う 6 にする関数

    Parameters
    ----------
    file: Path
        定義のファイル

    Returns
    ----------
    None
    """
    # 入力の形の最後の大きさだけ
    spec = yaml.safe_load(file.read_text(encoding="utf-8"))
    spec["inputs"]["x"]["shape"] = [8, 6]
    file.write_text(yaml.safe_dump(spec), encoding="utf-8")


@pytest.fixture
def group_run(tmp_path: Path) -> Path:
    """
    群分けの実験のディレクトリを作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    run: Path
        実験のディレクトリ
    """
    # 60 行の顧客の表
    run = write_group_run(tmp_path / "run", customer_rows())

    return run


def test_make_group_bundle(tmp_path: Path, group_run: Path) -> None:
    """
    実験から群分けの一式を作り, 写した成果物と説明と割り当て直しを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    group_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 学習の表を渡すと遠さの上限を持つ版 16. 列の符号化と群の中心の写し
    frame = customer_rows()
    root = tmp_path / "bundle"
    info = make_group_bundle(root, group_run, data=frame)
    assert info["format_version"] == 16
    assert (info["role"], info["output"]) == ("group", {"form": "cluster"})
    assert (root / "group_centers.json").is_file()
    group = info["group"]
    assert (group["k"], group["features"], group["width"]) == (3, 5, 16)
    assert group["columns"] == ["spend", "visits", "region"]
    assert (group["id"], group["rows"]) == ("customer", 60)

    # 群の安定性は実験の行の写しと比べる相手の名前
    stability = group["stability"]
    assert (stability["ari"], stability["ari_min_baseline"]) == (0.97, 0.98)
    assert stability["silhouette_diff"] == pytest.approx(0.07)
    assert (stability["baseline"], stability["epoch"]) == (BASELINE, 2)
    assert [row["group"] for row in group["summary"]] == [0, 1, 2, "全体"]

    # 学習の表を割り当て直すと実験の割り当てと同じ
    assert group["check"] == {"rows": 60, "agreement": 1.0}
    assert not any("割り当て直" in note for note in info["notes"])

    # 説明
    description = describe_bundle(root)
    assert (description["kind"], description["task"]) == ("group", "")
    assert description["group"]["k"] == 3
    assert description["rank"] is None
    assert load_bundle(root).from_group

    # 行の数の違う表は比べない. 渡さなければ check は null
    info = make_group_bundle(tmp_path / "other", group_run,
                             data=frame.iloc[:10])
    assert info["group"]["check"] is None
    assert any("行の数 10" in note for note in info["notes"])
    assert info["group"]["far"]["rows"] == 10
    info = make_group_bundle(tmp_path / "plain", group_run)
    assert info["group"]["check"] is None
    assert info["format_version"] == 9 and "far" not in info["group"]
    assert any("遠い行に印を付けません" in note for note in info["notes"])


def test_predict_group(tmp_path: Path, group_run: Path) -> None:
    """
    新しい行を最も近い群の中心に割り当て, 割り当てない行に理由を付ける
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    group_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 学習の行は実験の割り当てと同じ群
    root = tmp_path / "bundle"
    make_group_bundle(root, group_run)
    frame = customer_rows()
    result = predict(root, frame)
    expected = pd.read_csv(group_run / "group_assignments.csv")["group"]
    assert (result["status"] == "ok").all()
    assert result["group"].dtype == "Int64"
    assert list(result["group"]) == list(expected)
    assert (result["distance"] >= 0).all()
    assert list(result.columns[:4]) == ["customer", "spend", "visits",
                                        "region"]

    # 数にできない値の行は割り当てず, 知らない地域と欠けは割り当てる.
    # 予測の列と同じ名前の元の列は「(入力)」
    new = pd.DataFrame({
        "spend": [1000.0, "多い", None], "visits": [2, 3, 4],
        "region": ["north", "west", None], "status": ["a", "b", "c"],
    })
    result = predict(root, new)
    assert list(result["status"]) == ["ok", "skipped", "ok"]
    assert "列 spend の値 '多い' を数にできません" in result["reason"][1]
    assert pd.isna(result["group"][1]) and np.isnan(result["distance"][1])
    assert list(result["status (入力)"]) == ["a", "b", "c"]

    # ファイルも読める (読み込みの設定は load_data に渡す)
    path = tmp_path / "rows.csv"
    frame.to_csv(path, index=False)
    assert list(predict(root, path)["group"]) == list(expected)
    assert list(predict(root, path, options={"loader": "csv"})["group"]) == (
        list(expected)
    )

    # 群を作る列が無い, 表に読み込みの設定, 並べる件数は誤り
    with pytest.raises(ValueError, match="群を作る列 region"):
        predict(root, frame.drop(columns="region"))
    with pytest.raises(ValueError, match="options"):
        predict(root, frame, options={"loader": "csv"})
    with pytest.raises(ValueError, match="並べるの一式"):
        predict(root, frame, k=3)
    with pytest.raises(ValueError, match="表のファイル"):
        predict(root, [1, 2])


def test_group_summary(tmp_path: Path, group_run: Path) -> None:
    """
    群の要約が ML-EDA の describe_groups の行の形で, 安定性は学習の群の
    値, 行の無い群も並ぶことと, 結論と書き出しを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    group_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 北と南の行だけ (東の群には行が無い) と, 割り当てない 1 行
    root = tmp_path / "bundle"
    make_group_bundle(root, group_run)
    frame = customer_rows()
    new = pd.concat([frame.iloc[1:20], frame.iloc[20:40],
                     pd.DataFrame([{"customer": "x", "spend": "?",
                                    "visits": 1, "region": "north"}])],
                    ignore_index=True)
    prediction = predict(root, new)
    rows = group_summary(root, prediction)
    assert [row["group"] for row in rows] == [0, 1, 2, "全体"]
    assert list(rows[0]) == ["group", "rows", "share", "stability", "spend",
                             "visits", "region"]
    assert [row["rows"] for row in rows] == [19, 20, 0, 39]
    assert rows[0]["share"] == pytest.approx(19 / 39)
    assert [row["stability"] for row in rows] == [0.9, 0.91, 0.92, None]
    assert rows[0]["region"] == "north (100 %)"
    assert rows[0]["spend"] == pytest.approx(
        frame["spend"].iloc[1:20].mean()
    )
    assert rows[2]["spend"] is None and rows[2]["share"] == 0.0

    # ML-EDA の describe_groups に同じ群を渡した行と同じ値
    done = new.iloc[:39]
    direct = describe_groups(
        Data(kind="table", frame=done[["spend", "visits", "region"]]
             .astype({"spend": float})),
        columns=["spend", "visits", "region"],
        labels=list(prediction["group"][:39].astype(int))
    )["groups"]
    for mine, theirs in zip([rows[0], rows[1], rows[3]], direct):
        assert {k: v for k, v in mine.items() if k != "stability"} == {
            k: v for k, v in theirs.items() if k != "stability"
        }

    # 結論は件数と群の要約と群の安定性
    result = infer(root, new)
    summary = result["summary"]
    assert (summary["rows"], summary["predicted"], summary["skipped"],
            summary["k"]) == (40, 39, 1, 3)
    assert summary["groups"] == rows
    assert summary["stability"]["ari"] == 0.97
    assert result["sentences"][0] == (
        "40 行のうち 39 行を 3 個の群に割り当てた (1 行は割り当てられず)"
    )
    assert result["sentences"][1].startswith("群 0 19 行 (48.7 %)")
    assert "種を変えて作り直す, 5 回" in result["sentences"][2]
    assert BASELINE in result["sentences"][2]
    assert list(result["frame"]["decision"][-2:]) == ["", "予測なし"]

    # 書き出しは割り当てない行の群の番号の欠けの型と, 数にできない値の
    # 混ざる額を残す版 4 で, 読んだ表からも群の要約を作れる (CSV を通した
    # 数は丸めの差まで)
    out = tmp_path / "result"
    write_result(result, out)
    loaded = read_result(out)
    assert loaded["format_version"] == 4
    assert loaded["column_types"]["spend"] == "mixed"
    again = group_summary(root, loaded["frame"])
    assert [list(row) for row in again] == [list(row) for row in rows]
    for mine, theirs in zip(again, rows):
        for key, value in theirs.items():
            if isinstance(value, float):
                assert mine[key] == pytest.approx(value)
            else:
                assert mine[key] == value

    # 割り当てた行が無ければ空. 結論の決め方は使えない
    assert group_summary(root, prediction.iloc[39:]) == []
    with pytest.raises(ValueError, match="結論の決め方"):
        infer(root, new, rules={"min_confidence": 0.5})


def test_group_representatives(tmp_path: Path, group_run: Path) -> None:
    """
    群の代表の行が群ごとに中心に近い順で, 割り当てない行を除き, 結論と
    書き出しに群の要約と並ぶことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    group_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 北の 19 行と南の 2 行 (東の群には行が無い) と, 割り当てない 1 行
    root = tmp_path / "bundle"
    make_group_bundle(root, group_run)
    frame = customer_rows()
    new = pd.concat([pd.DataFrame([{"customer": "x", "spend": "?",
                                    "visits": 1, "region": "north"}]),
                     frame.iloc[1:20], frame.iloc[20:22]],
                    ignore_index=True)
    prediction = predict(root, new)
    groups = group_representatives(root, prediction)
    assert [entry["group"] for entry in groups] == [0, 1, 2]
    assert [len(entry["rows"]) for entry in groups] == [3, 2, 0]

    # 群の中で距離の近い順の行の位置と, 予測の表の距離と元の値
    north = prediction[prediction["group"] == 0]
    expected = list(north["distance"].sort_values(kind="stable").index[:3])
    first = groups[0]["rows"]
    assert [row["row"] for row in first] == expected
    assert 0 not in [row["row"] for entry in groups for row in entry["rows"]]
    assert [row["distance"] for row in first] == list(
        prediction["distance"][expected]
    )
    assert first[0]["distance"] <= first[1]["distance"] <= (
        first[2]["distance"]
    )
    place = expected[0]
    assert first[0]["values"] == {
        "spend": new["spend"][place], "visits": int(new["visits"][place]),
        "region": "north",
    }
    assert type(first[0]["values"]["visits"]) is int

    # 件数を選べる (群の行が少なければその数だけ)
    groups = group_representatives(root, prediction, count=30)
    assert [len(entry["rows"]) for entry in groups] == [19, 2, 0]
    assert [row["row"] for row in group_representatives(
        root, prediction, count=1
    )[0]["rows"]] == expected[:1]

    # 結論は群の要約と並べて既定の 3 件. 書き出して読んだ表からも作れる
    result = infer(root, new)
    assert result["summary"]["representatives"] == group_representatives(
        root, prediction
    )
    assert list(result["summary"])[:6] == [
        "rows", "predicted", "skipped", "k", "groups", "representatives"
    ]
    out = tmp_path / "result"
    write_result(result, out)
    loaded = read_result(out)
    assert loaded["format_version"] == 4
    assert loaded["summary"]["representatives"] == (
        result["summary"]["representatives"]
    )
    again = group_representatives(root, loaded["frame"])
    assert same_values(again, result["summary"]["representatives"])
    assert [[row["row"] for row in entry["rows"]] for entry in again] == [
        [row["row"] for row in entry["rows"]]
        for entry in result["summary"]["representatives"]
    ]

    # 割り当てた行が無ければ空. 件数, 列, 一式の誤り
    assert group_representatives(root, prediction.iloc[:1]) == []
    for count in (0, 1.5, True, "3"):
        with pytest.raises(ValueError, match="1 以上の整数"):
            group_representatives(root, prediction, count=count)
    for name in ("status", "group", "distance"):
        with pytest.raises(ValueError, match=f"{name} の列がありません"):
            group_representatives(root, prediction.drop(columns=name))
    with pytest.raises(ValueError, match="region"):
        group_representatives(root, prediction.drop(columns="region"))

    # 種類の列の欠けた行も代表に入り, 値は None
    gap = frame.iloc[40:60].reset_index(drop=True)
    gap["region"] = None
    found = group_representatives(root, predict(root, gap), count=30)
    assert sum(len(entry["rows"]) for entry in found) == 20
    assert all(row["values"]["region"] is None
               for entry in found for row in entry["rows"])


def test_group_representatives_input_names(
    tmp_path: Path, group_run: Path
) -> None:
    """
    予測の列と同じ名前の列を群を作る列に持つ一式で, 代表の行の値が
    「(入力)」付きの列の元の値になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    group_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 群を作る列 region を status に置き換えた実験
    renamed = {"region": "status"}
    centers = json.loads((group_run / "group_centers.json").read_text())
    categories = centers["encoding"]["categories"]
    categories["status"] = categories.pop("region")
    (group_run / "group_centers.json").write_text(json.dumps(centers))
    path = group_run / "evaluations" / "group_summary.json"
    evaluation = json.loads(path.read_text())
    evaluation["rows"] = [{renamed.get(k, k): v for k, v in row.items()}
                          for row in evaluation["rows"]]
    path.write_text(json.dumps(evaluation))
    root = tmp_path / "bundle"
    make_group_bundle(root, group_run)

    # 値は「(入力)」付きの列から, 鍵は群を作る列の名前
    frame = customer_rows().rename(columns=renamed)
    prediction = predict(root, frame)
    assert "status (入力)" in prediction.columns
    found = group_representatives(root, prediction)
    first = found[0]["rows"][0]
    assert list(first["values"]) == ["spend", "visits", "status"]
    assert first["values"]["status"] == frame["status"][first["row"]]
    assert {row["values"]["status"] for entry in found
            for row in entry["rows"]} == {"north", "south", "east"}


def test_group_bundle_errors(
    tmp_path: Path, group_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    実験の成果物の食い違い, 版の食い違い, 読めない版を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    group_run: Path
        実験のディレクトリ
    monkeypatch: pytest.MonkeyPatch
        版を差し替える

    Returns
    ----------
    None
    """
    # 版 8 までしか知らない読む側は止まる
    root = tmp_path / "bundle"
    make_group_bundle(root, group_run)
    monkeypatch.setattr(bundle_module, "BUNDLE_FORMAT_VERSION", 8)
    with pytest.raises(ValueError, match="format_version 9 の一式"):
        load_bundle(root)
    monkeypatch.undo()

    # group と版と出力の形の食い違い
    info = yaml.safe_load((root / "inference.yaml").read_text())
    for change, message in (
        ({"format_version": 8}, "group を持つ一式は版 9"),
        ({"output": {"form": "vector"}}, "出力の形 cluster"),
        ({"decision": {"type": "triage"}}, "null にしてください"),
        ({"group": {**info["group"], "k": 1}}, "group.k"),
        ({"group": {**info["group"], "width": 8}}, "group の欄と違います"),
    ):
        (root / "inference.yaml").write_text(
            yaml.safe_dump({**info, **change}, allow_unicode=True)
        )
        with pytest.raises(ValueError, match=message):
            load_bundle(root)

    # 成果物の食い違い (書き換えた実験の写しで)
    cases = [
        ("model_spec.yaml", lambda f: f.unlink(), "仮のモデル"),
        ("group_centers.json", lambda f: f.unlink(), "群の中心"),
        ("group_centers.json",
         rewrite(lambda d: d.update(centers=d["centers"][:2])), "centers"),
        ("group_centers.json",
         rewrite(lambda d: d["encoding"]["numeric"]["spend"].update(std=0)),
         "std"),
        ("group_assignments.csv",
         lambda f: f.write_text("customer,group,split\nc,7,train\n"),
         "0 から 2"),
        ("evaluations/group_stability.json",
         rewrite(lambda d: d["rows"][0].update(k=4)), "群の中心の数 3"),
        ("evaluations/group_stability.json",
         rewrite(lambda d: d["rows"][0].pop("ari_min")), "ari_min"),
        ("evaluations/group_summary.json",
         rewrite(lambda d: [r.pop("region") for r in d["rows"]]),
         "列の符号化の列"),
    ]
    for number, (name, edit, message) in enumerate(cases):
        run = broken_run(group_run, tmp_path / f"run{number}", name, edit)
        with pytest.raises((ValueError, FileNotFoundError), match=message):
            make_group_bundle(tmp_path / "x", run)

    # 定義の入力の幅が符号化と違えば作らない
    run = broken_run(group_run, tmp_path / "wide", "model_spec.yaml", widen)
    with pytest.raises(ValueError, match="符号化した行の幅 5"):
        make_group_bundle(tmp_path / "x", run)
    assert not (tmp_path / "x").exists()

    # 群分けの出力の形は make_bundle では作らない
    with pytest.raises(ValueError, match="make_group_bundle"):
        make_bundle(tmp_path / "y", encoder_spec(5), None,
                    group_run / "final_model.pt", form="cluster",
                    procedure={"name": "ddim", "steps": 2, "seed": 0},
                    output_options={"shape": [5]})


def same_values(mine: Any, theirs: Any) -> bool:
    """
    入れ子の値を, 小数は丸めの差まで, 他は型と値で比べる関数

    Parameters
    ----------
    mine: Any
        比べる値
    theirs: Any
        比べる相手

    Returns
    ----------
    same: bool
        同じか
    """
    # 小数は CSV を通した丸めの差まで, 他 (文字と整数) は型も同じ
    if isinstance(mine, dict) and isinstance(theirs, dict):
        same = list(mine) == list(theirs) and all(
            same_values(mine[k], theirs[k]) for k in mine
        )
    elif isinstance(mine, list) and isinstance(theirs, list):
        same = len(mine) == len(theirs) and all(
            same_values(a, b) for a, b in zip(mine, theirs)
        )
    elif isinstance(mine, float) and isinstance(theirs, float):
        same = mine == pytest.approx(theirs)
    else:
        same = type(mine) is type(theirs) and mine == theirs

    return same


def test_group_result_keeps_types(tmp_path: Path, group_run: Path) -> None:
    """
    数に見える種類の値と欠けのある整数の列を持つ群分けの結果を, 書き出して
    読み直しても値と群の要約と代表の行が変わらないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    group_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 番号の顧客 ("0001"), 数に見える地域 ("010". 学習に無い値), 欠けた
    # 来た回数と, 割り当てない行 (無限の額)
    root = tmp_path / "bundle"
    make_group_bundle(root, group_run)
    new = customer_rows().iloc[1:]
    new["customer"] = [f"{n:04d}" for n in range(len(new))]
    new["visits"] = new["visits"].astype(object)
    new.iloc[3, new.columns.get_loc("visits")] = None
    new["region"] = new["region"].map(
        {"north": "010", "south": "020", "east": "030"}
    )
    new = pd.concat([new, pd.DataFrame([{"customer": "0999", "spend": np.inf,
                                         "visits": 1, "region": "030"}])],
                    ignore_index=True)
    result = infer(root, new)
    assert result["frame"]["status"].iloc[-1] == "skipped"

    # 読み直した表の値と, そこから作る群の要約と代表の行は書き出す前と同じ
    out = tmp_path / "result"
    write_result(result, out)
    loaded = read_result(out)
    assert loaded["format_version"] == 3
    assert loaded["column_types"] == {
        "customer": "text", "visits": "integer", "region": "text",
        "group": "integer"
    }
    back = loaded["frame"]
    for name in ("customer", "region", "visits", "group"):
        assert [None if pd.isna(v) else v for v in back[name]] == [
            None if pd.isna(v) else v for v in result["frame"][name]
        ], name
    assert back["customer"].iloc[0] == "0000"
    assert back["region"].iloc[0] == "010"
    assert str(back["visits"].dtype) == "Int64"
    assert same_values(group_summary(root, back),
                       result["summary"]["groups"])
    assert same_values(group_representatives(root, back),
                       result["summary"]["representatives"])


def test_group_result_keeps_mixed(tmp_path: Path, group_run: Path) -> None:
    """
    数の列に数にできない値が混ざり割り当てない行を持つ群分けの結果を, 書き
    出して読み直しても値と群の要約と代表の行が変わらないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    group_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 額に数にできない値 (割り当てない行) と数に見える文字の混ざる表
    root = tmp_path / "bundle"
    make_group_bundle(root, group_run)
    new = customer_rows().iloc[1:].reset_index(drop=True)
    new["spend"] = new["spend"].astype(object)
    new.loc[2, "spend"] = "abc"
    new.loc[5, "spend"] = "1.5"
    result = infer(root, new)
    status = result["frame"]["status"]
    assert status.iloc[2] == "skipped" and (status == "skipped").sum() == 1
    assert result["summary"]["representatives"]

    # 読み直した額は値ごとの型のまま (数は数, 文字は文字)
    out = tmp_path / "result"
    write_result(result, out)
    loaded = read_result(out)
    assert loaded["format_version"] == 4
    assert loaded["column_types"]["spend"] == "mixed"
    assert loaded["text_rows"] == {"spend": [5]}
    back = loaded["frame"]
    assert back["spend"].tolist() == result["frame"]["spend"].tolist()
    assert [type(v) for v in back["spend"]] == [
        type(v) for v in result["frame"]["spend"]
    ]
    assert same_values(group_summary(root, back),
                       result["summary"]["groups"])
    assert same_values(group_representatives(root, back),
                       result["summary"]["representatives"])


def test_far_rows(tmp_path: Path, group_run: Path) -> None:
    """
    学習の表から求めた群ごとの距離の上限で, 学習の行には印が付かず, 学習の
    ときより遠い行に印と件数が付くことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    group_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 群の行が 100 以下なので, 上限は群の学習の行の距離の最も大きい値
    frame = customer_rows()
    root = tmp_path / "bundle"
    info = make_group_bundle(root, group_run, data=frame)
    far = info["group"]["far"]
    assert (far["quantile"], far["rows"]) == (0.99, 60)
    train = predict(root, frame)
    for key in range(GROUPS):
        assert far["thresholds"][key] == pytest.approx(
            train["distance"][train["group"] == key].max()
        )
    assert list(train.columns[-5:]) == ["status", "reason", "group",
                                        "distance", "far"]
    assert train["far"].dtype == bool and not train["far"].any()

    # 学習のときに無い大きな額と回数の行の印 (割り当てない行は印無し)
    new = pd.concat([frame.iloc[1:6], pd.DataFrame([
        {"customer": "big", "spend": 1e6, "visits": 500, "region": "north"},
        {"customer": "bad", "spend": "?", "visits": 1, "region": "north"},
    ])], ignore_index=True)
    result = infer(root, new)
    marks = result["frame"]["far"]
    assert list(marks) == [False] * 5 + [True, False]
    big = int(result["frame"]["group"][5])
    summary = result["summary"]["far"]
    assert (summary["quantile"], summary["rows"]) == (0.99, 1)
    assert [row["rows"] for row in summary["groups"]] == [
        int(key == big) for key in range(GROUPS)
    ]
    assert [row["threshold"] for row in summary["groups"]] == (
        far["thresholds"]
    )
    assert result["sentences"][2] == (
        "1 行は群の中心から学習のときより遠い (群ごとの学習の行の距離の "
        f"99 % 点を超える. 群 {big} 1 行)"
    )
    assert group_far(root, result["frame"]) == summary
    assert infer(root, frame)["sentences"][2] == (
        "群の中心から学習のときより遠い行は無い (群ごとの学習の行の距離の "
        "99 % 点まで)"
    )

    # 書き出して読んだ表の同じ印と件数
    out = tmp_path / "result"
    write_result(result, out)
    loaded = read_result(out)
    assert loaded["frame"]["far"].tolist() == marks.tolist()
    assert loaded["summary"]["far"] == summary
    assert group_far(root, loaded["frame"]) == summary
    with pytest.raises(ValueError, match="far の列がありません"):
        group_far(root, result["frame"].drop(columns="far"))


def test_far_training_rows_rarely(tmp_path: Path) -> None:
    """
    群ごとの行が多い学習の表でも, 上限を超える学習の行が群ごとに 1 % 以下
    であることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 群ごとに 300 行の学習の表
    frame = customer_rows(300)
    run = write_group_run(tmp_path / "run", frame)
    root = tmp_path / "bundle"
    make_group_bundle(root, run, data=frame)
    prediction = predict(root, frame)

    # 印の付く学習の行は群ごとに 1 % 以下 (99 % 点を超える行だけ)
    marks = prediction["far"]
    for key in range(GROUPS):
        inside = prediction["group"] == key
        assert marks[inside].mean() <= 0.01
    assert 0 < marks.sum() <= 0.01 * len(frame)


def test_far_without_limits(tmp_path: Path, group_run: Path) -> None:
    """
    上限を持たない版 9 の一式は印の列を持たず, 上限を持つ一式だけが元の
    far の列に「(入力)」を付けることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    group_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 学習の表を渡さない版 9 の一式の, そのままの元の far の列
    frame = customer_rows().assign(far="x")
    root = tmp_path / "plain"
    make_group_bundle(root, group_run)
    result = infer(root, frame)
    assert list(result["frame"]["far"]) == ["x"] * len(frame)
    assert "far (入力)" not in result["frame"].columns
    assert result["summary"]["far"] is None
    assert len(result["sentences"]) == 3
    assert group_far(root, result["frame"]) is None

    # 上限を持つ一式の「(入力)」付きの元の列と群の要約
    limited = tmp_path / "limited"
    make_group_bundle(limited, group_run, data=customer_rows())
    prediction = predict(limited, frame)
    assert list(prediction["far (入力)"]) == ["x"] * len(frame)
    assert prediction["far"].dtype == bool
    assert group_summary(limited, prediction)[-1]["rows"] == len(frame)


def test_far_errors(
    tmp_path: Path, group_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    上限の形の誤りと版の食い違いと, 上限の無い群の扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    group_run: Path
        実験のディレクトリ
    monkeypatch: pytest.MonkeyPatch
        版を差し替える

    Returns
    ----------
    None
    """
    # 版 15 までしか知らない読む側の停止
    frame = customer_rows()
    root = tmp_path / "bundle"
    make_group_bundle(root, group_run, data=frame)
    monkeypatch.setattr(bundle_module, "BUNDLE_FORMAT_VERSION", 15)
    with pytest.raises(ValueError, match="format_version 16 の一式"):
        load_bundle(root)
    monkeypatch.undo()

    # 上限の形と版の食い違い
    file = root / "inference.yaml"
    info = yaml.safe_load(file.read_text(encoding="utf-8"))
    group = info["group"]
    far = group["far"]
    plain = {key: value for key, value in group.items() if key != "far"}
    cases: list[tuple[dict, str]] = [
        ({"group": plain}, "group.far は quantile"),
        ({"format_version": 9}, "版 9 の一式は group.far を持てません"),
        ({"group": {**group, "far": {**far, "quantile": 1.5}}},
         "group.far.quantile"),
        ({"group": {**group, "far": {**far, "rows": 0}}}, "group.far.rows"),
        ({"group": {**group, "far": {**far,
                                     "thresholds": far["thresholds"][:2]}}},
         "group.far.thresholds"),
        ({"group": {**group, "far": {**far,
                                     "thresholds": [-1.0, None, 1.0]}}},
         "group.far.thresholds"),
        ({"group": {**group, "far": {**far, "extra": 1}}}, "知らない欄"),
    ]
    for change, message in cases:
        file.write_text(yaml.safe_dump({**info, **change},
                                       allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_bundle(root)

    # 上限の無い群の行の印無し
    limits = {**far, "thresholds": [None] * GROUPS}
    file.write_text(yaml.safe_dump(
        {**info, "group": {**group, "far": limits}}, allow_unicode=True
    ), encoding="utf-8")
    big = pd.DataFrame([{"customer": "big", "spend": 1e6, "visits": 500,
                         "region": "north"}])
    assert not predict(root, big)["far"].any()


def test_far_batch_size(tmp_path: Path, group_run: Path) -> None:
    """
    一度に渡す行の数が作るときと違っても, 上限ちょうどの学習の行に印が
    付かないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    group_run: Path
        実験のディレクトリ

    Returns
    ----------
    None
    """
    # 群の行が 100 以下なので, 群ごとに距離の最も大きい行が上限ちょうど
    frame = customer_rows()
    root = tmp_path / "bundle"
    make_group_bundle(root, group_run, data=frame, batch_size=256)

    # 1 行ずつ, 7 行ずつ, まとめての推論のどれでも学習の行は印無し
    for batch_size in (1, 7, 256):
        prediction = predict(root, frame, batch_size=batch_size)
        assert not prediction["far"].any(), batch_size

    # 上限を少し超える距離の行だけの印
    far = yaml.safe_load(
        (root / "inference.yaml").read_text(encoding="utf-8")
    )["group"]["far"]
    limit = far["thresholds"][0]
    distance = np.array([limit, limit * (1 + 1e-7), limit * 1.001])
    marks = far_marks(np.zeros(3, dtype=np.int64), distance,
                      far["thresholds"])
    assert list(marks) == [False, False, True]
