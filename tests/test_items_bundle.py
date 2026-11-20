import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from ml_eda import export_capture_clouds, export_image_masks, load_items
from ml_eda.synthetic import write_image_masks, write_shape_captures
from ml_modelcontrol import build_model

from ml_inference import (
    describe_bundle, infer, load_bundle, make_bundle, predict, read_result,
    write_result
)

# 画像と深度の読み書き (任意の依存. 無い環境では飛ばす)
Image = pytest.importorskip("PIL.Image")


def mask_spec() -> dict:
    """
    画素ごとに 3 クラスのスコアを出す畳み込みのモデルの定義を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    spec: dict
        ML-ModelControl の定義
    """
    # 3x3 の畳み込み 2 段
    spec = {
        "format_version": 1,
        "name": "segment",
        "inputs": {"image": {"shape": [2, 3, 24, 32]}},
        "nodes": {
            "conv1": {"unit": "nn.Conv2d", "inputs": "image",
                      "params": {"in_channels": 3, "out_channels": 8,
                                 "kernel_size": 3, "padding": 1}},
            "act": {"unit": "nn.ReLU", "inputs": "conv1"},
            "conv2": {"unit": "nn.Conv2d", "inputs": "act",
                      "params": {"in_channels": 8, "out_channels": 3,
                                 "kernel_size": 3, "padding": 1}},
        },
        "outputs": {"logits": "conv2"},
    }

    return spec


def cloud_spec(points: int) -> dict:
    """
    点ごとの全結合と平均から決まった数の点を出すモデルの定義を返す関数

    Parameters
    ----------
    points: int
        出す点の数

    Returns
    ----------
    spec: dict
        ML-ModelControl の定義
    """
    # 点ごとの特徴, 点の平均, 座標への戻し
    spec = {
        "format_version": 1,
        "name": "points",
        "inputs": {"x": {"shape": [2, 64, 3]}},
        "nodes": {
            "encode": {"unit": "nn.Linear", "inputs": "x",
                       "params": {"in_features": 3, "out_features": 16}},
            "act": {"unit": "nn.ReLU", "inputs": "encode"},
            "pool": {"unit": "op.Mean", "inputs": "act",
                     "params": {"dim": 1}},
            "decode": {"unit": "nn.Linear", "inputs": "pool",
                       "params": {"in_features": 16,
                                  "out_features": points * 3}},
            "shape": {"unit": "op.Reshape", "inputs": "decode",
                      "params": {"shape": [points, 3]}},
        },
        "outputs": {"points": "shape"},
    }

    return spec


def fake_run(path: Path, spec: dict, monitor: str, best: float) -> Path:
    """
    重みと run.json だけを持つ実験のディレクトリを作る関数

    Parameters
    ----------
    path: Path
        実験のディレクトリ
    spec: dict
        モデルの定義 (乱数の初期値の重みを保存)
    monitor: str
        監視した指標
    best: float
        ベストの値

    Returns
    ----------
    path: Path
        実験のディレクトリ
    """
    # 最終の重みがベストの終えた実験
    path.mkdir(parents=True, exist_ok=True)
    torch.save(build_model(spec, seed=0).state_dict(),
               path / "final_model.pt")
    state = {"format_version": 1, "status": "completed", "monitor": monitor,
             "best_value": best, "best_epoch": 1, "final_weights": "best"}
    (path / "run.json").write_text(json.dumps(state), encoding="utf-8")

    return path


def skeleton_iou(spec: dict, dataset: Path) -> float:
    """
    ML-Quest の骨組みと同じ求め方で, 検証用の分割の IoU を求める関数

    Parameters
    ----------
    spec: dict
        モデルの定義 (乱数の初期値の重み)
    dataset: Path
        版 2 の画像と領域のデータセット

    Returns
    ----------
    value: float
        背景を除いたクラスの IoU の平均
    """
    # 値の小さい順の番号 (背景 0 を含む), 画素ごとの重なりと和
    model = build_model(spec, seed=0).eval()
    part = load_items(dataset).split("valid")
    counts = torch.zeros(3, 3)
    with torch.no_grad():
        for index in range(len(part)):
            item = part[index]
            image = torch.from_numpy(item["image"]).permute(2, 0, 1)
            guess = model(image[None].float() / 255.0).argmax(dim=1)[0]
            truth = torch.from_numpy(item["mask"].astype(np.int64))
            for k in range(3):
                counts[0, k] += ((guess == k) & (truth == k)).sum()
                counts[1, k] += ((guess == k) | (truth == k)).sum()
    value = float(np.mean([
        (counts[0, k] / counts[1, k]).item() if counts[1, k] > 0 else 1.0
        for k in (1, 2)
    ]))

    return value


def skeleton_chamfer(
    spec: dict, dataset: Path, counts: tuple, seed: int
) -> float:
    """
    ML-Quest の骨組みと同じ選び方と求め方で, 検証用の分割の Chamfer 距離を
    求める関数

    Parameters
    ----------
    spec: dict
        モデルの定義 (乱数の初期値の重み)
    dataset: Path
        版 2 の撮影と点群のデータセット
    counts: tuple
        入力と正解から選ぶ点の数
    seed: int
        検証用の分割の点を選ぶ乱数の種

    Returns
    ----------
    value: float
        件ごとの距離の Chamfer の平均
    """
    # 深度から点に戻し, 件ごとの乱数で入力, 正解の順に選ぶ
    model = build_model(spec, seed=0).eval()
    part = load_items(dataset).split("valid")
    distances = []
    for index in range(len(part)):
        item = part[index]
        capture = item["capture"]
        intr = capture["intrinsics"]
        clouds = []
        for pose, depth in zip(capture["poses"], capture["depth"]):
            rows, columns = np.nonzero(depth > 0)
            d = depth[rows, columns].astype(np.float64)
            camera = np.stack([
                (columns + 0.5 - intr["cx"]) / intr["fl_x"] * d,
                -(rows + 0.5 - intr["cy"]) / intr["fl_y"] * d, -d,
            ], axis=1)
            clouds.append(camera @ pose[:3, :3].T + pose[:3, 3])
        points = np.concatenate(clouds)
        rng = np.random.default_rng([seed, index])
        x = points[rng.choice(len(points), counts[0],
                              replace=len(points) < counts[0])]
        cloud = item["cloud"]
        y = cloud[rng.choice(len(cloud), counts[1],
                             replace=len(cloud) < counts[1])]
        with torch.no_grad():
            guess = model(torch.from_numpy(x.astype(np.float32))[None])
        pairs = torch.cdist(guess, torch.from_numpy(y.astype(np.float32))
                            [None])
        distances.append(((pairs.min(dim=2).values.mean()
                           + pairs.min(dim=1).values.mean()) / 2).item())
    value = float(np.mean(distances))

    return value


@pytest.fixture
def masks(tmp_path: Path) -> dict:
    """
    合成の画像と領域から版 2 のデータセットを書き出す関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    places: dict
        raw (images と masks の場所) と dataset
    """
    # 12 組を学習用と検証用に
    raw = write_image_masks(tmp_path / "raw", count=12, seed=0)
    dataset = tmp_path / "masks"
    export_image_masks(raw["images"], raw["masks"], dataset,
                       classes=raw["classes"], origin={"type": "synthetic"})
    places = {"raw": raw, "dataset": dataset}

    return places


@pytest.fixture
def clouds(tmp_path: Path) -> dict:
    """
    合成の撮影と点群から版 2 のデータセットを書き出す関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    places: dict
        raw (captures と clouds の場所) と dataset
    """
    # 乱数の形 8 つ
    raw = write_shape_captures(tmp_path / "raw", count=8, seed=0,
                               points=400, views=6)
    dataset = tmp_path / "clouds"
    export_capture_clouds(raw["captures"], raw["clouds"], dataset,
                          split={"method": "random", "valid": 0.25,
                                 "test": 0.0, "seed": 0})
    places = {"raw": raw, "dataset": dataset}

    return places


def test_mask_bundle(tmp_path: Path, masks: dict) -> None:
    """
    画像と領域の版 2 のデータセットから一式を作り, 記録と比べ, 新しい画像
    を領域の値で推論して書き出せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    masks: dict
        合成のデータセット

    Returns
    ----------
    None
    """
    # 骨組みと同じ求め方の IoU を記録にした実験から, 欄を省いて作る
    spec = mask_spec()
    best = skeleton_iou(spec, masks["dataset"])
    run = fake_run(tmp_path / "run", spec, "valid/iou", best)
    info = make_bundle(tmp_path / "bundle", spec, masks["dataset"], run)
    assert info["format_version"] == 4
    assert info["items"] == {
        "input": "image", "target": "mask", "prepare": "image",
        "options": {}, "values": [0, 1, 2], "restore": None,
    }
    assert info["output"] == {"form": "mask",
                              "classes": ["background", "circle", "square"]}
    assert info["target"] == {"kind": "mask",
                              "origin": {"type": "synthetic", "name": None,
                                         "settings": {}}}
    assert info["check"]["metric"] == "iou"
    assert info["check"]["matches"] is True, info["check"]
    assert info["uncertainty"] is None
    assert sorted(p.name for p in (tmp_path / "bundle" / "dataset")
                  .iterdir()) == ["dataset.yaml"]

    # データセットと元の画像のディレクトリは同じ予測 (領域の値)
    bundle = load_bundle(tmp_path / "bundle")
    by_dataset = predict(bundle, masks["dataset"])
    by_files = predict(bundle, masks["raw"]["images"])
    assert (by_dataset["status"] == "ok").all() and len(by_files) == 12
    for a, b in zip(by_dataset["output"], by_files["output"]):
        assert a.dtype == np.int64 and np.array_equal(a, b)
        assert set(np.unique(a)) <= {0, 1, 2}

    # 結果は artifacts/ に配列で
    result = infer(bundle, masks["raw"]["images"])
    write_result(result, tmp_path / "result")
    back = read_result(tmp_path / "result")
    assert back["form"] == "mask" and len(back["frame"]) == 12
    described = describe_bundle(bundle)
    assert described["kind"] == "items" and described["target"] is None
    assert described["items"]["input_kind"] == "image"

    # 記録と合わなければ注意を残す (作るのは止めない)
    other = fake_run(tmp_path / "other", spec, "valid/iou", best + 0.3)
    info = make_bundle(tmp_path / "bundle", spec, masks["dataset"], other,
                       overwrite=True)
    assert info["check"]["matches"] is False
    assert any("1 件の扱い" in note for note in info["notes"])


def test_mask_bundle_skips_and_errors(tmp_path: Path, masks: dict) -> None:
    """
    チャンネルの数の違う画像を飛ばし, 作れない組と食い違う一式で止まる
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    masks: dict
        合成のデータセット

    Returns
    ----------
    None
    """
    # 灰色の画像は学習の 3 チャンネルと違うので飛ばす
    spec = mask_spec()
    run = fake_run(tmp_path / "run", spec, "valid/loss", 1.0)
    make_bundle(tmp_path / "bundle", spec, masks["dataset"], run)
    images = tmp_path / "new"
    images.mkdir()
    Image.fromarray(np.zeros((24, 32), dtype=np.uint8)).save(
        images / "gray.png"
    )
    Image.fromarray(np.zeros((24, 32, 3), dtype=np.uint8)).save(
        images / "color.png"
    )
    frame = predict(tmp_path / "bundle", images).set_index("path")
    assert frame.loc["gray.png", "status"] == "skipped"
    assert "チャンネル" in frame.loc["gray.png", "reason"]
    assert frame.loc["color.png", "status"] == "ok"

    # 版 2 に使えない引数と, 正解の種類と違う形
    cases = [
        ({"target_scale": "train"}, "target_scale"),
        ({"form": "point_cloud"}, "正解の種類"),
        ({"items": {"prepare": "points"}}, "扱い方"),
        ({"items": {"input": "nothing"}}, "項目にありません"),
        ({"items": {"options": {"points": 8}}}, "items.options"),
    ]
    for change, message in cases:
        with pytest.raises(ValueError, match=message):
            make_bundle(tmp_path / "again", spec, masks["dataset"], run,
                        **change)

    # 一式の版と items の食い違い
    file = tmp_path / "bundle" / "inference.yaml"
    original = yaml.safe_load(file.read_text(encoding="utf-8"))
    for change, message in (
        ({"format_version": 3}, "items を持てません"),
        ({"items": None}, "items が要り"),
        ({"items": {**original["items"], "prepare": "points",
                    "options": {"points": 8}}}, "扱い方"),
        ({"items": {**original["items"], "values": None}}, "values"),
    ):
        file.write_text(yaml.safe_dump({**original, **change},
                                       allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_bundle(tmp_path / "bundle")


def test_cloud_bundle(tmp_path: Path, clouds: dict) -> None:
    """
    撮影と点群の版 2 のデータセットから一式を作り, 記録と比べ, 元の撮影を
    元の座標の点群で推論できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    clouds: dict
        合成のデータセット

    Returns
    ----------
    None
    """
    # 点の数は省けない
    spec = cloud_spec(32)
    options = {"points": 64, "target_points": 128, "seed": 1}
    best = skeleton_chamfer(spec, clouds["dataset"], (64, 128), 1)
    run = fake_run(tmp_path / "run", spec, "valid/chamfer", best)
    with pytest.raises(ValueError, match="options.points"):
        make_bundle(tmp_path / "bundle", spec, clouds["dataset"], run)

    # 骨組みと同じ選び方なので記録と合う
    info = make_bundle(tmp_path / "bundle", spec, clouds["dataset"], run,
                       items={"options": options})
    assert info["items"]["prepare"] == "capture_points"
    assert info["items"]["restore"] == "shift"
    assert info["check"]["metric"] == "chamfer"
    assert info["check"]["matches"] is True, info["check"]

    # データセット (正規化した後) と元の撮影 (推論で正規化) は, 元の座標
    # で同じ点群. 正規化した座標の出力から shift を引いたもの
    by_dataset = predict(tmp_path / "bundle", clouds["dataset"])
    by_raw = predict(tmp_path / "bundle", clouds["raw"]["captures"])
    assert (by_raw["status"] == "ok").all() and len(by_raw) == 8
    raw = by_raw.assign(id=[Path(p).name for p in by_raw["path"]])
    raw = raw.set_index("id")
    for _, row in by_dataset.iterrows():
        other = raw.loc[row["id"]]
        assert np.allclose(other["shift_x"], row["shift_x"])
        assert np.allclose(other["output"], row["output"], atol=1e-5)
        assert row["output"].shape == (32, 3)

    # 戻さない一式は正規化した座標のまま
    make_bundle(tmp_path / "plain", spec, clouds["dataset"], run,
                items={"options": options, "restore": None})
    plain = predict(tmp_path / "plain", clouds["dataset"])
    shift = by_dataset[["shift_x", "shift_y", "shift_z"]].to_numpy()
    for index, output in enumerate(plain["output"]):
        assert np.allclose(output - shift[index], by_dataset["output"][index],
                           atol=1e-5)

    # 結果の書き出し
    result = infer(tmp_path / "bundle", clouds["raw"]["captures"])
    write_result(result, tmp_path / "result")
    back = read_result(tmp_path / "result")
    assert back["form"] == "point_cloud"
    assert (tmp_path / "result" / "artifacts" / "000000.npy").is_file()


def test_resized_mask_bundle(tmp_path: Path) -> None:
    """
    大きさをそろえた画像と領域の版 3 のデータセットから一式を作り, 大きさ
    の違う元の画像を, 書き出しと同じ手順をかけて推論できることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 24 x 32 と 30 x 40 の混ざった画像を 12 x 16 にそろえて書き出す
    raw = write_image_masks(tmp_path / "raw", count=12, seed=0,
                            sizes=[(24, 32), (30, 40)])
    steps = [{"op": "resize", "width": 16, "height": 12}]
    dataset = tmp_path / "small"
    export_image_masks(raw["images"], raw["masks"], dataset,
                       classes=raw["classes"], origin={"type": "synthetic"},
                       item_steps={"image": steps})
    assert load_items(dataset).info["format_version"] == 3

    # 一式の写しに手順が残り, 記録と合う
    spec = mask_spec()
    best = skeleton_iou(spec, dataset)
    run = fake_run(tmp_path / "run", spec, "valid/iou", best)
    info = make_bundle(tmp_path / "bundle", spec, dataset, run)
    assert info["format_version"] == 4
    assert info["check"]["matches"] is True, info["check"]
    copied = yaml.safe_load((tmp_path / "bundle" / "dataset" /
                             "dataset.yaml").read_text(encoding="utf-8"))
    assert copied["format_version"] == 3
    assert copied["entries"]["image"]["steps"] == steps

    # データセットの件と大きさの違う元の画像は, 同じ 12 x 16 の予測
    bundle = load_bundle(tmp_path / "bundle")
    by_dataset = predict(bundle, dataset)
    by_files = predict(bundle, raw["images"])
    assert (by_files["status"] == "ok").all() and len(by_files) == 12
    for a, b in zip(by_dataset["output"], by_files["output"]):
        assert a.shape == (12, 16) and np.array_equal(a, b)

    # 手順の違うデータセット (版 2) は, 黙って読まずに止める
    export_image_masks(raw["images"], raw["masks"], tmp_path / "full",
                       classes=raw["classes"], origin={"type": "synthetic"})
    with pytest.raises(ValueError, match="中身の手順が一式と違います"):
        predict(bundle, tmp_path / "full")


def test_resized_cloud_bundle(tmp_path: Path) -> None:
    """
    画像を縮めた撮影と点群の版 3 のデータセットから一式を作り, 元の撮影
    を, 書き出しと同じ手順をかけて推論できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 24 x 32 の撮影を 12 x 16 に縮めて書き出す
    raw = write_shape_captures(tmp_path / "raw", count=8, seed=0,
                               points=400, views=6)
    steps = [{"op": "resize", "width": 16, "height": 12}]
    dataset = tmp_path / "small"
    export_capture_clouds(raw["captures"], raw["clouds"], dataset,
                          split={"method": "random", "valid": 0.25,
                                 "test": 0.0, "seed": 0},
                          item_steps={"capture": steps})

    # 縮めた深度から選ぶ点で記録と合う
    spec = cloud_spec(32)
    options = {"points": 64, "target_points": 128, "seed": 1}
    best = skeleton_chamfer(spec, dataset, (64, 128), 1)
    run = fake_run(tmp_path / "run", spec, "valid/chamfer", best)
    info = make_bundle(tmp_path / "bundle", spec, dataset, run,
                       items={"options": options})
    assert info["check"]["matches"] is True, info["check"]

    # 元の撮影も同じ手順で縮めるので, データセットの件と同じ点群
    by_dataset = predict(tmp_path / "bundle", dataset)
    by_raw = predict(tmp_path / "bundle", raw["captures"])
    assert (by_raw["status"] == "ok").all() and len(by_raw) == 8
    raw_rows = by_raw.assign(id=[Path(p).name for p in by_raw["path"]])
    raw_rows = raw_rows.set_index("id")
    for _, row in by_dataset.iterrows():
        other = raw_rows.loc[row["id"]]
        assert np.allclose(other["output"], row["output"], atol=1e-5)
