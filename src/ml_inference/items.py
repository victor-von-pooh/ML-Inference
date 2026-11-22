from pathlib import Path
import tempfile
from typing import Any, Optional

import numpy as np
from ml_eda import ItemDataset, normalize_capture, read_capture

# 入力の 1 件の扱い方と, 扱える入力の項目の種類
PREPARE_KINDS = {
    "image": "image",
    "capture_points": "capture",
    "points": "point_cloud",
}
PREPARES = tuple(PREPARE_KINDS)

# 点を選ぶ扱い方
POINT_PREPARES = ("capture_points", "points")

# 版 2 のデータセットから作れる出力の形 (正解の種類と同じ)
ITEM_FORMS = ("mask", "point_cloud")

# 出力を元の座標に戻す方法 (正規化の平行移動を引く)
RESTORES = ("shift",)

# items の欄
ITEM_KEYS = ("input", "target", "prepare", "options", "values", "restore")

# 扱い方ごとの設定の名前
OPTION_NAMES = {
    "image": (),
    "capture_points": ("points", "target_points", "seed"),
    "points": ("points", "target_points", "seed"),
}

# 名前の無い値 0 の領域のクラスの名前
BACKGROUND = "background"


def check_items(items: Any) -> list[str]:
    """
    inference.yaml の items の形を確かめる関数

    Parameters
    ----------
    items: Any
        items の欄

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 欄の名前と型
    if not isinstance(items, dict):
        return ["items は辞書にしてください"]
    errors = []
    unknown = [str(k) for k in items if k not in ITEM_KEYS]
    if unknown:
        errors.append(f"items に知らない欄があります: {', '.join(unknown)}")
    if not isinstance(items.get("input"), str):
        errors.append("items.input (入力の項目の名前) が要ります")
    if items.get("target") is not None and not isinstance(
        items["target"], str
    ):
        errors.append("items.target は項目の名前にしてください")
    prepare = items.get("prepare")
    if prepare not in PREPARES:
        errors.append(
            f"items.prepare は {', '.join(PREPARES)} のどれかにしてください"
        )
    options = items.get("options") or {}
    if not isinstance(options, dict):
        errors.append("items.options は辞書にしてください")
        options = {}
    errors.extend(check_options(prepare, options))
    values = items.get("values")
    if values is not None and not (
        isinstance(values, list) and values and all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 0
            for v in values
        ) and len(set(values)) == len(values)
    ):
        errors.append("items.values は 0 以上の重ならない整数の並びにして"
                      "ください")
    if items.get("restore") not in (None, *RESTORES):
        errors.append("items.restore は shift か null にしてください")

    return errors


def check_options(prepare: Any, options: dict) -> list[str]:
    """
    1 件の扱い方の設定を確かめる関数

    Parameters
    ----------
    prepare: Any
        扱い方の名前
    options: dict
        設定

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 知らない設定と, 点の数と種の型
    if prepare not in OPTION_NAMES:
        return []
    errors = []
    unknown = [str(k) for k in options if k not in OPTION_NAMES[prepare]]
    if unknown:
        errors.append(
            f"items.options の {', '.join(unknown)} は扱い方 {prepare} に"
            "ありません"
        )
    if prepare in POINT_PREPARES:
        for key in ("points", "target_points"):
            value = options.get(key)
            if (key == "points" or value is not None) and not (
                isinstance(value, int) and not isinstance(value, bool)
                and value > 0
            ):
                errors.append(f"items.options.{key} は 1 以上の整数にして"
                              "ください")
        seed = options.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            errors.append("items.options.seed は 0 以上の整数にしてください")

    return errors


def mask_classes(entry: dict) -> tuple[list[int], list[str]]:
    """
    領域の項目の説明から, クラスの番号ごとの値と名前を決める関数

    ML-Quest の骨組みと同じ決まり. classes の値 (無ければ values) の小さい
    順で, 名前の無い値 0 は背景として足す

    Parameters
    ----------
    entry: dict
        データセットの領域の項目の説明

    Returns
    ----------
    classes: tuple[list[int], list[str]]
        番号ごとの領域の値と名前

    Raises
    ----------
    ValueError
        名前も値の一覧も無い場合と, クラスが 2 つ未満の場合
    """
    # 名前の付いた値 (名前が無ければ値の一覧の全て)
    values = [int(v) for v in entry.get("values") or []]
    named = entry.get("classes") or [
        {"value": v, "name": str(v)} for v in values
    ]
    if not named:
        raise ValueError(
            "領域の値の名前 (classes) も値の一覧 (values) もありません. "
            "items の values で, クラスの番号ごとの領域の値を渡してください"
        )
    names = {int(item["value"]): str(item["name"]) for item in named}
    if 0 in values and 0 not in names:
        names[0] = BACKGROUND
    if len(names) < 2:
        raise ValueError(f"領域のクラスが 2 つ未満です: {names}")
    ordered = sorted(names)
    classes = (ordered, [names[v] for v in ordered])

    return classes


def resolve_items(
    dataset: ItemDataset, form: str, given: Optional[dict]
) -> tuple[dict, Optional[list[str]]]:
    """
    版 2 のデータセットと渡した欄から, 一式の items を決める関数

    Parameters
    ----------
    dataset: ItemDataset
        版 2 のデータセット
    form: str
        出力の形 (mask か point_cloud)
    given: Optional[dict]
        make_bundle に渡した items (省いた欄はデータセットから)

    Returns
    ----------
    resolved: tuple[dict, Optional[list[str]]]
        items と, 領域のクラスの名前 (領域でなければ None)

    Raises
    ----------
    ValueError
        項目が無い場合, 扱い方と項目の種類が合わない場合, 欄に誤りが
        ある場合
    """
    # 入力と正解の項目
    given = dict(given or {})
    entries = dataset.entries
    inputs = [n for n, e in entries.items() if e.get("role") == "input"]
    targets = list((dataset.info.get("target") or {}).get("entries") or [])
    name = given.get("input", inputs[0] if inputs else None)
    target = given.get("target", targets[0] if targets else None)
    for label, entry in (("input", name), ("target", target)):
        if entry is not None and entry not in entries:
            raise ValueError(
                f"items.{label} の {entry} はデータセットの項目にありません "
                f"(項目: {', '.join(entries)})"
            )
    if name is None:
        raise ValueError("データセットに入力の項目がありません")

    # 扱い方は入力の項目の種類から
    kind = entries[name].get("kind")
    default = next((p for p, k in PREPARE_KINDS.items() if k == kind), None)
    prepare = given.get("prepare", default)
    if prepare is None or PREPARE_KINDS.get(prepare) != kind:
        raise ValueError(
            f"入力の項目 {name} (種類 {kind}) の扱い方を決められません "
            f"(扱えるのは {', '.join(sorted(set(PREPARE_KINDS.values())))})"
        )
    if form == "mask" and prepare != "image":
        raise ValueError("領域の一式の入力は画像 (扱い方 image) だけです")

    # 領域の値と名前, 点群の戻し方
    names = None
    values = given.get("values")
    if form == "mask":
        if target is None:
            raise ValueError("領域の正解の項目がありません")
        ordered, labels = mask_classes(entries[target])
        if values is None:
            values, names = ordered, labels
        else:
            values = [int(v) for v in values]
            lookup = dict(zip(ordered, labels))
            names = [lookup.get(v, str(v)) for v in values]
    normalized = bool(dataset.info.get("normalization"))
    restore = given.get("restore", "shift" if (
        form == "point_cloud" and prepare == "capture_points" and normalized
    ) else None)
    items = {
        "input": name, "target": target, "prepare": prepare,
        "options": dict(given.get("options") or {}),
        "values": values if form == "mask" else None,
        "restore": restore,
    }
    unknown = [k for k in given if k not in ITEM_KEYS]
    errors = check_items(items) + (
        [f"items に知らない欄があります: {', '.join(map(str, unknown))}"]
        if unknown else []
    )
    if errors:
        raise ValueError("items の誤り: " + " / ".join(errors))
    resolved = (items, names)

    return resolved


def defined_points(shape: Any) -> Optional[int]:
    """
    モデルの定義の入力の形から, 1 件の点の数を読む関数

    1 件の点群は (点の数, 3) なので, 件の軸の後ろの 2 つの軸のうち, 3 の
    軸を座標の軸 (後ろを先に見る. 前なら座標が前の並び), 他方を点の数の
    軸とする

    Parameters
    ----------
    shape: Any
        定義の入力の shape (件の軸を含む)

    Returns
    ----------
    points: Optional[int]
        点の数 (軸を決められない場合と, 可変 (None) の場合は None)
    """
    # 件の軸と, 点の数と座標の 2 つの軸の形だけを読む
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        return None
    first, second = shape[1], shape[2]
    if second == 3:
        size = first
    elif first == 3:
        size = second
    else:
        return None
    points = size if isinstance(size, int) and size > 0 else None

    return points


def check_point_count(spec: dict, inputs: Optional[dict], items: dict) -> None:
    """
    点を選ぶ一式で, 定義の入力の点の数と items.options.points を比べる
    関数

    Parameters
    ----------
    spec: dict
        モデルの定義 (ML-ModelControl の形)
    inputs: Optional[dict]
        モデルの入力ごとの source (書いていない入力は items)
    items: dict
        一式の 1 件の扱い

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        items を渡す入力の点の数が items.options.points と違う場合
    """
    # 点を選ぶ扱い方で, items を渡す入力だけを比べる
    if items.get("prepare") not in POINT_PREPARES:
        return
    chosen = int((items.get("options") or {})["points"])
    written = inputs or {}
    for name, defined in (spec.get("inputs") or {}).items():
        source = (written.get(name) or {}).get("source", "items")
        if source != "items" or not isinstance(defined, dict):
            continue
        points = defined_points(defined.get("shape"))
        if points is not None and points != chosen:
            raise ValueError(
                f"モデルの定義の入力 {name} の点の数は {points} (shape "
                f"{defined.get('shape')}) ですが, 一式の点の数 "
                f"items.options.points は {chosen} です. 学習のスクリプト"
                "が選んだ点の数が正しい方に合わせ, items.options.points "
                f"か定義の inputs.{name}.shape を直してください"
            )


def backproject(capture: dict) -> np.ndarray:
    """
    撮影の深度を, 撮影の世界の座標の点に戻す関数

    Parameters
    ----------
    capture: dict
        load_items か read_capture の撮影 (intrinsics, poses, depth,
        convention)

    Returns
    ----------
    points: np.ndarray
        (点の数, 3) の点

    Raises
    ----------
    ValueError
        扱えない座標の決まり, 深度が無い場合, 点が無い場合
    """
    # 扱うのは OpenGL の軸 (+X 右, +Y 上, -Z 前) と, カメラの前の向きの
    # 距離の深度 (z) だけ
    convention = capture["convention"]
    axes = str(convention.get("camera_axes", "OpenGL"))
    if not axes.startswith("OpenGL") or convention.get("depth") != "z":
        raise ValueError(f"扱えない撮影の座標の決まりです: {convention}")
    if capture["depth"] is None:
        raise ValueError("深度の無い枚があります")
    intr = capture["intrinsics"]
    center = float(convention.get("pixel_center", 0.5))
    clouds = []
    for pose, depth in zip(capture["poses"], capture["depth"]):
        if not np.all(np.isfinite(pose)):
            continue
        rows, columns = np.nonzero(depth > 0)
        value = depth[rows, columns].astype(np.float64)

        # 深度の大きさが画像と違う場合の, カメラの内部の値の縮め
        sx = depth.shape[1] / intr["w"] if intr.get("w") else 1.0
        sy = depth.shape[0] / intr["h"] if intr.get("h") else 1.0

        # 画素の中心を通る光線の点の, カメラの座標から世界の座標への変換
        camera = np.stack([
            (columns + center - intr["cx"] * sx) / (intr["fl_x"] * sx)
            * value,
            -(rows + center - intr["cy"] * sy) / (intr["fl_y"] * sy)
            * value,
            -value,
        ], axis=1)
        clouds.append(camera @ pose[:3, :3].T + pose[:3, 3])
    points = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    if len(points) == 0:
        raise ValueError("深度から戻せる点がありません")

    return points


def choose(
    points: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """
    点群から決まった数の点を選ぶ関数

    Parameters
    ----------
    points: np.ndarray
        (点の数, 3) の点
    count: int
        選ぶ数 (足りなければ重ねて選ぶ)
    rng: np.random.Generator
        選ぶ乱数

    Returns
    ----------
    chosen: np.ndarray
        (count, 3) の float32

    Raises
    ----------
    ValueError
        点が無い場合
    """
    # 点が足りないときだけ重ねての選び
    if len(points) == 0:
        raise ValueError("点の無い点群があります")
    index = rng.choice(len(points), count, replace=len(points) < count)
    chosen = np.asarray(points)[index].astype(np.float32)

    return chosen


def prepare_item(
    items: dict, content: Any, position: int
) -> tuple[np.ndarray, Optional[np.random.Generator]]:
    """
    1 件の入力の中身に, 学習のときと同じ扱いをかける関数

    Parameters
    ----------
    items: dict
        一式の items
    content: Any
        入力の項目の中身 (画像の配列, 撮影の辞書, 点の配列)
    position: int
        分割か渡したデータの中の位置 (点を選ぶ乱数に使う)

    Returns
    ----------
    prepared: tuple[np.ndarray, Optional[np.random.Generator]]
        モデルに渡す 1 件の配列と, 点を選んだ乱数 (続けて正解を選ぶ.
        点を選ばない扱い方は None)

    Raises
    ----------
    ValueError
        中身の形が扱い方に合わない場合
    """
    # 画像は (チャンネル, 高さ, 幅) の 0 から 1
    if items["prepare"] == "image":
        image = np.asarray(content)
        if image.ndim == 2:
            image = image[:, :, None]
        if image.ndim != 3:
            raise ValueError(f"画像の形が {image.shape} です")
        array = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        prepared = (array, None)
        return prepared

    # 点は件ごとに決まった乱数で選ぶ (撮影は深度から点に)
    options = items.get("options") or {}
    rng = np.random.default_rng([int(options.get("seed", 0)), position])
    points = backproject(content) if isinstance(content, dict) else (
        np.asarray(content)
    )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"点群の形が {points.shape} です ((点の数, 3) だけ)")
    prepared = (choose(points, int(options["points"]), rng), rng)

    return prepared


def target_points(
    items: dict, content: Any, rng: Optional[np.random.Generator]
) -> np.ndarray:
    """
    確かめで比べる正解の点を選ぶ関数

    Parameters
    ----------
    items: dict
        一式の items
    content: Any
        正解の点群 ((点の数, 3))
    rng: Optional[np.random.Generator]
        入力の点を選んだ乱数 (続けて選ぶ)

    Returns
    ----------
    points: np.ndarray
        target_points 個の点 (省いた場合は全ての点)
    """
    # 入力に続けて同じ乱数で選ぶ (学習のスクリプトと同じ順)
    count = (items.get("options") or {}).get("target_points")
    points = np.asarray(content, dtype=np.float32)
    if count is not None and rng is not None:
        points = choose(points, int(count), rng)

    return points


def mask_counts(
    predicted: np.ndarray, truth: np.ndarray, values: list[int]
) -> np.ndarray:
    """
    領域の値ごとの重なり, 和, 正解の画素数を数える関数

    Parameters
    ----------
    predicted: np.ndarray
        予測した領域の値 (高さ, 幅)
    truth: np.ndarray
        正解の領域の値 (高さ, 幅)
    values: list[int]
        クラスの番号ごとの領域の値 (無い値の正解の画素は数えない)

    Returns
    ----------
    counts: np.ndarray
        (3, クラスの数) の重なり, 和, 正解の画素数
    """
    # 正解の値が表にある画素だけ
    known = np.isin(truth, values)
    counts = np.zeros((3, len(values)))
    for index, value in enumerate(values):
        guess = (predicted == value) & known
        right = truth == value
        counts[0, index] = np.sum(guess & right)
        counts[1, index] = np.sum(guess | right)
        counts[2, index] = np.sum(right)

    return counts


def restore_size(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    領域を, 最も近い画素の値で元の画像の大きさに戻す関数

    Parameters
    ----------
    mask: np.ndarray
        予測した領域の値 (高さ, 幅)
    size: tuple[int, int]
        元の画像の (高さ, 幅)

    Returns
    ----------
    restored: np.ndarray
        元の画像の大きさの領域の値 (値を混ぜないので, mask の値だけ)
    """
    # 戻した画素の中心に最も近い出力の画素 (ML-EDA の書き出しが領域を
    # そろえる resize_mask と同じ式. テストが ML-EDA と比べる)
    height, width = int(size[0]), int(size[1])
    rows = np.minimum(
        ((np.arange(height) + 0.5) * mask.shape[0] / height).astype(np.int64),
        mask.shape[0] - 1
    )
    columns = np.minimum(
        ((np.arange(width) + 0.5) * mask.shape[1] / width).astype(np.int64),
        mask.shape[1] - 1
    )
    restored = mask[np.ix_(rows, columns)]

    return restored


def mean_iou(counts: np.ndarray, values: list[int]) -> float:
    """
    背景 (値 0) を除くクラスの IoU の平均を求める関数

    Parameters
    ----------
    counts: np.ndarray
        mask_counts の (3, クラスの数) の画素数
    values: list[int]
        クラスの番号ごとの領域の値

    Returns
    ----------
    value: float
        IoU の平均 (予測にも正解にも無いクラスは 1)
    """
    # 和が 0 のクラス (空と空) は 1
    scores = [
        counts[0, i] / counts[1, i] if counts[1, i] > 0 else 1.0
        for i, v in enumerate(values) if v != 0
    ]
    value = float(np.mean(scores)) if scores else 1.0

    return value


def chamfer(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    1 件の対称な Chamfer 距離を求める関数

    Parameters
    ----------
    predicted: np.ndarray
        (予測の点の数, 3)
    target: np.ndarray
        (正解の点の数, 3)

    Returns
    ----------
    distance: float
        両向きの最も近い点までの距離の平均の和の半分
    """
    # 全ての組の距離から, 両向きに最も近い点
    a = np.asarray(predicted, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    pairs = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    distance = float((pairs.min(axis=1).mean() + pairs.min(axis=0).mean())
                     / 2)

    return distance


def captured(
    path: Path, normalize: bool, work: Path, label: str,
    steps: Optional[list] = None
) -> tuple[dict, np.ndarray]:
    """
    元の撮影を, 学習と同じ正規化と中身の手順をかけて読む関数

    Parameters
    ----------
    path: Path
        撮影のディレクトリ
    normalize: bool
        正規化の平行移動をかけるか
    work: Path
        写す先を作る一時のディレクトリ
    label: str
        写す先の名前
    steps: Optional[list] = None
        各枚の画像にかける中身の手順 (版 3 のデータセットの撮影の項目の
        steps)

    Returns
    ----------
    capture: tuple[dict, np.ndarray]
        read_capture の撮影と, 足した平行移動 (かけない場合は 0)

    Raises
    ----------
    ValueError
        正規化しないデータセットの撮影に中身の手順がある場合
    """
    # 正規化と中身の手順は ML-EDA の書き出しと同じ決まり (写してから
    # 姿勢に足し, 画像と深度と内部パラメータを直す)
    if not normalize:
        if steps:
            raise ValueError(
                "正規化しないデータセットの撮影には, 中身の手順をかけられ"
                "ません"
            )
        capture = (read_capture(path), np.zeros(3))
        return capture
    target = work / label
    summary = normalize_capture(path, target, steps=steps)
    capture = (read_capture(target), np.asarray(summary["shift"], float))

    return capture


def work_directory() -> tempfile.TemporaryDirectory:
    """
    元の撮影を写す一時のディレクトリを作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    work: tempfile.TemporaryDirectory
        一時のディレクトリ (使い終えたら消す)
    """
    # 推論のあいだだけの置き場
    work = tempfile.TemporaryDirectory(prefix="ml_inference_capture_")

    return work
