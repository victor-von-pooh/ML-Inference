import copy
import datetime as dt
import json
import math
from pathlib import Path
import shutil
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from ml_eda import Data, Dataset, describe_groups, load_data
from ml_modelcontrol import dump_spec, load_spec

from ml_inference.bundle import (
    FAR_GROUP_BUNDLE_VERSION, FAR_GROUP_VERSIONS, GROUP_BUNDLE_VERSION,
    READY_FAR_GROUP_BUNDLE_VERSION, READY_GROUP_BUNDLE_VERSION,
    READY_GROUP_VERSIONS, Bundle, bundle_notes, check_sourced, inside,
    load_bundle, plan_weights, ready_record, ready_spec, run_summary
)
from ml_inference.conclude import NO_PREDICTION
from ml_inference.ready import (
    READY_LIBRARIES, ReadyEmbedding, check_head, check_ready, read_images
)
from ml_inference.units import check_units

# モデルの入力と出力の名前 (ML-Quest のまとめるの骨組みと, ML-ModelControl
# の同梱の例 contrastive_encoder と同じ)
FEATURE_INPUT = "x"
EMBEDDING_OUTPUT = "embedding"

# 実験の中の成果物 (ML-Quest のまとめるの骨組みが書く名前)
CENTERS_FILE = "group_centers.json"
ASSIGNMENT_FILE = "group_assignments.csv"
STABILITY_FILE = "evaluations/group_stability.json"
SUMMARY_FILE = "evaluations/group_summary.json"

# 群の割り当ての表の, 行を見分ける列が無いときの行の番号の列と, 他の列
ROW_KEY = "row"
ASSIGNMENT_COLUMNS = ("group", "split")

# 群の安定性の行の指標 (それぞれに _baseline と _diff が付く)
STABILITY_METRICS = ("ari", "ari_min", "silhouette")

# 群の要約の行の, 群を作る列でない鍵と, 全ての行の行の group (ML-EDA の
# describe_groups と骨組みの群の要約と同じ)
SUMMARY_KEYS = ("group", "rows", "share", "stability")
ALL_ROWS = "全体"

# 種類の列の欠けた値を 1 つの種類にする文字 (骨組みと同じ)
MISSING = "(欠損)"

# 群の安定性の数え方の名前 (骨組みの STABILITY_NAMES と同じ)
STABILITY_NAMES = {"seed": "種を変えて作り直す",
                   "sample": "標本を変えて作り直す"}

# group の欄
GROUP_KEYS = ("centers", "k", "columns", "features", "width", "id", "rows",
              "stability", "summary", "check", "far")

# group.far の欄 (分位点, 距離を数えた学習の行の数, 群ごとの距離の上限)
FAR_KEYS = ("quantile", "rows", "thresholds")

# 群ごとの距離の上限にする, 学習の行の距離の分位点
FAR_QUANTILE = 0.99

# 群の中心から遠い行の印の列
FAR_COLUMN = "far"

# 上限と比べるときの幅 (割合と絶対の値. 一度に渡す件数やデバイスで変わる
# 埋め込みの丸めの差で, 上限ちょうどの学習の行に印を付けないため)
FAR_TOLERANCE = (1e-6, 1e-9)

# 学習の表を渡さずに作った一式の注意
NO_FAR = "学習の表を渡さなかったので, 群の中心から遠い行に印を付けません"

# 作るときに確かめる 0 の行の数
PROBE_ROWS = 2

# 学習の表を割り当て直したときに, 合うとみなす一致の割合
AGREEMENT = 0.99

# 予測の表に足す列
GROUP_FRAME_COLUMNS = ("status", "reason", "group", "distance")

# 予測の状態 (predict と同じ値)
OK = "ok"
SKIPPED = "skipped"

# 割り当てない行の理由
NOT_FINITE = "埋め込みに数でない値があります"

# 群ごとに返す代表の行の既定の件数
REPRESENTATIVES = 3

# 既製の埋め込みの群分けの, 行を見分ける列と群の要約の列 (ML-EDA の画像の
# 読み込みの表の列. 写真の場所と, 幅と高さの画素の数)
READY_ID = "path"
READY_COLUMNS = ("width", "height")

# 群の中心のファイルの, 埋め込みを作った既製のモデルの欄
CENTERS_READY_KEYS = ("library", "model", "weights", "sha256")


def is_whole(value: Any, lowest: int) -> bool:
    """
    値が決まった値以上の整数 (真偽値でない) かを返す関数

    Parameters
    ----------
    value: Any
        確かめる値
    lowest: int
        いちばん小さい値

    Returns
    ----------
    whole: bool
        lowest 以上の整数なら True
    """
    # 真偽値は整数の子なので除く
    whole = isinstance(value, (int, np.integer)) and not isinstance(
        value, bool
    ) and value >= lowest

    return whole


def is_number(value: Any) -> bool:
    """
    値が有限の数 (真偽値でない) かを返す関数

    Parameters
    ----------
    value: Any
        確かめる値

    Returns
    ----------
    number: bool
        有限の数なら True
    """
    # 真偽値と inf, NaN は除外
    number = isinstance(value, (int, float, np.integer, np.floating)) and (
        not isinstance(value, bool)
    ) and math.isfinite(float(value))

    return number


def check_encoding(encoding: Any) -> list[str]:
    """
    列の符号化 (骨組みの make_encoding の形) を確かめる関数

    Parameters
    ----------
    encoding: Any
        numeric (数の列ごとの mean と std) と categories (種類の列ごとの
        値の文字の並び) の辞書

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 数の列は有限の平均と正の標準偏差, 種類の列は文字の並び
    if not isinstance(encoding, dict) or not isinstance(
        encoding.get("numeric"), dict
    ) or not isinstance(encoding.get("categories"), dict):
        return ["列の符号化は numeric と categories の辞書にしてください"]
    errors = []
    for name, stats in encoding["numeric"].items():
        if not isinstance(stats, dict) or not is_number(stats.get("mean")) or (
            not is_number(stats.get("std")) or stats["std"] <= 0
        ):
            errors.append(f"数の列 {name} の mean は数, std は正の数にして"
                          "ください")
    for name, values in encoding["categories"].items():
        if not isinstance(values, list) or not all(
            isinstance(v, str) for v in values
        ) or len(set(values)) != len(values):
            errors.append(f"種類の列 {name} の値は重ならない文字の並びに"
                          "してください")
    names = [*encoding["numeric"], *encoding["categories"]]
    if not names:
        errors.append("列の符号化に列がありません")
    if len(set(names)) != len(names):
        errors.append("列の符号化の数の列と種類の列に同じ列があります")

    return errors


def encoding_width(encoding: dict) -> int:
    """
    符号化した行の幅を返す関数

    Parameters
    ----------
    encoding: dict
        列の符号化

    Returns
    ----------
    width: int
        数の列の数と, 種類の列ごとの値の数の和
    """
    # 数の列は 1 列, 種類の列は値ごとに 1 列
    width = len(encoding["numeric"]) + sum(
        len(values) for values in encoding["categories"].values()
    )

    return width


def read_centers(path: Path) -> dict:
    """
    列の符号化と群の中心 (group_centers.json) を読んで確かめる関数

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    found: dict
        k, encoding (列の符号化. 既製の埋め込みは None), ready (埋め込みを
        作った既製のモデル. 列の符号化は None), centers ((群の数, 幅) の
        float64 の配列)

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        形が違う場合
    """
    # 群の数, 列の符号化, 群の中心
    if not path.is_file():
        raise FileNotFoundError(f"列の符号化と群の中心 {path} がありません")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} が辞書ではありません")
    errors = []
    k = loaded.get("k")
    if not is_whole(k, 2):
        errors.append("k (群の数) は 2 以上の整数にしてください")
    if ("encoding" in loaded) == ("ready" in loaded):
        errors.append("encoding (列の符号化) か ready (埋め込みを作った既製"
                      "のモデル) のどちらか 1 つが要ります")
    elif "encoding" in loaded:
        errors.extend(check_encoding(loaded["encoding"]))
    else:
        errors.extend(check_centers_ready(loaded["ready"]))
    try:
        centers = np.asarray(loaded.get("centers"), dtype=np.float64)
    except (TypeError, ValueError):
        centers = np.zeros((0,))
    if centers.ndim != 2 or centers.shape[1] < 1 or (
        is_whole(k, 2) and len(centers) != k
    ) or not np.isfinite(centers).all():
        errors.append("centers は (群の数, 埋め込みの幅) の数の並びにして"
                      "ください")
    if errors:
        raise ValueError(f"{path.name} の誤り: " + " / ".join(errors))
    found = {"k": int(k), "encoding": loaded.get("encoding"),
             "ready": loaded.get("ready"), "centers": centers}

    return found


def check_centers_ready(ready: Any) -> list[str]:
    """
    群の中心のファイルの, 埋め込みを作った既製のモデルの欄を確かめる関数

    Parameters
    ----------
    ready: Any
        group_centers.json の ready

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 出どころ, モデル, 重みの名前と, 重みの hash
    if not isinstance(ready, dict):
        return ["ready は library, model, weights, sha256 の辞書にして"
                "ください"]
    errors = []
    unknown = [str(k) for k in ready if k not in CENTERS_READY_KEYS]
    if unknown:
        errors.append(f"ready に知らない欄があります: {', '.join(unknown)}")
    for key in ("library", "model", "weights"):
        if not isinstance(ready.get(key), str) or not ready.get(key):
            errors.append(f"ready.{key} に名前の文字が要ります")
    sha256 = ready.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64 or any(
        c not in "0123456789abcdef" for c in sha256
    ):
        errors.append("ready.sha256 は重みの hash (64 文字の小文字の 16 進)"
                      "にしてください")

    return errors


def ready_centers(path: Path) -> bool:
    """
    群の中心のファイルが既製の埋め込みのものかを返す関数

    Parameters
    ----------
    path: Path
        group_centers.json

    Returns
    ----------
    ready: bool
        読める辞書で ready を持てば True (形の誤りは read_centers で)
    """
    # 読めないファイルは学習した符号化器のものとしての確かめ
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    ready = isinstance(loaded, dict) and "ready" in loaded

    return ready


def read_assignments(path: Path, k: int) -> tuple[Optional[str], pd.Series]:
    """
    行ごとの群の割り当ての表 (group_assignments.csv) を読んで確かめる関数

    Parameters
    ----------
    path: Path
        ファイル
    k: int
        群の数

    Returns
    ----------
    found: tuple[Optional[str], pd.Series]
        行を見分ける列の名前 (行の番号なら None) と, 行を見分ける値を
        添字にした群の番号

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        列が違う場合と, 群の番号が範囲の外の場合
    """
    # 行を見分ける列, group, split の 3 列 (行を見分ける値は文字のまま)
    if not path.is_file():
        raise FileNotFoundError(f"群の割り当ての表 {path} がありません")
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = list(table.columns)
    if len(columns) != 3 or tuple(columns[1:]) != ASSIGNMENT_COLUMNS:
        raise ValueError(
            f"{path.name} の列は, 行を見分ける列, group, split にして"
            f"ください (今は {', '.join(columns)})"
        )
    groups = pd.to_numeric(table["group"], errors="coerce")
    if table.empty or groups.isna().any() or not groups.between(
        0, k - 1
    ).all() or (groups % 1 != 0).any():
        raise ValueError(
            f"{path.name} の group は 0 から {k - 1} の整数にしてください"
        )
    key = None if columns[0] == ROW_KEY else columns[0]
    found = (key, pd.Series(groups.astype(int).to_numpy(),
                            index=table[columns[0]].to_numpy()))

    return found


def read_evaluation(path: Path) -> dict:
    """
    ML-Found の評価の表 (evaluations/<名前>.json) を読む関数

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    evaluation: dict
        評価の辞書 (rows は行の辞書の並び)

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        rows が行の辞書の並びでない場合
    """
    # ML-Found の save_evaluation の形
    if not path.is_file():
        raise FileNotFoundError(f"評価の表 {path} がありません")
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    rows = evaluation.get("rows") if isinstance(evaluation, dict) else None
    if not isinstance(rows, list) or not rows or not all(
        isinstance(row, dict) for row in rows
    ):
        raise ValueError(f"{path.name} の rows は行の辞書の並びにして"
                         "ください")

    return evaluation


def stability_of(evaluation: dict, k: int) -> dict:
    """
    群の安定性の評価の表から, 一式に写す群の安定性を作る関数

    Parameters
    ----------
    evaluation: dict
        evaluations/group_stability.json の中身
    k: int
        群の数

    Returns
    ----------
    stability: dict
        表の 1 行の写しと, 比べる相手の名前 (baseline) とエポック (epoch)

    Raises
    ----------
    ValueError
        群の数が違う場合と, 指標が数でない場合
    """
    # 骨組みが書くのは 1 行だけ
    row = evaluation["rows"][0]
    if row.get("k") != k:
        raise ValueError(
            f"群の安定性の k {row.get('k')} が, 群の中心の数 {k} と違います"
        )
    names = [f"{name}{end}" for name in STABILITY_METRICS
             for end in ("", "_baseline", "_diff")]
    wrong = [name for name in names if not is_number(row.get(name))]
    if wrong:
        raise ValueError(
            f"群の安定性の行の {', '.join(wrong)} が数ではありません"
        )
    stability = {**copy.deepcopy(row),
                 "baseline": evaluation.get("baseline"),
                 "epoch": evaluation.get("epoch")}

    return stability


def summary_of(evaluation: dict, k: int, names: list) -> tuple[list, list]:
    """
    群の要約の評価の表を確かめ, 行の写しと群を作る列の順を返す関数

    Parameters
    ----------
    evaluation: dict
        evaluations/group_summary.json の中身
    k: int
        群の数
    names: list
        列の符号化の列の名前

    Returns
    ----------
    found: tuple[list, list]
        行の写し (群の番号の順と, 最後に全体) と, 群を作る列の名前
        (群の要約の列の順)

    Raises
    ----------
    ValueError
        群の行が 0 から k - 1 と全体でない場合と, 列が符号化と違う場合
    """
    # 群の番号の順の行と全体の行
    rows = copy.deepcopy(evaluation["rows"])
    groups = [row.get("group") for row in rows]
    if groups != [*range(k), ALL_ROWS]:
        raise ValueError(
            f"群の要約の group は 0 から {k - 1} と最後の {ALL_ROWS} に"
            "してください"
        )
    columns = [key for key in rows[-1] if key not in SUMMARY_KEYS]
    if sorted(columns) != sorted(names):
        raise ValueError(
            f"群の要約の列 ({', '.join(columns)}) が, 列の符号化の列 "
            f"({', '.join(names)}) と違います"
        )
    found = (rows, columns)

    return found


def check_group_info(info: dict) -> list[str]:
    """
    群分けの一式 (版 9, 10, 16, 17) の形を確かめる関数

    Parameters
    ----------
    info: dict
        inference.yaml の中身

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # group が要り, データセット, items, 手順, 正解, 意思決定を持たない
    version = info.get("format_version")
    ready = version in READY_GROUP_VERSIONS
    group = info.get("group")
    if not isinstance(group, dict):
        return [f"版 {version} の一式には group (群の中心) が要ります"]
    errors = []
    unknown = [str(k) for k in group if k not in GROUP_KEYS]
    if unknown:
        errors.append(f"group の知らない項目: {', '.join(unknown)}")
    name = group.get("centers")
    if not isinstance(name, str) or not name or Path(name).name != name:
        errors.append("group.centers は一式の中のファイルの名前にして"
                      "ください")
    if not is_whole(group.get("k"), 2):
        errors.append("group.k は 2 以上の整数にしてください")
    for key in ("width", "rows") if ready else ("features", "width", "rows"):
        if not is_whole(group.get(key), 1):
            errors.append(f"group.{key} は 1 以上の整数にしてください")
    columns = group.get("columns")
    if not isinstance(columns, list) or not columns or not all(
        isinstance(c, str) for c in columns
    ) or len(set(columns)) != len(columns):
        errors.append("group.columns は重ならない列の名前の並びにして"
                      "ください")
    if group.get("id") is not None and not isinstance(group["id"], str):
        errors.append("group.id は列の名前か null にしてください")
    if not isinstance(group.get("stability"), dict):
        errors.append("group.stability は群の安定性の辞書にしてください")
    if not isinstance(group.get("summary"), list):
        errors.append("group.summary は群の要約の行の並びにしてください")
    check = group.get("check")
    if check is not None and not (
        isinstance(check, dict) and is_whole(check.get("rows"), 1)
        and is_number(check.get("agreement"))
    ):
        errors.append("group.check は rows と agreement の辞書か null に"
                      "してください")
    if version in FAR_GROUP_VERSIONS:
        errors.extend(check_far(group.get("far"), group.get("k")))
    elif group.get("far") is not None:
        errors.append(
            f"版 {version} の一式は group.far を持てません (遠さの上限を持つ"
            f"一式は版 {FAR_GROUP_BUNDLE_VERSION} か "
            f"{READY_FAR_GROUP_BUNDLE_VERSION})"
        )
    if ready:
        errors.extend(check_ready_group(info))
    held = [k for k in ("dataset", "items", "procedure", "decision",
                        "check", "uncertainty", "target", "calibrate",
                        "rank", *(() if ready else ("ready",)))
            if info.get(k) is not None]
    if held:
        errors.append(
            f"群分けの一式の {', '.join(held)} は null にしてください "
            "(データセットと正解を持たないため)"
        )
    if info.get("conclusion"):
        errors.append("群分けの一式は結論の決め方の既定値を持てません")
    if info.get("inputs"):
        errors.append("群分けの一式の inputs は空にしてください (入力は "
                      "符号化した行の特徴 x か画像に決まっています)")
    if info.get("role") != "group":
        errors.append("群分けの一式の role は group にしてください")
    model = info.get("model") if isinstance(info.get("model"), dict) else {}
    if model.get("target_scale") is not None or model.get("output"):
        errors.append("群分けの一式は model.target_scale と model.output を"
                      "持てません")

    return errors


def check_ready_group(info: dict) -> list[str]:
    """
    既製の埋め込みの群分けの一式 (版 10 と 17) の ready と group を確かめる
    関数

    Parameters
    ----------
    info: dict
        inference.yaml の中身

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # ready が要り, 重みは出どころから. 符号化した行の幅を持たず, 行を
    # 見分ける列と群の要約の列は画像の読み込みの表の列
    if info.get("ready") is None:
        return [f"版 {info.get('format_version')} の一式には ready (既製の"
                "モデルの出どころと前処理) が要ります"]
    errors = check_ready(info["ready"])
    model = info.get("model") if isinstance(info.get("model"), dict) else {}
    errors.extend(check_sourced(model))
    group = info["group"]
    if group.get("features") is not None:
        errors.append("既製の埋め込みの群分けの一式の group.features は null"
                      "にしてください")
    if group.get("id") != READY_ID:
        errors.append(f"既製の埋め込みの群分けの一式の group.id は "
                      f"{READY_ID} にしてください")
    if group.get("columns") != list(READY_COLUMNS):
        errors.append("既製の埋め込みの群分けの一式の group.columns は "
                      f"[{', '.join(READY_COLUMNS)}] にしてください")

    return errors


def check_far(far: Any, k: Any) -> list[str]:
    """
    群の中心から遠い行の上限 (group.far) を確かめる関数

    Parameters
    ----------
    far: Any
        group.far
    k: Any
        group.k (群の数)

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 分位点, 行の数, 群の数の長さの上限の並び (行の無い群は null)
    if not isinstance(far, dict):
        return ["group.far は quantile, rows, thresholds の辞書にしてください"]
    errors = []
    unknown = [str(key) for key in far if key not in FAR_KEYS]
    if unknown:
        errors.append(f"group.far の知らない欄: {', '.join(unknown)}")
    quantile = far.get("quantile")
    if not is_number(quantile) or not 0 < quantile < 1:
        errors.append("group.far.quantile は 0 より大きく 1 より小さい数に"
                      "してください")
    if not is_whole(far.get("rows"), 1):
        errors.append("group.far.rows は 1 以上の整数にしてください")
    thresholds = far.get("thresholds")
    if not isinstance(thresholds, list) or (
        is_whole(k, 2) and len(thresholds) != k
    ) or not all(t is None or (is_number(t) and t >= 0)
                 for t in thresholds):
        errors.append("group.far.thresholds は群の数の長さの, 0 以上の数か "
                      "null の並びにしてください")

    return errors


def check_group_spec(spec: dict, features: int) -> None:
    """
    モデルの定義が行の特徴の入力と埋め込みの出力の約束に合うかを確かめる
    関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    features: int
        符号化した行の幅

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        入力が x の 1 つでない場合, 出力に embedding が無い場合, 入力の
        形の最後の大きさが符号化した行の幅と違う場合
    """
    # 入力は行の特徴の 1 つ, 出力に埋め込み
    inputs = list(spec.get("inputs") or {})
    if inputs != [FEATURE_INPUT]:
        raise ValueError(
            f"群分けの一式のモデルの入力は {FEATURE_INPUT} の 1 つにして"
            f"ください (今は {', '.join(map(str, inputs)) or '無し'})"
        )
    if EMBEDDING_OUTPUT not in (spec.get("outputs") or {}):
        raise ValueError(
            f"群分けの一式のモデルの出力に {EMBEDDING_OUTPUT} が要ります "
            "(行の埋め込みを群の中心に割り当てるため)"
        )
    shape = (spec["inputs"][FEATURE_INPUT] or {}).get("shape")
    if isinstance(shape, list) and shape and shape[-1] is not None and (
        shape[-1] != features
    ):
        raise ValueError(
            f"モデルの入力 {FEATURE_INPUT} の形の最後の大きさ {shape[-1]} "
            f"が, 符号化した行の幅 {features} と違います"
        )


def category_text(value: Any) -> str:
    """
    種類の列の値を, 符号化に使う文字にする関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    text: str
        文字 (欠けた値は MISSING. 骨組みと同じ)
    """
    # 欠けた値も 1 つの種類
    text = MISSING if pd.isna(value) else str(value)

    return text


def encode_rows(
    frame: pd.DataFrame, encoding: dict
) -> tuple[np.ndarray, list[str]]:
    """
    行を, 列の符号化で数の並びにする関数 (骨組みの encode_rows と同じ式)

    Parameters
    ----------
    frame: pd.DataFrame
        群を作る列を持つ表
    encoding: dict
        列の符号化

    Returns
    ----------
    encoded: tuple[np.ndarray, list[str]]
        (行の数, 符号化した行の幅) の float32 (数の列は標準化して欠けた
        値は 0, 種類の列は値ごとの 0 と 1. 学習用に無い値は全て 0) と,
        行ごとの割り当てない理由 (割り当てる行は空)
    """
    # 数の列で, 欠けでないのに数にできない値の行は割り当ての外
    reasons = [""] * len(frame)
    parts = [np.zeros((len(frame), 0))]
    for name, stats in encoding["numeric"].items():
        column = frame[name]
        values = pd.to_numeric(
            column.map(lambda v: float(v) if isinstance(v, bool) else v),
            errors="coerce"
        ).astype(float)
        wrong = values.isna().to_numpy() & column.notna().to_numpy()
        for row in np.flatnonzero(wrong):
            reasons[row] = reasons[row] or (
                f"列 {name} の値 {column.iloc[row]!r} を数にできません"
            )
        values = (values - stats["mean"]) / stats["std"]
        parts.append(values.fillna(0.0).to_numpy()[:, None])
    for name, values in encoding["categories"].items():
        text = frame[name].map(category_text).to_numpy()
        parts.append(np.stack([text == value for value in values], 1)
                     if values else np.zeros((len(frame), 0)))
    features = np.concatenate(parts, 1).astype(np.float32)
    encoded = (features, reasons)

    return encoded


def embed(
    model: Any, features: np.ndarray, batch_size: int, device: str,
    named: bool = True
) -> np.ndarray:
    """
    符号化した行か前処理した画像の埋め込みを, まとまりごとに求める関数

    Parameters
    ----------
    model: Any
        組み立てたモデル
    features: np.ndarray
        (件数, 符号化した行の幅) か (件数, 3, crop, crop) の float32
    batch_size: int
        1 回に渡す件数
    device: str
        使うデバイス
    named: bool = True
        出力が名前の辞書 (embedding を使う) か. 既製のモデルの埋め込みは
        False (出力が埋め込み)

    Returns
    ----------
    points: np.ndarray
        (件数, 埋め込みの幅) の float64

    Raises
    ----------
    ValueError
        出力が埋め込みを持つ名前の辞書でない場合と, 埋め込みが 2 次元で
        ない場合
    """
    # まとまりごとの埋め込みの出力
    parts = []
    for start in range(0, len(features), batch_size):
        chunk = torch.from_numpy(features[start:start + batch_size])
        with torch.no_grad():
            out = model(chunk.to(device))
        if named and (
            not isinstance(out, dict) or EMBEDDING_OUTPUT not in out
        ):
            raise ValueError(
                f"モデルの出力に {EMBEDDING_OUTPUT} がありません"
            )
        out = out[EMBEDDING_OUTPUT] if named else out
        found = out.detach().double().cpu().numpy()
        if found.ndim != 2 or len(found) != len(chunk):
            raise ValueError(
                f"埋め込みの形 {tuple(found.shape)} が (件数, 幅) では"
                "ありません"
            )
        parts.append(found)
    points = np.concatenate(parts) if parts else np.zeros((0, 0))

    return points


def assign(
    points: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    点を最も近い群の中心に割り当てる関数 (骨組みの assign と同じ式)

    Parameters
    ----------
    points: np.ndarray
        (件数, 幅) の点
    centers: np.ndarray
        (群の数, 幅) の群の中心

    Returns
    ----------
    assigned: tuple[np.ndarray, np.ndarray]
        (件数,) の群の番号 (同じ距離なら番号の小さい群) と, その中心まで
        の距離
    """
    # 内積で求める二乗距離
    points = np.asarray(points, dtype=np.float64)
    distance = ((points ** 2).sum(1)[:, None] - 2 * points @ centers.T
                + (centers ** 2).sum(1)[None, :])
    labels = distance.argmin(1)
    nearest = np.sqrt(np.maximum(
        np.take_along_axis(distance, labels[:, None], 1)[:, 0], 0.0
    ))
    assigned = (labels, nearest)

    return assigned


def read_table(
    source: Any, options: Optional[dict], what: str
) -> pd.DataFrame:
    """
    群に割り当てる表を読む関数

    Parameters
    ----------
    source: Any
        表のファイル, 表, 表の ML-EDA の Data
    options: Optional[dict]
        ファイルの読み込みの設定 (load_data に渡す. loader も書ける)
    what: str
        誤りの文に使う表の呼び名

    Returns
    ----------
    frame: pd.DataFrame
        表 (添字は 0 から)

    Raises
    ----------
    ValueError
        表でない場合と, 表と Data に読み込みの設定を渡した場合
    """
    # ファイルは ML-EDA の読み込み
    if isinstance(source, (str, Path)):
        data = load_data(Path(source).expanduser(), **dict(options or {}))
        source = data
    elif options:
        raise ValueError(
            f"{what}に表か Data を渡す場合は, 読み込みの設定 (options) を"
            "渡さないでください"
        )
    if isinstance(source, Data):
        if not isinstance(source.frame, pd.DataFrame) or source.kind not in (
            "table", "timeseries"
        ):
            raise ValueError(
                f"{what}は表にしてください ({source.kind} のデータです)"
            )
        source = source.frame
    if not isinstance(source, pd.DataFrame):
        raise ValueError(
            f"{what}は表のファイル, 表 (DataFrame), 表の Data にして"
            "ください"
        )
    frame = source.reset_index(drop=True)

    return frame


def assign_frame(
    bundle: Bundle, frame: pd.DataFrame, batch_size: int, device: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    表の行を符号化し, 埋め込みを群の中心に割り当てる関数

    Parameters
    ----------
    bundle: Bundle
        群分けの一式
    frame: pd.DataFrame
        群を作る列を持つ表
    batch_size: int
        1 回に渡す行の数
    device: str
        使うデバイス

    Returns
    ----------
    assigned: tuple[list[str], np.ndarray, np.ndarray]
        行ごとの割り当てない理由 (割り当てた行は空), 群の番号 (割り当て
        ない行は -1), 距離 (割り当てない行は NaN)

    Raises
    ----------
    ValueError
        群を作る列が無い場合
    """
    # 群を作る列
    group = bundle.info["group"]
    missing = [c for c in group["columns"] if c not in frame.columns]
    if missing:
        raise ValueError(
            f"表に群を作る列 {', '.join(missing)} がありません (列: "
            f"{', '.join(map(str, frame.columns))})"
        )
    encoding = bundle.cache["group_encoding"]
    features, reasons = encode_rows(frame, encoding)
    labels, distance = assign_kept(bundle, features, reasons, batch_size,
                                   device)
    assigned = (reasons, labels, distance)

    return assigned


def assign_kept(
    bundle: Bundle, features: np.ndarray, reasons: list[str],
    batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    割り当てる件の埋め込みを, 最も近い群の中心に割り当てる関数

    Parameters
    ----------
    bundle: Bundle
        群分けの一式
    features: np.ndarray
        モデルの入力 (符号化した行か前処理した画像)
    reasons: list[str]
        件ごとの割り当てない理由 (数でない埋め込みの件に理由を足す)
    batch_size: int
        1 回に渡す件数
    device: str
        使うデバイス

    Returns
    ----------
    assigned: tuple[np.ndarray, np.ndarray]
        群の番号 (割り当てない件は -1) と距離 (割り当てない件は NaN)
    """
    # 割り当てる件だけの埋め込み (数でない埋め込みの件は割り当ての外)
    labels = np.full(len(features), -1, dtype=np.int64)
    distance = np.full(len(features), math.nan)
    keep = np.array([not r for r in reasons], dtype=bool)
    if keep.any():
        model = bundle.model().to(device)
        try:
            points = embed(model, features[keep], batch_size, device,
                           named=not bundle.from_ready)
        finally:
            model.to("cpu")
        finite = np.isfinite(points).all(1)
        rows = np.flatnonzero(keep)
        for row in rows[~finite]:
            reasons[row] = NOT_FINITE
        found, near = assign(points[finite], bundle.cache["group_centers"])
        labels[rows[finite]] = found
        distance[rows[finite]] = near
    assigned = (labels, distance)

    return assigned


def assign_images(
    bundle: Bundle, source: Any, batch_size: int, device: str
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """
    写真を読み, 既製のモデルの埋め込みを群の中心に割り当てる関数

    Parameters
    ----------
    bundle: Bundle
        既製の埋め込みの群分けの一式
    source: Any
        画像のファイルかディレクトリ
    batch_size: int
        1 回に渡す枚数
    device: str
        使うデバイス

    Returns
    ----------
    assigned: tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]
        読み込みの表 (path など), 写真ごとの割り当てない理由 (読めない
        画像も), 群の番号, 距離

    Raises
    ----------
    ValueError
        画像のファイルかディレクトリでない場合と, 読めない場合
    """
    # 写真の読み込み (読めない画像は理由付きで割り当ての外)
    frame, images, reasons = read_images(source,
                                         bundle.info["ready"]["preprocess"])
    labels, distance = assign_kept(bundle, images, reasons, batch_size,
                                   device)
    assigned = (frame, reasons, labels, distance)

    return assigned


def predict_group(
    bundle: Bundle, source: Any, options: Optional[dict] = None,
    batch_size: int = 256, device: str = "cpu"
) -> pd.DataFrame:
    """
    新しい行を, 埋め込みの最も近い群の中心に割り当てる関数

    Parameters
    ----------
    bundle: Bundle
        群分けの一式
    source: Any
        表のファイル, 表, 表の ML-EDA の Data (既製の埋め込みは画像の
        ファイルかディレクトリ)
    options: Optional[dict] = None
        ファイルの読み込みの設定 (load_data に渡す. 既製の埋め込みは使え
        ない)
    batch_size: int = 256
        1 回に渡す行の数
    device: str = "cpu"
        使うデバイス

    Returns
    ----------
    result: pd.DataFrame
        元の列 (既製の埋め込みは画像の読み込みの表. 予測の列と同じ名前は
        「(入力)」付き) と, status, reason, group (Int64), distance と,
        遠さの上限を持つ一式は far (真偽値)

    Raises
    ----------
    ValueError
        表 (既製の埋め込みは画像) でない場合と, 群を作る列が無い場合
    """
    # 表 (既製の埋め込みは写真) の読み込みと割り当て
    if bundle.from_ready:
        if options:
            raise ValueError(
                "既製の埋め込みの群分けの一式は画像を一式の前処理で読むので, "
                "読み込みの設定 (options) は使えません"
            )
        frame, reasons, labels, distance = assign_images(
            bundle, source, batch_size, device
        )
    else:
        frame = read_table(source, options, "群に割り当てる表")
        reasons, labels, distance = assign_frame(bundle, frame, batch_size,
                                                 device)

    # 元の列の後ろに予測の列 (同じ名前の元の列は「(入力)」)
    added = frame_columns(bundle)
    result = frame.rename(
        columns={c: f"{c} (入力)" for c in frame.columns if c in added}
    )
    ok = np.array([not r for r in reasons], dtype=bool)
    result["status"] = np.where(ok, OK, SKIPPED)
    result["reason"] = pd.Series(reasons, dtype=object)
    result["group"] = pd.array(
        [int(v) if v >= 0 else None for v in labels], dtype="Int64"
    )
    result["distance"] = distance.astype(float)
    far = bundle.info["group"].get("far")
    if far is not None:
        result[FAR_COLUMN] = far_marks(labels, distance, far["thresholds"])

    return result


def far_marks(
    labels: np.ndarray, distance: np.ndarray, thresholds: list
) -> np.ndarray:
    """
    群の距離の上限より遠い行に印を付ける関数

    Parameters
    ----------
    labels: np.ndarray
        群の番号 (割り当てない行は -1)
    distance: np.ndarray
        群の中心までの距離
    thresholds: list
        群ごとの距離の上限 (行の無かった群は None)

    Returns
    ----------
    marks: np.ndarray
        距離が群の上限 t の少し外 (t × (1 + 1e-6) + 1e-9) より大きい行が
        True (割り当てない行と, 上限の無い群の行は False)
    """
    # 群ごとの上限に丸めの差の幅を足した値 (上限が無ければ無限) との比べ
    relative, absolute = FAR_TOLERANCE
    limits = np.array([math.inf if t is None
                       else float(t) * (1 + relative) + absolute
                       for t in thresholds])
    marks = np.zeros(len(labels), dtype=bool)
    kept = labels >= 0
    marks[kept] = distance[kept] > limits[labels[kept]]

    return marks


def frame_columns(bundle: Bundle) -> list[str]:
    """
    群分けの一式の予測の表に足す列の名前を返す関数

    Parameters
    ----------
    bundle: Bundle
        群分けの一式

    Returns
    ----------
    added: list[str]
        status, reason, group, distance と, 遠さの上限を持つ一式は far
    """
    # 遠さの上限を持つ一式だけ印の列
    added = list(GROUP_FRAME_COLUMNS)
    if bundle.info["group"].get("far") is not None:
        added.append(FAR_COLUMN)

    return added


def run_sources(run: Union[str, Path]) -> dict:
    """
    群分けの実験のディレクトリ (既製の埋め込みは成果物のディレクトリ) から,
    一式に要るものを読む関数

    Parameters
    ----------
    run: Union[str, Path]
        ML-Found の実験のディレクトリか, 骨組みが既製の埋め込みの群分けの
        成果物を書いたディレクトリ

    Returns
    ----------
    sources: dict
        weights, spec (既製の埋め込みは None), centers_file (写すファイル),
        centers (read_centers), ready (埋め込みを作った既製のモデル. 列の
        符号化は None), id, assignments, stability, summary, columns, run
        (実験の状態. 既製の埋め込みは None)

    Raises
    ----------
    ValueError
        実験のディレクトリでない場合, 重みか定義が無い場合, 成果物の形が
        違う場合
    FileNotFoundError
        成果物が無い場合
    """
    # 重みと定義 (仮のモデルで学習した実験には定義が無い. 既製の埋め込みは
    # 重みも定義も出どころから)
    root = Path(run).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"実験のディレクトリ {root} がありません")
    ready = ready_centers(root / CENTERS_FILE)
    weights = None if ready else root / "final_model.pt"
    if weights is not None and not weights.is_file():
        raise ValueError(f"重み {weights} がありません")
    spec = None if ready else root / "model_spec.yaml"
    if spec is not None and not spec.is_file():
        raise ValueError(
            f"モデルの定義 {spec} がありません (仮のモデルで学習した実験"
            "は一式にできません. 定義を置いて学習し直してください)"
        )

    # 列の符号化 (か既製のモデル) と群の中心, 割り当て, 群の安定性と
    # 群の要約
    centers = read_centers(root / CENTERS_FILE)
    k = centers["k"]
    key, assignments = read_assignments(root / ASSIGNMENT_FILE, k)
    if ready and key != READY_ID:
        raise ValueError(
            f"既製の埋め込みの群の割り当ての表の行を見分ける列は {READY_ID} "
            f"(写真の場所) にしてください (今は {key or ROW_KEY})"
        )
    stability = stability_of(read_evaluation(root / STABILITY_FILE), k)
    names = list(READY_COLUMNS) if ready else [
        *centers["encoding"]["numeric"], *centers["encoding"]["categories"]
    ]
    summary, columns = summary_of(read_evaluation(root / SUMMARY_FILE), k,
                                  names)
    sources = {
        "weights": weights, "spec": spec, "centers_file": root / CENTERS_FILE,
        "centers": centers, "ready": centers["ready"], "id": key,
        "assignments": assignments, "stability": stability,
        "summary": summary,
        "columns": list(READY_COLUMNS) if ready else columns,
        "run": None if ready else run_summary(root),
    }

    return sources


def assign_data(
    bundle: Bundle, data: Any, options: Optional[dict], batch_size: int
) -> tuple[str, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    学習の表 (既製の埋め込みは群に分けた写真) を一式で割り当て直す関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    data: Any
        学習に使った表 (ファイル, 表, 表の Data. 既製の埋め込みは群に分けた
        写真のディレクトリ)
    options: Optional[dict]
        ファイルの読み込みの設定
    batch_size: int
        1 回に渡す行の数

    Returns
    ----------
    assigned: tuple[str, pd.DataFrame, np.ndarray, np.ndarray]
        誤りの文に使う呼び名, 表 (既製の埋め込みは読めた写真の読み込みの
        表), 群の番号 (割り当てない行は -1), 距離 (割り当てない行は NaN)
    """
    # 既製の埋め込みは読めた写真だけ (骨組みの割り当ての表も読めた写真
    # だけ)
    if bundle.from_ready:
        what = "群に分けた写真"
        frame, reasons, labels, distance = assign_images(bundle, data,
                                                         batch_size, "cpu")
        read = np.array([r in ("", NOT_FINITE) for r in reasons],
                        dtype=bool)
        frame = frame[read].reset_index(drop=True)
        labels, distance = labels[read], distance[read]
    else:
        what = "学習の表"
        frame = read_table(data, options, what)
        _, labels, distance = assign_frame(bundle, frame, batch_size, "cpu")
    assigned = (what, frame, labels, distance)

    return assigned


def data_check(
    what: str, frame: pd.DataFrame, labels: np.ndarray, sources: dict
) -> tuple[Optional[dict], list[str]]:
    """
    割り当て直した群と, 実験の割り当ての一致を数える関数

    Parameters
    ----------
    what: str
        誤りの文に使う表の呼び名
    frame: pd.DataFrame
        割り当て直した表
    labels: np.ndarray
        割り当て直した群の番号 (割り当てない行は -1)
    sources: dict
        run_sources の結果

    Returns
    ----------
    found: tuple[Optional[dict], list[str]]
        rows と agreement (比べられなければ None) と注意の文
    """
    # 比べるのは, 行の数と行を見分ける列の値の並びがそろう場合だけ
    expected = sources["assignments"]
    if len(frame) != len(expected):
        note = (f"{what}の行の数 {len(frame)} が, 実験の群の割り当ての行の"
                f"数 {len(expected)} と違うので, 割り当て直しを比べません")
        return None, [note]
    key = sources["id"]
    if key is not None and (key not in frame.columns or list(
        frame[key].astype(str)
    ) != list(map(str, expected.index))):
        note = (f"{what}の行を見分ける列 {key} の値の並びが実験の群の割り"
                "当てと違うので, 割り当て直しを比べません")
        return None, [note]
    agreement = float((labels == expected.to_numpy()).mean())
    notes = []
    if agreement < AGREEMENT:
        notes.append(
            f"{what}を一式で割り当て直した群が, 実験の群の割り当てと "
            f"{agreement * 100:.1f} % しか同じになりません (列の符号化か重み"
            "が学習と違う可能性があります)"
        )
    found = ({"rows": len(frame), "agreement": agreement}, notes)

    return found


def far_of(
    labels: np.ndarray, distance: np.ndarray, k: int
) -> Optional[dict]:
    """
    学習の行の距離から, 群ごとの距離の上限を求める関数

    Parameters
    ----------
    labels: np.ndarray
        割り当て直した群の番号 (割り当てない行は -1)
    distance: np.ndarray
        群の中心までの距離
    k: int
        群の数

    Returns
    ----------
    far: Optional[dict]
        quantile, rows (距離を数えた行の数), thresholds (群ごとの上限.
        行の無い群は None) の辞書. 割り当てた行が無ければ None
    """
    # 群ごとの距離の分位点 (並べた距離のうち分位点以上で最も小さい値)
    kept = labels >= 0
    if not kept.any():
        return None
    thresholds = []
    for key in range(k):
        inside_group = distance[labels == key]
        thresholds.append(
            float(np.quantile(inside_group, FAR_QUANTILE, method="higher"))
            if inside_group.size else None
        )
    far = {"quantile": FAR_QUANTILE, "rows": int(kept.sum()),
           "thresholds": thresholds}

    return far


def place_far(info: dict, far: Optional[dict], data: Any) -> None:
    """
    群ごとの距離の上限を一式の説明に置き, 版を決める関数

    Parameters
    ----------
    info: dict
        作っている一式の説明 (版 9 か 10. 上限があれば版 16 か 17 にする)
    far: Optional[dict]
        far_of の結果
    data: Any
        学習の表 (渡さなければ None)

    Returns
    ----------
    None
    """
    # 上限が無ければ版 9 か 10 のままで, 理由は注意に
    if far is None:
        info["notes"].append(
            NO_FAR if data is None else
            "学習の表で群に割り当てた行が無いので, 群の中心から遠い行に"
            "印を付けません"
        )
        return
    info["group"]["far"] = far
    info["format_version"] = (
        READY_FAR_GROUP_BUNDLE_VERSION
        if info["format_version"] == READY_GROUP_BUNDLE_VERSION
        else FAR_GROUP_BUNDLE_VERSION
    )


def make_group_bundle(
    path: Union[str, Path], run: Union[str, Path],
    name: Optional[str] = None, data: Any = None,
    options: Optional[dict] = None, copy_weights: bool = True,
    overwrite: bool = False, pretrained: Optional[dict] = None,
    batch_size: int = 256, license: Optional[dict] = None
) -> dict:
    """
    新しい行 (既製の埋め込みは写真) を群に割り当てる一式を, 群分けの実験
    (既製の埋め込みは骨組みの成果物) から作る関数

    Parameters
    ----------
    path: Union[str, Path]
        作るディレクトリ
    run: Union[str, Path]
        ML-Found の実験のディレクトリ (final_model.pt, model_spec.yaml,
        group_centers.json, group_assignments.csv, evaluations の
        group_stability.json と group_summary.json を使い, 実験の状態を
        記録する). 既製の埋め込みは, ready を持つ group_centers.json と
        他の 3 つの成果物のディレクトリ
    name: Optional[str] = None
        一式の名前 (省略するとディレクトリの名前)
    data: Any = None
        学習に使った表 (既製の埋め込みは群に分けた写真のディレクトリ.
        渡すと割り当て直して実験の割り当てと比べ, 群ごとの距離の上限を
        持つ版 16 (既製の埋め込みは版 17) の一式を作る)
    options: Optional[dict] = None
        data のファイルの読み込みの設定 (load_data に渡す. 既製の埋め込み
        は使えない)
    copy_weights: bool = True
        重みを一式に写すか (写さない場合は絶対パスで指す. 既製の埋め込みは
        いつも写さない)
    overwrite: bool = False
        既にある一式を上書きするか
    pretrained: Optional[dict] = None
        ノードの名前ごとの重みのライセンスの情報 (make_bundle と同じ. 既製
        の埋め込みは license)
    batch_size: int = 256
        1 回に渡す行の数
    license: Optional[dict] = None
        既製のモデルの重みのライセンスの情報 (make_ready_bundle と同じ.
        既製の埋め込みだけ)

    Returns
    ----------
    info: dict
        書き出した inference.yaml の中身

    Raises
    ----------
    FileExistsError
        ディレクトリが既にあり, overwrite が False の場合と, 一式ではない
        ディレクトリを上書きしようとした場合
    FileNotFoundError
        成果物が無い場合
    ValueError
        定義, 重み, 成果物が約束に合わない場合
    """
    # 置き換えられるのは, 前に作った一式だけ
    final = Path(path).expanduser().resolve()
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"{final} は既にあります")
        if not (final / "inference.yaml").is_file():
            raise FileExistsError(f"{final} は推論の一式ではないので上書きしません")
    if batch_size < 1:
        raise ValueError("batch_size は 1 以上にしてください")

    # 実験の成果物 (既製の埋め込みは別の作り方)
    sources = run_sources(run)
    if sources["ready"] is not None:
        if options or pretrained is not None:
            raise ValueError(
                "既製の埋め込みの群分けの一式には, 読み込みの設定 (options) "
                "と pretrained を渡せません (写真は一式の前処理で読み, "
                "ライセンスの情報は license で渡す)"
            )
        info = make_ready_group(final, sources, name, data, license,
                                batch_size)
        return info
    if license is not None:
        raise ValueError(
            "license は既製の埋め込みの群分けの一式だけに渡せます (学習した"
            "重みのライセンスは pretrained)"
        )

    # 入力と出力の約束
    centers = sources["centers"]
    features = encoding_width(centers["encoding"])
    spec_dict = load_spec(sources["spec"])
    check_units(spec_dict)
    check_group_spec(spec_dict, features)
    spec_dict, records, kept, weight_notes = plan_weights(
        spec_dict, sources["weights"], pretrained, copy_weights
    )

    # 隣の一時ディレクトリに作り, 確かめてから入れ替え
    writing = final.with_name(f".{final.name}.writing")
    if writing.exists():
        shutil.rmtree(writing)
    writing.mkdir(parents=True)
    try:
        dump_spec(spec_dict, writing / "model_spec.yaml")
        if copy_weights and kept is not None:
            torch.save(kept, writing / "final_model.pt")
        elif copy_weights:
            shutil.copy(sources["weights"], writing / "final_model.pt")
        shutil.copy(sources["centers_file"], writing / CENTERS_FILE)
        empty = Dataset(path=writing, info={}, frames={}, recipe={})
        info = {
            "format_version": GROUP_BUNDLE_VERSION,
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "role": "group",
            "output": {"form": "cluster"},
            "target": None,
            "procedure": None,
            "model": {
                "spec": "model_spec.yaml",
                "weights": "final_model.pt" if copy_weights
                else str(sources["weights"]),
                "output": None,
                "kind": "value",
                "target_scale": None,
                "pretrained": records,
            },
            "inputs": {},
            "dataset": None,
            "run": sources["run"],
            "conclusion": {},
            "decision": None,
            "check": None,
            "uncertainty": None,
            "notes": bundle_notes(empty, sources["run"]) + weight_notes,
        }

        # 0 の行の埋め込みの形と幅
        building = Bundle(path=writing, info=info, spec=spec_dict,
                          dataset=empty)
        probe = embed(building.model(),
                      np.zeros((PROBE_ROWS, features), dtype=np.float32),
                      batch_size, "cpu")
        width = centers["centers"].shape[1]
        if probe.shape != (PROBE_ROWS, width) or not np.isfinite(
            probe
        ).all():
            raise ValueError(
                f"0 の行の埋め込みの形 {tuple(probe.shape)} が (件数, 群の"
                f"中心の幅 {width}) の数の配列ではありません"
            )
        info["group"] = {
            "centers": CENTERS_FILE,
            "k": centers["k"],
            "columns": sources["columns"],
            "features": features,
            "width": int(width),
            "id": sources["id"],
            "rows": len(sources["assignments"]),
            "stability": sources["stability"],
            "summary": sources["summary"],
            "check": None,
        }

        # 学習の表を渡した場合は, 割り当て直しと実験の割り当ての比べと,
        # 群ごとの距離の上限
        far = None
        if data is not None:
            building.cache["group_encoding"] = centers["encoding"]
            building.cache["group_centers"] = centers["centers"]
            what, table, labels, distance = assign_data(building, data,
                                                        options, batch_size)
            found, notes = data_check(what, table, labels, sources)
            info["group"]["check"] = found
            info["notes"].extend(notes)
            far = far_of(labels, distance, centers["k"])
        place_far(info, far, data)
        (writing / "inference.yaml").write_text(
            yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
            encoding="utf-8"
        )
        load_bundle(writing)
    except Exception:
        shutil.rmtree(writing, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(final)
    writing.rename(final)

    return info


def make_ready_group(
    final: Path, sources: dict, name: Optional[str], data: Any,
    license: Optional[dict], batch_size: int
) -> dict:
    """
    既製の埋め込みの群分けの一式 (版 10. 写真を渡せば版 17) を, 骨組みの
    成果物から作る関数

    Parameters
    ----------
    final: Path
        作るディレクトリ
    sources: dict
        run_sources の結果 (ready を持つもの)
    name: Optional[str]
        一式の名前 (省略するとディレクトリの名前)
    data: Any
        群に分けた写真のディレクトリ (渡すと割り当て直して比べ, 群ごとの
        距離の上限を求める)
    license: Optional[dict]
        重みのライセンスの情報 (省略すると pretrained_info で読む)
    batch_size: int
        1 回に渡す枚数

    Returns
    ----------
    info: dict
        書き出した inference.yaml の中身

    Raises
    ----------
    ValueError
        出どころ, モデル, 重みの名前が無い場合, 重みの hash が群の中心を
        作ったものと違う場合, 埋め込みを取り出せない場合, 埋め込みの幅が
        群の中心と違う場合
    """
    # 出どころの重みの説明, 1 つのノードの定義, 出どころの重みの hash
    wanted = sources["ready"]
    library, model = wanted["library"], wanted["model"]
    if library not in READY_LIBRARIES:
        raise ValueError(
            f"既製のモデルの出どころ {library} はありません (出どころ: "
            f"{', '.join(READY_LIBRARIES)})"
        )
    found = READY_LIBRARIES[library](model, wanted["weights"])
    spec = ready_spec(found, model)
    record, ready, notes, built = ready_record(spec, found, library, model,
                                               license)
    if record["sha256"] != wanted["sha256"]:
        raise ValueError(
            f"{library} の {model} の重み {record['pinned_revision']} の hash "
            f"({record['sha256'][:12]}…) が, 群の中心を作った重みの hash "
            f"({wanted['sha256'][:12]}…) と違います"
        )

    # 分類の最後の層の手前の出力の幅と, 群の中心の幅
    centers = sources["centers"]
    width = int(centers["centers"].shape[1])
    embedder = ReadyEmbedding(built.eval())
    found_width = check_head(embedder, int(ready["preprocess"]["crop"]))
    if found_width != width:
        raise ValueError(
            f"既製のモデルの埋め込みの幅 {found_width} が, 群の中心の幅 "
            f"{width} と違います"
        )

    # 隣の一時ディレクトリでの作成と, 確かめた後の入れ替え (重みは空)
    writing = final.with_name(f".{final.name}.writing")
    if writing.exists():
        shutil.rmtree(writing)
    writing.mkdir(parents=True)
    try:
        dump_spec(spec, writing / "model_spec.yaml")
        torch.save({}, writing / "final_model.pt")
        shutil.copy(sources["centers_file"], writing / CENTERS_FILE)
        info = {
            "format_version": READY_GROUP_BUNDLE_VERSION,
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "role": "group",
            "output": {"form": "cluster"},
            "target": None,
            "procedure": None,
            "model": {
                "spec": "model_spec.yaml",
                "weights": "final_model.pt",
                "output": None,
                "kind": "value",
                "target_scale": None,
                "pretrained": [record],
            },
            "inputs": {},
            "dataset": None,
            "run": None,
            "conclusion": {},
            "decision": None,
            "check": None,
            "uncertainty": None,
            "notes": notes,
            "ready": ready,
            "group": {
                "centers": CENTERS_FILE,
                "k": centers["k"],
                "columns": list(READY_COLUMNS),
                "features": None,
                "width": width,
                "id": READY_ID,
                "rows": len(sources["assignments"]),
                "stability": sources["stability"],
                "summary": sources["summary"],
                "check": None,
            },
        }

        # 写真を渡した場合は, 割り当て直しと骨組みの割り当ての比べと群
        # ごとの距離の上限 (読んだ重みのモデルのまま)
        far = None
        if data is not None:
            building = Bundle(path=writing, info=info, spec=spec,
                              dataset=Dataset(path=writing, info={},
                                              frames={}, recipe={}))
            building.cache["model"] = embedder
            building.cache["group_centers"] = centers["centers"]
            what, table, labels, distance = assign_data(building, data,
                                                        None, batch_size)
            checked, more = data_check(what, table, labels, sources)
            info["group"]["check"] = checked
            info["notes"].extend(more)
            far = far_of(labels, distance, centers["k"])
        place_far(info, far, data)
        (writing / "inference.yaml").write_text(
            yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
            encoding="utf-8"
        )
        load_bundle(writing)
    except Exception:
        shutil.rmtree(writing, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(final)
    writing.rename(final)

    return info


def check_group_bundle(bundle: Bundle) -> None:
    """
    読んだ群分けの一式の定義と群の中心が group の欄と合うかを確かめる関数

    Parameters
    ----------
    bundle: Bundle
        読んだ一式

    Returns
    ----------
    None

    Raises
    ----------
    FileNotFoundError
        群の中心のファイルが無い場合
    ValueError
        定義, 列の符号化, 群の中心の大きさが group の欄と違う場合
    """
    # 群の数, 列, 符号化した行の幅, 埋め込みの幅 (既製の埋め込みは
    # 埋め込みを作った既製のモデル)
    group = bundle.info["group"]
    found = read_centers(inside(bundle.path, group["centers"]))
    if bundle.from_ready:
        check_ready_centers(bundle, found)
        bundle.cache["group_encoding"] = None
        bundle.cache["group_centers"] = found["centers"]
        return
    encoding = found["encoding"]
    if encoding is None:
        raise ValueError(
            f"{group['centers']} に列の符号化 (encoding) がありません"
        )
    names = [*encoding["numeric"], *encoding["categories"]]
    width = found["centers"].shape[1]
    if (found["k"], encoding_width(encoding), width) != (
        group["k"], group["features"], group["width"]
    ) or sorted(names) != sorted(group["columns"]):
        raise ValueError(
            f"{group['centers']} の群の数, 列, 符号化した行の幅, 埋め込みの"
            "幅が group の欄と違います"
        )
    check_group_spec(bundle.spec, group["features"])
    bundle.cache["group_encoding"] = encoding
    bundle.cache["group_centers"] = found["centers"]


def check_ready_centers(bundle: Bundle, found: dict) -> None:
    """
    既製の埋め込みの群の中心が一式と合うかを確かめる関数

    Parameters
    ----------
    bundle: Bundle
        読んだ一式
    found: dict
        read_centers の結果

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        群の中心が既製の埋め込みのものでない場合, 群の数と幅が group と
        違う場合, 埋め込みを作った既製のモデルと重みが一式と違う場合
    """
    # 群の数と幅, 出どころとモデルの名前, 重みの hash
    group = bundle.info["group"]
    wanted = found["ready"]
    if wanted is None:
        raise ValueError(
            f"{group['centers']} に埋め込みを作った既製のモデル (ready) が"
            "ありません"
        )
    if (found["k"], found["centers"].shape[1]) != (group["k"],
                                                     group["width"]):
        raise ValueError(
            f"{group['centers']} の群の数と埋め込みの幅が group の欄と"
            "違います"
        )
    ready = bundle.info["ready"]
    hashes = {r.get("sha256") for r in bundle.info["model"]["pretrained"]}
    if (wanted["library"], wanted["model"]) != (
        ready["library"], ready["model"]
    ) or hashes != {wanted["sha256"]}:
        raise ValueError(
            f"{group['centers']} の群の中心を作った既製のモデルか重みの hash"
            " が, 一式の ready と model.pretrained と違います"
        )


def frame_column(
    prediction: pd.DataFrame, name: str, added: list[str]
) -> pd.Series:
    """
    予測の表から, 群を作る列の元の値を取り出す関数

    Parameters
    ----------
    prediction: pd.DataFrame
        予測の表
    name: str
        群を作る列の名前
    added: list[str]
        予測の表に足した列の名前 (frame_columns)

    Returns
    ----------
    column: pd.Series
        元の値 (予測の列と同じ名前の列は「(入力)」付きの列から)

    Raises
    ----------
    ValueError
        列が無い場合
    """
    # 予測の列と同じ名前の元の列は「(入力)」付き
    key = f"{name} (入力)" if name in added else name
    if key not in prediction.columns:
        raise ValueError(f"予測の表に群を作る列 {key} がありません")
    column = prediction[key]

    return column


def group_summary(
    bundle: Union[str, Path, Bundle], prediction: pd.DataFrame
) -> list[dict]:
    """
    割り当てた行の群の要約 (ML-EDA の describe_groups の行の形) を返す関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        群分けの一式かそのディレクトリ
    prediction: pd.DataFrame
        predict の表 (read_result で読んだ表も)

    Returns
    ----------
    rows: list[dict]
        群の番号の順の行と, 最後に全体の行 (stability は学習のときの群の
        値. 割り当てた行が無ければ空)

    Raises
    ----------
    ValueError
        群分けの一式でない場合と, 表に列が無い場合
    """
    # 割り当てた行だけ
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    if not bundle.from_group:
        raise ValueError("group_summary は群分けの一式だけに使えます")
    group = bundle.info["group"]
    done = prediction[prediction["status"] == OK]
    if done.empty:
        return []

    # 一時の名前の列で作る ML-EDA の群の要約 (数の列は数, 種類の列は
    # 文字. 欠けは欠けのまま)
    numeric = READY_COLUMNS if bundle.from_ready else (
        bundle.cache["group_encoding"]["numeric"]
    )
    temporary = {name: f"c{n}" for n, name in enumerate(group["columns"])}
    added = frame_columns(bundle)
    frame = pd.DataFrame(index=range(len(done)))
    for name, short in temporary.items():
        column = frame_column(done, name, added).reset_index(drop=True)
        if name in numeric:
            frame[short] = pd.to_numeric(
                column.map(lambda v: float(v) if isinstance(v, bool) else v),
                errors="coerce"
            ).astype(float)
        else:
            frame[short] = column.map(
                lambda v: v if pd.isna(v) else str(v)
            ).astype(object)
    labels = [int(v) for v in done["group"]]
    found = describe_groups(Data(kind="table", frame=frame),
                            columns=list(temporary.values()), labels=labels)

    # 群の番号の順の並び (行の無い群も. 安定性は学習のときの群の値)
    back = {short: name for name, short in temporary.items()}
    given = {row["group"]: row for row in found["groups"]}
    learned = {row["group"]: row.get("stability")
               for row in group["summary"]}
    rows = []
    for key in [*range(group["k"]), ALL_ROWS]:
        row = given.get(key)
        if row is None:
            row = {"group": key, "rows": 0, "share": 0.0, "stability": None,
                   **{short: None for short in back}}
        row = {back.get(k, k): v for k, v in row.items()}
        row["stability"] = None if key == ALL_ROWS else learned.get(key)
        rows.append(row)

    return rows


def plain_value(value: Any) -> Any:
    """
    表の値を, result.yaml に書ける値にする関数

    Parameters
    ----------
    value: Any
        表の 1 つの値

    Returns
    ----------
    plain: Any
        欠けは None, 真偽値と数はそのまま (numpy の値は Python の値),
        他は文字
    """
    # 欠け, numpy の値, 真偽値と数, 他 (並びの値は文字)
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, float, str)):
        return value
    plain = str(value)

    return plain


def group_representatives(
    bundle: Union[str, Path, Bundle], prediction: pd.DataFrame,
    count: int = REPRESENTATIVES
) -> list[dict]:
    """
    群ごとに, 群の中心に近い順の割り当てた行 (代表の行) を返す関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        群分けの一式かそのディレクトリ
    prediction: pd.DataFrame
        predict の表 (read_result で読んだ表も)
    count: int = REPRESENTATIVES
        群ごとの代表の行の件数の上限

    Returns
    ----------
    groups: list[dict]
        群の番号の順の group と rows (row (予測の表の行の位置),
        distance, values (群を作る列の元の値. 既製の埋め込みは写真の
        場所 path も)) の辞書. 割り当てた行が無ければ空

    Raises
    ----------
    ValueError
        群分けの一式でない場合, 件数が 1 以上の整数でない場合, 表に列が
        無い場合
    """
    # 割り当てた行だけ
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    if not bundle.from_group:
        raise ValueError(
            "group_representatives は群分けの一式だけに使えます"
        )
    if not is_whole(count, 1):
        raise ValueError(
            f"代表の行の件数は 1 以上の整数にしてください ({count!r})"
        )
    missing = [c for c in ("status", "group", "distance")
               if c not in prediction.columns]
    if missing:
        raise ValueError(
            f"予測の表に {', '.join(missing)} の列がありません"
        )
    group = bundle.info["group"]
    names = [READY_ID, *group["columns"]] if bundle.from_ready else (
        group["columns"]
    )
    added = frame_columns(bundle)
    columns = {name: frame_column(prediction, name, added)
               for name in names}
    positions = np.flatnonzero((prediction["status"] == OK).to_numpy())
    if positions.size == 0:
        return []

    # 群ごとに距離の近い順 (同じ距離なら行の位置の順)
    labels = np.array([int(v) for v in prediction["group"].iloc[positions]])
    distance = pd.to_numeric(
        prediction["distance"].iloc[positions], errors="coerce"
    ).to_numpy(dtype=float)
    groups = []
    for key in range(group["k"]):
        inside_group = np.flatnonzero(labels == key)
        order = inside_group[np.argsort(distance[inside_group],
                                        kind="stable")][:count]
        rows = [{
            "row": int(positions[n]),
            "distance": float(distance[n]),
            "values": {name: plain_value(column.iloc[positions[n]])
                       for name, column in columns.items()},
        } for n in order]
        groups.append({"group": key, "rows": rows})

    return groups


def group_far(
    bundle: Union[str, Path, Bundle], prediction: pd.DataFrame
) -> Optional[dict]:
    """
    群の中心から学習のときより遠い行の件数を, 群ごとに数える関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        群分けの一式かそのディレクトリ
    prediction: pd.DataFrame
        predict の表 (read_result で読んだ表も)

    Returns
    ----------
    far: Optional[dict]
        quantile, rows (印の付いた行の数), groups (群の番号の順の group,
        rows (群の印の付いた行の数), threshold (群の距離の上限)) の辞書.
        遠さの上限を持たない一式は None

    Raises
    ----------
    ValueError
        群分けの一式でない場合と, 表に列が無い場合
    """
    # 上限を持たない一式は None
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    if not bundle.from_group:
        raise ValueError("group_far は群分けの一式だけに使えます")
    limits = bundle.info["group"].get("far")
    if limits is None:
        return None
    missing = [c for c in ("status", "group", FAR_COLUMN)
               if c not in prediction.columns]
    if missing:
        raise ValueError(
            f"予測の表に {', '.join(missing)} の列がありません"
        )

    # 割り当てた行の印の, 群ごとの件数
    done = prediction[prediction["status"] == OK]
    labels = np.array([int(v) for v in done["group"]], dtype=np.int64)
    marks = done[FAR_COLUMN].fillna(False).astype(bool).to_numpy()
    groups = [{"group": key,
               "rows": int((marks & (labels == key)).sum()),
               "threshold": threshold}
              for key, threshold in enumerate(limits["thresholds"])]
    far = {"quantile": limits["quantile"], "rows": int(marks.sum()),
           "groups": groups}

    return far


def far_sentence(far: dict) -> str:
    """
    群の中心から遠い行の件数を述べる文を返す関数

    Parameters
    ----------
    far: dict
        group_far の結果

    Returns
    ----------
    sentence: str
        遠い行の件数と, 印の付いた群ごとの件数 (無ければ無いこと)
    """
    # 分位点は百分率
    point = f"{far['quantile'] * 100:g} % 点"
    if not far["rows"]:
        sentence = ("群の中心から学習のときより遠い行は無い (群ごとの学習の"
                    f"行の距離の {point}まで)")
        return sentence
    counts = ", ".join(f"群 {row['group']} {row['rows']:,} 行"
                       for row in far["groups"] if row["rows"])
    sentence = (f"{far['rows']:,} 行は群の中心から学習のときより遠い (群ごと"
                f"の学習の行の距離の {point}を超える. {counts})")

    return sentence


def stability_sentence(stability: dict) -> str:
    """
    一式の群の安定性を述べる文を返す関数

    Parameters
    ----------
    stability: dict
        group.stability

    Returns
    ----------
    sentence: str
        作り直し方と回数, 調整ランド指数, シルエット係数と比べる相手の値
    """
    # 骨組みの数え方の名前 (知らない名前はそのまま)
    method = STABILITY_NAMES.get(str(stability.get("method")),
                                 str(stability.get("method")))
    baseline = stability.get("baseline") or "比べる相手"
    sentence = (
        f"学習の群の安定性 ({method}, {stability.get('repeats')} 回): 調整"
        f"ランド指数 {stability['ari']:.3f} (最も低い "
        f"{stability['ari_min']:.3f}), シルエット係数 "
        f"{stability['silhouette']:.3f}. {baseline}は "
        f"{stability['ari_baseline']:.3f} "
        f"({stability['ari_min_baseline']:.3f}), "
        f"{stability['silhouette_baseline']:.3f}"
    )

    return sentence


def conclude_groups(bundle: Bundle, prediction: pd.DataFrame) -> dict:
    """
    群の割り当てから, 件数と群の要約の結論を出す関数

    Parameters
    ----------
    bundle: Bundle
        群分けの一式
    prediction: pd.DataFrame
        predict_group の表

    Returns
    ----------
    conclusion: dict
        frame (decision を足した表. 割り当てた行は空, しなかった行は
        予測なし), summary (rows, predicted, skipped, k, groups,
        representatives, stability, far), sentences, rules (空)
    """
    # 割り当てたかと, 群ごとの行の数
    frame = prediction.copy()
    ok = frame["status"] == OK
    frame["decision"] = np.where(ok, "", NO_PREDICTION)
    group = bundle.info["group"]
    groups = group_summary(bundle, prediction)
    summary = {
        "rows": int(len(frame)), "predicted": int(ok.sum()),
        "skipped": int(len(frame) - ok.sum()), "k": group["k"],
        "groups": groups,
        "representatives": group_representatives(bundle, prediction),
        "stability": copy.deepcopy(group["stability"]),
        "far": group_far(bundle, prediction),
    }
    head = (f"{summary['rows']:,} 行のうち {summary['predicted']:,} 行を "
            f"{group['k']} 個の群に割り当てた")
    if summary["skipped"]:
        head += f" ({summary['skipped']:,} 行は割り当てられず)"
    sentences = [head]
    if groups:
        sentences.append(", ".join(
            f"群 {row['group']} {row['rows']:,} 行 "
            f"({row['share'] * 100:.1f} %)" for row in groups[:-1]
        ))
        if summary["far"] is not None:
            sentences.append(far_sentence(summary["far"]))
    sentences.append(stability_sentence(group["stability"]))
    conclusion = {"frame": frame, "summary": summary,
                  "sentences": sentences, "rules": {}}

    return conclusion


def group_of_bundle(bundle: Bundle) -> Optional[dict]:
    """
    群分けの一式の group の欄を, 説明の形で返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    group: Optional[dict]
        group の欄の写し (群分けの一式でなければ None)
    """
    # 画面が群の安定性と学習のときの群の要約を出すための写し
    if not bundle.from_group:
        return None
    group = copy.deepcopy(bundle.info["group"])

    return group
