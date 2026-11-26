import copy
import datetime as dt
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from ml_modelcontrol import dump_spec

from ml_inference.bundle import (
    SEARCH_BUNDLE_VERSION, Bundle, check_sourced, inside, load_bundle, plain,
    ready_record, ready_spec
)
from ml_inference.conclude import NO_PREDICTION
from ml_inference.grouping import is_whole, read_evaluation
from ml_inference.ready import (
    READY_LIBRARIES, ReadyEmbedding, check_head, check_ready, read_images
)

# 成果物と一式のファイルの名前 (ML-Quest の似た写真を探す骨組みと同じ)
INDEX_FILE = "search_index.json"
EMBEDDINGS_FILE = "search_embeddings.npy"
EVALUATION_FILE = "evaluations/search_valid.json"

# 索引の説明の項目と, 写真ごとの項目
INDEX_KEYS = ("ready", "width", "k", "max_latency_ms", "root", "items")
ITEM_KEYS = ("path", "label", "sha256")
INDEX_READY_KEYS = ("library", "model", "weights", "sha256")

# 評価の行の指標 (比べる相手と差も)
EVALUATION_METRICS = ("recall_at_k", "ndcg_at_k")

# inference.yaml の search の項目
SEARCH_KEYS = ("index", "embeddings", "items", "width", "k",
               "max_latency_ms", "labeled", "root", "evaluation")

# 予測の表に足す列
SEARCH_FRAME_COLUMNS = ("status", "reason", "rank", "item", "label",
                        "similarity", "latency_ms")

# 行の状態と, 並べられない理由
OK = "ok"
SKIPPED = "skipped"
NOT_FINITE = "埋め込みに数でない値があります"
NOTHING_LEFT = "中身が同じ写真を除くと, 並べる写真が残りません"

# 検索の判断の項目
SEARCH_DECISION_KEYS = ("type", "k")

# 64 文字の小文字の 16 進
HEX64 = re.compile(r"[0-9a-f]{64}")


def is_hash(value: Any) -> bool:
    """
    64 文字の小文字の 16 進の hash かを返す関数

    Parameters
    ----------
    value: Any
        調べる値

    Returns
    ----------
    found: bool
        hash の文字なら True
    """
    # 文字で, 全体が 16 進の 64 文字
    found = isinstance(value, str) and HEX64.fullmatch(value) is not None

    return found


def is_limit(value: Any) -> bool:
    """
    待ち時間の上限 (正の有限の数か null) かを返す関数

    Parameters
    ----------
    value: Any
        調べる値

    Returns
    ----------
    found: bool
        null か, 真偽値でない正の有限の数なら True
    """
    # 上限は無くてもよい
    if value is None:
        return True
    found = isinstance(value, (int, float)) and not isinstance(
        value, bool
    ) and math.isfinite(value) and value > 0

    return found


def check_items(items: Any) -> list[str]:
    """
    索引の写真の並びを確かめる関数

    Parameters
    ----------
    items: Any
        search_index.json の items

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 1 枚以上の辞書の並びで, 場所は重ならない文字
    if not isinstance(items, list) or not items or not all(
        isinstance(item, dict) for item in items
    ):
        return ["items は 1 枚以上の写真の辞書の並びにしてください"]
    errors = []
    for index, item in enumerate(items):
        unknown = [key for key in item if key not in ITEM_KEYS]
        if unknown:
            errors.append(
                f"items の {index} 番目に知らない項目です: "
                f"{', '.join(map(str, unknown))}"
            )
        if not isinstance(item.get("path"), str) or not item["path"]:
            errors.append(f"items の {index} 番目の path は空でない文字に"
                          "してください")
        label = item.get("label")
        if label is not None and (not isinstance(label, str) or not label):
            errors.append(f"items の {index} 番目の label は空でない文字か "
                          "null にしてください")
        if not is_hash(item.get("sha256")):
            errors.append(f"items の {index} 番目の sha256 は 64 文字の小文字"
                          "の 16 進にしてください")
    paths = [item.get("path") for item in items]
    if len(set(map(str, paths))) != len(paths):
        errors.append("items の path が重なっています")
    labeled = {item.get("label") is not None for item in items}
    if len(labeled) > 1:
        errors.append("items の label は全ての写真に書くか, 全ての写真で "
                      "null にしてください")

    return errors


def check_index(index: Any) -> list[str]:
    """
    索引の説明 (search_index.json) の形を確かめる関数

    Parameters
    ----------
    index: Any
        search_index.json の中身

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 項目と, 埋め込みを作った既製のモデル
    if not isinstance(index, dict):
        return [f"{INDEX_FILE} は辞書にしてください"]
    errors = [
        f"{INDEX_FILE} に知らない項目です: {key}"
        for key in index if key not in INDEX_KEYS
    ]
    errors.extend(
        f"{INDEX_FILE} に {key} がありません"
        for key in INDEX_KEYS if key not in index
    )
    ready = index.get("ready")
    if not isinstance(ready, dict) or sorted(ready) != sorted(
        INDEX_READY_KEYS
    ):
        errors.append(
            f"{INDEX_FILE} の ready は {', '.join(INDEX_READY_KEYS)} の辞書"
            "にしてください"
        )
    else:
        if not all(isinstance(ready[key], str) and ready[key]
                   for key in ("library", "model", "weights")):
            errors.append(f"{INDEX_FILE} の ready の library, model, weights"
                          " は空でない文字にしてください")
        if not is_hash(ready["sha256"]):
            errors.append(f"{INDEX_FILE} の ready の sha256 は 64 文字の"
                          "小文字の 16 進にしてください")

    # 幅, 件数, 上限, 写真のディレクトリ, 写真の並び
    if "width" in index and not is_whole(index["width"], 1):
        errors.append(f"{INDEX_FILE} の width は 1 以上の整数にしてください")
    if "k" in index and not is_whole(index["k"], 1):
        errors.append(f"{INDEX_FILE} の k は 1 以上の整数にしてください")
    if not is_limit(index.get("max_latency_ms")):
        errors.append(f"{INDEX_FILE} の max_latency_ms は正の数か null に"
                      "してください")
    root = index.get("root")
    if root is not None and (not isinstance(root, str) or not root):
        errors.append(f"{INDEX_FILE} の root は空でない文字か null にして"
                      "ください")
    if "items" in index:
        errors.extend(check_items(index["items"]))

    return errors


def read_index(path: Path) -> dict:
    """
    索引の説明 (search_index.json) を読む関数

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    index: dict
        索引の説明

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        形が違う場合
    """
    # JSON の外の NaN と Infinity は読まない
    if not path.is_file():
        raise FileNotFoundError(f"索引の説明 {path} がありません")
    try:
        index = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=reject_constant)
    except ValueError as error:
        raise ValueError(f"{path.name} を読めません ({error})") from error
    errors = check_index(index)
    if errors:
        raise ValueError(f"{path.name} の誤り: " + " / ".join(errors))

    return index


def reject_constant(name: str) -> Any:
    """
    JSON の外の定数 (NaN, Infinity) を誤りにする関数

    Parameters
    ----------
    name: str
        定数の名前

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        いつも
    """
    # json.loads の parse_constant に渡す
    raise ValueError(f"JSON の外の値 {name} があります")


def read_embeddings(path: Path, items: int, width: int) -> np.ndarray:
    """
    索引の埋め込み (search_embeddings.npy) を読む関数

    Parameters
    ----------
    path: Path
        ファイル
    items: int
        索引の写真の数
    width: int
        埋め込みの幅

    Returns
    ----------
    embeddings: np.ndarray
        (写真の数, 幅) の float64

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        形が違う場合, 数でない値がある場合, 長さが 0 の行がある場合
    """
    # pickle の任意のコードは読まない
    if not path.is_file():
        raise FileNotFoundError(f"索引の埋め込み {path} がありません")
    try:
        found = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"{path.name} を読めません ({error})") from error
    if found.shape != (items, width) or not np.issubdtype(
        found.dtype, np.number
    ):
        raise ValueError(
            f"{path.name} の形 {found.shape} が, 索引の写真の数と幅 "
            f"({items}, {width}) の数の配列ではありません"
        )
    embeddings = found.astype(np.float64)
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{path.name} に数でない値があります")
    if (np.linalg.norm(embeddings, axis=1) == 0).any():
        raise ValueError(f"{path.name} に長さが 0 の埋め込みがあります")

    return embeddings


def evaluation_of(path: Path, k: int) -> dict:
    """
    骨組みの評価の表から, 一式に写す評価を作る関数

    Parameters
    ----------
    path: Path
        evaluations/search_valid.json
    k: int
        索引の k

    Returns
    ----------
    evaluation: dict
        split, queries, k, 指標とその比べる相手と差, baseline

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        行が 1 つでない場合, 指標が数でない場合, k が索引と違う場合
    """
    # ML-Found の evaluations の形の 1 行
    found = read_evaluation(path)
    rows = found["rows"]
    if len(rows) != 1:
        raise ValueError(f"{path.name} の rows は 1 行にしてください")
    row = rows[0]
    names = [f"{name}{tail}" for name in EVALUATION_METRICS
             for tail in ("", "_baseline", "_diff")]
    bad = [
        name for name in names
        if not isinstance(row.get(name), (int, float))
        or isinstance(row.get(name), bool)
        or not math.isfinite(row[name])
    ]
    if bad:
        raise ValueError(
            f"{path.name} の行の {', '.join(bad)} を数にしてください"
        )
    if not is_whole(row.get("queries"), 1) or row.get("k") != k:
        raise ValueError(
            f"{path.name} の行の queries は 1 以上の整数, k は索引の k ({k})"
            " にしてください"
        )
    baseline = found.get("baseline")
    evaluation = {
        "split": "valid", "queries": int(row["queries"]), "k": int(k),
        **{name: float(row[name]) for name in names},
        "baseline": str(baseline) if baseline else None,
    }

    return evaluation


def run_sources(run: Union[str, Path]) -> dict:
    """
    似た写真を探す骨組みの成果物のディレクトリから, 一式に要るものを読む
    関数

    Parameters
    ----------
    run: Union[str, Path]
        成果物のディレクトリ

    Returns
    ----------
    sources: dict
        root (ディレクトリ), index (索引の説明), embeddings, evaluation
        (種類が無ければ None)

    Raises
    ----------
    ValueError
        ディレクトリでない場合と, 成果物の形が違う場合
    FileNotFoundError
        成果物が無い場合
    """
    # 索引の説明と埋め込み
    root = Path(run).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"成果物のディレクトリ {root} がありません")
    index = read_index(root / INDEX_FILE)
    embeddings = read_embeddings(root / EMBEDDINGS_FILE,
                                 len(index["items"]), int(index["width"]))

    # 種類のある索引だけが評価を持つ
    labeled = index["items"][0].get("label") is not None
    evaluation_path = root / EVALUATION_FILE
    if labeled:
        evaluation = evaluation_of(evaluation_path, int(index["k"]))
    elif evaluation_path.exists():
        raise ValueError(
            f"種類の無い索引に評価 {EVALUATION_FILE} があります (正解が無い"
            "ので評価は書きません)"
        )
    else:
        evaluation = None
    sources = {"root": root, "index": index, "embeddings": embeddings,
               "evaluation": evaluation}

    return sources


def search_section(sources: dict) -> dict:
    """
    成果物から inference.yaml の search の欄を作る関数

    Parameters
    ----------
    sources: dict
        run_sources の結果

    Returns
    ----------
    search: dict
        SEARCH_KEYS の辞書
    """
    # 索引の大きさ, 既定の件数と上限, 評価の写し
    index = sources["index"]
    search = {
        "index": INDEX_FILE,
        "embeddings": EMBEDDINGS_FILE,
        "items": len(index["items"]),
        "width": int(index["width"]),
        "k": int(index["k"]),
        "max_latency_ms": index["max_latency_ms"],
        "labeled": index["items"][0].get("label") is not None,
        "root": index["root"],
        "evaluation": sources["evaluation"],
    }

    return search


def make_search_bundle(
    path: Union[str, Path], run: Union[str, Path],
    name: Optional[str] = None, overwrite: bool = False,
    license: Optional[dict] = None
) -> dict:
    """
    問いの写真に近い索引の写真を返す一式を, 似た写真を探す骨組みの成果物
    から作る関数

    Parameters
    ----------
    path: Union[str, Path]
        作るディレクトリ
    run: Union[str, Path]
        成果物のディレクトリ (search_index.json, search_embeddings.npy と,
        種類があれば evaluations/search_valid.json)
    name: Optional[str] = None
        一式の名前 (省略するとディレクトリの名前)
    overwrite: bool = False
        既にある一式を置き換えるか
    license: Optional[dict] = None
        既製のモデルの重みのライセンスの情報 (make_ready_bundle と同じ)

    Returns
    ----------
    info: dict
        書き出した inference.yaml の中身

    Raises
    ----------
    FileExistsError
        ディレクトリが既にあり, overwrite が False の場合と, 一式ではない
        ディレクトリを置き換えようとした場合
    FileNotFoundError
        成果物が無い場合
    ValueError
        成果物が約束に合わない場合, 重みの hash が索引を作ったものと違う
        場合, 埋め込みを取り出せない場合, 埋め込みの幅が索引と違う場合
    """
    # 置き換えられるのは, 前に作った一式だけ
    final = Path(path).expanduser().resolve()
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"{final} は既にあります")
        if not (final / "inference.yaml").is_file():
            raise FileExistsError(f"{final} は推論の一式ではないので置き換えません")
    sources = run_sources(run)

    # 出どころの重みの説明, 1 つのノードの定義, 出どころの重みの hash
    wanted = sources["index"]["ready"]
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
            f"({record['sha256'][:12]}…) が, 索引を作った重みの hash "
            f"({wanted['sha256'][:12]}…) と違います"
        )
    if ready["weights"] != wanted["weights"]:
        raise ValueError(
            f"重みの名前 {ready['weights']} が, 索引を作った重みの名前 "
            f"{wanted['weights']} と違います (版の決まった名前を書いて"
            "ください)"
        )

    # 分類の最後の層の手前の出力の幅と, 索引の幅
    width = int(sources["index"]["width"])
    found_width = check_head(ReadyEmbedding(built.eval()),
                             int(ready["preprocess"]["crop"]))
    if found_width != width:
        raise ValueError(
            f"既製のモデルの埋め込みの幅 {found_width} が, 索引の幅 {width} "
            "と違います"
        )

    # 隣の一時ディレクトリでの作成と, 確かめた後の入れ替え (重みは空)
    writing = final.with_name(f".{final.name}.writing")
    if writing.exists():
        shutil.rmtree(writing)
    writing.mkdir(parents=True)
    try:
        dump_spec(spec, writing / "model_spec.yaml")
        torch.save({}, writing / "final_model.pt")
        shutil.copy(sources["root"] / INDEX_FILE, writing / INDEX_FILE)
        shutil.copy(sources["root"] / EMBEDDINGS_FILE,
                    writing / EMBEDDINGS_FILE)
        search = search_section(sources)
        info = {
            "format_version": SEARCH_BUNDLE_VERSION,
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "role": "rank",
            "output": {"form": "ranking"},
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
            "decision": {"type": "search", "k": search["k"]},
            "check": None,
            "uncertainty": None,
            "notes": notes,
            "ready": ready,
            "search": search,
        }
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


def check_search_info(info: dict) -> list[str]:
    """
    検索の一式 (版 15) の ready と search の形を確かめる関数

    Parameters
    ----------
    info: dict
        inference.yaml の中身

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # ready と出どころから読む重みが要る
    errors = []
    if info.get("ready") is None:
        errors.append(f"版 {SEARCH_BUNDLE_VERSION} の一式には ready (既製の"
                      "モデルの出どころと前処理) が要ります")
    else:
        errors.extend(check_ready(info["ready"]))
    model = info.get("model") if isinstance(info.get("model"), dict) else {}
    errors.extend(check_sourced(model))
    if info.get("role") != "rank":
        errors.append("検索の一式の role は rank にしてください")
    if info.get("rank") is not None:
        errors.append("検索の一式は rank を持てません")

    # search の項目と値
    search = info.get("search")
    if not isinstance(search, dict):
        errors.append(f"版 {SEARCH_BUNDLE_VERSION} の一式には search が"
                      "要ります")
        return errors
    errors.extend(f"search に知らない項目です: {key}"
                  for key in search if key not in SEARCH_KEYS)
    errors.extend(f"search に {key} がありません"
                  for key in SEARCH_KEYS if key not in search)
    for key in ("index", "embeddings"):
        if not isinstance(search.get(key), str) or not search[key]:
            errors.append(f"search.{key} は一式の中のファイルの名前にして"
                          "ください")
    for key in ("items", "width", "k"):
        if not is_whole(search.get(key), 1):
            errors.append(f"search.{key} は 1 以上の整数にしてください")
    if not is_limit(search.get("max_latency_ms")):
        errors.append("search.max_latency_ms は正の数か null にしてください")
    if not isinstance(search.get("labeled"), bool):
        errors.append("search.labeled は真偽値にしてください")
    root = search.get("root")
    if root is not None and (not isinstance(root, str) or not root):
        errors.append("search.root は空でない文字か null にしてください")
    evaluation = search.get("evaluation")
    if evaluation is not None and not isinstance(evaluation, dict):
        errors.append("search.evaluation は辞書か null にしてください")
    if isinstance(search.get("labeled"), bool) and (
        evaluation is None
    ) == search["labeled"]:
        errors.append("search.evaluation は種類のある索引 (labeled が真) "
                      "だけに書いてください")
    decision = info.get("decision")
    if decision is not None:
        errors.extend(check_search_decision(decision))

    return errors


def check_search_bundle(bundle: Bundle) -> None:
    """
    読んだ検索の一式の索引が search の欄と ready と合うかを確かめる関数

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
        索引のファイルが無い場合
    ValueError
        索引の写真の数, 幅, 埋め込みを作った既製のモデルと重みが一式と
        違う場合
    """
    # 索引の説明と, 埋め込みを作った既製のモデルと重み
    search = bundle.info["search"]
    index = read_index(inside(bundle.path, search["index"]))
    if (len(index["items"]), index["width"]) != (search["items"],
                                                 search["width"]):
        raise ValueError(
            f"{search['index']} の写真の数と幅が search の欄と違います"
        )
    ready = bundle.info["ready"]
    wanted = index["ready"]
    hashes = {r.get("sha256") for r in bundle.info["model"]["pretrained"]}
    if (wanted["library"], wanted["model"], wanted["weights"]) != (
        ready["library"], ready["model"], ready["weights"]
    ) or hashes != {wanted["sha256"]}:
        raise ValueError(
            f"{search['index']} の索引を作った既製のモデルか重みの hash が, "
            "一式の ready と model.pretrained と違います"
        )
    labeled = index["items"][0].get("label") is not None
    if labeled != search["labeled"]:
        raise ValueError(
            f"{search['index']} の写真の種類の有無が search.labeled と"
            "違います"
        )

    # 長さで割った索引の埋め込み (余弦の近さは内積)
    embeddings = read_embeddings(inside(bundle.path, search["embeddings"]),
                                 search["items"], search["width"])
    bundle.cache["search_items"] = index["items"]
    bundle.cache["search_unit"] = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )


def search_k(bundle: Bundle, k: Any) -> int:
    """
    返す件数を確かめて返す関数

    Parameters
    ----------
    bundle: Bundle
        検索の一式
    k: Any
        渡した件数 (None なら一式の k)

    Returns
    ----------
    k: int
        1 以上の整数

    Raises
    ----------
    ValueError
        1 以上の整数でない場合
    """
    # 省くと一式の既定
    if k is None:
        return int(bundle.info["search"]["k"])
    if not is_whole(k, 1):
        raise ValueError("k は 1 以上の整数にしてください")
    k = int(k)

    return k


def top_items(
    unit: np.ndarray, point: np.ndarray, excluded: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    1 つの問いの埋め込みに余弦の近さで近い索引の写真を上位 K 件選ぶ関数

    Parameters
    ----------
    unit: np.ndarray
        長さで割った索引の埋め込み ((写真の数, 幅))
    point: np.ndarray
        問いの埋め込み ((幅,))
    excluded: np.ndarray
        除く索引の写真の印 ((写真の数,) の真偽値)
    k: int
        返す件数

    Returns
    ----------
    found: tuple[np.ndarray, np.ndarray]
        近い順の索引の番号と余弦の近さ (同じ近さなら番号の小さい写真)
    """
    # 余弦の近さ (問いの長さで割る) と, 除く写真
    similarity = unit @ (point / np.linalg.norm(point))
    kept = np.flatnonzero(~excluded)
    values = similarity[kept]

    # 近い順 (安定な並べ替えで番号の小さい写真が先)
    order = np.argsort(-values, kind="stable")[:k]
    found = (kept[order], values[order])

    return found


def predict_search(
    bundle: Bundle, source: Any, k: Optional[int] = None,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    問いの写真ごとに, 余弦の近さで近い索引の写真を上位 K 件並べる関数

    Parameters
    ----------
    bundle: Bundle
        検索の一式
    source: Any
        問いの画像のファイルかディレクトリ
    k: Optional[int] = None
        問いごとに返す件数 (省くと一式の search.k)
    device: str = "cpu"
        使うデバイス

    Returns
    ----------
    result: pd.DataFrame
        問いと順位ごとの行. 画像の読み込みの表の列 (予測の列と同じ名前は
        「(入力)」付き) と, status, reason, rank (Int64), item, label,
        similarity, latency_ms. attrs に k

    Raises
    ----------
    ValueError
        画像のファイルかディレクトリでない場合と, k が 1 以上の整数でない
        場合
    """
    # 件数, 読み込みの表, 組み立てたモデル (組み立ては待ち時間に含めない)
    k = search_k(bundle, k)
    preprocess = bundle.info["ready"]["preprocess"]
    started = time.perf_counter()
    frame, images, reasons = read_images(source, preprocess)
    read_ms = (time.perf_counter() - started) * 1000 / max(len(frame), 1)
    model = bundle.model().to(device)
    unit = bundle.cache["search_unit"]
    items = bundle.cache["search_items"]
    digests = np.array([item["sha256"] for item in items])
    queries = frame["digest"] if "digest" in frame.columns else (
        pd.Series([None] * len(frame))
    )

    # 1 枚ずつの埋め込みと上位 K 件 (待ち時間は読み込みの 1 枚分を足す)
    rows = []
    try:
        for index in range(len(frame)):
            base = frame.iloc[index].to_dict()
            if reasons[index]:
                rows.append((base, reasons[index], None, math.nan))
                continue
            begun = time.perf_counter()
            chunk = torch.from_numpy(images[index:index + 1]).to(device)
            with torch.no_grad():
                point = model(chunk).detach().double().cpu().numpy()[0]
            if not np.isfinite(point).all() or not np.linalg.norm(point):
                rows.append((base, NOT_FINITE, None, math.nan))
                continue
            excluded = digests == queries.iloc[index]
            if excluded.all():
                rows.append((base, NOTHING_LEFT, None, math.nan))
                continue
            found = top_items(unit, point, excluded, k)
            latency = (time.perf_counter() - begun) * 1000 + read_ms
            rows.append((base, "", found, latency))
    finally:
        model.to("cpu")

    # 問いと順位ごとの行 (並べられない問いは 1 行)
    result = expand_rows(frame, rows, items)
    result.attrs["k"] = k

    return result


def expand_rows(
    frame: pd.DataFrame, rows: list, items: list
) -> pd.DataFrame:
    """
    問いごとの上位 K 件を, 問いと順位ごとの表にする関数

    Parameters
    ----------
    frame: pd.DataFrame
        画像の読み込みの表
    rows: list
        問いごとの (元の行, 理由, (番号, 近さ) か None, 待ち時間)
    items: list
        索引の写真の並び

    Returns
    ----------
    result: pd.DataFrame
        元の列と SEARCH_FRAME_COLUMNS の表
    """
    # 元の列 (予測の列と同じ名前は「(入力)」)
    renamed = {c: f"{c} (入力)" for c in frame.columns
               if c in SEARCH_FRAME_COLUMNS}
    records = []
    for base, reason, found, latency in rows:
        source = {renamed.get(key, key): value for key, value in base.items()}
        if found is None:
            records.append({**source, "status": SKIPPED, "reason": reason,
                            "rank": None, "item": None, "label": None,
                            "similarity": math.nan, "latency_ms": latency})
            continue
        for place, (number, value) in enumerate(zip(*found), start=1):
            item = items[int(number)]
            records.append({**source, "status": OK, "reason": "",
                            "rank": place, "item": item["path"],
                            "label": item.get("label"),
                            "similarity": float(value),
                            "latency_ms": latency})
    columns = [renamed.get(c, c) for c in frame.columns]
    result = pd.DataFrame(records, columns=[*columns,
                                            *SEARCH_FRAME_COLUMNS])
    result["rank"] = pd.array(result["rank"].tolist(), dtype="Int64")
    for name in ("item", "label", "reason", "status"):
        result[name] = result[name].astype(object)
    for name in ("similarity", "latency_ms"):
        result[name] = result[name].astype(float)

    return result


def query_latency(prediction: pd.DataFrame) -> pd.Series:
    """
    並べた問いごとの待ち時間を返す関数

    Parameters
    ----------
    prediction: pd.DataFrame
        predict_search の表

    Returns
    ----------
    latency: pd.Series
        並べた問いの最初の行の待ち時間 (問いの行の位置の順)
    """
    # 同じ問いの行は同じ待ち時間なので, 1 位の行だけ
    first = prediction[(prediction["status"] == OK)
                       & (prediction["rank"] == 1)]
    latency = first["latency_ms"].astype(float)

    return latency


def latency_summary(
    bundle: Bundle, prediction: pd.DataFrame
) -> tuple[Optional[dict], Optional[int]]:
    """
    待ち時間の要約と, 上限を超えた問いの数を返す関数

    Parameters
    ----------
    bundle: Bundle
        検索の一式
    prediction: pd.DataFrame
        predict_search の表

    Returns
    ----------
    found: tuple[Optional[dict], Optional[int]]
        median と max (並べた問いが無ければ None) と, 上限を超えた問いの数
        (上限が無ければ None)
    """
    # 並べた問いの待ち時間
    latency = query_latency(prediction)
    summary = {"median": float(latency.median()),
               "max": float(latency.max())} if len(latency) else None
    limit = bundle.info["search"]["max_latency_ms"]
    over = None if limit is None else int((latency > limit).sum())
    found = (summary, over)

    return found


def evaluation_sentence(evaluation: Optional[dict]) -> str:
    """
    骨組みの評価の文を返す関数

    Parameters
    ----------
    evaluation: Optional[dict]
        一式の評価の写し

    Returns
    ----------
    sentence: str
        評価の値と比べる相手の値の文 (評価が無ければ無い旨)
    """
    # 種類の無い索引は正解が無い
    if evaluation is None:
        sentence = ("索引の写真に種類が無いので評価は無い. 並べた写真と近さを"
                    "見て確かめる")
        return sentence
    name = evaluation.get("baseline") or "比べる相手"
    sentence = (
        f"骨組みの評価 (検証用の問い {evaluation['queries']:,} 枚, 上位 "
        f"{evaluation['k']} 件): Recall@K {evaluation['recall_at_k']:.3f}, "
        f"NDCG@K {evaluation['ndcg_at_k']:.3f}. {name}は "
        f"{evaluation['recall_at_k_baseline']:.3f}, "
        f"{evaluation['ndcg_at_k_baseline']:.3f} (差 "
        f"{evaluation['recall_at_k_diff']:+.3f}, "
        f"{evaluation['ndcg_at_k_diff']:+.3f})"
    )

    return sentence


def conclude_search(bundle: Bundle, prediction: pd.DataFrame) -> dict:
    """
    問いごとの上位 K 件から, 件数と待ち時間の結論を出す関数

    Parameters
    ----------
    bundle: Bundle
        検索の一式
    prediction: pd.DataFrame
        predict_search の表

    Returns
    ----------
    conclusion: dict
        frame (decision を足した表), summary (rows, predicted, skipped, k,
        items, latency, max_latency_ms, over_latency, evaluation),
        sentences, rules (空)
    """
    # 問いの数 (並べた問いは 1 位の行, 並べられない問いは 1 行)
    search = bundle.info["search"]
    frame = prediction.copy()
    ok = (frame["status"] == OK).to_numpy()
    frame["decision"] = np.where(ok, "", NO_PREDICTION)
    predicted = int((ok & (frame["rank"] == 1).fillna(False)
                     .to_numpy(dtype=bool)).sum())
    skipped = int((~ok).sum())
    k = int(prediction.attrs.get("k", search["k"]))
    latency, over = latency_summary(bundle, prediction)
    summary = {
        "rows": predicted + skipped, "predicted": predicted,
        "skipped": skipped, "k": k, "items": int(search["items"]),
        "latency": latency, "max_latency_ms": search["max_latency_ms"],
        "over_latency": over,
        "evaluation": copy.deepcopy(search["evaluation"]),
    }

    # 件数, 待ち時間, 上限, 索引の少なさ, 骨組みの評価の文
    head = (f"問いの写真 {summary['rows']:,} 枚のうち {predicted:,} 枚に, "
            f"索引の写真 {summary['items']:,} 枚から近い順に上位 {k} 件を"
            "並べた")
    if skipped:
        head += f" ({skipped:,} 枚は並べられず)"
    sentences = [head]
    if latency is not None:
        sentences.append(
            f"1 件の問いの待ち時間は中央値 {latency['median']:.1f} ms, "
            f"最大 {latency['max']:.1f} ms"
        )
    if over is not None:
        sentences.append(
            f"待ち時間の上限 {search['max_latency_ms']:g} ms を超えた問いは "
            f"{over:,} 枚"
        )
    if summary["items"] < k:
        sentences.append(
            f"索引の写真が {summary['items']:,} 枚なので, 問いごとに "
            f"{summary['items']:,} 件まで並べた"
        )
    sentences.append(evaluation_sentence(search["evaluation"]))
    conclusion = {"frame": frame, "summary": summary,
                  "sentences": sentences, "rules": {}}

    return conclusion


def check_search_decision(decision: Any) -> list[str]:
    """
    検索の一式の検索の判断の決め方の誤りを一覧で返す関数

    Parameters
    ----------
    decision: Any
        意思決定の決め方

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 検索の判断の型と書ける項目 (type, k)
    if not isinstance(decision, dict) or decision.get("type") != "search":
        return ["検索の一式の意思決定は検索の判断 (type: search) だけです"]
    errors = [
        f"検索の一式の検索の判断に書けない項目です: {key} (書けるのは type,"
        " k だけ)"
        for key in decision if key not in SEARCH_DECISION_KEYS
    ]
    if not is_whole(decision.get("k"), 1):
        errors.append("検索の判断の k は 1 以上の整数にしてください")

    return errors


def decision_notes(bundle: Bundle) -> list[str]:
    """
    検索の判断の前提の文を返す関数

    Parameters
    ----------
    bundle: Bundle
        検索の一式

    Returns
    ----------
    notes: list[str]
        近さと待ち時間の前提の文
    """
    # 近さの意味と, 測った場所
    ready = bundle.info["ready"]
    notes = [
        f"近さは既製のモデル ({ready['library']} の {ready['model']}) の埋め"
        "込みの余弦の近さで, 見た目が近いことを表し, 同じ商品であることは"
        "保証しない",
        "待ち時間はこの場所で測った値なので, 本番の場所では確かめ直す",
    ]

    return notes


def decide_search(
    bundle: Bundle, prediction: pd.DataFrame, decision: dict
) -> dict:
    """
    問いごとに, 近い順の上位 K 件を返すと決める関数

    Parameters
    ----------
    bundle: Bundle
        検索の一式
    prediction: pd.DataFrame
        predict_search の表 (か conclude_search の表)
    decision: dict
        {type: search, k}

    Returns
    ----------
    result: dict
        frame (action を足した表), summary (queries, returned, k,
        max_latency_ms, over_latency, latency), sentences, notes (前提),
        decision (使った決め方)

    Raises
    ----------
    ValueError
        決め方に誤りがある場合, 表に列が無い場合, 並べた件数より大きい k
        の場合
    """
    # 決め方と表の確認
    decision = plain(dict(decision))
    errors = check_search_decision(decision)
    missing = [c for c in ("status", "rank", "latency_ms")
               if c not in prediction.columns]
    if missing:
        errors.append(f"予測の表に {', '.join(missing)} の列がありません")
    listed = prediction.attrs.get("k")
    if not errors and listed is not None and decision["k"] > listed:
        errors.append(
            f"検索の判断の k ({decision['k']}) が, 予測の表で並べた件数 "
            f"({listed}) より大きいので決められません (k を渡して並べ直して"
            "ください)"
        )
    if errors:
        raise ValueError("意思決定の決め方の誤り: " + " / ".join(errors))

    # 上位 K 件の行を返す (並べられない問いの行は空)
    k = int(decision["k"])
    ok = (prediction["status"] == OK).to_numpy()
    rank = prediction["rank"].to_numpy(dtype=float, na_value=math.nan)
    action = pd.array([None] * len(prediction), dtype="boolean")
    action[ok] = rank[ok] <= k
    frame = prediction.copy()
    frame["action"] = action
    latency, over = latency_summary(bundle, prediction)
    search = bundle.info["search"]
    summary = {
        "queries": int((ok & (rank == 1)).sum()),
        "returned": int((ok & (rank <= k)).sum()),
        "k": k, "max_latency_ms": search["max_latency_ms"],
        "over_latency": over, "latency": latency,
    }

    # 前提, 返す件数, 上限の文 (前提は notes にも)
    notes = decision_notes(bundle)
    sentences = [
        *notes,
        f"問いの写真 {summary['queries']:,} 枚それぞれに, 近い順に {k} 件を"
        f"返す (返す行 {summary['returned']:,} 行)",
    ]
    if over is not None:
        sentences.append(
            f"待ち時間の上限 {search['max_latency_ms']:g} ms (守る指標) を"
            f"超えた問いは {over:,} 枚"
        )
        if over:
            sentences.append(
                "上限を超えた問いがあるので, 写真の大きさ, 索引の数, 動かす"
                "場所を確かめる"
            )
    result = {"frame": frame, "summary": summary, "sentences": sentences,
              "notes": notes, "decision": decision}

    return result


def search_of_bundle(bundle: Bundle) -> Optional[dict]:
    """
    検索の一式の search の欄を, 説明の形で返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    search: Optional[dict]
        search の欄の写し (ファイルの名前を除く. 検索の一式でなければ
        None)
    """
    # 画面が件数, 上限, 索引の大きさと評価を出すための写し
    if not bundle.from_search:
        return None
    search = {key: copy.deepcopy(value)
              for key, value in bundle.info["search"].items()
              if key not in ("index", "embeddings")}

    return search
