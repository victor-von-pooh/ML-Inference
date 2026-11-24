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
from ml_eda import Dataset, load_data
from ml_modelcontrol import dump_spec, load_spec

from ml_inference.bundle import (
    RANK_BUNDLE_VERSION, Bundle, bundle_notes, inside, load_bundle,
    plan_weights, run_summary
)
from ml_inference.units import check_units

# モデルの入力と出力の名前 (ML-Quest の並べるの骨組みと, ML-ModelControl
# の同梱の例 two_tower_retrieval と同じ)
USER_INPUT = "user"
ITEM_INPUT = "item"
SCORE_OUTPUT = "score"
USER_EMBEDDING = "user_embedding"
ITEM_EMBEDDING = "item_embedding"
RANK_OUTPUTS = (SCORE_OUTPUT, USER_EMBEDDING, ITEM_EMBEDDING)

# 実験と一式の中の, 利用者と品物の番号の表と品物の埋め込みの索引
INDEX_FILE = "rank_index.json"
EMBEDDINGS_FILE = "item_embeddings.npy"

# 反応の表の列の名前の既定 (要る列と任意の列)
RANK_COLUMNS = {"user": "user", "item": "item", "reaction": None}

# rank の欄
RANK_KEYS = ("index", "embeddings", "users", "items", "width", "scale", "k",
             "columns")

# 点数と内積の定数倍を確かめる見本の利用者と品物の, 先頭と等間隔の
# それぞれの数, 合うとみなす差
SCALE_USERS = 8
SCALE_ITEMS = 256
SCALE_TOLERANCE = 1e-4

# 予測の表の列
RANK_FRAME_COLUMNS = ("user", "status", "reason", "rank", "item", "score")

# 予測の状態 (predict と同じ値)
OK = "ok"
SKIPPED = "skipped"

# 並べられない利用者の理由
UNKNOWN_USER = (
    "番号の表に無い利用者です (学習用の反応に無い利用者は並べられません)"
)
EMPTY_USER = "利用者が空です"
NOTHING_LEFT = "前に反応した品物を除くと, 並べる品物が残りません"

# 結論の判断 (conclude の NO_PREDICTION と同じ値)
NOT_RANKED = "予測なし"


def is_count(value: Any) -> bool:
    """
    値が 1 以上の整数 (真偽値でない) かを返す関数

    Parameters
    ----------
    value: Any
        確かめる値

    Returns
    ----------
    count: bool
        1 以上の整数なら True
    """
    # 真偽値は整数の子なので除く
    count = isinstance(value, int) and not isinstance(value, bool) and (
        value >= 1
    )

    return count


def check_columns(columns: Any) -> list[str]:
    """
    反応の表の列の名前の形を確かめる関数

    Parameters
    ----------
    columns: Any
        user, item, reaction (反応の強さ. 無ければ None) の辞書

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 利用者と品物は要る. 反応の強さは任意. 同じ列は使えない
    if not isinstance(columns, dict):
        return ["rank.columns は user, item, reaction の辞書にしてください"]
    errors = []
    unknown = [str(k) for k in columns if k not in RANK_COLUMNS]
    if unknown:
        errors.append(f"rank.columns の知らない項目: {', '.join(unknown)}")
    for key in ("user", "item"):
        if not isinstance(columns.get(key), str) or not columns[key]:
            errors.append(f"rank.columns.{key} は列の名前にしてください")
    reaction = columns.get("reaction")
    if reaction is not None and (not isinstance(reaction, str)
                                 or not reaction):
        errors.append("rank.columns.reaction は列の名前か null にして"
                      "ください")
    named = [columns.get(k) for k in RANK_COLUMNS if columns.get(k)]
    if len(set(map(str, named))) != len(named):
        errors.append("rank.columns に同じ列を 2 つ書けません")

    return errors


def check_rank_info(info: dict) -> list[str]:
    """
    並べるの一式 (版 8) の形を確かめる関数

    Parameters
    ----------
    info: dict
        inference.yaml の中身

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # rank が要り, データセット, items, 手順, 正解, 意思決定を持たない
    rank = info.get("rank")
    if not isinstance(rank, dict):
        return [f"版 {RANK_BUNDLE_VERSION} の一式には rank (番号の表と品物"
                "の埋め込みの索引) が要ります"]
    errors = []
    unknown = [str(k) for k in rank if k not in RANK_KEYS]
    if unknown:
        errors.append(f"rank の知らない項目: {', '.join(unknown)}")
    for key in ("index", "embeddings"):
        name = rank.get(key)
        if not isinstance(name, str) or not name or Path(name).name != name:
            errors.append(f"rank.{key} は一式の中のファイルの名前にして"
                          "ください")
    for key in ("users", "items", "width", "k"):
        if not is_count(rank.get(key)):
            errors.append(f"rank.{key} は 1 以上の整数にしてください")
    scale = rank.get("scale")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or (
        not math.isfinite(scale) or scale <= 0
    ):
        errors.append("rank.scale は 0 より大きい数にしてください")
    errors.extend(check_columns(rank.get("columns")))
    held = [k for k in ("dataset", "items", "procedure", "decision",
                        "check", "uncertainty", "target", "ready",
                        "calibrate")
            if info.get(k) is not None]
    if held:
        errors.append(
            f"並べるの一式の {', '.join(held)} は null にしてください "
            "(データセットと正解を持たないため)"
        )
    if info.get("conclusion"):
        errors.append("並べるの一式は結論の決め方の既定値を持てません")
    if info.get("inputs"):
        errors.append("並べるの一式の inputs は空にしてください (入力は "
                      "user と item のワンホットに決まっています)")
    if info.get("role") != "rank":
        errors.append("並べるの一式の role は rank にしてください")
    model = info.get("model") if isinstance(info.get("model"), dict) else {}
    if model.get("target_scale") is not None or model.get("output"):
        errors.append("並べるの一式は model.target_scale と model.output を"
                      "持てません")

    return errors


def check_rank_spec(spec: dict, users: int, items: int) -> None:
    """
    モデルの定義がワンホットの入力と点数と埋め込みの出力の約束に合うかを
    確かめる関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    users: int
        利用者の数 (番号の表の長さ)
    items: int
        品物の数

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        入力が user と item でない場合, 出力に score, user_embedding,
        item_embedding が無い場合, 入力の形の最後の大きさが数と違う場合
    """
    # 入力は user と item の 2 つ, 出力は点数と 2 つの埋め込み
    inputs = list(spec.get("inputs") or {})
    if sorted(inputs) != sorted([USER_INPUT, ITEM_INPUT]):
        raise ValueError(
            f"並べるの一式のモデルの入力は {USER_INPUT} と {ITEM_INPUT} の"
            f" 2 つにしてください (今は {', '.join(inputs) or '無し'})"
        )
    missing = [name for name in RANK_OUTPUTS
               if name not in (spec.get("outputs") or {})]
    if missing:
        raise ValueError(
            f"並べるの一式のモデルの出力に {', '.join(missing)} が要ります "
            "(品物の埋め込みを索引にするため)"
        )

    # ワンホットの幅は番号の表の数
    for name, count in ((USER_INPUT, users), (ITEM_INPUT, items)):
        shape = (spec["inputs"][name] or {}).get("shape")
        if isinstance(shape, list) and shape and shape[-1] is not None and (
            shape[-1] != count
        ):
            raise ValueError(
                f"モデルの入力 {name} の形の最後の大きさ {shape[-1]} が, "
                f"番号の表の数 {count} と違います"
            )


def read_index(path: Path) -> dict:
    """
    利用者と品物の番号の表 (rank_index.json) を読んで確かめる関数

    Parameters
    ----------
    path: Path
        番号の表のファイル

    Returns
    ----------
    index: dict
        users と items (番号の順の元の値)

    Raises
    ----------
    FileNotFoundError
        ファイルが無い場合
    ValueError
        形が違う場合と, 文字にすると重なる値がある場合
    """
    # 番号の順の元の値の並び (文字にして引くので, 文字で重ならない)
    if not path.is_file():
        raise FileNotFoundError(f"番号の表 {path} がありません")
    index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise ValueError(f"番号の表 {path.name} が辞書ではありません")
    for key in ("users", "items"):
        values = index.get(key)
        if not isinstance(values, list) or not values or any(
            isinstance(v, (dict, list)) or v is None for v in values
        ):
            raise ValueError(
                f"番号の表の {key} は, 空でない値 (文字か数) の並びにして"
                "ください"
            )
        if len({str(v) for v in values}) != len(values):
            raise ValueError(
                f"番号の表の {key} に, 文字にすると同じになる値があります"
            )
    index = {"users": index["users"], "items": index["items"]}

    return index


def one_hot(numbers: np.ndarray, size: int) -> torch.Tensor:
    """
    番号をワンホットにする関数

    Parameters
    ----------
    numbers: np.ndarray
        (件数,) の番号
    size: int
        番号の数

    Returns
    ----------
    x: torch.Tensor
        (件数, 番号の数) の float32 (番号の所だけ 1)
    """
    # 番号の所だけ 1
    x = torch.zeros(len(numbers), size)
    x[torch.arange(len(numbers)), torch.as_tensor(numbers)] = 1.0

    return x


def tower_outputs(
    model: Any, spec: dict, users: np.ndarray, items: np.ndarray,
    sizes: tuple, device: str
) -> dict:
    """
    利用者と品物の番号をワンホットにしてモデルに渡し, 出力を返す関数

    Parameters
    ----------
    model: Any
        組み立てたモデル
    spec: dict
        モデルの定義 (入力の順)
    users: np.ndarray
        利用者の番号
    items: np.ndarray
        品物の番号
    sizes: tuple
        (利用者の数, 品物の数)
    device: str
        使うデバイス

    Returns
    ----------
    outputs: dict
        score, user_embedding, item_embedding の float64 の配列

    Raises
    ----------
    ValueError
        出力が名前の辞書でない場合
    """
    # 定義の入力の順に渡す
    given = {USER_INPUT: one_hot(users, sizes[0]),
             ITEM_INPUT: one_hot(items, sizes[1])}
    with torch.no_grad():
        out = model(*[given[name].to(device) for name in spec["inputs"]])
    if not isinstance(out, dict) or any(k not in out for k in RANK_OUTPUTS):
        raise ValueError(
            f"モデルの出力に {', '.join(RANK_OUTPUTS)} がありません"
        )
    outputs = {
        name: out[name].detach().double().cpu().numpy()
        for name in RANK_OUTPUTS
    }

    return outputs


def item_index(
    model: Any, spec: dict, sizes: tuple, batch_size: int, device: str
) -> np.ndarray:
    """
    全ての品物の埋め込み (索引) を品物のまとまりごとに求める関数

    Parameters
    ----------
    model: Any
        組み立てたモデル
    spec: dict
        モデルの定義
    sizes: tuple
        (利用者の数, 品物の数)
    batch_size: int
        1 回に渡す品物の数
    device: str
        使うデバイス

    Returns
    ----------
    embeddings: np.ndarray
        (品物の数, 幅) の float32

    Raises
    ----------
    ValueError
        埋め込みが (件数, 幅) でない場合と, 数でない値がある場合
    """
    # 利用者は 1 人 (番号 0) だけ渡し, 品物の塔の出力を集める
    user = np.zeros(1, dtype=np.int64)
    parts = []
    for start in range(0, sizes[1], batch_size):
        items = np.arange(start, min(start + batch_size, sizes[1]))
        found = tower_outputs(model, spec, user, items, sizes, device)
        embedding = found[ITEM_EMBEDDING]
        if embedding.ndim != 2 or len(embedding) != len(items):
            raise ValueError(
                f"{ITEM_EMBEDDING} の形 {tuple(embedding.shape)} が (品物の"
                "件数, 幅) ではありません"
            )
        parts.append(embedding)
    embeddings = np.concatenate(parts).astype(np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{ITEM_EMBEDDING} に数でない値があります")

    return embeddings


def sample_numbers(size: int, count: int) -> np.ndarray:
    """
    確かめの見本の番号 (先頭の count 個と, 全体から等間隔の count 個) を
    返す関数

    Parameters
    ----------
    size: int
        番号の数
    count: int
        先頭と等間隔のそれぞれの数

    Returns
    ----------
    numbers: np.ndarray
        重ならない番号の小さい順 (size が count 以下なら全ての番号)
    """
    # 先頭だけでは後ろの番号の重みの誤りに気付けないため, 等間隔も混ぜる
    head = np.arange(min(count, size))
    spread = np.linspace(0, size - 1, min(count, size)).round()
    numbers = np.unique(np.concatenate([head, spread.astype(np.int64)]))

    return numbers


def score_scale(
    model: Any, spec: dict, sizes: tuple, embeddings: np.ndarray, device: str
) -> float:
    """
    点数が埋め込みの内積の正の定数倍であることを確かめ, その倍率を返す関数

    Parameters
    ----------
    model: Any
        組み立てたモデル
    spec: dict
        モデルの定義
    sizes: tuple
        (利用者の数, 品物の数)
    embeddings: np.ndarray
        品物の埋め込みの索引
    device: str
        使うデバイス

    Returns
    ----------
    scale: float
        点数 = 倍率 × 内積 の倍率

    Raises
    ----------
    ValueError
        点数の形が違う場合, 品物の埋め込みが利用者に依る場合, 点数が内積の
        正の定数倍でない場合
    """
    # 見本 (先頭と等間隔の利用者と品物) の全ての組の点数と, 埋め込みの
    # 内積
    users = sample_numbers(sizes[0], SCALE_USERS)
    items = sample_numbers(sizes[1], SCALE_ITEMS)
    chosen = embeddings[items].astype(np.float64)
    found = tower_outputs(model, spec, users, items, sizes, device)
    score = found[SCORE_OUTPUT]
    if score.shape != (len(users), len(items)):
        raise ValueError(
            f"{SCORE_OUTPUT} の形 {tuple(score.shape)} が (利用者の件数, "
            f"品物の件数) = ({len(users)}, {len(items)}) ではありません"
        )
    if not np.allclose(found[ITEM_EMBEDDING], chosen,
                       rtol=SCALE_TOLERANCE, atol=SCALE_TOLERANCE):
        raise ValueError(
            f"{ITEM_EMBEDDING} が一緒に渡した利用者で変わります (品物の塔"
            "が品物の入力だけから埋め込みを作るモデルにしてください)"
        )
    inner = found[USER_EMBEDDING] @ chosen.T

    # 最小二乗の倍率で, 全ての組が合うか
    power = float((inner * inner).sum())
    scale = float((score * inner).sum()) / power if power > 0 else math.nan
    limit = SCALE_TOLERANCE * max(1.0, float(np.abs(score).max()))
    if not (math.isfinite(scale) and scale > 0) or not np.allclose(
        score, scale * inner, rtol=SCALE_TOLERANCE, atol=limit
    ):
        raise ValueError(
            f"点数 ({SCORE_OUTPUT}) が, 利用者と品物の埋め込みの内積の正の"
            "定数倍になっていません (並べるの一式は品物の埋め込みの内積で"
            "並べるため)"
        )

    return scale


def rank_sources(
    weights: Union[str, Path], spec: Union[str, Path, dict, None],
    index: Union[str, Path, None]
) -> tuple[Path, Union[Path, dict], Path, Optional[dict]]:
    """
    重み, モデルの定義, 番号の表の場所と実験の状態を決める関数

    Parameters
    ----------
    weights: Union[str, Path]
        ML-Found の実験のディレクトリか重みのファイル
    spec: Union[str, Path, dict, None]
        モデルの定義 (省くと実験の model_spec.yaml)
    index: Union[str, Path, None]
        番号の表 (省くと実験の rank_index.json)

    Returns
    ----------
    sources: tuple[Path, Union[Path, dict], Path, Optional[dict]]
        重みのファイル, 定義, 番号の表, 実験の状態 (重みのファイルを
        渡した場合は None)

    Raises
    ----------
    ValueError
        重み, 定義, 番号の表が無い場合
    """
    # 実験のディレクトリなら, 中の 3 つのファイルと状態
    source = Path(weights).expanduser().resolve()
    run = None
    if source.is_dir():
        run = run_summary(source)
        spec = source / "model_spec.yaml" if spec is None else spec
        index = source / INDEX_FILE if index is None else index
        source = source / "final_model.pt"
    if not source.is_file():
        raise ValueError(f"重み {source} がありません")
    if spec is None or index is None:
        raise ValueError(
            "重みのファイルを渡す場合は, モデルの定義 (spec) と番号の表 "
            "(index) も渡してください"
        )
    if not isinstance(spec, dict):
        spec = Path(spec).expanduser()
        if not spec.is_file():
            raise ValueError(
                f"モデルの定義 {spec} がありません (仮のモデルで学習した実験"
                "は一式にできません. 定義を置いて学習し直してください)"
            )
    sources = (source, spec, Path(index).expanduser(), run)

    return sources


def make_rank_bundle(
    path: Union[str, Path], weights: Union[str, Path],
    spec: Union[str, Path, dict, None] = None,
    index: Union[str, Path, None] = None, k: int = 10,
    columns: Optional[dict] = None, name: Optional[str] = None,
    copy_weights: bool = True, overwrite: bool = False,
    pretrained: Optional[dict] = None, batch_size: int = 256
) -> dict:
    """
    利用者ごとに上位 K 件の品物を並べる一式を, 品物の埋め込みの索引ととも
    に作る関数

    Parameters
    ----------
    path: Union[str, Path]
        作るディレクトリ
    weights: Union[str, Path]
        ML-Found の実験のディレクトリ (final_model.pt, model_spec.yaml,
        rank_index.json を使い, 実験の状態を記録する) か重みのファイル
    spec: Union[str, Path, dict, None] = None
        モデルの定義 (ML-ModelControl の形. 省くと実験の model_spec.yaml)
    index: Union[str, Path, None] = None
        利用者と品物の番号の表 (省くと実験の rank_index.json)
    k: int = 10
        利用者ごとに並べる件数の既定 (問題の設定の rank.k)
    columns: Optional[dict] = None
        前に反応した品物の表の列の名前 (user, item, reaction. 問題の設定の
        rank の列. 省くと user と item)
    name: Optional[str] = None
        一式の名前 (省略するとディレクトリの名前)
    copy_weights: bool = True
        重みを一式に写すか (写さない場合は絶対パスで指す)
    overwrite: bool = False
        既にある一式を置き換えるか
    pretrained: Optional[dict] = None
        ノードの名前ごとの重みのライセンスの情報 (make_bundle と同じ)
    batch_size: int = 256
        品物の埋め込みを求めるときに 1 回に渡す品物の数

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
        番号の表が無い場合
    ValueError
        定義, 重み, 番号の表がワンホットの入力の約束に合わない場合と,
        点数が埋め込みの内積の定数倍でない場合
    """
    # 置き換えられるのは, 前に作った一式だけ
    final = Path(path).expanduser().resolve()
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"{final} は既にあります")
        if not (final / "inference.yaml").is_file():
            raise FileExistsError(f"{final} は推論の一式ではないので置き換えません")
    if not is_count(k):
        raise ValueError(f"k は 1 以上の整数にしてください (今は {k})")
    if batch_size < 1:
        raise ValueError("batch_size は 1 以上にしてください")
    named = {**RANK_COLUMNS, **dict(columns or {})}
    errors = check_columns(named)
    if errors:
        raise ValueError("反応の表の列の誤り: " + " / ".join(errors))

    # 重み, 定義, 番号の表と, 入力と出力の約束
    source, spec_source, index_file, run = rank_sources(weights, spec, index)
    found = read_index(index_file)
    sizes = (len(found["users"]), len(found["items"]))
    spec_dict = load_spec(spec_source)
    check_units(spec_dict)
    check_rank_spec(spec_dict, *sizes)
    spec_dict, records, kept, weight_notes = plan_weights(
        spec_dict, source, pretrained, copy_weights
    )

    # 隣の一時ディレクトリに作り, 確かめてから入れ替える
    writing = final.with_name(f".{final.name}.writing")
    if writing.exists():
        shutil.rmtree(writing)
    writing.mkdir(parents=True)
    try:
        dump_spec(spec_dict, writing / "model_spec.yaml")
        if copy_weights and kept is not None:
            torch.save(kept, writing / "final_model.pt")
        elif copy_weights:
            shutil.copy(source, writing / "final_model.pt")
        (writing / INDEX_FILE).write_text(
            json.dumps(found, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )
        info = {
            "format_version": RANK_BUNDLE_VERSION,
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "role": "rank",
            "output": {"form": "ranking"},
            "target": None,
            "procedure": None,
            "model": {
                "spec": "model_spec.yaml",
                "weights": "final_model.pt" if copy_weights else str(source),
                "output": None,
                "kind": "value",
                "target_scale": None,
                "pretrained": records,
            },
            "inputs": {},
            "dataset": None,
            "run": run,
            "conclusion": {},
            "decision": None,
            "check": None,
            "uncertainty": None,
            "notes": bundle_notes(
                Dataset(path=writing, info={}, frames={}, recipe={}), run
            ) + weight_notes,
        }

        # 品物の埋め込みの索引と, 点数の倍率
        building = Bundle(
            path=writing, info=info, spec=spec_dict,
            dataset=Dataset(path=writing, info={}, frames={}, recipe={})
        )
        model = building.model()
        embeddings = item_index(model, spec_dict, sizes, batch_size, "cpu")
        scale = score_scale(model, spec_dict, sizes, embeddings, "cpu")
        np.save(writing / EMBEDDINGS_FILE, embeddings, allow_pickle=False)
        info["rank"] = {
            "index": INDEX_FILE,
            "embeddings": EMBEDDINGS_FILE,
            "users": sizes[0],
            "items": sizes[1],
            "width": int(embeddings.shape[1]),
            "scale": scale,
            "k": k,
            "columns": named,
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


def check_rank_bundle(bundle: Bundle) -> None:
    """
    読んだ並べるの一式の定義, 番号の表, 索引が rank の欄と合うかを確かめる
    関数

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
        番号の表か索引が無い場合
    ValueError
        定義, 番号の表, 索引の大きさが rank の欄と違う場合
    """
    # 番号の表と定義の約束
    rank = bundle.info["rank"]
    found = read_index(inside(bundle.path, rank["index"]))
    sizes = (len(found["users"]), len(found["items"]))
    if sizes != (rank["users"], rank["items"]):
        raise ValueError(
            f"番号の表の利用者と品物の数 {sizes} が rank の users と items "
            f"({rank['users']}, {rank['items']}) と違います"
        )
    check_rank_spec(bundle.spec, *sizes)

    # 索引は (品物の数, 幅) の数の配列
    file = inside(bundle.path, rank["embeddings"])
    if not file.is_file():
        raise FileNotFoundError(f"品物の埋め込みの索引 {file} がありません")
    embeddings = np.load(file, allow_pickle=False)
    if embeddings.shape != (rank["items"], rank["width"]) or not (
        np.issubdtype(embeddings.dtype, np.floating)
        and np.isfinite(embeddings).all()
    ):
        raise ValueError(
            f"品物の埋め込みの索引の形 {tuple(embeddings.shape)} が "
            f"(品物の数, 幅) = ({rank['items']}, {rank['width']}) の数の"
            "配列ではありません"
        )
    bundle.cache["rank_index"] = found
    bundle.cache["rank_embeddings"] = embeddings


def request_users(bundle: Bundle, source: Any) -> list:
    """
    並べる利用者の並びを返す関数

    Parameters
    ----------
    bundle: Bundle
        並べるの一式
    source: Any
        利用者の値の並び (list, tuple, numpy の配列, pandas の Series) か,
        利用者の列を持つ表

    Returns
    ----------
    users: list
        渡した順の利用者の値

    Raises
    ----------
    ValueError
        並びでも表でもない場合と, 表に利用者の列が無い場合
    """
    # 表は利用者の列 (問題の設定の rank.user と同じ名前)
    column = bundle.info["rank"]["columns"]["user"]
    if isinstance(source, pd.DataFrame):
        if column not in source.columns:
            raise ValueError(
                f"表に利用者の列 {column} がありません (列: "
                f"{', '.join(map(str, source.columns))})"
            )
        users = source[column].tolist()
        return users
    if isinstance(source, (str, Path, dict)) or not isinstance(
        source, (list, tuple, np.ndarray, pd.Series)
    ):
        raise ValueError(
            "並べるの一式には, 利用者の値の並びか, 利用者の列 "
            f"({column}) を持つ表を渡してください"
        )
    users = list(source.tolist() if hasattr(source, "tolist") else source)

    return users


def is_missing(value: Any) -> bool:
    """
    値が欠損 (None か NaN) かを返す関数

    Parameters
    ----------
    value: Any
        確かめる値

    Returns
    ----------
    missing: bool
        欠損なら True
    """
    # 1 つの値だけを見る (並びの値は欠損でない)
    if value is None:
        return True
    missing = np.ndim(value) == 0 and not isinstance(value, dict) and bool(
        pd.isna(value)
    )

    return missing


def reaction_pairs(
    bundle: Bundle, table: Union[str, Path, pd.DataFrame], what: str
) -> dict:
    """
    反応の表から, 利用者の番号ごとの品物の番号の集合を作る関数

    Parameters
    ----------
    bundle: Bundle
        並べるの一式
    table: Union[str, Path, pd.DataFrame]
        反応の表かそのファイル (ML-EDA の load_data で読む)
    what: str
        誤りの文に使う表の呼び名

    Returns
    ----------
    pairs: dict
        利用者の番号から品物の番号の集合 (番号の表に無い利用者と品物の行
        と, 反応の強さが 0 以下か数でない行は除く)

    Raises
    ----------
    ValueError
        表でない場合と, 列が無い場合
    """
    # 表かファイル
    if isinstance(table, (str, Path)):
        table = load_data(Path(table).expanduser()).frame
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"{what}は表かそのファイルの場所にしてください")
    columns = bundle.info["rank"]["columns"]
    needed = [columns[k] for k in RANK_COLUMNS if columns.get(k)]
    missing = [c for c in needed if c not in table.columns]
    if missing:
        raise ValueError(
            f"{what}に列 {', '.join(missing)} がありません (列: "
            f"{', '.join(map(str, table.columns))})"
        )

    # 反応の強さが 0 より大きい行だけ (問題の設定の rank.reaction)
    frame = table
    if columns.get("reaction"):
        strength = pd.to_numeric(frame[columns["reaction"]], errors="coerce")
        frame = frame[strength > 0]
    index = bundle.cache["rank_index"]
    users = {str(v): n for n, v in enumerate(index["users"])}
    items = {str(v): n for n, v in enumerate(index["items"])}
    pairs: dict = {}
    for user, item in zip(frame[columns["user"]], frame[columns["item"]]):
        if is_missing(user) or is_missing(item):
            continue
        u, i = users.get(str(user)), items.get(str(item))
        if u is not None and i is not None:
            pairs.setdefault(u, set()).add(i)

    return pairs


def user_vectors(
    bundle: Bundle, numbers: np.ndarray, batch_size: int, device: str
) -> np.ndarray:
    """
    利用者の番号から利用者の埋め込みを求める関数

    Parameters
    ----------
    bundle: Bundle
        並べるの一式
    numbers: np.ndarray
        利用者の番号
    batch_size: int
        1 回に渡す利用者の数
    device: str
        使うデバイス

    Returns
    ----------
    vectors: np.ndarray
        (利用者の件数, 幅) の float64
    """
    # 品物は 1 つ (番号 0) だけ渡し, 利用者の塔の出力を集める
    rank = bundle.info["rank"]
    sizes = (rank["users"], rank["items"])
    model = bundle.model().to(device)
    item = np.zeros(1, dtype=np.int64)
    parts = [np.zeros((0, rank["width"]))]
    try:
        for start in range(0, len(numbers), batch_size):
            chunk = numbers[start:start + batch_size]
            found = tower_outputs(model, bundle.spec, chunk, item, sizes,
                                  device)
            parts.append(found[USER_EMBEDDING])
    finally:
        model.to("cpu")
    vectors = np.concatenate(parts)

    return vectors


def top_of(
    scores: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    利用者ごとに点数の高い順に K 件の品物を選ぶ関数

    Parameters
    ----------
    scores: np.ndarray
        (利用者の件数, 品物の数) の点数 (除く品物は -inf)
    k: int
        並べる件数

    Returns
    ----------
    picked: tuple[np.ndarray, np.ndarray]
        (利用者の件数, K) の品物の番号と点数 (同じ点数なら番号の小さい
        品物が先. 除いた品物は点数が -inf)
    """
    # 全ての品物は並べ替えず, K 番目の点数以上の品物 (同じ点数の品物は
    # 全て) だけを, 点数の高い順, 同じ点数は番号の順に並べる
    size = scores.shape[1]
    k = min(k, size)
    order = np.zeros((len(scores), k), dtype=np.int64)
    for row, values in enumerate(-scores):
        found = np.arange(size)
        if k < size:
            cut = np.partition(values, k - 1)[k - 1]
            found = np.flatnonzero(values <= cut)
        order[row] = found[np.lexsort((found, values[found]))][:k]
    picked = (order, np.take_along_axis(scores, order, axis=1))

    return picked


def predict_rank(
    bundle: Bundle, source: Any, k: Optional[int] = None,
    seen: Union[str, Path, pd.DataFrame, None] = None,
    batch_size: int = 256, device: str = "cpu"
) -> pd.DataFrame:
    """
    利用者ごとに, 品物の埋め込みの索引から上位 K 件の品物と点数を返す関数

    Parameters
    ----------
    bundle: Bundle
        並べるの一式
    source: Any
        利用者の値の並びか, 利用者の列を持つ表
    k: Optional[int] = None
        並べる件数 (省くと一式の rank.k)
    seen: Union[str, Path, pd.DataFrame, None] = None
        前に反応した品物の表 (渡すと, 利用者ごとに表の品物を除いて並べる.
        省くと除かない)
    batch_size: int = 256
        1 回に点数を求める利用者の数
    device: str = "cpu"
        使うデバイス

    Returns
    ----------
    result: pd.DataFrame
        利用者と順位ごとの user, status, reason, rank (1 から), item,
        score の表 (並べられない利用者は 1 行で, status が skipped)

    Raises
    ----------
    ValueError
        利用者の並びか表でない場合, k が 1 以上の整数でない場合, 前に
        反応した品物の表に列が無い場合
    """
    # 並べる件数と, 前に反応した品物
    rank = bundle.info["rank"]
    k = rank["k"] if k is None else k
    if not is_count(k):
        raise ValueError(f"k は 1 以上の整数にしてください (今は {k})")
    users = request_users(bundle, source)
    excluded = reaction_pairs(bundle, seen, "前に反応した品物の表") if (
        seen is not None
    ) else {}

    # 知っている利用者の番号 (文字にして引く)
    numbers = {str(v): n for n, v in enumerate(bundle.cache["rank_index"][
        "users"
    ])}
    keys = [None if is_missing(u) else numbers.get(str(u)) for u in users]
    known = np.array(sorted({n for n in keys if n is not None}),
                     dtype=np.int64)

    # 利用者のまとまりごとに内積の点数を求め, 上位 K 件だけを残す
    items = bundle.cache["rank_index"]["items"]
    embeddings = bundle.cache["rank_embeddings"].astype(np.float64)
    picked: dict = {}
    for start in range(0, len(known), batch_size):
        chunk = known[start:start + batch_size]
        scores = rank["scale"] * (
            user_vectors(bundle, chunk, batch_size, device) @ embeddings.T
        )
        for row, user in enumerate(chunk):
            scores[row, sorted(excluded.get(int(user), ()))] = -math.inf
        order, values = top_of(scores, k)
        for row, user in enumerate(chunk):
            kept = np.isfinite(values[row])
            picked[int(user)] = (order[row][kept], values[row][kept])

    # 渡した順の利用者ごとに行を並べる
    rows = []
    for user, number in zip(users, keys):
        reason = EMPTY_USER if is_missing(user) else UNKNOWN_USER
        if number is not None and len(picked[number][0]):
            for place, (item, score) in enumerate(zip(*picked[number]), 1):
                rows.append((user, OK, "", place, items[int(item)],
                             float(score)))
            continue
        if number is not None:
            reason = NOTHING_LEFT
        rows.append((user, SKIPPED, reason, None, None, math.nan))
    result = pd.DataFrame(rows, columns=list(RANK_FRAME_COLUMNS))
    result["rank"] = result["rank"].astype("Int64")
    result["score"] = result["score"].astype(float)
    result["reason"] = result["reason"].astype(object)
    result.attrs["k"] = k
    result.attrs["excluded"] = seen is not None

    return result


def conclude_ranking(prediction: pd.DataFrame) -> dict:
    """
    並べるの予測から, 利用者の人数だけの結論を出す関数

    Parameters
    ----------
    prediction: pd.DataFrame
        predict_rank の表

    Returns
    ----------
    conclusion: dict
        frame (decision を足した表. 並べた行は空, 並べられない行は予測
        なし), summary (rows (渡した利用者の数), predicted (並べた利用者
        の数), skipped, k, excluded), sentences, rules (空)
    """
    # 利用者ごとに 1 度だけ数える (並べた利用者は K 行, 他は 1 行)
    frame = prediction.copy()
    ok = frame["status"] == OK
    frame["decision"] = np.where(ok, "", NOT_RANKED)
    first = frame["rank"].isna() | (frame["rank"] == 1)
    rows = int(first.sum())
    predicted = int((first & ok).sum())
    k = prediction.attrs.get("k")
    excluded = bool(prediction.attrs.get("excluded"))
    summary = {"rows": rows, "predicted": predicted,
               "skipped": rows - predicted, "k": k, "excluded": excluded}
    head = (f"利用者 {rows:,} 人のうち {predicted:,} 人に, 上位 {k} 件の"
            "品物を並べた")
    if summary["skipped"]:
        head += f" ({summary['skipped']:,} 人は並べられず)"
    sentences = [head]
    if excluded:
        sentences.append("前に反応した品物は除いて並べた")
    conclusion = {"frame": frame, "summary": summary,
                  "sentences": sentences, "rules": {}}

    return conclusion


def rank_metrics(
    bundle: Union[str, Path, Bundle], prediction: pd.DataFrame,
    truth: Union[str, Path, pd.DataFrame],
    seen: Union[str, Path, pd.DataFrame, None] = None,
    k: Optional[int] = None
) -> dict:
    """
    並べた品物と後の反応から, Recall@K と NDCG を数える関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        並べるの一式かそのディレクトリ
    prediction: pd.DataFrame
        predict の表 (read_result で読んだ表も)
    truth: Union[str, Path, pd.DataFrame]
        正解の反応の表 (並べた後の反応)
    seen: Union[str, Path, pd.DataFrame, None] = None
        前に反応した品物の表 (渡すと, 正解から除く)
    k: Optional[int] = None
        K (省くと予測の表の attrs の k, 無ければ表の最大の順位, 並べた
        行が無ければ一式の rank.k)

    Returns
    ----------
    metrics: dict
        users (数えた利用者の数), k, recall_at_k, ndcg (数える利用者が
        いなければ 0.0)
    """
    # 正解は, 番号の表にある品物のうち前に反応していないもの
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    if not bundle.from_rank:
        raise ValueError("rank_metrics は並べるの一式だけに使えます")
    done = prediction[prediction["status"] == OK]
    if k is None:
        k = prediction.attrs.get("k")
    if k is None and len(done):
        k = int(done["rank"].max())
    if k is None:
        k = bundle.info["rank"]["k"]
    if not is_count(k):
        raise ValueError(f"k は 1 以上の整数にしてください (今は {k})")
    relevant = reaction_pairs(bundle, truth, "正解の反応の表")
    if seen is not None:
        before = reaction_pairs(bundle, seen, "前に反応した品物の表")
        relevant = {u: items - before.get(u, set())
                    for u, items in relevant.items()}

    # 並べた利用者ごとの上位 K 件 (表の順は利用者ごとの順位の順. 同じ
    # 利用者を 2 度渡した表は初めの並びだけ). 前に反応した品物を除くと
    # 残りが無い利用者は, 空の上位として数える (ML-Quest の骨組みと同じ)
    index = bundle.cache["rank_index"]
    users = {str(v): n for n, v in enumerate(index["users"])}
    items = {str(v): n for n, v in enumerate(index["items"])}
    counted = prediction[(prediction["status"] == OK) | (
        prediction["reason"] == NOTHING_LEFT
    )]
    top: dict = {}
    current = None
    for user, status, place, item in zip(
        counted["user"], counted["status"], counted["rank"], counted["item"]
    ):
        if status != OK or int(place) == 1:
            current = users.get(str(user))
            current = None if current in top else current
            if current is not None:
                top[current] = []
        if current is not None and status == OK:
            top[current].append(items.get(str(item)))

    # Recall@K は正解の全数で割り, NDCG は min(K, 正解の数) 件の理想と比べる
    recalls, gains = [], []
    for user, chosen in top.items():
        wanted = relevant.get(user) or set()
        if not wanted:
            continue
        hits = [r for r, item in enumerate(chosen[:k], 1) if item in wanted]
        recalls.append(len(hits) / len(wanted))
        ideal = sum(1 / math.log2(r + 1)
                    for r in range(1, min(k, len(wanted)) + 1))
        gains.append(sum(1 / math.log2(r + 1) for r in hits) / ideal)
    # 数える利用者がいなければ 0 (ML-Quest の骨組みと同じ)
    count = max(len(recalls), 1)
    metrics = {
        "users": len(recalls),
        "k": k,
        "recall_at_k": float(sum(recalls) / count),
        "ndcg": float(sum(gains) / count),
    }

    return metrics


def rank_of_bundle(bundle: Bundle) -> Optional[dict]:
    """
    並べるの一式の rank の欄を, 説明の形で返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    rank: Optional[dict]
        rank の欄の写し (並べるの一式でなければ None)
    """
    # 画面が件数の既定と前に反応した品物の表の列を出すため
    if not bundle.from_rank:
        return None
    rank = copy.deepcopy(bundle.info["rank"])

    return rank
