import copy
import hashlib
from typing import Any, Optional

import torch

# 学習済みの重みの置き方 (一式の重みに含める, 出どころから読む)
PLACES = ("bundle", "source")

# 記録に写す pretrained_info の項目
INFO_KEYS = (
    "unit", "model", "hub_id", "revision", "pinned_revision", "license",
    "status", "redistributable", "source"
)


def node_keys(state: dict, node: str) -> list[str]:
    """
    重みのうち, 定義の一番上のノードのキーを返す関数

    Parameters
    ----------
    state: dict
        state_dict
    node: str
        ノードの名前

    Returns
    ----------
    keys: list[str]
        nodes.<ノード>. で始まるキー (並べた順)
    """
    # ML-ModelControl のモデルはノードを nodes の下に持つ
    prefix = f"nodes.{node}."
    keys = sorted(k for k in state if k.startswith(prefix))

    return keys


def state_hash(state: dict, keys: list[str]) -> str:
    """
    重みの一部の hash を求める関数

    Parameters
    ----------
    state: dict
        state_dict
    keys: list[str]
        hash に含めるキー

    Returns
    ----------
    digest: str
        キーの名前, 型, 形, 中身からの sha256
    """
    # キーの順に, 名前, 型, 形, 中身のバイト
    hasher = hashlib.sha256()
    for key in sorted(keys):
        tensor = state[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        flat = tensor.reshape(-1)
        hasher.update(flat.view(torch.uint8).numpy().tobytes()
                      if flat.numel() else b"")
    digest = hasher.hexdigest()

    return digest


def lookup_pretrained(
    spec: dict, given: Optional[dict]
) -> tuple[dict, list[str]]:
    """
    学習済みの重みを読むノードと, その重みのライセンスの情報を集める
    関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    given: Optional[dict]
        ノードの名前ごとの pretrained_info の形の情報 (渡せばそれを使い,
        None なら ML-ModelControl の pretrained_info で読む)

    Returns
    ----------
    infos: dict
        学習済みの重みを読むノードごとの情報
    notes: list[str]
        読めなかったことの注意

    Raises
    ----------
    ValueError
        渡した情報のノードが定義に無い場合
    """
    # 渡した情報 (画面が読んだもの, テスト) はそのまま
    nodes = spec.get("nodes") or {}
    if given is not None:
        unknown = [str(n) for n in given if n not in nodes]
        if unknown:
            raise ValueError(
                f"pretrained の {', '.join(unknown)} はモデルの定義のノードに"
                "ありません"
            )
        infos = {
            str(node): dict(info) for node, info in given.items()
            if info is not None
        }
        return infos, []

    # 入っている ML-ModelControl で読む (古いものには関数が無い)
    try:
        from ml_modelcontrol import pretrained_info
    except ImportError:
        notes = [
            "入っている ML-ModelControl は学習済みの重みのライセンスを"
            "読めない (pretrained_info が無い) ので, 重みは全て一式に"
            "写しました"
        ]
        return {}, notes
    infos = {}
    for node, item in nodes.items():
        unit = item.get("unit") if isinstance(item, dict) else None
        if not unit:
            continue
        try:
            info = pretrained_info(unit, dict(item.get("params") or {}))
        except Exception as error:
            info = {
                "unit": unit, "status": "unknown", "redistributable": False,
                "source": f"読めなかった ({type(error).__name__}: {error})",
            }
        if info is not None:
            infos[str(node)] = info

    return infos, []


def pin_spec(spec: dict, infos: dict) -> dict:
    """
    出どころから読むかもしれないノードの版を, 読んだ版に固定した定義を
    返す関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    infos: dict
        ノードごとの重みのライセンスの情報

    Returns
    ----------
    pinned: dict
        revision の引数を持つ部品で, 写せない重みの版を pinned_revision に
        替えた定義 (元の定義は変えない)
    """
    from ml_modelcontrol import get_unit

    # 写せない重みで, 版の識別子が分かり, 部品が revision を受け取るもの
    pinned = copy.deepcopy(spec)
    for node, info in infos.items():
        revision = info.get("pinned_revision")
        item = pinned["nodes"][node]
        if info.get("redistributable") or not revision:
            continue
        try:
            names = [param.name for param in get_unit(item["unit"]).params]
        except KeyError:
            continue
        if "revision" in names:
            item.setdefault("params", {})
            item["params"] = {**(item["params"] or {}),
                              "revision": str(revision)}

    return pinned


def place_weights(
    infos: dict, state: dict, fresh: dict, copy_weights: bool
) -> tuple[list[dict], dict, list[str]]:
    """
    学習済みの重みを読むノードごとに, 一式に写すか出どころから読むかを
    決める関数

    Parameters
    ----------
    infos: dict
        ノードごとの重みのライセンスの情報
    state: dict
        渡した重み (学習した state_dict)
    fresh: dict
        定義から組み立てた直後の state_dict (出どころの重み)
    copy_weights: bool
        重みを一式に写すか (写さない場合は除かない)

    Returns
    ----------
    records: list[dict]
        ノードごとの記録 (inference.yaml の model.pretrained)
    kept: dict
        一式に写す重み (出どころから読むノードを除いたもの)
    notes: list[str]
        注意の文
    """
    # 写せないもので, 学習で変わっていないノードだけを除く
    records = []
    kept = dict(state)
    notes = []
    for node, info in infos.items():
        keys = node_keys(state, node)
        unchanged = bool(keys) and keys == node_keys(fresh, node) and all(
            state[k].shape == fresh[k].shape and torch.equal(
                state[k].detach().cpu(), fresh[k].detach().cpu()
            ) for k in keys
        )
        redistributable = bool(info.get("redistributable"))
        place = "source" if (
            copy_weights and not redistributable and unchanged
        ) else "bundle"
        if place == "source":
            for key in keys:
                kept.pop(key)
        elif not redistributable and copy_weights:
            notes.append(
                f"ノード {node} の学習済みの重み ({info.get('status')}) は"
                "再配布できるか分からないか, できませんが, 学習で変わったので"
                "一式の重みに含めました. 元の重みのライセンスを確かめてから"
                "一式を渡してください"
            )
        record: dict[str, Any] = {"node": node}
        record.update({k: info.get(k) for k in INFO_KEYS})
        record["redistributable"] = redistributable
        record["place"] = place
        record["sha256"] = state_hash(state, keys)
        record["keys"] = len(keys)
        records.append(record)

    return records, kept, notes
