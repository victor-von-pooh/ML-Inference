"""
強化学習の方策の一式 (ML-Found の policy/) の読み込みと守りの適用

一式の形と守りの決まりは ML-Found の方策の一式 (版 1) と同じで, ML-Found の
コードに頼らずに読む. モデルは, 渡されたものか, 一式に写したモデルの定義
(ML-ModelControl の形) から組み立てたものを使う
"""
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from ml_modelcontrol import build_model, load_spec
from torch import nn

from ml_inference.bundle import inside
from ml_inference.units import check_units

# 読める方策の一式 (policy.json) の形の版
POLICY_FORMAT_VERSION = 1

# 一式のファイルの名前
SPEC_FILE = "policy.json"
WEIGHTS_FILE = "policy.pt"

# 外れたときの動き
FALLBACK_KINDS = {"stop", "handover", "action"}

# 空間, 守り, 外れたときの動きに書ける項目 (書き損じを見つけるため,
# 他の項目は誤り)
SPACE_KEYS = {
    "discrete": {"type", "n", "start"},
    "box": {"type", "shape", "low", "high"}
}
GUARD_KEYS = {
    "action_low", "action_high", "allowed_actions", "observation_low",
    "observation_high", "fallback"
}
FALLBACK_KEYS = {"kind", "action"}

# 数として読める配列の種類 (真偽値, 整数, 小数)
NUMERIC_KINDS = "biuf"


def is_int(value: Any) -> bool:
    """
    真偽値を除く整数かを確かめる関数

    Parameters
    ----------
    value: Any
        確かめる値

    Returns
    ----------
    result: bool
        int (と NumPy の整数) で, 真偽値でない場合は True
    """
    # bool は int の子なので除外
    result = isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )

    return result


def bounds_from_json(
    items: Optional[list], size: int, fill: float, name: str
) -> np.ndarray:
    """
    JSON の範囲のリストを, 制限なしを無限大にした配列に戻す関数

    Parameters
    ----------
    items: Optional[list]
        範囲の値のリスト, None の場合は全て制限なし
    size: int
        要素の数
    fill: float
        制限なしの値 (下限は -inf, 上限は inf)
    name: str
        誤りの文に出す名前

    Returns
    ----------
    bounds: np.ndarray
        平らな float64 の配列

    Raises
    ----------
    ValueError
        要素の数が違うか, 有限の数と null 以外の要素がある場合
    """
    # 無い場合は全て制限なし, ある場合は要素の数の確認
    if items is None:
        bounds = np.full(size, fill, dtype=np.float64)
        return bounds
    if not isinstance(items, list) or len(items) != size:
        raise ValueError(
            f"{name} は要素が {size} 個のリストにしてください: {items!r}"
        )

    # 要素は有限の数か None だけ (nan, 無限大, 真偽値, 文字は誤り)
    for value in items:
        is_number = isinstance(value, (int, float)) and not isinstance(
            value, bool
        )
        if value is not None and not (is_number and math.isfinite(value)):
            raise ValueError(
                f"{name} の要素は有限の数か null にしてください: {value!r}"
            )
    bounds = np.array(
        [fill if value is None else float(value) for value in items],
        dtype=np.float64
    )

    return bounds


def space_size(space: dict) -> int:
    """
    box の空間の要素の数を返す関数

    Parameters
    ----------
    space: dict
        box の空間の辞書

    Returns
    ----------
    size: int
        shape の積
    """
    # 各次元の大きさの積
    size = int(np.prod(space["shape"])) if space["shape"] else 1

    return size


def check_space(space: Any) -> None:
    """
    空間の辞書の形を確かめる関数

    Parameters
    ----------
    space: Any
        policy.json の空間

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        辞書でない, 種類が違う, 知らない項目がある, 値が合わない場合
    """
    # 辞書であることと, 種類ごとに書ける項目だけであることの確認
    if not isinstance(space, dict):
        raise ValueError(f"空間は辞書にしてください: {space!r}")
    if space.get("type") not in SPACE_KEYS:
        raise ValueError(f"空間の type は discrete か box にしてください: {space}")
    unknown = sorted(set(space) - SPACE_KEYS[space["type"]])
    if unknown:
        raise ValueError(f"空間に知らない項目 {unknown} があります: {space}")

    # 種類ごとに要る項目の確認 (discrete の start は省略で 0)
    if space["type"] == "discrete":
        if not is_int(space.get("n")) or space["n"] < 1:
            raise ValueError(f"discrete の n は 1 以上の整数にしてください: {space}")
        space.setdefault("start", 0)
        if not is_int(space["start"]):
            raise ValueError(f"discrete の start は整数にしてください: {space}")
        return
    shape = space.get("shape")
    if not isinstance(shape, list) or not all(
        is_int(size) and size >= 1 for size in shape
    ):
        raise ValueError(f"box の shape は正の整数のリストにしてください: {space}")
    size = space_size(space)
    low = bounds_from_json(space.get("low"), size, -math.inf, "low")
    high = bounds_from_json(space.get("high"), size, math.inf, "high")
    if np.any(low > high):
        raise ValueError(f"box の low が high より大きい要素があります: {space}")


def action_limits(
    guards: dict, act_space: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    box の行動の下限と上限を, 空間と守りの狭い方で求める関数

    Parameters
    ----------
    guards: dict
        守りの辞書
    act_space: dict
        box の行動の空間の辞書

    Returns
    ----------
    low: np.ndarray
        行動の下限
    high: np.ndarray
        行動の上限
    """
    # 空間の範囲と守りの範囲の狭い方
    size = space_size(act_space)
    low = np.maximum(
        bounds_from_json(act_space.get("low"), size, -math.inf, "low"),
        bounds_from_json(
            guards.get("action_low"), size, -math.inf, "action_low"
        )
    )
    high = np.minimum(
        bounds_from_json(act_space.get("high"), size, math.inf, "high"),
        bounds_from_json(
            guards.get("action_high"), size, math.inf, "action_high"
        )
    )

    return low, high


def float32_inside(
    values: np.ndarray, low: np.ndarray, high: np.ndarray
) -> Optional[np.ndarray]:
    """
    行動を float32 にし, 丸めで範囲の外に出た要素を内側に寄せる関数

    0.1 のような範囲は float32 で少し大きな値 (0.10000000149) になるので,
    丸めた後に範囲を確かめ直し, 外に出た要素は内側の隣の float32 に寄せる

    Parameters
    ----------
    values: np.ndarray
        範囲の中の行動 (float64)
    low: np.ndarray
        行動の下限
    high: np.ndarray
        行動の上限

    Returns
    ----------
    action: Optional[np.ndarray]
        float32 の行動, 有限でないか範囲の中に表せない場合は None
    """
    # 大きすぎる値は無限大になるので, 警告を出さずに丸めて確認
    with np.errstate(over="ignore"):
        action = np.asarray(values, dtype=np.float64).astype(np.float32)
    if not np.all(np.isfinite(action)):
        action = None
        return action

    # 丸めで外に出た要素を, 内側の隣の float32 に移動
    above = action.astype(np.float64) > high
    action[above] = np.nextafter(action[above], np.float32(-np.inf))
    below = action.astype(np.float64) < low
    action[below] = np.nextafter(action[below], np.float32(np.inf))

    # それでも外の要素 (範囲に float32 が無い場合) は表せない
    widened = action.astype(np.float64)
    if np.any((widened > high) | (widened < low)):
        action = None
        return action

    return action


def check_guards(guards: Any, obs_space: dict, act_space: dict) -> None:
    """
    守りの辞書が空間と食い違わないことを確かめる関数

    Parameters
    ----------
    guards: Any
        policy.json の守り
    obs_space: dict
        観測の空間の辞書
    act_space: dict
        行動の空間の辞書

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        知らない項目がある, 範囲が空間と合わない, 外れたときの動きが
        違う場合
    """
    # 守りと外れたときの動きは辞書で, 書ける項目だけ
    if not isinstance(guards, dict):
        raise ValueError(f"guards は辞書にしてください: {guards!r}")
    unknown = sorted(set(guards) - GUARD_KEYS)
    if unknown:
        raise ValueError(f"guards に知らない項目 {unknown} があります")
    fallback = guards.get("fallback")
    if not isinstance(fallback, dict):
        raise ValueError(f"guards の fallback は辞書にしてください: {fallback!r}")
    unknown = sorted(set(fallback) - FALLBACK_KEYS)
    if unknown:
        raise ValueError(f"fallback に知らない項目 {unknown} があります")

    # 観測は box に限り, 止める条件の要素の数と向きを確認
    if obs_space["type"] != "box":
        raise ValueError("観測の空間は box にしてください")
    size = space_size(obs_space)
    low = bounds_from_json(
        guards.get("observation_low"), size, -math.inf, "observation_low"
    )
    high = bounds_from_json(
        guards.get("observation_high"), size, math.inf, "observation_high"
    )
    if np.any(low > high):
        raise ValueError("observation_low が observation_high より大きい要素があります")

    # 行動の範囲は空間の中で, 空間の種類に合うものだけ
    if act_space["type"] == "discrete":
        check_discrete_guards(guards, act_space)
    else:
        check_box_guards(guards, act_space)

    # 外れたときの動きの種類
    if fallback.get("kind") not in FALLBACK_KINDS:
        raise ValueError(
            f"fallback は {sorted(FALLBACK_KINDS)} のどれかにしてください: "
            f"{fallback.get('kind')!r}"
        )


def check_discrete_guards(guards: dict, act_space: dict) -> None:
    """
    discrete の行動の守りを確かめる関数

    Parameters
    ----------
    guards: dict
        守りの辞書
    act_space: dict
        行動の空間の辞書

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        box の範囲がある, 選んでよい行動か決めた行動が空間の外の場合
    """
    # box の範囲は書けず, 選んでよい行動と決めた行動は空間の中
    if guards.get("action_low") is not None or (
        guards.get("action_high") is not None
    ):
        raise ValueError("discrete の行動には action_low と action_high を書けません")
    start = act_space["start"]
    valid = set(range(start, start + act_space["n"]))
    allowed = guards.get("allowed_actions")
    if allowed is not None and (
        not isinstance(allowed, list) or not allowed
        or not all(is_int(action) for action in allowed)
        or set(allowed) - valid
    ):
        raise ValueError(
            f"allowed_actions は {sorted(valid)} の中から 1 つ以上にしてください: "
            f"{allowed}"
        )
    fallback = guards["fallback"]
    if fallback.get("kind") == "action":
        action = fallback.get("action")
        usable = valid if allowed is None else set(allowed)
        if not is_int(action) or action not in usable:
            raise ValueError(
                f"fallback の行動 {action!r} は選んでよい行動にしてください"
            )


def check_box_guards(guards: dict, act_space: dict) -> None:
    """
    box の行動の守りを確かめる関数

    Parameters
    ----------
    guards: dict
        守りの辞書
    act_space: dict
        行動の空間の辞書

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        選んでよい行動がある, 範囲が逆向き, 決めた行動が範囲の外の場合
    """
    # 選んでよい行動は書けず, 範囲は空間の範囲の中
    if guards.get("allowed_actions") is not None:
        raise ValueError("box の行動には allowed_actions を書けません")
    low, high = action_limits(guards, act_space)
    if np.any(low > high):
        raise ValueError("action_low が action_high より大きい要素があります")

    # 決めた行動は範囲の中で, float32 でも範囲の中に表せること
    fallback = guards["fallback"]
    if fallback.get("kind") != "action":
        return
    action = fallback.get("action")
    if not isinstance(action, list) or len(action) != low.size:
        raise ValueError(
            f"fallback の行動は要素が {low.size} 個のリストにしてください"
        )
    values = bounds_from_json(action, low.size, math.nan, "fallback")
    if not np.all(np.isfinite(values)) or np.any(
        (values < low) | (values > high)
    ):
        raise ValueError(
            f"fallback の行動 {action} は行動の範囲の中にしてください"
        )
    if float32_inside(values, low, high) is None:
        raise ValueError(
            f"fallback の行動 {action} は float32 で行動の範囲の中に表せません"
        )


def check_policy_spec(spec: Any) -> None:
    """
    policy.json の中身の版と形を確かめる関数

    Parameters
    ----------
    spec: Any
        policy.json を読んだもの

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        版が無いか新しい, 形が違う場合 (守りを読み違えないように止める)
    """
    # 版が無いものと新しい版は, 守りを読み違えないように停止
    if not isinstance(spec, dict):
        raise ValueError("policy.json は辞書にしてください")
    version = spec.get("format_version")
    if not is_int(version) or version < 1:
        raise ValueError(
            "policy.json に format_version (1 以上の整数) がありません"
        )
    if version > POLICY_FORMAT_VERSION:
        raise ValueError(
            f"policy.json の版 {version} はこの ML-Inference が読める版 "
            f"{POLICY_FORMAT_VERSION} より新しいです. ML-Inference を新しく"
            "してください (守りを読み違えないように止めます)"
        )
    if spec.get("kind") != "policy":
        raise ValueError(
            f"policy.json の kind は policy にしてください: {spec.get('kind')!r}"
        )

    # 出力の読み方と行動の空間の組み合わせ, 空間と守り
    obs_space = spec.get("observation_space")
    act_space = spec.get("action_space")
    check_space(obs_space)
    check_space(act_space)
    expected = "categorical" if act_space["type"] == "discrete" else (
        "deterministic"
    )
    if spec.get("head") != expected:
        raise ValueError(
            f"{act_space['type']} の行動の head は {expected} にしてください: "
            f"{spec.get('head')!r}"
        )
    check_guards(spec.get("guards"), obs_space, act_space)

    # 重みは一式の中の決まったファイルだけで, 中身の sha256 を持つこと
    if spec.get("weights") != WEIGHTS_FILE:
        raise ValueError(
            f"policy.json の weights は {WEIGHTS_FILE} にしてください: "
            f"{spec.get('weights')!r}"
        )
    digest = spec.get("weights_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or not all(
        char in "0123456789abcdef" for char in digest
    ):
        raise ValueError(
            "policy.json の weights_sha256 は 16 進の小文字 64 文字に"
            f"してください: {digest!r}"
        )


def reject_constant(name: str) -> Any:
    """
    JSON の外の値 (NaN, Infinity) を読んだら誤りにする関数

    Parameters
    ----------
    name: str
        読んだ値の名前

    Returns
    ----------
    value: Any
        返らない

    Raises
    ----------
    ValueError
        常に (守りの範囲を黙って外さないため)
    """
    # 守りの範囲を黙って外さないための停止
    raise ValueError(f"policy.json に {name} は書けません (制限なしは null)")


def read_policy_spec(path: Union[str, Path]) -> dict[str, Any]:
    """
    方策の一式のディレクトリから policy.json を読んで確かめる関数

    Parameters
    ----------
    path: Union[str, Path]
        方策の一式のディレクトリ

    Returns
    ----------
    spec: dict[str, Any]
        policy.json の中身

    Raises
    ----------
    ValueError
        policy.json が一式の外を指す場合と, 版か形が違う場合
    """
    # 一式の外を指すリンクは読まず, NaN と Infinity は JSON の外なので
    # 拒んで読み, 版と形を確認
    spec = json.loads(
        inside(Path(path), SPEC_FILE).read_text(encoding="utf-8"),
        parse_constant=reject_constant
    )
    check_policy_spec(spec)

    return spec


def check_observation(spec: dict, observation: Any) -> list[str]:
    """
    観測が止める条件に当たるかを確かめる関数

    Parameters
    ----------
    spec: dict
        policy.json の中身
    observation: Any
        環境の観測

    Returns
    ----------
    reasons: list[str]
        当たった条件, 当たらなければ空
    """
    # 数の配列にできること, 形と有限であることの確認
    # 形は空間と同じか, 同じ大きさの平らな配列だけ
    obs_space = spec["observation_space"]
    size = space_size(obs_space)
    try:
        raw = np.asarray(observation)
        obs = raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        # 桁の大きな整数は float にできず OverflowError
        reasons = [f"観測を数の配列にできない ({exc})"]
        return reasons
    if raw.dtype.kind not in NUMERIC_KINDS:
        reasons = [f"観測が数の配列でない ({raw.dtype})"]
        return reasons
    if obs.shape not in (tuple(obs_space["shape"]), (size,)):
        reasons = [
            f"観測の形 {obs.shape} が {tuple(obs_space['shape'])} と違う"
        ]
        return reasons

    # モデルには float32 で渡すので, 丸めた後も有限であることを確認
    # (1e39 は float64 では有限でも float32 では無限大)
    obs = obs.reshape(-1)
    with np.errstate(over="ignore"):
        obs32 = obs.astype(np.float32)
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(obs32)):
        reasons = ["観測に有限でない値がある (float32 での丸めも含む)"]
        return reasons

    # 止める条件の範囲の外の要素 (範囲は元の値で比べる)
    guards = spec["guards"]
    low = bounds_from_json(
        guards.get("observation_low"), size, -math.inf, "observation_low"
    )
    high = bounds_from_json(
        guards.get("observation_high"), size, math.inf, "observation_high"
    )
    outside = np.nonzero((obs < low) | (obs > high))[0]
    reasons = [
        f"観測の要素 {int(index)} が範囲の外 ({obs[index]:.6g})"
        for index in outside
    ]

    return reasons


def to_action(spec: dict, value: Any) -> Union[int, np.ndarray]:
    """
    行動の値を空間の型にする関数

    Parameters
    ----------
    spec: dict
        policy.json の中身
    value: Any
        行動の値

    Returns
    ----------
    action: Union[int, np.ndarray]
        discrete なら整数, box なら shape の形の float32 の配列
    """
    # 空間の種類ごとの型, box は float32 にしても範囲の中に収める
    act_space = spec["action_space"]
    if act_space["type"] == "discrete":
        action = int(value)
        return action
    low, high = action_limits(spec["guards"], act_space)
    inside_values = float32_inside(
        np.asarray(value, dtype=np.float64), low, high
    )
    if inside_values is None:
        raise ValueError(f"行動 {value} は float32 で範囲の中に表せません")
    action = inside_values.reshape(act_space["shape"])

    return action


def fallback_result(spec: dict, reasons: list[str]) -> dict[str, Any]:
    """
    外れたときの動きの結果を作る関数

    Parameters
    ----------
    spec: dict
        policy.json の中身
    reasons: list[str]
        外れた理由

    Returns
    ----------
    result: dict[str, Any]
        action (stop と handover は None), status (stop, handover,
        fallback), reasons
    """
    # 止める, 元の制御に戻す, 決めた行動にする, のどれか
    fallback = spec["guards"]["fallback"]
    action = None
    status = fallback["kind"]
    if fallback["kind"] == "action":
        status = "fallback"
        action = to_action(spec, fallback["action"])
    result = {"action": action, "status": status, "reasons": reasons}

    return result


def guard_discrete(spec: dict, scores: np.ndarray) -> dict[str, Any]:
    """
    discrete の行動を, 選んでよい行動の中の最もスコアの高いものにする関数

    Parameters
    ----------
    spec: dict
        policy.json の中身
    scores: np.ndarray
        行動ごとのスコア

    Returns
    ----------
    result: dict[str, Any]
        action, status, reasons
    """
    # スコアの数の確認
    act_space = spec["action_space"]
    if scores.size != act_space["n"]:
        result = fallback_result(
            spec,
            [f"モデルの出力の大きさ {scores.size} が {act_space['n']} と違う"]
        )
        return result

    # 選んでよい行動以外を除いて最もスコアの高い行動
    start = act_space["start"]
    best = int(np.argmax(scores)) + start
    allowed = spec["guards"].get("allowed_actions")
    if allowed is None or best in allowed:
        result = {"action": best, "status": "ok", "reasons": []}
        return result
    masked = np.full(scores.shape, -math.inf)
    indices = [action - start for action in allowed]
    masked[indices] = scores[indices]
    action = int(np.argmax(masked)) + start
    result = {
        "action": action, "status": "clipped",
        "reasons": [f"行動 {best} は選んでよい行動でない"]
    }

    return result


def guard_box(spec: dict, values: np.ndarray) -> dict[str, Any]:
    """
    box の行動を範囲の中に clip する関数

    Parameters
    ----------
    spec: dict
        policy.json の中身
    values: np.ndarray
        モデルが出した行動 (平均)

    Returns
    ----------
    result: dict[str, Any]
        action, status, reasons
    """
    # 大きさの確認
    act_space = spec["action_space"]
    low, high = action_limits(spec["guards"], act_space)
    if values.size != low.size:
        result = fallback_result(
            spec, [f"モデルの出力の大きさ {values.size} が {low.size} と違う"]
        )
        return result

    # 範囲の中に clip し, float32 にしても有限で範囲の中であることを確認
    clipped = np.clip(values, low, high)
    inside_values = float32_inside(clipped, low, high)
    if inside_values is None:
        result = fallback_result(
            spec, ["clip した行動が float32 で有限な範囲の中に表せない"]
        )
        return result
    changed = np.nonzero(clipped != values)[0]
    reasons = [
        f"行動の要素 {int(index)} を {values[index]:.6g} から "
        f"{clipped[index]:.6g} に clip"
        for index in changed
    ]
    result = {
        "action": inside_values.reshape(act_space["shape"]),
        "status": "clipped" if reasons else "ok",
        "reasons": reasons
    }

    return result


def apply_guards(
    spec: dict, observation: Any, output: Any
) -> dict[str, Any]:
    """
    観測とモデルの出力に守りを当てて, 行動を決める関数 (純粋な関数)

    Parameters
    ----------
    spec: dict
        policy.json の中身 (read_policy_spec で読んだもの)
    observation: Any
        環境の観測
    output: Any
        モデルの出力 (discrete なら行動ごとのスコア, box なら行動)

    Returns
    ----------
    result: dict[str, Any]
        action (行動. 止めるか元の制御に戻す場合は None), status (ok,
        clipped, fallback, stop, handover), reasons (守りが働いた理由)
    """
    # 止める条件に当たった場合は外れたときの動き
    reasons = check_observation(spec, observation)
    if reasons:
        result = fallback_result(spec, reasons)
        return result

    # 出力が数の配列でないか, 有限でない場合も外れたときの動き
    try:
        values = np.asarray(output, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        result = fallback_result(spec, [f"モデルの出力を数にできない ({exc})"])
        return result
    if not np.all(np.isfinite(values)):
        result = fallback_result(spec, ["モデルの出力に有限でない値がある"])
        return result

    # 空間の種類ごとに, 行動の範囲を当てて決める
    if spec["action_space"]["type"] == "discrete":
        result = guard_discrete(spec, values)
    else:
        result = guard_box(spec, values)

    return result


class GuardedPolicy:
    def __init__(
        self, model: nn.Module, spec: dict,
        unused_keys: Optional[list[str]] = None
    ):
        # cpu の推論のモードのモデル, policy.json の中身, 一式の重みのうち
        # モデルに無く読まなかったキー (log_std, 価値の関数など) を保存
        self.model = model.cpu().eval()
        self.spec = spec
        self.unused_keys = list(unused_keys or [])

    def act(self, observation: Any) -> dict[str, Any]:
        """
        観測から, 守りを当てた行動を決める関数

        Parameters
        ----------
        observation: Any
            環境の観測

        Returns
        ----------
        result: dict[str, Any]
            action, status (ok, clipped, fallback, stop, handover), reasons
        """
        # 止める条件に当たる場合はモデルを動かさずに外れたときの動き
        reasons = check_observation(self.spec, observation)
        if reasons:
            result = fallback_result(self.spec, reasons)
            return result

        # 勾配を記録せずにモデルを動かし, forward の出力に守りを適用.
        # モデルの誤りも外れたときの動き
        shape = self.spec["observation_space"]["shape"]
        obs = torch.as_tensor(
            np.asarray(observation, dtype=np.float32).reshape(shape)
        )
        try:
            with torch.no_grad():
                output = self.model(obs.unsqueeze(0))[0]
        except Exception as exc:
            result = fallback_result(
                self.spec, [f"モデルのエラー ({type(exc).__name__}: {exc})"]
            )
            return result
        result = apply_guards(
            self.spec, observation, output.detach().cpu().numpy()
        )

        return result


def file_sha256(path: Path) -> str:
    """
    ファイルの中身の sha256 を求める関数

    Parameters
    ----------
    path: Path
        ファイルのパス

    Returns
    ----------
    digest: str
        16 進の小文字 64 文字
    """
    # 大きな重みでも少しずつ読んで計算
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()

    return digest


def policy_model(directory: Path, spec: dict) -> nn.Module:
    """
    一式に写したモデルの定義 (metadata.model.spec) からモデルを組み立てる関数

    Parameters
    ----------
    directory: Path
        方策の一式のディレクトリ
    spec: dict
        policy.json の中身

    Returns
    ----------
    model: nn.Module
        組み立てたモデル (重みはまだ読まない)

    Raises
    ----------
    ValueError
        定義の場所が無い, 一式の外を指す, ファイルでない, 入力か出力が
        1 つでない場合
    """
    # 組み立て方は一式に写したモデルの定義だけ (クラスの名前からは
    # コードを読み込まない)
    metadata = spec.get("metadata")
    how = metadata.get("model") if isinstance(metadata, dict) else None
    relative = how.get("spec") if isinstance(how, dict) else None
    if not isinstance(relative, str) or not relative:
        raise ValueError(
            "policy.json の metadata.model.spec (一式に写したモデルの定義の"
            "場所) がありません. 書き出したときと同じ組み立てのモデルを "
            "model に渡してください (クラスの名前からはコードを読み込み"
            "ません)"
        )
    location = inside(directory, relative)
    if not location.is_file():
        raise ValueError(
            f"metadata.model.spec の {relative} はファイルではありません"
        )
    definition = load_spec(location)

    # 観測の 1 つの入力から行動の 1 つの出力
    inputs = definition.get("inputs") or {}
    outputs = definition.get("outputs") or {}
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"方策のモデルの定義は入力と出力を 1 つずつにしてください "
            f"(入力 {len(inputs)} つ, 出力 {len(outputs)} つ)"
        )
    check_units(definition)
    model = build_model(definition, seed=0, load_weights=False)

    return model


def load_policy(
    path: Union[str, Path], model: Optional[nn.Module] = None
) -> GuardedPolicy:
    """
    方策の一式を読み, 守りを当てて行動を返す方策を作る関数

    実機での実行はしない. 返す方策は, 観測から行動と守りの結果を返すだけで,
    環境や機械には触れない

    Parameters
    ----------
    path: Union[str, Path]
        方策の一式のディレクトリ
    model: Optional[nn.Module] = None
        書き出したときと同じ組み立てのモデル (重みは一式から読む), None の
        場合は一式に写したモデルの定義 (metadata.model.spec) から組み立てる

    Returns
    ----------
    policy: GuardedPolicy
        act(observation) で行動を返す方策 (cpu で動かす). unused_keys に
        一式の重みのうち読まなかったキー

    Raises
    ----------
    ValueError
        一式の形が違う, 一式の外を指す, 重みの sha256 が違う, 重みが
        モデルと合わない (足りないか形が違う) 場合
    """
    # 版と形と, 重みのファイルの sha256 を確かめてから読み込み
    directory = Path(path)
    spec = read_policy_spec(directory)
    weights = inside(directory, spec["weights"])
    if file_sha256(weights) != spec["weights_sha256"]:
        raise ValueError(
            f"{spec['weights']} の sha256 が policy.json と違います "
            "(書き出した後に重みが変わっています)"
        )
    state = torch.load(weights, map_location="cpu", weights_only=True)

    # 重みはモデルの state_dict 全体なので, 行動に使わない重み (log_std,
    # 価値の関数など) は余ってよく, モデルの重みが足りなければ誤り
    if model is None:
        model = policy_model(directory, spec)
    model = model.cpu()
    try:
        loaded = model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        raise ValueError(
            f"{spec['weights']} の重みの形がモデルと合いません ({error}). "
            "書き出したときと同じ組み立てのモデルを使ってください"
        ) from error
    if loaded.missing_keys:
        raise ValueError(
            f"{spec['weights']} にモデルの重み "
            f"{', '.join(loaded.missing_keys)} がありません. 書き出した"
            "ときと同じ組み立てのモデルを使ってください"
        )
    policy = GuardedPolicy(model, spec, sorted(loaded.unexpected_keys))

    return policy
