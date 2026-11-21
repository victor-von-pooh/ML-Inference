"""
テストの方策の一式と, ML-Found で求めた期待の結果を作るスクリプト

ML-Found (と ML-ModelControl) の入った環境で, このディレクトリを今の
ディレクトリにせずに `python tests/policies/make_policies.py` で動かす.
テストはこのスクリプトを動かさず, 書き出したファイルだけを読む
(ML-Inference は ML-Found のコードに頼らないため)
"""
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ml_found.rl import apply_guards, export_policy, load_policy, make_guards
from ml_modelcontrol import build_model, dump_spec
from torch import nn

# 書き出す場所
HERE = Path(__file__).resolve().parent

# float32 の最大の値 (CartPole の速度の範囲)
FLOAT32_MAX = float(np.finfo(np.float32).max)


def mlp_spec(obs_dim: int, outputs: int) -> dict:
    """
    観測から全結合 2 層のモデルの定義を返す関数

    Parameters
    ----------
    obs_dim: int
        観測の次元数
    outputs: int
        出力の幅

    Returns
    ----------
    spec: dict
        ML-ModelControl の定義
    """
    # 全結合, tanh, 全結合
    spec = {
        "format_version": 1,
        "name": "policy",
        "inputs": {"obs": {"shape": [1, obs_dim]}},
        "nodes": {
            "hidden": {
                "unit": "nn.Linear",
                "params": {"in_features": obs_dim, "out_features": 8},
                "inputs": "obs"
            },
            "act": {"unit": "nn.Tanh", "inputs": "hidden"},
            "head": {
                "unit": "nn.Linear",
                "params": {"in_features": 8, "out_features": outputs},
                "inputs": "act"
            }
        },
        "outputs": {"out": "head"}
    }

    return spec


def encode(value: Any) -> Any:
    """
    観測と出力を, 有限でない値も書ける JSON の形にする関数

    Parameters
    ----------
    value: Any
        観測か出力

    Returns
    ----------
    item: Any
        array (数の並び. 有限でない値は文字) か raw (そのまま)
    """
    # 数の配列は形と値, それ以外はそのまま
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        item = {"kind": "raw", "value": value}
        return item
    values = [
        float(number) if math.isfinite(number) else str(float(number))
        for number in array.astype(np.float64).reshape(-1)
    ]
    item = {"kind": "array", "shape": list(array.shape), "values": values}

    return item


def encode_result(result: dict) -> dict:
    """
    守りの結果を JSON の形にする関数

    Parameters
    ----------
    result: dict
        action, status, reasons

    Returns
    ----------
    item: dict
        action を数のリスト (box) か整数か None にしたもの
    """
    # box の行動は float32 の値を float64 で正確に書く
    action = result["action"]
    if isinstance(action, np.ndarray):
        action = {
            "shape": list(action.shape), "dtype": str(action.dtype),
            "values": [float(number) for number in action.reshape(-1)]
        }
    item = {
        "action": action, "status": result["status"],
        "reasons": result["reasons"]
    }

    return item


def write_policy(
    name: str, spec: dict, observation_space: dict, action_space: dict,
    guards: dict, env_id: str, extra: bool, seed: int
) -> tuple[Path, nn.Module]:
    """
    定義から組み立てたモデルを方策の一式に書き出す関数

    Parameters
    ----------
    name: str
        一式のディレクトリの名前
    spec: dict
        モデルの定義
    observation_space: dict
        観測の空間
    action_space: dict
        行動の空間
    guards: dict
        守り
    env_id: str
        環境の名前 (metadata に書く)
    extra: bool
        行動に使わない重み (log_std と価値の関数) を足すか
    seed: int
        重みの初期値の乱数の種

    Returns
    ----------
    directory: Path
        書き出した一式
    model: nn.Module
        書き出したモデル
    """
    # 定義から組み立て, 連続の行動の方策のように行動に使わない重みを足す
    model = build_model(spec, seed=seed, load_weights=False)
    if extra:
        torch.manual_seed(seed)
        model.log_std = nn.Parameter(torch.full((1,), -0.5))
        model.value = nn.Linear(spec["inputs"]["obs"]["shape"][1], 1)

    # 定義を一式に写し, その場所を metadata に
    directory = HERE / name
    directory.mkdir(parents=True, exist_ok=True)
    dump_spec(spec, directory / "model_spec.yaml")
    export_policy(
        directory, model, observation_space, action_space, guards,
        metadata={
            "env_id": env_id, "algorithm": "REINFORCE",
            "model": {"spec": "model_spec.yaml"}
        }
    )

    return directory, model


def cartpole() -> list[dict]:
    """
    CartPole の方策の一式を書き出し, 観測ごとの期待の結果を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    cases: list[dict]
        観測と, ML-Found の load_policy(...).act の結果
    """
    # CartPole-v1 の空間と, 位置と角度を外れたら元の制御に戻す守り
    angle = float(np.float32(12 * 2 * math.pi / 360 * 2))
    observation_space = {
        "type": "box", "shape": [4],
        "low": [-4.8, -FLOAT32_MAX, -angle, -FLOAT32_MAX],
        "high": [4.8, FLOAT32_MAX, angle, FLOAT32_MAX]
    }
    action_space = {"type": "discrete", "n": 2, "start": 0}
    guards = make_guards(
        observation_space, action_space,
        observation_low=[-2.4, None, -0.21, None],
        observation_high=[2.4, None, 0.21, None], fallback="handover"
    )
    directory, model = write_policy(
        "cartpole", mlp_spec(4, 2), observation_space, action_space,
        guards, "CartPole-v1", False, 3
    )

    # 範囲の中の観測と, 守りが働く観測
    rng = np.random.default_rng(0)
    observations: list[Any] = [
        rng.uniform([-2, -2, -0.2, -2], [2, 2, 0.2, 2]).tolist()
        for _ in range(40)
    ]
    observations += [
        [2.5, 0.0, 0.0, 0.0], [0.0, 0.0, -0.3, 0.0], [math.nan, 0, 0, 0],
        [0.0, math.inf, 0.0, 0.0], [1e39, 0, 0, 0], [0.0, 0.0, 0.0],
        [[0.0, 0.0], [0.0, 0.0]], [True, False, True, False]
    ]
    policy = load_policy(directory, model)
    cases = [
        {
            "observation": encode(observation),
            "result": encode_result(policy.act(observation))
        }
        for observation in observations
    ]

    return cases


def box() -> list[dict]:
    """
    連続の行動の方策の一式を書き出し, 観測ごとの期待の結果を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    cases: list[dict]
        観測と, ML-Found の load_policy(...).act の結果
    """
    # 振り子の形の空間と, 空間より狭い行動の範囲, 外れたら 0 の行動
    observation_space = {
        "type": "box", "shape": [3],
        "low": [-1.0, -1.0, -8.0], "high": [1.0, 1.0, 8.0]
    }
    action_space = {
        "type": "box", "shape": [1], "low": [-2.0], "high": [2.0]
    }
    guards = make_guards(
        observation_space, action_space, action_low=[-0.1],
        action_high=[0.1], observation_low=[None, None, -6.0],
        observation_high=[None, None, 6.0], fallback="action"
    )
    directory, model = write_policy(
        "box", mlp_spec(3, 1), observation_space, action_space, guards,
        "Pendulum-v1", True, 1
    )

    # 範囲の中の観測と, 止める条件に当たる観測
    rng = np.random.default_rng(1)
    observations: list[Any] = [
        rng.uniform([-1, -1, -5], [1, 1, 5]).tolist() for _ in range(40)
    ]
    observations += [[0.0, 0.0, 7.0], [0.0, math.nan, 0.0], [0.0, 0.0]]
    policy = load_policy(directory, model)
    cases = [
        {
            "observation": encode(observation),
            "result": encode_result(policy.act(observation))
        }
        for observation in observations
    ]

    return cases


def guard_cases() -> list[dict]:
    """
    いろいろな守りと出力で, ML-Found の apply_guards の結果を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    cases: list[dict]
        policy.json の中身, 観測, 出力, 結果
    """
    # 選んでよい行動と決めた行動の discrete, 狭い範囲と止める box
    # (数にできない出力は誤りの文が NumPy の版で変わるので除く)
    discrete = {"type": "discrete", "n": 3, "start": 1}
    obs2 = {
        "type": "box", "shape": [2], "low": [-5.0, -5.0], "high": [5.0, 5.0]
    }
    wide = {"type": "box", "shape": [2], "low": None, "high": None}
    specs = {
        "allowed": (obs2, discrete, make_guards(
            obs2, discrete, allowed_actions=[1, 3],
            observation_high=[1.0, None], fallback="action",
            fallback_action=3
        )),
        "stop": (obs2, wide, make_guards(
            obs2, wide, action_low=[-0.1, None], action_high=[0.1, 1e38],
            observation_low=[-1.0, -1.0], fallback="stop"
        )),
        "zero": (obs2, wide, make_guards(
            obs2, wide, action_low=[-1.0, -1.0], action_high=[1.0, 1.0],
            fallback="action"
        ))
    }
    outputs = {
        "allowed": [
            [0.1, 0.9, 0.2], [0.9, 0.1, 0.2], [0.1, 0.2, 0.9],
            [0.5, 0.5, 0.1], [0.1, 0.2], [math.nan, 0.0, 0.0]
        ],
        "stop": [
            [0.05, 0.5], [0.3, -0.2], [-3.0, 1e39], [1e300, 0.0],
            [0.0, 0.0, 0.0], [math.inf, 0.0]
        ],
        "zero": [[0.5, -0.5], [2.0, -3.0], [[1.0, 1.0]], [0.0]]
    }
    observations = [[0.0, 0.0], [2.0, 0.0], [0.0, -2.0], [math.nan, 0.0]]
    cases = []
    for name, (obs_space, act_space, guards) in specs.items():
        spec = {
            "format_version": 1, "kind": "policy",
            "head": "categorical" if act_space["type"] == "discrete"
            else "deterministic",
            "observation_space": obs_space, "action_space": act_space,
            "guards": guards, "weights": "policy.pt",
            "weights_sha256": "0" * 64, "metadata": {}
        }
        for observation in observations:
            for output in outputs[name]:
                result = apply_guards(spec, observation, output)
                cases.append({
                    "spec": spec, "observation": encode(observation),
                    "output": encode(output),
                    "result": encode_result(result)
                })

    return cases


def main() -> None:
    """
    一式と期待の結果を書き出す関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 一式ごとの観測の結果と, 守りだけの結果
    expected = {
        "cartpole": cartpole(), "box": box(), "guards": guard_cases()
    }
    text = json.dumps(expected, ensure_ascii=False, indent=1)
    (HERE / "expected.json").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
