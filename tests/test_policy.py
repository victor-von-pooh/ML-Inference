import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest
import torch
from ml_modelcontrol import build_model, load_spec
from torch import nn

from ml_inference import (
    POLICY_FORMAT_VERSION, apply_guards, load_policy, read_policy_spec
)

# ML-Found で書き出した一式と期待の結果 (policies/make_policies.py)
POLICIES = Path(__file__).resolve().parent / "policies"
EXPECTED = json.loads(
    (POLICIES / "expected.json").read_text(encoding="utf-8")
)


def decode(item: dict) -> Any:
    """
    期待の結果のファイルの観測と出力を元の値に戻す関数

    Parameters
    ----------
    item: dict
        array (数の並び. 有限でない値は文字) か raw (そのまま)

    Returns
    ----------
    value: Any
        元の観測か出力 (数の並びは形の付いた入れ子のリスト)
    """
    # 有限でない値の文字を数に戻して形を付ける
    if item["kind"] == "raw":
        value = item["value"]
        return value
    values = np.array([float(number) for number in item["values"]])
    value = values.reshape(item["shape"]).tolist()

    return value


def assert_same(
    result: dict, expected: dict, exact: bool = True
) -> None:
    """
    守りの結果が ML-Found の結果と同じことを確かめる関数

    Parameters
    ----------
    result: dict
        ML-Inference の結果
    expected: dict
        ML-Found の結果 (期待の結果のファイルの形)
    exact: bool = True
        box の行動の値を完全に比べるか. モデルを動かした結果は計算の
        仕組み (OS や CPU) で最後の桁が変わるので, 相対の 1e-5 と絶対の 1e-6 で比べる

    Returns
    ----------
    None
    """
    # 状態と理由と行動 (box は float32 の形と値) の一致
    assert result["status"] == expected["status"]
    assert result["reasons"] == expected["reasons"]
    action = expected["action"]
    if isinstance(action, dict):
        assert isinstance(result["action"], np.ndarray)
        assert result["action"].dtype == np.float32
        assert list(result["action"].shape) == action["shape"]
        values = result["action"].astype(np.float64).reshape(-1)
        if exact:
            assert values.tolist() == action["values"]
            return
        assert np.allclose(values, action["values"], rtol=1e-5, atol=1e-6)
        return
    assert result["action"] == action
    assert type(result["action"]) is type(action)


def copy_policy(name: str, tmp_path: Path) -> Path:
    """
    テストの一式を一時の場所に写す関数

    Parameters
    ----------
    name: str
        一式の名前
    tmp_path: Path
        一時の場所

    Returns
    ----------
    directory: Path
        写した一式
    """
    # 一式のディレクトリごと
    directory = tmp_path / name
    shutil.copytree(POLICIES / name, directory)

    return directory


def rewrite_spec(directory: Path, change: dict) -> None:
    """
    一式の policy.json の一番上の項目を書き換える関数

    Parameters
    ----------
    directory: Path
        一式のディレクトリ
    change: dict
        書き換える項目と値

    Returns
    ----------
    None
    """
    # 読んで書き換えて書き戻し
    path = directory / "policy.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec.update(change)
    path.write_text(json.dumps(spec), encoding="utf-8")


@pytest.mark.parametrize("name", ["cartpole", "box"])
def test_act_matches_ml_found(name: str) -> None:
    """
    定義から組み立てた方策の行動と守りの結果が ML-Found と同じことを
    確かめる関数

    Parameters
    ----------
    name: str
        一式の名前

    Returns
    ----------
    None
    """
    # 観測ごとの結果の比べ
    policy = load_policy(POLICIES / name)
    for case in EXPECTED[name]:
        result = policy.act(decode(case["observation"]))
        assert_same(result, case["result"], exact=False)


def test_cartpole_statuses() -> None:
    """
    CartPole の結果に両方の行動と元の制御に戻す結果が入ることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 比べが意味を持つように, 期待の結果の中身の確認
    results = [case["result"] for case in EXPECTED["cartpole"]]
    assert {result["action"] for result in results} == {0, 1, None}
    assert {result["status"] for result in results} == {"ok", "handover"}


def test_box_ignores_unused_weights() -> None:
    """
    行動に使わない重み (log_std, 価値の関数) を含む一式を読めることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 一式の重みには余る重みがあり, 結果には clip と決めた行動が入る
    state = torch.load(POLICIES / "box" / "policy.pt", weights_only=True)
    assert "log_std" in state and "value.weight" in state
    statuses = {case["result"]["status"] for case in EXPECTED["box"]}
    assert statuses == {"ok", "clipped", "fallback"}
    policy = load_policy(POLICIES / "box")
    assert "log_std" not in policy.model.state_dict()
    assert policy.unused_keys == ["log_std", "value.bias", "value.weight"]
    assert load_policy(POLICIES / "cartpole").unused_keys == []


def test_apply_guards_matches_ml_found() -> None:
    """
    いろいろな守りと出力で apply_guards が ML-Found と同じことを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 守りだけの結果の比べ
    for case in EXPECTED["guards"]:
        result = apply_guards(
            case["spec"], decode(case["observation"]),
            decode(case["output"])
        )
        assert_same(result, case["result"])


def test_passed_model(tmp_path: Path) -> None:
    """
    渡したモデルで読み, 定義の場所が無くても動くことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時の場所

    Returns
    ----------
    None
    """
    # 定義の場所を消した一式に, 同じ組み立てのモデルを渡す
    directory = copy_policy("cartpole", tmp_path)
    model = build_model(load_spec(directory / "model_spec.yaml"))
    rewrite_spec(directory, {"metadata": {"env_id": "CartPole-v1"}})
    policy = load_policy(directory, model)
    for case in EXPECTED["cartpole"]:
        assert_same(policy.act(decode(case["observation"])), case["result"])

    # モデルを渡さなければ, 名前からコードを読まずに止まる
    with pytest.raises(ValueError, match="metadata.model.spec"):
        load_policy(directory)


def test_model_error_falls_back(tmp_path: Path) -> None:
    """
    モデルの誤りと数にできない観測が外れたときの動きになることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時の場所

    Returns
    ----------
    None
    """
    class Broken(nn.Module):
        def __init__(self):
            # 親クラスのコンストラクタを呼び出す
            super(Broken, self).__init__()

            # 一式の重みと同じ名前の層
            self.nodes = nn.ModuleDict({
                "hidden": nn.Linear(4, 8), "head": nn.Linear(8, 2)
            })

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            常に誤りを出す関数

            Parameters
            ----------
            x: torch.Tensor
                観測

            Returns
            ----------
            out: torch.Tensor
                返らない
            """
            raise RuntimeError("壊れた")

    # モデルの誤りと数にできない観測は元の制御に戻す
    policy = load_policy(POLICIES / "cartpole", Broken())
    result = policy.act([0.0, 0.0, 0.0, 0.0])
    assert result == {
        "action": None, "status": "handover",
        "reasons": ["モデルの計算の失敗 (RuntimeError: 壊れた)"]
    }
    assert policy.act("abc")["status"] == "handover"


def test_rejects_changed_weights(tmp_path: Path) -> None:
    """
    sha256 の違う重みと, モデルの重みが足りない一式を読まないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時の場所

    Returns
    ----------
    None
    """
    # 書き出した後に変わった重み
    directory = copy_policy("cartpole", tmp_path)
    state = torch.load(directory / "policy.pt", weights_only=True)
    state["nodes.head.bias"] = state["nodes.head.bias"] + 1
    torch.save(state, directory / "policy.pt")
    with pytest.raises(ValueError, match="sha256"):
        load_policy(directory)

    # 一式の重みに無い層を持つモデル
    model = build_model(load_spec(POLICIES / "cartpole" / "model_spec.yaml"))
    model.extra = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="extra.weight"):
        load_policy(POLICIES / "cartpole", model)

    # 一式の重みと形の違う層を持つモデル
    model = build_model(load_spec(POLICIES / "box" / "model_spec.yaml"))
    with pytest.raises(ValueError, match="形がモデルと合いません"):
        load_policy(POLICIES / "cartpole", model)


@pytest.mark.parametrize("name", ["policy.json", "policy.pt"])
def test_rejects_links_outside(tmp_path: Path, name: str) -> None:
    """
    一式の外を指すリンクの policy.json と policy.pt を読まないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時の場所
    name: str
        リンクにするファイルの名前

    Returns
    ----------
    None
    """
    # ファイルを一式の外に移し, 一式の中にリンクを置く
    directory = copy_policy("cartpole", tmp_path)
    outside = tmp_path / name
    shutil.move(directory / name, outside)
    (directory / name).symlink_to(outside)
    with pytest.raises(ValueError, match="一式の外"):
        load_policy(directory)


@pytest.mark.parametrize("change, message", [
    ({"format_version": POLICY_FORMAT_VERSION + 1}, "新しい"),
    ({"format_version": None}, "format_version"),
    ({"kind": "bundle"}, "kind"),
    ({"head": "categorical"}, "head"),
    ({"weights": "../policy.pt"}, "weights"),
    ({"weights_sha256": "ABC"}, "weights_sha256"),
    (
        {"metadata": {"model": {"spec": "../cartpole/model_spec.yaml"}}},
        "一式の外"
    ),
    ({"metadata": {"model": {"spec": "."}}}, "ファイルではありません")
])
def test_rejects_bad_spec(tmp_path: Path, change: dict, message: str) -> None:
    """
    形の違う policy.json を読まないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時の場所
    change: dict
        書き換える項目と値
    message: str
        誤りの文に入る言葉

    Returns
    ----------
    None
    """
    # box の一式を書き換えて読む
    directory = copy_policy("box", tmp_path)
    rewrite_spec(directory, change)
    with pytest.raises(ValueError, match=message):
        load_policy(directory)


@pytest.mark.parametrize("change, message", [
    ({"observation_hihg": None}, "知らない項目"),
    ({"action_low": ["0.1"]}, "有限の数か null"),
    ({"allowed_actions": [0]}, "allowed_actions"),
    ({"fallback": {"kind": "action", "action": [5.0]}}, "範囲の中"),
    ({"fallback": {"kind": "wait", "action": None}}, "fallback"),
    ({"fallback": {"kind": "stop", "extra": 1}}, "知らない項目")
])
def test_rejects_bad_guards(
    tmp_path: Path, change: dict, message: str
) -> None:
    """
    書き損じの守りを読まないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時の場所
    change: dict
        守りの書き換える項目と値
    message: str
        誤りの文に入る言葉

    Returns
    ----------
    None
    """
    # box の一式の守りを書き換えて読む
    directory = copy_policy("box", tmp_path)
    guards = read_policy_spec(directory)["guards"]
    guards.update(change)
    rewrite_spec(directory, {"guards": guards})
    with pytest.raises(ValueError, match=message):
        read_policy_spec(directory)


def test_rejects_nan_in_json(tmp_path: Path) -> None:
    """
    JSON の外の NaN を書いた policy.json を読まないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時の場所

    Returns
    ----------
    None
    """
    # 止める条件に NaN を書いた一式
    directory = copy_policy("box", tmp_path)
    path = directory / "policy.json"
    text = path.read_text(encoding="utf-8").replace("-6.0", "NaN", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="NaN"):
        read_policy_spec(directory)


def test_huge_integer_falls_back() -> None:
    """
    float にできない桁の大きな整数の観測と出力は, 誤りにせずに外れたときの
    動きにすることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 観測 (JSON の大きな整数のリテラルを読んだ値) と, apply_guards の出力
    huge = 10 ** 400
    policy = load_policy(POLICIES / "cartpole")
    result = policy.act([1, 1.0, 0, huge])
    assert result["action"] is None and result["status"] == "handover"
    assert result["reasons"][0].startswith("観測を数の配列にできない")
    spec = read_policy_spec(POLICIES / "cartpole")
    result = apply_guards(spec, [0.0, 0.0, 0.0, 0.0], [0.0, huge])
    assert result["status"] == "handover"
    assert result["reasons"][0].startswith("モデルの出力を数にできない")
