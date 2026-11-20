import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import torch
from ml_eda import Dataset, export_dataset, load_data, load_dataset
from ml_modelcontrol import build_model
from torch import nn, optim

from ml_inference.bundle import make_bundle

# 見本のモデルの学習の回数 (学習用の全ての行を 1 度に使う)
SAMPLE_STEPS = 300

# 見本のモデルの学習率と, 隠れ層と埋め込みの幅
SAMPLE_RATE = 0.02
SAMPLE_WIDTH = 16


def sample_spec(dataset: Dataset, outputs: int) -> dict[str, Any]:
    """
    見本のデータセットに合わせた小さなモデルの定義を返す関数

    Parameters
    ----------
    dataset: Dataset
        ML-EDA のデータセット (表か文)
    outputs: int
        出力の幅 (分類はクラスの数, 回帰と予測は 1)

    Returns
    ----------
    spec: dict[str, Any]
        ML-ModelControl の定義 (文は埋め込みの平均, 表は全結合 2 層)

    Raises
    ----------
    ValueError
        表と文でないデータセットの場合
    """
    # 文は番号の並び, 表は学習に使う列
    kind = dataset.info.get("kind")
    if kind == "text":
        length = int(dataset.info["splits"]["train"]["shape"][1])
        vocabulary = next(
            (len(step["state"]["vocabulary"])
             for step in dataset.recipe.get("item_steps", [])
             if step["op"] == "vocabulary"), None
        )
        if vocabulary is None:
            raise ValueError("文の見本は, 中身の手順に vocabulary が要ります")
        nodes: dict[str, Any] = {
            "bag": {"unit": "nn.EmbeddingBag",
                    "params": {"num_embeddings": vocabulary,
                               "embedding_dim": SAMPLE_WIDTH,
                               "mode": "mean", "padding_idx": 0},
                    "inputs": {"input": "ids"}},
            "head": {"unit": "nn.Linear",
                     "params": {"in_features": SAMPLE_WIDTH,
                                "out_features": outputs},
                     "inputs": "bag"},
        }
        inputs = {"ids": {"shape": [2, length], "dtype": "int64",
                          "fill": "ones"}}
    elif kind in ("table", "timeseries"):
        width = len(dataset.features)
        nodes = {
            "fc": {"unit": "nn.Linear",
                   "params": {"in_features": width,
                              "out_features": SAMPLE_WIDTH},
                   "inputs": "x"},
            "act": {"unit": "nn.ReLU", "inputs": "fc"},
            "head": {"unit": "nn.Linear",
                     "params": {"in_features": SAMPLE_WIDTH,
                                "out_features": outputs},
                     "inputs": "act"},
        }
        inputs = {"x": {"shape": [2, width]}}
    else:
        raise ValueError(
            f"見本の一式は表と文のデータだけで作れます (このデータは {kind})"
        )
    spec = {"format_version": 1, "name": "sample", "inputs": inputs,
            "nodes": nodes, "outputs": {"y": "head"}}

    return spec


def train_sample(
    spec: dict[str, Any], dataset: Dataset, steps: int = SAMPLE_STEPS,
    seed: int = 0
) -> tuple[nn.Module, Optional[dict[str, float]]]:
    """
    見本のモデルを学習用の分割で学習する関数

    Parameters
    ----------
    spec: dict[str, Any]
        モデルの定義
    dataset: Dataset
        ML-EDA のデータセット
    steps: int = SAMPLE_STEPS
        学習の回数 (学習用の全ての行を 1 度に使う)
    seed: int = 0
        重みの初期値の乱数の種

    Returns
    ----------
    result: tuple[nn.Module, Optional[dict[str, float]]]
        学習したモデルと, 回帰と予測で目的の値を標準化した中心と幅
        (一式の target_scale. 分類は None)
    """
    # 回帰と予測の目的の値の標準化 (一式の target_scale で戻す)
    model = build_model(spec, seed=seed)
    x, y = dataset.tensors("train")
    scale: Optional[dict[str, float]] = None
    if dataset.info["task"] == "classification":
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    else:
        scale = {"center": y.mean().item(), "scale": y.std().item()}
        y = ((y - scale["center"]) / scale["scale"]).unsqueeze(1)
        loss_fn = nn.MSELoss()

    # 全ての行を 1 度に使う学習
    optimizer = optim.Adam(model.parameters(), lr=SAMPLE_RATE)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    result = (model, scale)

    return result


def make_sample_bundle(
    path: Union[str, Path], source: Union[str, Path], recipe: dict,
    options: Optional[dict] = None, conclusion: Optional[dict] = None,
    decision: Optional[dict] = None, name: Optional[str] = None,
    steps: int = SAMPLE_STEPS, seed: int = 0, overwrite: bool = False
) -> dict:
    """
    表か文のデータから, 小さなモデルをその場で学習して見本の推論の一式を
    作る関数

    画面で操作を試すための一式で, 課題のモデルではない. データセットの
    書き出し (ML-EDA), モデルの定義と組み立て (ML-ModelControl), 学習,
    一式づくり (make_bundle) までを行う

    Parameters
    ----------
    path: Union[str, Path]
        作る一式のディレクトリ
    source: Union[str, Path]
        学習に使うデータのファイル (表か, 文の列を持つ表)
    recipe: dict
        ML-EDA のレシピ (target, task, steps, item_steps, split)
    options: Optional[dict] = None
        ML-EDA の読み込みの設定 (文の列 text_column など)
    conclusion: Optional[dict] = None
        結論の決め方の既定値
    decision: Optional[dict] = None
        意思決定の決め方の既定値
    name: Optional[str] = None
        一式の名前 (省略するとディレクトリの名前)
    steps: int = SAMPLE_STEPS
        学習の回数
    seed: int = 0
        重みの初期値の乱数の種
    overwrite: bool = False
        既にある一式を置き換えるか

    Returns
    ----------
    info: dict
        書き出した inference.yaml の中身

    Raises
    ----------
    ValueError
        表と文でないデータの場合と, make_bundle の誤り
    """
    # データセットと重みは一時の場所に書き, 一式に写す (一式は写した
    # データセットの説明とレシピと重みだけで読める)
    with tempfile.TemporaryDirectory() as work:
        place = Path(work)
        data = load_data(Path(source).expanduser(), **(options or {}))
        export_dataset(data, {"format_version": 1, **recipe},
                       place / "dataset")
        dataset = load_dataset(place / "dataset")
        classes = dataset.info.get("classes")
        spec = sample_spec(dataset, len(classes) if classes else 1)
        model, scale = train_sample(spec, dataset, steps, seed)
        torch.save(model.state_dict(), place / "weights.pt")
        info = make_bundle(
            path, spec, place / "dataset", place / "weights.pt",
            target_scale=scale, conclusion=conclusion, decision=decision,
            name=name, overwrite=overwrite
        )

    return info
