from dataclasses import dataclass, field
import datetime as dt
import json
from pathlib import Path
import shutil
from typing import Any, Optional, Union

import numpy as np
import torch
import yaml
from ml_eda import Dataset, load_dataset
from ml_eda.pipeline import DATASET_FORMAT_VERSION
from ml_modelcontrol import build_model, dump_spec, load_spec

from ml_inference.conclude import check_rules, plain

# 推論の一式の形の版
BUNDLE_FORMAT_VERSION = 1

# inference.yaml の一番上の項目
TOP_KEYS = (
    "format_version", "name", "created_at", "model", "inputs", "dataset",
    "run", "conclusion", "check", "notes"
)

# 出力の扱い方
OUTPUT_KINDS = ("logits", "probabilities", "value")

# 入力の中身の出どころ
INPUT_SOURCES = ("features", "items")

# 一式を作るときに確かめる見本の件数
SAMPLE_ROWS = 8

# 読める ML-Found の run.json の版
RUN_FORMAT_VERSION = 1

# 実験の記録と比べるときに推論する行の上限, 比べられないときの行の数,
# 1 回に推論する行の数, 合うとみなす差 (割合)
CHECK_ROWS = 50000
CHECK_SAMPLE_ROWS = 256
CHECK_BATCH = 256
CHECK_TOLERANCE = 0.02

# 実験の監視した指標の名前 (/ の後) から, 比べられる指標
METRIC_NAMES = {"acc": "acc", "accuracy": "acc", "mae": "mae"}


@dataclass
class Bundle:
    # 一式のディレクトリ
    path: Path
    # inference.yaml の中身
    info: dict
    # 前処理 (ML-EDA のデータセットの説明とレシピ. 分割の表は持たない)
    dataset: Dataset
    # モデルの定義
    spec: dict
    # 組み立てて重みを読んだモデル (使うときに作る)
    cache: dict = field(default_factory=dict)

    @property
    def task(self) -> str:
        """
        タスクの種類を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        task: str
            classification, regression, forecasting など
        """
        # データセットの説明の task
        task = str(self.dataset.info.get("task", ""))

        return task

    @property
    def classes(self) -> list[str]:
        """
        分類のクラスの名前を番号の順で返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        classes: list[str]
            クラスの名前 (分類でなければ空)
        """
        # データセットの説明の classes
        classes = [str(c) for c in self.dataset.info.get("classes") or []]

        return classes

    @property
    def kind(self) -> str:
        """
        データの種類を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        kind: str
            table, timeseries, image, text, pointcloud
        """
        # データセットの説明の kind
        kind = str(self.dataset.info.get("kind", "table"))

        return kind

    @property
    def output_kind(self) -> str:
        """
        モデルの出力の扱い方を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        kind: str
            logits, probabilities, value のどれか
        """
        # 書かれていなければ, 分類は logits, それ以外は値
        kind = self.info["model"].get("kind") or (
            "logits" if self.task == "classification" else "value"
        )

        return kind

    @property
    def weights_path(self) -> Path:
        """
        重みのファイルの場所を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        path: Path
            重みのファイル (一式からの相対パスは一式の中に解決)
        """
        # 絶対パスはそのまま, 相対パスは一式から
        path = Path(self.info["model"]["weights"]).expanduser()
        if not path.is_absolute():
            path = inside(self.path, str(self.info["model"]["weights"]))

        return path

    def input_plan(self) -> dict:
        """
        モデルの入力ごとの中身の出どころを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        plan: dict
            入力の名前ごとの source と prepend (定義の inputs の順)
        """
        # 書かれていない入力は, データの種類から選ぶ
        written = self.info.get("inputs") or {}
        default = "features" if self.kind in ("table", "timeseries") else (
            "items"
        )
        plan = {}
        for name in self.spec["inputs"]:
            item = written.get(name) or {}
            plan[name] = {
                "source": item.get("source", default),
                "prepend": [int(v) for v in item.get("prepend") or []],
            }

        return plan

    def model(self) -> Any:
        """
        定義からモデルを組み立て, 重みを読み込んで返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        model: nn.Module
            推論のモードのモデル (cpu)
        """
        # 2 回目からは作ったものを使う
        if "model" in self.cache:
            return self.cache["model"]

        # 定義から組み立て, 重みだけを読み込む (pickle の任意のコードは不可)
        model = build_model(self.spec, seed=0)
        state = torch.load(
            self.weights_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        model.eval()
        self.cache["model"] = model

        return model


def inside(root: Path, relative: str) -> Path:
    """
    一式からの相対パスを, 一式の外に出ないことを確かめて解決する関数

    Parameters
    ----------
    root: Path
        一式のディレクトリ
    relative: str
        一式からの相対パス

    Returns
    ----------
    path: Path
        解決したパス

    Raises
    ----------
    ValueError
        絶対パスか, 一式の外を指す場合
    """
    # .. や絶対パスで外に出るものは誤り
    base = root.resolve()
    path = (base / relative).resolve()
    if Path(relative).is_absolute() or (
        path != base and base not in path.parents
    ):
        raise ValueError(f"{relative} は一式の外を指しています")

    return path


def check_info(info: Any) -> list[str]:
    """
    inference.yaml の中身の形を確かめる関数

    Parameters
    ----------
    info: Any
        読み込んだ inference.yaml

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 一番上の形と必須の項目
    if not isinstance(info, dict):
        return ["inference.yaml の中身が辞書ではありません"]
    errors = []
    unknown = [k for k in info if k not in TOP_KEYS]
    if unknown:
        errors.append(f"知らない項目があります: {', '.join(map(str, unknown))}")
    version = info.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool):
        errors.append("format_version が整数ではありません")
    model = info.get("model")
    if not isinstance(model, dict) or not model.get("spec") or (
        not model.get("weights")
    ):
        errors.append("model に spec と weights が要ります")
    elif model.get("kind") not in (None, *OUTPUT_KINDS):
        errors.append(
            f"model.kind は {', '.join(OUTPUT_KINDS)} のどれかにしてください"
        )
    if isinstance(model, dict) and model.get("target_scale") is not None:
        scale = model["target_scale"]
        if not isinstance(scale, dict) or not all(
            isinstance(scale.get(k), (int, float))
            and not isinstance(scale.get(k), bool)
            for k in ("center", "scale")
        ):
            errors.append(
                "model.target_scale は center と scale の数にしてください"
            )
    if not info.get("dataset"):
        errors.append("dataset (データセットの説明の場所) が要ります")

    # 入力ごとの出どころ
    inputs = info.get("inputs") or {}
    if not isinstance(inputs, dict):
        errors.append("inputs は入力の名前ごとの辞書にしてください")
        inputs = {}
    for name, item in inputs.items():
        if not isinstance(item, dict):
            errors.append(f"inputs.{name} が辞書ではありません")
            continue
        if item.get("source", "features") not in INPUT_SOURCES:
            errors.append(
                f"inputs.{name}.source は {', '.join(INPUT_SOURCES)} の"
                "どちらかにしてください"
            )
        prepend = item.get("prepend") or []
        if not isinstance(prepend, list) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in prepend
        ):
            errors.append(f"inputs.{name}.prepend は整数の並びにしてください")

    return errors


def load_bundle(path: Union[str, Path]) -> Bundle:
    """
    推論の一式を読み込む関数

    Parameters
    ----------
    path: Union[str, Path]
        一式のディレクトリ (inference.yaml を持つ)

    Returns
    ----------
    bundle: Bundle
        一式 (モデルは使うときに組み立てる)

    Raises
    ----------
    FileNotFoundError
        inference.yaml が無い場合
    ValueError
        新しい版の場合と, 中身に誤りがある場合
    """
    # 説明と版
    root = Path(path).expanduser().resolve()
    file = root / "inference.yaml"
    if not file.is_file():
        raise FileNotFoundError(f"{root} に inference.yaml がありません")
    info = yaml.safe_load(file.read_text(encoding="utf-8"))
    version = info.get("format_version") if isinstance(info, dict) else None
    if isinstance(version, int) and version > BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"format_version {version} の一式は, このライブラリ (版 "
            f"{BUNDLE_FORMAT_VERSION} まで) では読めません"
        )
    errors = check_info(info)
    if errors:
        raise ValueError("inference.yaml の誤り: " + " / ".join(errors))

    # モデルの定義と前処理 (データセットの説明とレシピ)
    spec = load_spec(inside(root, str(info["model"]["spec"])))
    data_dir = inside(root, str(info["dataset"]))
    dataset = read_dataset(data_dir, frames=False)
    unknown = [
        name for name in (info.get("inputs") or {})
        if name not in spec["inputs"]
    ]
    if unknown:
        raise ValueError(
            f"inputs の {', '.join(unknown)} はモデルの入力にありません "
            f"(入力: {', '.join(spec['inputs'])})"
        )
    bundle = Bundle(path=root, info=info, dataset=dataset, spec=spec)
    if not bundle.weights_path.is_file():
        raise FileNotFoundError(
            f"重み {bundle.weights_path} がありません (実験のディレクトリを"
            "動かしたか消した場合は, 一式を作り直してください)"
        )
    kind = info["model"].get("kind")
    classify = bundle.task == "classification"
    if kind is not None and (kind == "value") == classify:
        raise ValueError(
            f"model.kind {kind} は {bundle.task} に使えません (分類は "
            "logits か probabilities, 回帰と予測は value)"
        )

    return bundle


def read_dataset(path: Path, frames: bool) -> Dataset:
    """
    ML-EDA のデータセットの説明とレシピを読む関数

    Parameters
    ----------
    path: Path
        dataset.yaml と recipe.yaml を持つディレクトリ
    frames: bool
        分割の表も読むか (一式の中の写しには表が無い)

    Returns
    ----------
    dataset: Dataset
        データセット

    Raises
    ----------
    FileNotFoundError
        dataset.yaml が無い場合
    """
    # 表まで要る場合は ML-EDA の読み込み (版の確認も含む)
    if not (path / "dataset.yaml").is_file():
        raise FileNotFoundError(f"{path} に dataset.yaml がありません")
    if frames:
        dataset = load_dataset(path)
        return dataset

    # 写しは説明とレシピだけ (版の確認と, レシピが外を指さないことの確認)
    info = yaml.safe_load((path / "dataset.yaml").read_text(encoding="utf-8"))
    version = info.get("format_version") if isinstance(info, dict) else None
    if not isinstance(version, int) or version > DATASET_FORMAT_VERSION:
        raise ValueError(
            f"format_version {version} のデータセットは, 入っている ML-EDA "
            f"(版 {DATASET_FORMAT_VERSION} まで) では読めません"
        )
    file = inside(path, str(info.get("recipe", "recipe.yaml")))
    recipe = yaml.safe_load(file.read_text(encoding="utf-8"))
    dataset = Dataset(path=path, info=info, frames={}, recipe=recipe)

    return dataset


def run_summary(run_dir: Path) -> dict:
    """
    ML-Found の実験のディレクトリから, 記録に残す状態を取り出す関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ

    Returns
    ----------
    summary: dict
        場所, 状態, 監視した指標とベストの値とエポック (無いものは None)
    """
    # run.json が無い (古い実験など) 場合は場所だけ
    summary: dict = {"path": str(run_dir.resolve())}
    file = run_dir / "run.json"
    if file.is_file():
        state = json.loads(file.read_text(encoding="utf-8"))
        for key in ("format_version", "status", "monitor", "best_value",
                    "best_epoch", "final_weights", "finished_at"):
            summary[key] = state.get(key)

    return summary


def bundle_notes(dataset: Dataset, run: Optional[dict]) -> list[str]:
    """
    一式を作るときの注意 (戻せない変換, 学習を終えていない実験) を返す関数

    Parameters
    ----------
    dataset: Dataset
        データセット
    run: Optional[dict]
        実験の状態 (run_summary の結果. 重みのファイルを渡した場合は None)

    Returns
    ----------
    notes: list[str]
        注意の文
    """
    # 実験の状態 (知らない版は読める項目だけを使ったことも)
    notes = []
    if run is not None:
        version = run.get("format_version")
        if isinstance(version, int) and version > RUN_FORMAT_VERSION:
            notes.append(
                f"実験の run.json は新しい版 ({version}) です. 読める項目だけを"
                "記録しました"
            )
        if run.get("status") not in (None, "completed"):
            notes.append(
                f"実験の状態は {run.get('status')} です (学習を終えていない"
                "重みの可能性があります)"
            )
        if run.get("final_weights") == "partial":
            notes.append("重みは途中で止めたエポックのものです")

    # 外れ値の切り詰めは戻せない
    target = dataset.info.get("target")
    for spec in dataset.recipe.get("steps", []):
        columns = spec.get("state", {}).get("columns") or spec.get("columns")
        columns = [columns] if isinstance(columns, str) else columns or []
        if spec["op"] == "clip" and target in columns:
            notes.append(
                f"目的の列 {target} を外れ値で切り詰めているので, 予測は "
                "切り詰めた範囲の値になります"
            )

    return notes


def make_bundle(
    path: Union[str, Path], spec: Union[str, Path, dict],
    dataset: Union[str, Path], weights: Union[str, Path],
    inputs: Optional[dict] = None, output: Optional[str] = None,
    kind: Optional[str] = None, target_scale: Optional[dict] = None,
    conclusion: Optional[dict] = None,
    name: Optional[str] = None, copy_weights: bool = True,
    overwrite: bool = False
) -> dict:
    """
    推論の一式を作り, 見本を推論できることを確かめる関数

    Parameters
    ----------
    path: Union[str, Path]
        作るディレクトリ
    spec: Union[str, Path, dict]
        モデルの定義 (ML-ModelControl の形) かそのファイル
    dataset: Union[str, Path]
        ML-EDA で書き出したデータセットのディレクトリ
    weights: Union[str, Path]
        重み (state_dict) のファイルか, ML-Found の実験のディレクトリ
        (final_model.pt を使い, 実験の状態を記録する)
    inputs: Optional[dict] = None
        モデルの入力ごとの source と prepend (入力が 1 つなら省略可)
    output: Optional[str] = None
        出力が辞書のモデルで使う出力の名前
    kind: Optional[str] = None
        出力の扱い方 (logits, probabilities, value. 省略するとタスクから)
    target_scale: Optional[dict] = None
        学習のスクリプトが目的の値を (y - center) / scale にして学習した
        場合の center と scale (出力を元に戻す)
    conclusion: Optional[dict] = None
        結論の決め方の既定値
    name: Optional[str] = None
        一式の名前 (省略するとディレクトリの名前)
    copy_weights: bool = True
        重みを一式に写すか (写さない場合は絶対パスで指す)
    overwrite: bool = False
        既にある一式を置き換えるか

    Returns
    ----------
    info: dict
        書き出した inference.yaml の中身

    Raises
    ----------
    FileExistsError
        ディレクトリが既にあり, overwrite が False の場合と, 一式ではない
        ディレクトリを置き換えようとした場合
    ValueError
        定義, 重み, 入力と出力のつなぎ方が合わない場合
    """
    # 置き換えられるのは, 前に作った一式だけ
    final = Path(path).expanduser().resolve()
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"{final} は既にあります")
        if not (final / "inference.yaml").is_file():
            raise FileExistsError(f"{final} は推論の一式ではないので置き換えません")

    # 重みの場所と, 実験のディレクトリなら記録する状態
    source = Path(weights).expanduser().resolve()
    run = None
    if source.is_dir():
        run = run_summary(source)
        source = source / "final_model.pt"
    if not source.is_file():
        raise ValueError(f"重み {source} がありません")
    data_dir = Path(dataset).expanduser().resolve()
    full = read_dataset(data_dir, frames=True)

    # 隣の一時ディレクトリに作り, 確かめてから入れ替える
    target = final.with_name(f".{final.name}.writing")
    if target.exists():
        shutil.rmtree(target)
    (target / "dataset").mkdir(parents=True)
    try:
        dump_spec(load_spec(spec), target / "model_spec.yaml")
        shutil.copy(data_dir / "dataset.yaml", target / "dataset")
        recipe_name = str(full.info.get("recipe", "recipe.yaml"))
        if Path(recipe_name).name != recipe_name:
            raise ValueError(f"レシピの名前 {recipe_name} がファイルの名前では"
                             "ありません")
        shutil.copy(data_dir / recipe_name, target / "dataset" / recipe_name)
        if copy_weights:
            shutil.copy(source, target / "final_model.pt")
        info = {
            "format_version": BUNDLE_FORMAT_VERSION,
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "model": {
                "spec": "model_spec.yaml",
                "weights": "final_model.pt" if copy_weights else str(source),
                "output": output,
                "kind": kind,
                "target_scale": target_scale,
            },
            "inputs": inputs or {},
            "dataset": "dataset",
            "run": run,
            "conclusion": plain(conclusion or {}),
            "notes": bundle_notes(full, run),
        }
        (target / "inference.yaml").write_text(
            yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
            encoding="utf-8"
        )
        bundle = load_bundle(target)
        # 結論の決め方は重い推論の前に確かめる
        if conclusion:
            errors = check_rules(conclusion, bundle.task, bundle.classes)
            if errors:
                raise ValueError("結論の決め方の誤り: " + " / ".join(errors))
        check_sample(bundle, full)
        # 記録との比べは注意のためなので, 失敗しても作るのは止めない
        try:
            info["check"] = check_against_run(bundle, full, run)
        except Exception as error:
            info["check"] = None
            info["notes"].append(
                f"検証用の分割を推論して記録と比べられませんでした ({error})"
            )
        if info["check"] and info["check"]["matches"] is False:
            info["notes"].append(check_note(info["check"]))
        (target / "inference.yaml").write_text(
            yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
            encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(final)
    target.rename(final)

    return info


def split_sources(
    bundle: Bundle, dataset: Dataset, split: str, start: int, stop: int
) -> dict:
    """
    前処理を済ませた分割の一部を, 中身の出どころごとの配列にする関数

    Parameters
    ----------
    bundle: Bundle
        一式
    dataset: Dataset
        分割の表を持つデータセット
    split: str
        分割の名前
    start: int
        最初の行
    stop: int
        最後の行の次

    Returns
    ----------
    sources: dict
        features と items の配列 (あるものだけ)
    """
    # 中身の配列はファイルから必要な行だけを読む
    x, _ = dataset.arrays(split)
    sources = {}
    if bundle.kind in ("table", "timeseries"):
        sources["features"] = np.asarray(x[start:stop])
    else:
        sources["items"] = np.asarray(x[start:stop])
        if dataset.features:
            sources["features"] = dataset.frames[split][
                dataset.features
            ].to_numpy(dtype=np.float32)[start:stop]

    return sources


def check_sample(bundle: Bundle, dataset: Dataset) -> None:
    """
    データセットの学習用の見本を推論し, 出力の形を確かめる関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    dataset: Dataset
        分割の表を持つデータセット

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        モデルに見本を渡せない場合と, 出力の形がタスクに合わない場合
    """
    # 予測の関数はこのモジュールを読むので, 使うときに読み込む
    from ml_inference.predict import decode, run_model, shape_inputs

    # 前処理を済ませた学習用の最初の数件 (分割の名前は train を優先)
    split = "train" if "train" in dataset.frames else next(
        iter(dataset.frames)
    )
    sources = split_sources(bundle, dataset, split, 0, SAMPLE_ROWS)

    # 形の誤りは分かりやすい文に (つなぎ方の誤りの文はそのまま)
    try:
        outputs = run_model(bundle, shape_inputs(bundle, sources), 64, "cpu")
        decode(bundle, outputs)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            f"見本の推論に失敗しました ({type(error).__name__}: {error}). "
            "入力のつなぎ方 (inputs) と定義の入力の形を確かめてください"
        ) from error


def check_against_run(
    bundle: Bundle, dataset: Dataset, run: Optional[dict]
) -> Optional[dict]:
    """
    データセットの分割を一式で推論した評価を, 実験の記録と比べる関数

    学習のスクリプトが前処理の外でしたこと (並びの先頭の番号, 目的の値の
    標準化など) を一式に書き忘れると, 形は合っても予測が誤る. 検証用の
    分割の正解率か平均絶対誤差が, 実験の記録 (ベストの値) と合うかで見つける

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    dataset: Dataset
        分割の表を持つデータセット
    run: Optional[dict]
        実験の状態 (run_summary の結果)

    Returns
    ----------
    check: Optional[dict]
        split, rows, metric, value, recorded (実験の記録. 比べられなければ
        None), matches (合うか. 比べられなければ None). 目的の値の無い
        データセットは None
    """
    # 予測の関数はこのモジュールを読むので, 使うときに読み込む
    from ml_inference.predict import (
        decode, inverse_target, run_model, shape_inputs
    )

    # 検証用の分割 (無ければテスト用, 学習用) と指標
    if not dataset.target or not dataset.frames:
        return None
    split = next(
        (s for s in ("valid", "test", "train") if s in dataset.frames), None
    )
    if split is None:
        return None
    _, y = dataset.arrays(split)
    if y is None or len(y) == 0:
        return None
    metric = "acc" if bundle.task == "classification" else "mae"

    # 監視した指標が同じで, 最終の重みがベストのときだけ記録と比べる.
    # 比べられない場合は, 値を残すだけなので少しの行で済ませる
    monitor = str((run or {}).get("monitor") or "")
    phase, _, name = monitor.partition("/")
    comparable = (
        run is not None and phase == split and len(y) <= CHECK_ROWS
        and METRIC_NAMES.get(name) == metric
        and run.get("final_weights") == "best"
        and isinstance(run.get("best_value"), (int, float))
    )
    rows = len(y) if comparable else min(len(y), CHECK_SAMPLE_ROWS)

    # 少しずつ推論して, 正解の数か誤差の合計だけを積み上げる
    total = 0.0
    for start in range(0, rows, CHECK_BATCH):
        stop = min(rows, start + CHECK_BATCH)
        sources = split_sources(bundle, dataset, split, start, stop)
        columns = decode(
            bundle,
            run_model(bundle, shape_inputs(bundle, sources), stop, "cpu")
        )
        if metric == "acc":
            answers = [bundle.classes[int(i)] for i in y[start:stop]]
            total += float(np.sum(
                np.array(columns["prediction"]) == np.array(answers)
            ))
        else:
            answers = inverse_target(bundle, np.asarray(y[start:stop]))
            total += float(np.sum(np.abs(columns["prediction"] - answers)))
    value = total / rows
    recorded, matches = None, None
    if comparable:
        recorded = float(run["best_value"])
        gap = abs(value - recorded) / max(abs(recorded), 1e-12)
        matches = bool(gap <= CHECK_TOLERANCE)
    check = {
        "split": split, "rows": int(rows), "metric": metric, "value": value,
        "recorded": recorded, "matches": matches,
    }

    return check


def check_note(check: dict) -> str:
    """
    実験の記録と合わない評価の注意の文を返す関数

    Parameters
    ----------
    check: dict
        check_against_run の結果

    Returns
    ----------
    note: str
        注意の文
    """
    # 前処理の外でしたことの書き忘れを疑う
    note = (
        f"{check['split']} の分割を一式で推論した {check['metric']} "
        f"{check['value']:.4g} が, 実験の記録 {check['recorded']:.4g} と"
        "合いません. 学習のスクリプトが前処理の外でしたこと (並びの先頭の"
        "番号 prepend, 目的の値の標準化 target_scale など) を一式に書いたか"
        "確かめてください (学習のスクリプトが指標を元の単位でない値や違う"
        "決まりで求めている場合も合いません)"
    )

    return note
