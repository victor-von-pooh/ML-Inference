from dataclasses import dataclass, field
import datetime as dt
import json
from pathlib import Path
import shutil
from typing import Any, Optional, Union
import warnings

import numpy as np
import torch
import yaml
from ml_eda import Dataset, ItemDataset, load_dataset, load_items
from ml_eda.items import PLACEMENTS
from ml_eda.pipeline import DATASET_FORMAT_VERSION, ITEMS_FORMAT_VERSIONS
from ml_modelcontrol import build_model, dump_spec, load_spec

from ml_inference.conclude import check_rules, plain
from ml_inference.items import ITEM_FORMS, PREPARE_KINDS, check_items
from ml_inference.procedures import get_procedure, option_names
from ml_inference.units import check_units
from ml_inference.weights import (
    PLACES, place_weights, lookup_pretrained, node_keys, pin_spec,
    state_hash
)

# 推論の一式の形の版 (版 4 は版 2 のデータセットの一式だけ. 版 1 の
# データセットの一式は版 3 で書く)
BUNDLE_FORMAT_VERSION = 4
COLUMN_BUNDLE_VERSION = 3

# inference.yaml の一番上の項目
TOP_KEYS = (
    "format_version", "name", "created_at", "role", "output", "target",
    "procedure", "model", "inputs", "dataset", "run", "conclusion",
    "decision", "check", "uncertainty", "notes", "items"
)

# 出力の扱い方
OUTPUT_KINDS = ("logits", "probabilities", "value")

# 受け持ち (ML-Quest の受け持ちと同じ値)
ROLES = ("predict", "generate", "rank", "group", "effect")

# 出力の形と, 形ごとに使える出力の扱い方 (先頭が既定)
FORM_KINDS = {
    "class": ("logits", "probabilities"),
    "value": ("value",),
    "distribution": ("value",),
    "mask": ("logits", "probabilities"),
    "point_cloud": ("value",),
    "vector": ("value",),
    "text": ("logits",),
    "image": ("value",),
}
FORMS = tuple(FORM_KINDS)

# 結論と意思決定を使える列の出力の形
COLUMN_FORMS = ("class", "value", "distribution")

# 出力の形ごとに要るデータセットのタスク (無い形はタスクを問わない形)
FORM_TASKS = {
    "class": ("classification",),
    "value": ("regression", "forecasting"),
    "distribution": ("regression", "forecasting"),
}

# 正解の出どころの種類 (ML-EDA の版 2 の target.origin と同じ)
ORIGIN_TYPES = ("process", "annotation", "measurement", "synthetic",
                "unknown")

# 入力の中身の出どころ (state と time は繰り返しの手順の中身と時刻)
INPUT_SOURCES = ("features", "items", "state", "time")

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
METRIC_NAMES = {
    "acc": "acc", "accuracy": "acc", "mae": "mae", "iou": "iou",
    "chamfer": "chamfer",
}

# 版 2 のデータセットの一式の, 出力の形ごとの確かめの指標
ITEM_METRICS = {"mask": "iou", "point_cloud": "chamfer"}


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
        # 版 2 のデータセットの一式はファイルの組
        if self.from_items:
            kind = "items"
            return kind
        kind = str(self.dataset.info.get("kind", "table"))

        return kind

    @property
    def from_items(self) -> bool:
        """
        版 2 のデータセットの一式かを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        items: bool
            items (1 件の扱い) を持つ一式なら True
        """
        # 版 4 の items の有無
        items = self.info.get("items") is not None

        return items

    @property
    def mask_values(self) -> Optional[list[int]]:
        """
        領域のクラスの番号ごとの領域の値を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        values: Optional[list[int]]
            番号ごとの値 (版 2 のデータセットの領域の一式でなければ None)
        """
        # items の values
        values = (self.info.get("items") or {}).get("values")

        return values

    @property
    def role(self) -> str:
        """
        受け持ちを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        role: str
            predict, generate, rank, group, effect (版 2 までは predict)
        """
        # 版 2 までの一式は列の予測
        role = str(self.info.get("role") or "predict")

        return role

    @property
    def output_info(self) -> dict:
        """
        出力の形の説明を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        output: dict
            form と形ごとの欄 (版 2 までは form をタスクから)
        """
        # 版 2 までは分類がクラス, 他は値
        output = dict(self.info.get("output") or {})
        if not output.get("form"):
            output["form"] = form_of_task(self.task)

        return output

    @property
    def form(self) -> str:
        """
        出力の形を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        form: str
            class, value, distribution, mask, point_cloud, vector, text,
            image のどれか
        """
        # 出力の形の説明の form
        form = str(self.output_info["form"])

        return form

    @property
    def column_output(self) -> bool:
        """
        結論と意思決定を使える列の出力かを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        column: bool
            class, value, distribution なら True
        """
        # 1 行に 1 つのクラスか値
        column = self.form in COLUMN_FORMS

        return column

    @property
    def procedure(self) -> Optional[dict]:
        """
        繰り返しの手順の設定を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        procedure: Optional[dict]
            name, steps, seed, options (無ければ None)
        """
        # 書かれていなければ 1 回の推論
        procedure = self.info.get("procedure") or None

        return procedure

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
        # 書かれていなければ, 出力の形の既定 (分類は logits, 値は value)
        kind = self.info["model"].get("kind") or FORM_KINDS[self.form][0]

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

    def require_column(self, what: str) -> None:
        """
        列の出力 (クラスか値) の一式でなければ誤りにする関数

        Parameters
        ----------
        what: str
            使おうとした機能の名前

        Returns
        ----------
        None

        Raises
        ----------
        ValueError
            列の出力でない一式の場合
        """
        # 結論と意思決定は 1 行に 1 つのクラスか値が前提
        if not self.column_output:
            raise ValueError(
                f"出力の形が {self.form} の一式には {what} を使えません "
                f"(使えるのは {', '.join(COLUMN_FORMS)} の一式だけ)"
            )

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

        # 定義から組み立て, 重みだけを読み込む (pickle の任意のコードは不可).
        # 外部の重みは学習した重みに含まれるので読まない. 出どころから読む
        # ノードがあれば, 読めないときにノードと出どころを添えた誤り
        sourced = [
            record for record in self.info["model"].get("pretrained") or []
            if record.get("place") == "source"
        ]
        try:
            model = build_model(self.spec, seed=0, load_weights=False)
        except Exception as error:
            if not sourced:
                raise
            names = ", ".join(
                f"{r['node']} ({r.get('hub_id') or r.get('model') or ''} "
                f"版 {r.get('pinned_revision')})" for r in sourced
            )
            raise ValueError(
                f"出どころから読むノード {names} を含むモデルを組み立てられ"
                f"ません ({type(error).__name__}: {error}). ネットワーク, "
                "トークン (HF_TOKEN), 出どころの版が残っているかを確かめて"
                "ください"
            ) from error
        state = torch.load(
            self.weights_path, map_location="cpu", weights_only=True
        )
        if sourced:
            self.cache["pretrained_warnings"] = load_sourced(
                model, state, sourced
            )
        else:
            model.load_state_dict(state)
        model.eval()
        self.cache["model"] = model

        return model


def load_sourced(model: Any, state: dict, sourced: list) -> list[str]:
    """
    出どころから読むノードの重みを hash で確かめる関数

    出どころから読むノードを除いた一式の重みを読み込み, 出どころの重みを
    記録の hash と比べる

    Parameters
    ----------
    model: nn.Module
        定義から組み立てたモデル (出どころの重みを読んだもの)
    state: dict
        一式の重み (出どころから読むノードを除いたもの)
    sourced: list
        出どころから読むノードの記録

    Returns
    ----------
    messages: list[str]
        hash の違ったノードの警告の文 (無ければ空)

    Raises
    ----------
    ValueError
        足りない重みが出どころから読むノードのものだけでない場合
    """
    # 足りないキーは出どころから読むノードのものだけ
    built = model.state_dict()
    expected = set()
    for record in sourced:
        expected.update(node_keys(built, str(record["node"])))
    missing = set(built) - set(state)
    extra = set(state) - set(built)
    if missing != expected or extra:
        raise ValueError(
            "一式の重みが定義と合いません (足りない重み "
            f"{len(missing - expected)} 個, 余る重み {len(extra)} 個)"
        )
    model.load_state_dict(state, strict=False)

    # 出どころの重みと一式を作ったときの重みの比べ (違えば警告. 推論は続行)
    messages = []
    for record in sourced:
        node = str(record["node"])
        found = state_hash(built, node_keys(built, node))
        if found != record.get("sha256"):
            message = (
                f"ノード {node} の出どころから読んだ重み "
                f"({record.get('hub_id') or record.get('model') or ''} "
                f"版 {record.get('pinned_revision')}) が, 一式を作ったとき"
                "と違います. 出どころの重みが入れ替わったか, 版を固定できない"
                "部品です. 予測が一式を作ったときと変わることがあります"
            )
            warnings.warn(message)
            messages.append(message)

    return messages


def form_of_task(task: str) -> str:
    """
    版 2 までの一式の出力の形を, タスクから決める関数

    Parameters
    ----------
    task: str
        データセットのタスク

    Returns
    ----------
    form: str
        分類は class, 他は value
    """
    # 版 2 までは列の予測だけ
    form = "class" if task == "classification" else "value"

    return form


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
    elif model.get("pretrained") is not None:
        errors.extend(check_pretrained(model["pretrained"]))
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

    # 版 3 の受け持ち, 出力の形, 正解, 繰り返しの手順と, 版 4 の 1 件の扱い
    errors.extend(check_kinds(info))
    errors.extend(check_procedure(info.get("procedure"), info))
    if info.get("items") is not None:
        if isinstance(version, int) and version < 4:
            errors.append(f"版 {version} の一式は items を持てません")
        errors.extend(check_items(info["items"]))

    return errors


def check_kinds(info: dict) -> list[str]:
    """
    受け持ち, 出力の形, 正解の種類と出どころの形を確かめる関数

    Parameters
    ----------
    info: dict
        inference.yaml の中身

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 受け持ち
    errors = []
    role = info.get("role")
    if role is not None and role not in ROLES:
        errors.append(f"role は {', '.join(ROLES)} のどれかにしてください")

    # 出力の形と形ごとの欄
    output = info.get("output")
    if output is not None:
        if not isinstance(output, dict) or output.get("form") not in FORMS:
            errors.append(
                f"output.form は {', '.join(FORMS)} のどれかにしてください"
            )
            output = {}
        form = output.get("form")
        kind = (info.get("model") or {}).get("kind") if isinstance(
            info.get("model"), dict
        ) else None
        if form in FORM_KINDS and kind is not None and (
            kind not in FORM_KINDS[form]
        ):
            errors.append(
                f"model.kind {kind} は出力の形 {form} に使えません "
                f"({', '.join(FORM_KINDS[form])})"
            )
        quantiles = output.get("quantiles")
        if form == "distribution" and not (
            isinstance(quantiles, list) and quantiles
            and all(is_level(q) for q in quantiles)
            and all(a < b for a, b in zip(quantiles, quantiles[1:]))
        ):
            errors.append(
                "output.quantiles は 0 から 1 の間の小さい順の数の並びに"
                "してください"
            )
        shape = output.get("shape")
        if shape is not None and not (
            isinstance(shape, list) and shape and all(
                isinstance(v, int) and not isinstance(v, bool) and v > 0
                for v in shape
            )
        ):
            errors.append("output.shape は正の整数の並びにしてください")

    # 正解の種類と出どころ
    target = info.get("target")
    if target is not None:
        if not isinstance(target, dict) or not isinstance(
            target.get("kind"), str
        ):
            errors.append("target は kind (文字) を持つ辞書にしてください")
        elif target.get("origin") is not None:
            origin = target["origin"]
            if not isinstance(origin, dict) or origin.get("type") not in (
                ORIGIN_TYPES
            ):
                errors.append(
                    "target.origin.type は "
                    f"{', '.join(ORIGIN_TYPES)} のどれかにしてください"
                )

    return errors


def is_level(value: Any) -> bool:
    """
    分位点の水準 (0 と 1 の間の数) かを確かめる関数

    Parameters
    ----------
    value: Any
        確かめる値

    Returns
    ----------
    result: bool
        0 より大きく 1 より小さい数なら True
    """
    # 真偽値を除いた数
    result = isinstance(value, (int, float)) and not isinstance(
        value, bool
    ) and 0 < value < 1

    return result


def check_procedure(procedure: Any, info: dict) -> list[str]:
    """
    繰り返しの手順の設定と, 手順の中身と時刻の入力を確かめる関数

    Parameters
    ----------
    procedure: Any
        inference.yaml の procedure
    info: dict
        inference.yaml の中身

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 手順の中身と時刻の入力の数
    inputs = info.get("inputs") if isinstance(info.get("inputs"), dict) else {}
    sources = [
        item.get("source") for item in inputs.values()
        if isinstance(item, dict)
    ]
    errors: list[str] = []
    if procedure is None:
        if "state" in sources or "time" in sources:
            errors.append("入力の state と time は procedure が要ります")
        return errors
    if not isinstance(procedure, dict):
        errors.append("procedure は name, steps, seed の辞書にしてください")
        return errors

    # 名前は登録簿に, 回数と種は整数, 設定は手順の引数
    name = procedure.get("name")
    try:
        found = get_procedure(str(name))
    except KeyError as error:
        errors.append(str(error.args[0]))
        return errors
    steps = procedure.get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        errors.append("procedure.steps は 1 以上の整数にしてください")
    seed = procedure.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append("procedure.seed は整数にしてください")
    options = procedure.get("options") or {}
    known = option_names(str(name))
    if not isinstance(options, dict):
        errors.append("procedure.options は辞書にしてください")
    elif known is not None:
        unknown = [k for k in options if k not in known]
        if unknown:
            errors.append(
                f"procedure.options の {', '.join(map(str, unknown))} は "
                f"{name} の設定にありません "
                f"({', '.join(known) or '設定の無い手順'})"
            )
    extra = [k for k in procedure if k not in ("name", "steps", "seed",
                                               "options")]
    if extra:
        errors.append(
            f"procedure の {', '.join(map(str, extra))} は知らない項目です"
        )

    # 中身の入力は 1 つ, 時刻は時刻を使う手順だけ
    if sources.count("state") != 1:
        errors.append("procedure を使う一式は source が state の入力を 1 つ"
                      "持ってください")
    if "time" in sources and not found.time:
        errors.append(f"手順 {name} は時刻を渡さないので time の入力を使えません")
    if found.time and sources.count("time") != 1:
        errors.append(f"手順 {name} は source が time の入力を 1 つ要ります")
    output = info.get("output") if isinstance(info.get("output"), dict) else {}
    if found.start == "noise" and not output.get("shape"):
        errors.append(
            f"手順 {name} は雑音から始まるので output.shape (1 件の形) が"
            "要ります"
        )

    # 手順の始まりと出力の形の組 (雑音からは配列, 番号の並びからは文)
    form = output.get("form")
    if form is not None and found.start == "noise" and form in (
        "class", "value", "distribution", "text"
    ):
        errors.append(f"雑音から始まる手順 {name} は出力の形 {form} に"
                      "使えません")
    if form is not None and found.start == "given" and form != "text":
        errors.append(f"番号の並びから始まる手順 {name} は出力の形 text に"
                      "だけ使えます")
    if isinstance(options, dict):
        errors.extend(check_options(str(name), options, steps))

    return errors


def check_options(name: str, options: dict, steps: Any) -> list[str]:
    """
    組み込みの手順の設定の値を確かめる関数

    Parameters
    ----------
    name: str
        手順の名前
    options: dict
        手順の設定
    steps: Any
        繰り返しの回数

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 数の設定の型と範囲
    errors = []
    numbers = {
        "train_steps": 1, "beta_start": 0, "beta_end": 0, "eta": 0,
        "t_start": None, "t_end": None, "temperature": 0,
    }
    for key, low in numbers.items():
        if key not in options:
            continue
        value = options[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"procedure.options.{key} は数にしてください")
        elif key == "train_steps" and not isinstance(value, int):
            errors.append("procedure.options.train_steps は整数にしてください")
        elif low is not None and (
            value < low or (key == "temperature" and value == 0)
        ):
            errors.append(f"procedure.options.{key} が範囲の外です")
    eos = options.get("eos")
    if eos is not None and (not isinstance(eos, int) or isinstance(eos,
                                                                   bool)):
        errors.append("procedure.options.eos は整数にしてください")
    limit = options.get("train_steps", 1000)
    if name == "ddim" and isinstance(steps, int) and isinstance(
        limit, (int, float)
    ) and steps > limit:
        errors.append("ddim の steps は train_steps 以下にしてください")

    return errors


def check_pretrained(records: Any) -> list[str]:
    """
    学習済みの重みの置き方の記録の形を確かめる関数

    Parameters
    ----------
    records: Any
        inference.yaml の model.pretrained

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # ノードの名前, 置き方, 出どころから読むものは hash
    errors: list[str] = []
    if not isinstance(records, list):
        errors.append("model.pretrained はノードごとの記録の並びにして"
                      "ください")
        return errors
    for record in records:
        if not isinstance(record, dict) or not record.get("node") or (
            record.get("place") not in PLACES
        ):
            errors.append(
                "model.pretrained の記録は node と place "
                f"({', '.join(PLACES)}) を持つ辞書にしてください"
            )
        elif record["place"] == "source" and not record.get("sha256"):
            errors.append(
                f"model.pretrained の {record['node']} は sha256 が要ります"
            )

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
        新しい版の場合と, 中身に誤りがある場合 (登録されていない部品は
        子の UnregisteredUnitError)
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

    # モデルの定義 (部品は登録簿から引くだけ. 名前から何も読み込まない)
    # と前処理 (データセットの説明とレシピ)
    spec = load_spec(inside(root, str(info["model"]["spec"])))
    check_units(spec)
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
    form = bundle.form
    if info.get("output") is None:
        classify = bundle.task == "classification"
        if kind is not None and (kind == "value") == classify:
            raise ValueError(
                f"model.kind {kind} は {bundle.task} に使えません (分類は "
                "logits か probabilities, 回帰と予測は value)"
            )
    elif form in FORM_TASKS and bundle.task not in FORM_TASKS[form]:
        raise ValueError(
            f"出力の形 {form} は {bundle.task or 'タスクの無い'} の"
            f"データセットに使えません ({', '.join(FORM_TASKS[form])} だけ)"
        )
    procedure = info.get("procedure")
    if procedure and get_procedure(str(procedure["name"])).start == (
        "given"
    ) and bundle.kind != "text":
        raise ValueError(
            f"手順 {procedure['name']} は文の番号の並びから始めるので, 文の"
            f"データセットの一式だけに使えます (今は {bundle.kind})"
        )
    check_items_bundle(bundle)
    names = set(spec["nodes"])
    for record in info["model"].get("pretrained") or []:
        if record["node"] not in names:
            raise ValueError(
                f"model.pretrained の {record['node']} はモデルの定義の"
                "ノードにありません"
            )

    return bundle


def check_items_bundle(bundle: Bundle) -> None:
    """
    一式の版, items, データセットの版と出力の形が食い違わないかを確かめる
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
    ValueError
        版 2 のデータセットの一式が items を持たない場合と, その逆, 出力
        の形や項目が合わない場合
    """
    # 版 2 と 3 のデータセット と items と版 4 は組
    info = bundle.info
    items = info.get("items")
    data_version = bundle.dataset.info.get("format_version")
    if (data_version in ITEMS_FORMAT_VERSIONS) != (items is not None):
        raise ValueError(
            "版 2 と 3 のデータセットの一式には items が要り, 版 1 のデータ"
            "セットの一式は items を持てません"
        )
    if items is None and info.get("format_version") == BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"版 {BUNDLE_FORMAT_VERSION} の一式には items (版 2 のデータ"
            "セットの 1 件の扱い) が要ります"
        )
    if items is None:
        return
    if info.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"items を持つ一式は版 {BUNDLE_FORMAT_VERSION} にしてください"
        )
    if bundle.form not in ITEM_FORMS:
        raise ValueError(
            f"版 2 のデータセットの一式の出力の形は {', '.join(ITEM_FORMS)}"
            f" だけです (今は {bundle.form})"
        )

    # 項目の名前と種類 (データセットの写しの entries)
    entries = bundle.dataset.info.get("entries") or {}
    for key in ("input", "target"):
        name = items.get(key)
        if name is not None and name not in entries:
            raise ValueError(
                f"items.{key} の {name} はデータセットの項目にありません"
            )
    kind = entries[items["input"]].get("kind")
    if PREPARE_KINDS[items["prepare"]] != kind:
        raise ValueError(
            f"扱い方 {items['prepare']} は入力の項目の種類 {kind} に使え"
            "ません"
        )
    if bundle.form == "mask" and not items.get("values"):
        raise ValueError("領域の一式の items には values が要ります")


def check_entries(info: dict) -> None:
    """
    版 2 のデータセットの説明の項目が読める形かを確かめる関数

    一式の写しは件の表を持たないので, load_items の代わりに説明だけを見る

    Parameters
    ----------
    info: dict
        版 2 の dataset.yaml の中身

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        項目が無い場合と, 知らない置き方の項目がある場合
    """
    # 項目の辞書と置き方 (知らない置き方は読み違えないよう止まる)
    entries = info.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("dataset.yaml に entries がありません")
    for name, spec in entries.items():
        placement = spec.get("placement") if isinstance(spec, dict) else None
        if placement not in PLACEMENTS:
            raise ValueError(
                f"項目 {name} の置き方 {placement} は, 入っている ML-EDA "
                f"では読めません ({', '.join(PLACEMENTS)} だけ)"
            )


def read_dataset(
    path: Path, frames: bool
) -> Union[Dataset, ItemDataset]:
    """
    ML-EDA のデータセットの説明とレシピを読む関数

    Parameters
    ----------
    path: Path
        dataset.yaml (と版 1 は recipe.yaml) を持つディレクトリ
    frames: bool
        分割の表 (版 2 と 3 は件の表) も読むか (一式の中の写しには表が
        無い)

    Returns
    ----------
    dataset: Union[Dataset, ItemDataset]
        データセット (表まで読む版 2 と 3 は ItemDataset. 写しは版に
        依らず, 説明だけの Dataset で, 版 2 と 3 のレシピは空)

    Raises
    ----------
    FileNotFoundError
        dataset.yaml が無い場合
    """
    # 版を見て, 表まで要る場合は ML-EDA の読み込み (版の確認も含む)
    if not (path / "dataset.yaml").is_file():
        raise FileNotFoundError(f"{path} に dataset.yaml がありません")
    info = yaml.safe_load((path / "dataset.yaml").read_text(encoding="utf-8"))
    version = info.get("format_version") if isinstance(info, dict) else None
    if frames:
        dataset = load_items(path) if version in ITEMS_FORMAT_VERSIONS else (
            load_dataset(path)
        )
        return dataset

    # 版 2 と 3 の写しは説明だけ (読める版と項目の置き方の確認は ML-EDA)
    if version in ITEMS_FORMAT_VERSIONS:
        check_entries(info)
        dataset = Dataset(path=path, info=info, frames={}, recipe={})
        return dataset

    # 版 1 の写しは説明とレシピ (版の確認と, レシピが外を指さないことの
    # 確認)
    if not isinstance(version, int) or version > DATASET_FORMAT_VERSION:
        raise ValueError(
            f"format_version {version} のデータセットは, 入っている ML-EDA "
            f"(版 {DATASET_FORMAT_VERSION} と "
            f"{' と '.join(str(v) for v in ITEMS_FORMAT_VERSIONS)}) では"
            "読めません"
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


def bundle_notes(
    dataset: Union[Dataset, ItemDataset], run: Optional[dict]
) -> list[str]:
    """
    一式を作るときの注意 (戻せない変換, 学習を終えていない実験) を返す関数

    Parameters
    ----------
    dataset: Union[Dataset, ItemDataset]
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

    # 外れ値の切り詰めは戻せない (レシピは版 1 のデータセットだけ)
    target = dataset.info.get("target")
    recipe = dataset.recipe if isinstance(dataset, Dataset) else {}
    for spec in recipe.get("steps", []):
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
    kind: Optional[str] = None,
    target_scale: Union[dict, str, None] = None,
    conclusion: Optional[dict] = None, decision: Optional[dict] = None,
    name: Optional[str] = None, copy_weights: bool = True,
    overwrite: bool = False, role: Optional[str] = None,
    form: Optional[str] = None, output_options: Optional[dict] = None,
    target: Optional[dict] = None, procedure: Optional[dict] = None,
    pretrained: Optional[dict] = None, items: Optional[dict] = None
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
        ML-EDA で書き出したデータセットのディレクトリ (版 1 か, 版 2 の
        ファイルの組)
    weights: Union[str, Path]
        重み (state_dict) のファイルか, ML-Found の実験のディレクトリ
        (final_model.pt を使い, 実験の状態を記録する)
    inputs: Optional[dict] = None
        モデルの入力ごとの source と prepend (入力が 1 つなら省略可)
    output: Optional[str] = None
        出力が辞書のモデルで使う出力の名前
    kind: Optional[str] = None
        出力の扱い方 (logits, probabilities, value. 省略するとタスクから)
    target_scale: Union[dict, str, None] = None
        学習のスクリプトが目的の値を (y - center) / scale にして学習した
        場合の center と scale (出力を元に戻す). "train" は, データセット
        の学習用の分割の目的の値の平均と標準偏差 (よくある決め方)
    conclusion: Optional[dict] = None
        結論の決め方の既定値
    decision: Optional[dict] = None
        意思決定の決め方の既定値 (損得と制約. 将来は問題の設定から)
    name: Optional[str] = None
        一式の名前 (省略するとディレクトリの名前)
    copy_weights: bool = True
        重みを一式に写すか (写さない場合は絶対パスで指す)
    overwrite: bool = False
        既にある一式を置き換えるか
    role: Optional[str] = None
        受け持ち (predict, generate, rank, group, effect. 省略すると
        predict)
    form: Optional[str] = None
        出力の形 (class, value, distribution, mask, point_cloud, vector,
        text, image. 省略するとタスクから)
    output_options: Optional[dict] = None
        出力の形ごとの欄 (distribution の quantiles, 雑音から始まる手順の
        shape, mask の classes)
    target: Optional[dict] = None
        正解の種類 (kind) と出どころ (origin. ML-EDA の版 2 の
        target.origin の形). 省略すると列の出力は kind: column, 他は null
    procedure: Optional[dict] = None
        繰り返しの手順 (name, steps, seed, options)
    pretrained: Optional[dict] = None
        ノードの名前ごとの重みのライセンスの情報 (ML-ModelControl の
        pretrained_info の形). 省略すると pretrained_info で読む
    items: Optional[dict] = None
        版 2 のデータセットの 1 件の扱い (input, target, prepare,
        options, values, restore. 省いた欄はデータセットから. 点を選ぶ
        扱い方は options.points が要る)

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
    if isinstance(spec, (str, Path)) and not Path(spec).expanduser().is_file():
        raise ValueError(f"モデルの定義 {spec} はファイルではありません")
    data_dir = Path(dataset).expanduser().resolve()
    if not data_dir.is_dir():
        raise ValueError(f"データセット {data_dir} はディレクトリではありません")
    spec_dict = load_spec(spec)
    check_units(spec_dict)
    full = read_dataset(data_dir, frames=True)
    item_info = None
    if isinstance(full, ItemDataset):
        if target_scale is not None:
            raise ValueError("版 2 のデータセットの一式に target_scale は"
                             "使えません")
        kinds, item_info = item_fields(
            full, role, form, output_options, target, items
        )
    else:
        if items is not None:
            raise ValueError("items は版 2 のデータセットの一式だけに使え"
                             "ます")
        if isinstance(target_scale, str):
            target_scale = train_scale(full, target_scale)
        kinds = output_fields(
            str(full.info.get("task", "")), role, form, output_options,
            target
        )

    # 学習済みの重みを読むノードの置き方 (無ければ丸ごとの写し)
    infos, weight_notes = lookup_pretrained(spec_dict, pretrained)
    records: list = []
    kept = None
    if infos:
        spec_dict = pin_spec(spec_dict, infos)
        state = torch.load(source, map_location="cpu", weights_only=True)
        fresh = build_model(
            spec_dict, seed=0, load_weights=False
        ).state_dict()
        records, kept, notes = place_weights(
            infos, state, fresh, copy_weights
        )
        weight_notes.extend(notes)

    # 隣の一時ディレクトリに作り, 確かめてから入れ替える
    writing = final.with_name(f".{final.name}.writing")
    if writing.exists():
        shutil.rmtree(writing)
    (writing / "dataset").mkdir(parents=True)
    try:
        dump_spec(spec_dict, writing / "model_spec.yaml")
        shutil.copy(data_dir / "dataset.yaml", writing / "dataset")
        if item_info is None:
            recipe_name = str(full.info.get("recipe", "recipe.yaml"))
            if Path(recipe_name).name != recipe_name:
                raise ValueError(f"レシピの名前 {recipe_name} がファイルの"
                                 "名前ではありません")
            shutil.copy(data_dir / recipe_name,
                        writing / "dataset" / recipe_name)
        if copy_weights and kept is not None:
            torch.save(kept, writing / "final_model.pt")
        elif copy_weights:
            shutil.copy(source, writing / "final_model.pt")
        info = {
            "format_version": COLUMN_BUNDLE_VERSION if item_info is None
            else BUNDLE_FORMAT_VERSION,
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            **kinds,
            "procedure": plain(procedure) if procedure else None,
            "model": {
                "spec": "model_spec.yaml",
                "weights": "final_model.pt" if copy_weights else str(source),
                "output": output,
                "kind": kind,
                "target_scale": target_scale,
                "pretrained": records,
            },
            "inputs": inputs or {},
            "dataset": "dataset",
            "run": run,
            "conclusion": plain(conclusion or {}),
            "decision": plain(decision) if decision else None,
            "notes": bundle_notes(full, run) + weight_notes,
        }
        if item_info is not None:
            info["items"] = item_info
        (writing / "inference.yaml").write_text(
            yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
            encoding="utf-8"
        )
        bundle = load_bundle(writing)
        # 結論と意思決定の決め方は重い推論の前に確かめる (列の出力だけ)
        if conclusion or decision:
            bundle.require_column("結論の決め方と意思決定")
        if conclusion:
            errors = check_rules(conclusion, bundle.task, bundle.classes)
            if errors:
                raise ValueError("結論の決め方の誤り: " + " / ".join(errors))
        if decision:
            from ml_inference.decision import check_decision

            # 既定値の今のやり方は列の名前だけ (一式の版 3 の形のまま.
            # 決まりはたどり直しで渡す)
            errors = check_decision(decision, bundle)
            if isinstance(decision.get("current"), dict):
                errors.append(
                    "一式の既定値の current は列の名前にしてください "
                    "(決まりで書いた今のやり方は, たどり直しで渡す)"
                )
            if errors:
                raise ValueError(
                    "意思決定の決め方の誤り: " + " / ".join(errors)
                )
        if bundle.from_items:
            check_item_sample(bundle, full)
            measure_items(bundle, full, run, info)
        elif bundle.column_output:
            check_sample(bundle, full)
            measure_column(bundle, full, run, info)
        else:
            check_sample(bundle, full)
            info["check"] = None
            info["uncertainty"] = None
        (writing / "inference.yaml").write_text(
            yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
            encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(writing, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(final)
    writing.rename(final)

    return info


def measure_column(
    bundle: Bundle, dataset: Dataset, run: Optional[dict], info: dict
) -> None:
    """
    列の出力の一式の, 記録との比べと外れ方を測る関数

    実験の記録との比べと予測の外れ方を測って説明に書く

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    dataset: Dataset
        分割の表を持つデータセット
    run: Optional[dict]
        実験の状態 (run_summary の結果)
    info: dict
        書き足す inference.yaml の中身 (check, uncertainty, notes)

    Returns
    ----------
    None
    """
    # 記録との比べは注意のためなので, 失敗しても作るのは止めない
    try:
        info["check"] = check_against_run(bundle, dataset, run)
    except Exception as error:
        info["check"] = None
        info["notes"].append(
            f"検証用の分割を推論して記録と比べられませんでした ({error})"
        )
    if info["check"] and info["check"]["matches"] is False:
        info["notes"].append(check_note(info["check"]))

    # 意思決定のための外れ方 (失敗しても作るのは止めない)
    from ml_inference.uncertainty import measure_uncertainty

    try:
        info["uncertainty"] = measure_uncertainty(bundle, dataset)
    except Exception as error:
        info["uncertainty"] = None
        info["notes"].append(f"予測の外れ方を測れませんでした ({error})")


def output_fields(
    task: str, role: Optional[str], form: Optional[str],
    options: Optional[dict], target: Optional[dict]
) -> dict:
    """
    一式の受け持ち, 出力の形, 正解の項目を決める関数

    Parameters
    ----------
    task: str
        データセットのタスク
    role: Optional[str]
        受け持ち (省略すると predict)
    form: Optional[str]
        出力の形 (省略するとタスクから)
    options: Optional[dict]
        出力の形ごとの欄
    target: Optional[dict]
        正解の種類と出どころ

    Returns
    ----------
    fields: dict
        role, output, target

    Raises
    ----------
    ValueError
        タスクから出力の形を決められない場合と, 欄に誤りがある場合
    """
    # 省略した出力の形はタスクから (正解の無いデータセットでは必須)
    if form is None:
        if task not in ("classification", "regression", "forecasting"):
            raise ValueError(
                f"タスクが {task or '無い'} のデータセットでは出力の形 "
                f"(form. {', '.join(FORMS)}) を選んでください"
            )
        form = form_of_task(task)
    output = {"form": form, **plain(dict(options or {}))}
    if "form" in (options or {}):
        raise ValueError("output_options に form は書けません (form で選ぶ)")

    # 省略した正解は, 列の出力なら列, 他は不明
    if target is None and form in COLUMN_FORMS:
        target = {"kind": "column"}
    fields = {
        "role": role or "predict",
        "output": output,
        "target": plain(dict(target)) if target is not None else None,
    }
    errors = check_kinds(fields)
    if errors:
        raise ValueError("出力の形の誤り: " + " / ".join(errors))

    return fields


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


def train_scale(dataset: Dataset, how: str) -> dict:
    """
    学習用の分割の目的の値から, 目的の値の標準化の中心と幅を求める関数

    Parameters
    ----------
    dataset: Dataset
        分割の表を持つデータセット
    how: str
        求め方 (今は train だけ. 平均と標準偏差 (不偏))

    Returns
    ----------
    scale: dict
        center と scale

    Raises
    ----------
    ValueError
        知らない求め方の場合と, 学習用の目的の値が無い場合
    """
    # 学習のスクリプトが使う PyTorch の std と同じ不偏の標準偏差
    if how != "train":
        raise ValueError(f"target_scale の {how} は知りません (train か辞書)")
    if "train" not in dataset.frames or not dataset.target:
        raise ValueError("学習用の分割の目的の値が無いので求められません")
    _, y = dataset.tensors("train")
    values = y.float()
    scale = {"center": values.mean().item(), "scale": values.std().item()}

    return scale


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

    # 形の誤りは分かりやすい文に (つなぎ方の誤りの文はそのまま. 長い誤り
    # の文は始めだけ)
    try:
        outputs = run_model(bundle, shape_inputs(bundle, sources), 64, "cpu")
        decode(bundle, outputs)
    except ValueError:
        raise
    except Exception as error:
        text = str(error)
        text = text if len(text) <= 300 else text[:300] + " …"
        hint = (
            "重みが定義と合いません. 学習に使った定義と重みの組を選んで"
            "ください" if "state_dict" in text else
            "入力のつなぎ方 (inputs) と定義の入力の形を確かめてください"
        )
        raise ValueError(
            f"見本の推論に失敗しました ({type(error).__name__}: {text}). "
            f"{hint}"
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
    comparable = comparable_run(run, split, len(y), metric)
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
    check = check_record(run, comparable, split, rows, metric, value)

    return check


def comparable_run(
    run: Optional[dict], split: str, rows: int, metric: str
) -> bool:
    """
    実験の記録と一式の評価を比べられるかを返す関数

    Parameters
    ----------
    run: Optional[dict]
        実験の状態 (run_summary の結果)
    split: str
        評価する分割
    rows: int
        分割の行 (件) の数
    metric: str
        一式の評価の指標 (acc, mae, iou, chamfer)

    Returns
    ----------
    comparable: bool
        監視した指標が同じ分割の同じ指標で, 最終の重みがベストで, 行の数
        が上限までなら True
    """
    # 監視した指標の分割と名前 (<分割>/<名前>)
    monitor = str((run or {}).get("monitor") or "")
    phase, _, name = monitor.partition("/")
    comparable = (
        run is not None and phase == split and rows <= CHECK_ROWS
        and METRIC_NAMES.get(name) == metric
        and run.get("final_weights") == "best"
        and isinstance(run.get("best_value"), (int, float))
    )

    return comparable


def check_record(
    run: Optional[dict], comparable: bool, split: str, rows: int,
    metric: str, value: float
) -> dict:
    """
    一式の評価と実験の記録の比べを, 説明の check の形にする関数

    Parameters
    ----------
    run: Optional[dict]
        実験の状態
    comparable: bool
        記録と比べられるか
    split: str
        評価した分割
    rows: int
        評価した行 (件) の数
    metric: str
        指標
    value: float
        一式で求めた値

    Returns
    ----------
    check: dict
        split, rows, metric, value, recorded, matches
    """
    # 比べられれば, 記録との差の割合が許す範囲か
    recorded, matches = None, None
    if comparable and run is not None:
        recorded = float(run["best_value"])
        gap = abs(value - recorded) / max(abs(recorded), 1e-12)
        matches = bool(gap <= CHECK_TOLERANCE)
    check = {
        "split": split, "rows": int(rows), "metric": metric,
        "value": float(value), "recorded": recorded, "matches": matches,
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


def item_fields(
    dataset: ItemDataset, role: Optional[str], form: Optional[str],
    options: Optional[dict], target: Optional[dict], items: Optional[dict]
) -> tuple[dict, dict]:
    """
    版 2 のデータセットから, 一式の受け持ち, 出力の形, 正解, 1 件の扱いを
    決める関数

    Parameters
    ----------
    dataset: ItemDataset
        版 2 のデータセット
    role: Optional[str]
        受け持ち (省くとデータセットの role)
    form: Optional[str]
        出力の形 (省くと正解の種類)
    options: Optional[dict]
        出力の形ごとの欄 (省いた領域の classes は値の名前)
    target: Optional[dict]
        正解の種類と出どころ (省くとデータセットの target)
    items: Optional[dict]
        1 件の扱い (省いた欄はデータセットから)

    Returns
    ----------
    fields: tuple[dict, dict]
        role, output, target の辞書と items

    Raises
    ----------
    ValueError
        予測するの領域か点群のデータセットでない場合と, 出力の形が正解の
        種類と違う場合
    """
    # 作れるのは予測するで, 正解の種類が領域か点群のものだけ
    from ml_inference.items import resolve_items

    described = dataset.info.get("target") or {}
    kind = described.get("kind")
    if dataset.info.get("role") != "predict" or kind not in ITEM_FORMS:
        raise ValueError(
            f"版 2 のデータセットから一式を作れるのは, 予測する (role: "
            f"predict) で正解の種類が {', '.join(ITEM_FORMS)} のものだけ"
            f"です (今は role: {dataset.info.get('role')}, 正解の種類: "
            f"{kind})"
        )
    form = form or kind
    if form != kind:
        raise ValueError(
            f"出力の形 {form} は正解の種類 {kind} と同じにしてください"
        )

    # 1 件の扱いと, 正解と領域のクラスの名前の既定
    resolved, names = resolve_items(dataset, form, items)
    if target is None:
        target = {"kind": kind}
        if described.get("origin") is not None:
            target["origin"] = described["origin"]
    options = dict(options or {})
    if names is not None and "classes" not in options:
        options["classes"] = names
    kinds = output_fields(
        "", role or "predict", form, options, target
    )
    fields = (kinds, resolved)

    return fields


def item_batches(
    bundle: Bundle, dataset: ItemDataset, split: str, start: int,
    stop: int
) -> tuple[list, list]:
    """
    分割の件に学習と同じ扱いをかけ, 入力と比べる正解を返す関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    dataset: ItemDataset
        版 2 のデータセット
    split: str
        分割の名前
    start: int
        最初の件
    stop: int
        最後の件の次

    Returns
    ----------
    batches: tuple[list, list]
        件ごとのモデルの入力の配列と, 比べる正解 (領域の値か点. 正解の
        項目が無ければ None)
    """
    # 件の位置は分割の中の番号 (点を選ぶ乱数に使う)
    from ml_inference.items import prepare_item, target_points

    items = bundle.info["items"]
    part = dataset.split(split)
    arrays, truths = [], []
    for position in range(start, stop):
        item = part[position]
        array, rng = prepare_item(items, item[items["input"]], position)
        arrays.append(array)
        truth = item.get(items["target"]) if items.get("target") else None
        if truth is not None and bundle.form == "point_cloud":
            truth = target_points(items, truth, rng)
        truths.append(truth)
    batches = (arrays, truths)

    return batches


def check_item_sample(bundle: Bundle, dataset: ItemDataset) -> None:
    """
    版 2 のデータセットの学習用の見本を推論し, 出力の形を確かめる関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    dataset: ItemDataset
        版 2 のデータセット

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        見本を扱えない場合と, モデルに渡せない場合, 出力の形が合わない
        場合
    """
    # 予測の関数はこのモジュールを読むので, 使うときに読み込む
    from ml_inference.predict import infer_arrays

    # 学習用の最初の数件 (分割の名前は train を優先)
    splits = list(dataset.info.get("splits") or {})
    if not splits:
        raise ValueError("データセットに件がありません")
    split = "train" if "train" in splits else splits[0]
    count = min(SAMPLE_ROWS, len(dataset.split(split)))
    try:
        arrays, _ = item_batches(bundle, dataset, split, 0, count)
        infer_arrays(bundle, arrays, 64, "cpu")
    except ValueError:
        raise
    except Exception as error:
        text = str(error)
        text = text if len(text) <= 300 else text[:300] + " …"
        hint = (
            "重みが定義と合いません. 学習に使った定義と重みの組を選んで"
            "ください" if "state_dict" in text else
            "1 件の扱い (items) と定義の入力の形を確かめてください"
        )
        raise ValueError(
            f"見本の推論に失敗しました ({type(error).__name__}: {text}). "
            f"{hint}"
        ) from error


def check_items_against_run(
    bundle: Bundle, dataset: ItemDataset, run: Optional[dict]
) -> Optional[dict]:
    """
    版 2 のデータセットの分割を一式で推論した評価を, 実験の記録と比べる
    関数

    1 件の扱い (点の数, 選ぶ乱数, 領域の値) の書き違いは, 形が合っても
    予測を黙って変える. 検証用の分割の IoU か Chamfer 距離が, 実験の記録
    (ベストの値) と合うかで見つける

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    dataset: ItemDataset
        版 2 のデータセット
    run: Optional[dict]
        実験の状態 (run_summary の結果)

    Returns
    ----------
    check: Optional[dict]
        split, rows, metric, value, recorded, matches (正解の項目か分割が
        無ければ None)
    """
    # 予測の関数はこのモジュールを読むので, 使うときに読み込む
    from ml_inference.items import chamfer, mask_counts, mean_iou
    from ml_inference.predict import infer_arrays

    # 検証用の分割 (無ければテスト用, 学習用) と指標
    splits = dataset.info.get("splits") or {}
    split = next((s for s in ("valid", "test", "train") if s in splits),
                 None)
    if split is None or not bundle.info["items"].get("target"):
        return None
    total = len(dataset.split(split))
    if total == 0:
        return None
    metric = ITEM_METRICS[bundle.form]
    comparable = comparable_run(run, split, total, metric)
    rows = total if comparable else min(total, CHECK_SAMPLE_ROWS)

    # 少しずつ推論して, 画素数か距離の合計だけを積み上げる (正規化した
    # 座標のまま比べる)
    values = bundle.mask_values or []
    counts = np.zeros((3, len(values)))
    distance = 0.0
    for start in range(0, rows, CHECK_BATCH):
        stop = min(rows, start + CHECK_BATCH)
        arrays, truths = item_batches(bundle, dataset, split, start, stop)
        outputs, _ = infer_arrays(bundle, arrays, CHECK_BATCH, "cpu")
        for output, truth in zip(outputs, truths):
            if bundle.form == "mask":
                counts += mask_counts(output, np.asarray(truth), values)
            else:
                distance += chamfer(output, truth)
    value = mean_iou(counts, values) if bundle.form == "mask" else (
        distance / rows
    )
    check = check_record(run, comparable, split, rows, metric, value)

    return check


def measure_items(
    bundle: Bundle, dataset: ItemDataset, run: Optional[dict], info: dict
) -> None:
    """
    版 2 のデータセットの一式の, 記録との比べを測る関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    dataset: ItemDataset
        版 2 のデータセット
    run: Optional[dict]
        実験の状態
    info: dict
        書き足す inference.yaml の中身 (check, uncertainty, notes)

    Returns
    ----------
    None
    """
    # 記録との比べは注意のためなので, 失敗しても作るのは止めない. 外れ方
    # は列の出力だけ
    info["uncertainty"] = None
    try:
        info["check"] = check_items_against_run(bundle, dataset, run)
    except Exception as error:
        info["check"] = None
        info["notes"].append(
            f"検証用の分割を推論して記録と比べられませんでした ({error})"
        )
    if info["check"] and info["check"]["matches"] is False:
        info["notes"].append(item_check_note(info["check"]))


def item_check_note(check: dict) -> str:
    """
    版 2 のデータセットの一式の, 記録と合わない評価の注意の文を返す関数

    Parameters
    ----------
    check: dict
        check_items_against_run の結果

    Returns
    ----------
    note: str
        注意の文
    """
    # 1 件の扱いの書き違いを疑う
    note = (
        f"{check['split']} の分割を一式で推論した {check['metric']} "
        f"{check['value']:.4g} が, 実験の記録 {check['recorded']:.4g} と"
        "合いません. 1 件の扱い (items の prepare, 点の数 points と "
        "target_points, 乱数の種 seed, 領域の値 values) が学習のスクリプト"
        "と同じか確かめてください (学習のスクリプトが指標を違う決まりで"
        "求めている場合も合いません)"
    )

    return note
