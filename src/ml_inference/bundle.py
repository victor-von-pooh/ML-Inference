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
from ml_eda import (
    DATASET_FORMAT_VERSION, ITEMS_FORMAT_VERSIONS, PLACEMENTS, Dataset,
    ItemDataset, load_dataset, load_items
)
from ml_modelcontrol import build_model, dump_spec, load_spec

from ml_inference.calibrate import CALIBRATE_METHODS, check_calibrate
from ml_inference.conclude import check_rules, plain
from ml_inference.items import (
    ITEM_FORMS, PREPARE_KINDS, check_items, check_point_count
)
from ml_inference.procedures import PROCEDURES, get_procedure, option_names
from ml_inference.ready import (
    READY_FORMS, READY_LIBRARIES, ReadyEmbedding, check_head, check_ready,
    ready_license
)
from ml_inference.tables import (
    check_table_columns, read_table_columns, table_width
)
from ml_inference.units import check_units
from ml_inference.weights import (
    INFO_KEYS, PLACES, layers_weights, place_weights, lookup_pretrained,
    node_items, node_keys, pin_spec, state_hash, unchecked_notes
)

# 推論の一式の形の版 (読める最も新しい版. 版 1 のデータセットの一式は
# 版 3, 版 2 のデータセットの一式は版 4, 雑音だけの一式は版 5 (表の出力は
# 版 7), 既製のモデルの一式は版 6, 並べるの一式は版 8, 群分けの一式は
# 版 9, 既製の埋め込みの群分けの一式は版 10, 異常の度合いの一式は版 11,
# 効果の一式は版 12 で書く)
BUNDLE_FORMAT_VERSION = 12
COLUMN_BUNDLE_VERSION = 3
ITEMS_BUNDLE_VERSION = 4
NOISE_BUNDLE_VERSION = 5
READY_BUNDLE_VERSION = 6
TABLE_BUNDLE_VERSION = 7
RANK_BUNDLE_VERSION = 8
GROUP_BUNDLE_VERSION = 9
READY_GROUP_BUNDLE_VERSION = 10
ANOMALY_BUNDLE_VERSION = 11
UPLIFT_BUNDLE_VERSION = 12

# 群分けの一式の版 (列の符号化か, 既製のモデルの埋め込み)
GROUP_VERSIONS = (GROUP_BUNDLE_VERSION, READY_GROUP_BUNDLE_VERSION)

# データセットを持たない一式の版
DATASETLESS_VERSIONS = (
    NOISE_BUNDLE_VERSION, READY_BUNDLE_VERSION, TABLE_BUNDLE_VERSION,
    RANK_BUNDLE_VERSION, GROUP_BUNDLE_VERSION, READY_GROUP_BUNDLE_VERSION,
    ANOMALY_BUNDLE_VERSION, UPLIFT_BUNDLE_VERSION
)

# inference.yaml の一番上の項目
TOP_KEYS = (
    "format_version", "name", "created_at", "role", "output", "target",
    "procedure", "model", "inputs", "dataset", "run", "conclusion",
    "decision", "check", "uncertainty", "notes", "items", "ready",
    "calibrate", "rank", "group", "anomaly", "uplift"
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
    "table": ("value",),
    "ranking": ("value",),
    "cluster": ("value",),
    "effect": ("value",),
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

# 一式を作るときに確かめる見本の件数 (雑音だけの一式は作る件数)
SAMPLE_ROWS = 8
NOISE_SAMPLE_ROWS = 2

# 雑音だけの一式の入力の出どころ
NOISE_SOURCES = ("state", "time")

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

# 並べるの出力の形を make_bundle に渡したときの誤り
RANKING_ONLY = (
    "並べるの出力の形 (ranking) の一式は make_rank_bundle で作ってください"
)

# 群分けの出力の形を make_bundle に渡したときの誤り
GROUP_ONLY = (
    "群分けの出力の形 (cluster) の一式は make_group_bundle で作ってください"
)

# 効果の出力の形を make_bundle に渡したときの誤り
EFFECT_ONLY = (
    "効果の出力の形 (effect) の一式は make_uplift_bundle で作ってください"
)


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
        # データセットの説明の task (既製のモデルの一式は出力の形から)
        if self.from_ready:
            task = "classification" if self.form == "class" else ""
            return task
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
        # データセットの説明の classes (既製のモデルの一式は出力の形の
        # classes. 重みの説明の種類の名前の表)
        if self.from_ready:
            classes = [
                str(c) for c in self.output_info.get("classes") or []
            ]
            return classes
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
            table, timeseries, image, text, pointcloud, audio, items
            (版 2 のデータセットの一式), noise (雑音だけの一式), ready
            (既製のモデルの一式), rank (並べるの一式), group (群分けの
            一式), anomaly (異常の度合いの一式), uplift (効果の一式)
        """
        # 版 2 のデータセットの一式はファイルの組, 雑音だけの一式は雑音
        if self.from_uplift:
            kind = "uplift"
            return kind
        if self.from_anomaly:
            kind = "anomaly"
            return kind
        if self.from_items:
            kind = "items"
            return kind
        if self.from_group:
            kind = "group"
            return kind
        if self.from_ready:
            kind = "ready"
            return kind
        if self.from_rank:
            kind = "rank"
            return kind
        if self.from_noise:
            kind = "noise"
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
    def from_noise(self) -> bool:
        """
        データセットを持たない雑音だけの一式かを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        noise: bool
            dataset が null の一式 (版 5) なら True
        """
        # dataset が null で, 既製のモデルの一式でも並べるの一式でも群分け
        # の一式でも異常の度合いの一式でも効果の一式でもない一式 (版 5 と,
        # 表の出力の版 7)
        noise = self.info.get("dataset") is None and not (
            self.from_ready or self.from_rank or self.from_group
            or self.from_anomaly or self.from_uplift
        )

        return noise

    @property
    def from_ready(self) -> bool:
        """
        学習しない既製のモデルの一式かを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        ready: bool
            ready (既製のモデルの出どころと前処理) を持つ一式 (版 6) なら
            True
        """
        # 版 6 の ready の有無
        ready = self.info.get("ready") is not None

        return ready

    @property
    def from_rank(self) -> bool:
        """
        利用者ごとに品物を並べる一式かを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        rank: bool
            rank (番号の表と品物の埋め込みの索引) を持つ一式 (版 8) なら
            True
        """
        # 版 8 の rank の有無
        rank = self.info.get("rank") is not None

        return rank

    @property
    def from_group(self) -> bool:
        """
        行を群に割り当てる一式かを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        group: bool
            group (列の符号化と群の中心) を持つ一式 (版 9) なら True
        """
        # 版 9 の group の有無
        group = self.info.get("group") is not None

        return group

    @property
    def from_anomaly(self) -> bool:
        """
        行ごとの異常の度合いを出す一式かを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        anomaly: bool
            anomaly (度合いの決め方と前処理のレシピ) を持つ一式 (版 11)
            なら True
        """
        # 版 11 の anomaly の有無
        anomaly = self.info.get("anomaly") is not None

        return anomaly

    @property
    def from_uplift(self) -> bool:
        """
        行ごとの効果を見積もる一式かを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        uplift: bool
            uplift (効果の決め方と前処理のレシピ) を持つ一式 (版 12) なら
            True
        """
        # 版 12 の uplift の有無
        uplift = self.info.get("uplift") is not None

        return uplift

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
    def embedding_width(self) -> Optional[int]:
        """
        既製のモデルの埋め込みの幅を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        width: Optional[int]
            埋め込みの一式は output.shape の幅, 既製の埋め込みの群分けの
            一式は group.width (他は None)
        """
        # 1 件の形は (幅,). 群分けは群の中心の幅
        if self.from_group:
            width = int(self.info["group"]["width"])
            return width
        shape = self.output_info.get("shape")
        width = int(shape[0]) if self.form == "vector" and isinstance(
            shape, list
        ) and shape else None

        return width

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
            image, table, ranking, cluster のどれか
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
            class, value, distribution なら True (異常の度合いの一式は
            False)
        """
        # 1 行に 1 つのクラスか値 (確率でも目的の列の値でもない異常の度合い
        # は除外)
        column = self.form in COLUMN_FORMS and not self.from_anomaly

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
        # 書かれていなければ, 出力の形の既定 (分類は logits, 値は value.
        # モデルを持たない一式も)
        kind = (self.info.get("model") or {}).get("kind") or FORM_KINDS[
            self.form
        ][0]

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
        # 書かれていない入力は, データの種類から選ぶ (雑音だけの一式は
        # 手順の中身)
        written = self.info.get("inputs") or {}
        default = "features" if self.kind in ("table", "timeseries") else (
            "state" if self.from_noise else "items"
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
        # 結論と意思決定は 1 行に 1 つのクラスか値が前提. 既製のモデルの
        # 一式は測るデータ無し. 異常の度合いは確率でなく, 効果は 2 つの予測
        # の差なので誤り
        if self.from_uplift:
            raise ValueError(
                f"効果の一式には {what} を使えません (使えるのは効果の上位"
                "から配る判断 {type: treat, budget} だけ)"
            )
        if self.from_anomaly:
            raise ValueError(
                f"異常の度合いの一式には {what} を使えません (使えるのは度合い"
                "の上位から回す振り分け {type: triage, capacity} だけ)"
            )
        if self.from_ready:
            raise ValueError(
                f"既製のモデルの一式はデータセットと外れ方の記録を持たない"
                f"ので, {what} を使えません"
            )
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
                model, state, sourced, self.spec, self.from_ready
            )
        else:
            model.load_state_dict(state)
        model.eval()
        if self.from_ready and self.form != "class":
            # 既製のモデルの埋め込みは分類の最後の層の入力
            model = ReadyEmbedding(model, self.embedding_width).eval()
        self.cache["model"] = model

        return model


def load_sourced(
    model: Any, state: dict, sourced: list, spec: dict, pinned: bool = False
) -> list[str]:
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
    spec: dict
        一式のモデルの定義
    pinned: bool = False
        hash の違いを誤りにするか (既製のモデルの一式は重みを hash で
        固定する)

    Returns
    ----------
    messages: list[str]
        hash の違ったノードの警告の文 (無ければ空)

    Raises
    ----------
    ValueError
        足りない重みが出どころから読むノードのものだけでない場合と, hash の
        違ったノードが外部の重み (weights) を重ねる場合 (外部の重みを
        読まずに組み立てるので, 出どころの重みと違う) と, 既製のモデルの
        一式で hash が違う場合
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

    # 出どころの重みと一式を作ったときの重みの比べ (違えば警告. 推論は続行.
    # 外部の重みを重ねたノードは, 重ねた重みが一式に無いので誤り)
    messages = []
    for record in sourced:
        node = str(record["node"])
        found = state_hash(built, node_keys(built, node))
        if found != record.get("sha256") and layers_weights(spec, node):
            raise ValueError(
                f"ノード {node} は出どころから読む記録ですが, 定義で外部の"
                "重み (weights) を重ねています. 一式は外部の重みを読まずに"
                "組み立てるので, 一式を作ったときの重みになりません. 今の"
                "ライブラリで一式を作り直してください (重ねたノードの重みは"
                "一式に含めます)"
            )
        if found != record.get("sha256") and pinned:
            raise ValueError(
                f"既製のモデルの一式のノード {node} の出どころから読んだ重み "
                f"({record.get('model') or ''} 版 "
                f"{record.get('pinned_revision')}) の hash が一式と違います "
                f"(一式 {str(record.get('sha256'))[:12]}…, 読んだもの "
                f"{found[:12]}…). 置いた重みのファイルが違うか, 出どころの"
                "重みが入れ替わりました"
            )
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
    # (距離に基づく度合いの一式はモデル無しの空の欄)
    distance = version == ANOMALY_BUNDLE_VERSION and isinstance(
        info.get("anomaly"), dict
    ) and info["anomaly"].get("score") == "distance"
    if model is None and distance:
        model = {}
    elif not isinstance(model, dict) or not model.get("spec") or (
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
    if info.get("dataset") is None and version not in DATASETLESS_VERSIONS:
        errors.append("dataset (データセットの説明の場所) が要ります")
    elif info.get("dataset") is not None and not info.get("dataset"):
        errors.append("dataset (データセットの説明の場所) が空です")

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
    if version in (NOISE_BUNDLE_VERSION, TABLE_BUNDLE_VERSION):
        errors.extend(check_noise(info))
    output = info.get("output") if isinstance(info.get("output"), dict) else {}
    if (output.get("form") == "table") != (version == TABLE_BUNDLE_VERSION):
        errors.append(
            f"出力の形 table は版 {TABLE_BUNDLE_VERSION} の雑音だけの一式"
            f"だけで, 版 {TABLE_BUNDLE_VERSION} の一式の出力の形は table "
            "だけです"
        )
    if info.get("ready") is not None and version not in (
        READY_BUNDLE_VERSION, READY_GROUP_BUNDLE_VERSION
    ):
        errors.append(
            f"ready を持つ一式は版 {READY_BUNDLE_VERSION} にしてください "
            f"(既製の埋め込みの群分けは版 {READY_GROUP_BUNDLE_VERSION})"
        )
    if version == READY_BUNDLE_VERSION:
        errors.extend(check_ready_info(info))
    if (output.get("form") == "ranking") != (version == RANK_BUNDLE_VERSION):
        errors.append(
            f"出力の形 ranking は版 {RANK_BUNDLE_VERSION} の並べるの一式だけ"
            f"で, 版 {RANK_BUNDLE_VERSION} の一式の出力の形は ranking だけ"
            "です"
        )
    if info.get("rank") is not None and version != RANK_BUNDLE_VERSION:
        errors.append(
            f"rank を持つ一式は版 {RANK_BUNDLE_VERSION} にしてください"
        )
    if version == RANK_BUNDLE_VERSION:
        # 並べるの一式の決まりは並べるのモジュールが持つ
        from ml_inference.ranking import check_rank_info

        errors.extend(check_rank_info(info))
    if (output.get("form") == "cluster") != (version in GROUP_VERSIONS):
        errors.append(
            f"出力の形 cluster は版 {GROUP_BUNDLE_VERSION} と "
            f"{READY_GROUP_BUNDLE_VERSION} の群分けの一式だけで, この版の一式"
            "の出力の形は cluster だけです"
        )
    if info.get("group") is not None and version not in GROUP_VERSIONS:
        errors.append(
            f"group を持つ一式は版 {GROUP_BUNDLE_VERSION} (既製の埋め込みは"
            f"版 {READY_GROUP_BUNDLE_VERSION}) にしてください"
        )
    if version in GROUP_VERSIONS:
        # 群分けの一式の決まりは群分けのモジュールが持つ
        from ml_inference.grouping import check_group_info

        errors.extend(check_group_info(info))
    if info.get("anomaly") is not None and version != ANOMALY_BUNDLE_VERSION:
        errors.append(
            f"anomaly を持つ一式は版 {ANOMALY_BUNDLE_VERSION} にしてください"
        )
    if version == ANOMALY_BUNDLE_VERSION:
        # 異常の度合いの一式の決まりは異常のモジュールの受け持ち
        from ml_inference.anomaly import check_anomaly_info

        errors.extend(check_anomaly_info(info))
    if (output.get("form") == "effect") != (version == UPLIFT_BUNDLE_VERSION):
        errors.append(
            f"出力の形 effect は版 {UPLIFT_BUNDLE_VERSION} の効果の一式だけ"
            f"で, 版 {UPLIFT_BUNDLE_VERSION} の一式の出力の形は effect だけ"
            "です"
        )
    if info.get("uplift") is not None and version != UPLIFT_BUNDLE_VERSION:
        errors.append(
            f"uplift を持つ一式は版 {UPLIFT_BUNDLE_VERSION} にしてください"
        )
    if version == UPLIFT_BUNDLE_VERSION:
        # 効果の一式の決まりは効果のモジュールの受け持ち
        from ml_inference.uplift import check_uplift_info

        errors.extend(check_uplift_info(info))

    return errors


def check_ready_info(info: dict) -> list[str]:
    """
    既製のモデルの一式 (版 6) の形を確かめる関数

    Parameters
    ----------
    info: dict
        inference.yaml の中身

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # ready が要り, データセット, items, 手順を持たない
    if info.get("ready") is None:
        return [f"版 {READY_BUNDLE_VERSION} の一式には ready (既製のモデル"
                "の出どころと前処理) が要ります"]
    errors = check_ready(info["ready"])
    held = [k for k in ("dataset", "items", "procedure", "decision",
                        "check", "uncertainty", "target")
            if info.get(k) is not None]
    if held:
        errors.append(
            f"既製のモデルの一式の {', '.join(held)} は null にしてください "
            "(データセットと正解を持たず, 学習しないため)"
        )

    # 出力は種類の名前の表を持つ分類か幅の決まった埋め込み, 入力は画像,
    # 重みは出どころから
    output = info.get("output") if isinstance(info.get("output"), dict) else {}
    classes = output.get("classes")
    shape = output.get("shape")
    if output.get("form") not in READY_FORMS:
        errors.append(
            f"既製のモデルの一式の出力の形は {', '.join(READY_FORMS)} だけ"
            "です"
        )
    elif output["form"] == "vector":
        if not (isinstance(shape, list) and len(shape) == 1 and isinstance(
            shape[0], int
        ) and not isinstance(shape[0], bool) and shape[0] >= 1):
            errors.append(
                "既製のモデルの埋め込みの一式の output.shape は [埋め込みの"
                "幅] にしてください"
            )
        if classes is not None:
            errors.append(
                "既製のモデルの埋め込みの一式は output.classes を持てません"
            )
        if info.get("conclusion"):
            errors.append(
                "既製のモデルの埋め込みの一式は結論の決め方の既定値を持てま"
                "せん"
            )
    elif not isinstance(classes, list) or not classes or not all(
        isinstance(c, str) for c in classes
    ) or len(set(classes)) != len(classes):
        errors.append(
            "既製のモデルの一式の output.classes は重ならない種類の名前の"
            "並びにしてください"
        )
    inputs = info.get("inputs") if isinstance(info.get("inputs"), dict) else {}
    if any(isinstance(item, dict) and item.get("source", "items") != "items"
           for item in inputs.values()):
        errors.append("既製のモデルの一式の入力の source は items だけです")
    model = info.get("model") if isinstance(info.get("model"), dict) else {}
    if model.get("target_scale") is not None:
        errors.append("既製のモデルの一式は model.target_scale を持てません")
    errors.extend(check_sourced(model))

    return errors


def check_sourced(model: dict) -> list[str]:
    """
    既製のモデルの重みが出どころから読む記録かを確かめる関数

    Parameters
    ----------
    model: dict
        inference.yaml の model

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 重みは写さず, 全てのノードを出どころから
    errors = []
    records = model.get("pretrained")
    if not isinstance(records, list) or not records or not all(
        isinstance(r, dict) and r.get("place") == "source" for r in records
    ):
        errors.append(
            "既製のモデルの一式の model.pretrained は, 出どころから読む "
            "(place: source) ノードの記録にしてください"
        )

    return errors


def check_noise(info: dict) -> list[str]:
    """
    雑音だけの一式 (版 5 と, 表の出力の版 7) の形を確かめる関数

    Parameters
    ----------
    info: dict
        inference.yaml の中身

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # データセットと items を持たず, 雑音から始まる手順だけ
    errors = []
    version = info.get("format_version")
    if info.get("dataset") is not None:
        errors.append(
            f"版 {version} の一式は dataset を持てません "
            "(雑音だけの一式だけを書く版)"
        )
    if info.get("items") is not None:
        errors.append(f"版 {version} の一式は items を持てません")
    procedure = info.get("procedure")
    name = procedure.get("name") if isinstance(procedure, dict) else None
    if name is None:
        errors.append("雑音だけの一式には procedure (雑音から始まる手順) が"
                      "要ります")
    elif str(name) in PROCEDURES and PROCEDURES[str(name)].start != "noise":
        errors.append(
            f"雑音だけの一式の手順 {name} は雑音から始まる手順にしてください"
        )

    # 入力は手順の中身と時刻だけ (書かない入力は中身)
    inputs = info.get("inputs") if isinstance(info.get("inputs"), dict) else {}
    others = [
        str(name) for name, item in inputs.items()
        if isinstance(item, dict)
        and item.get("source", "state") not in NOISE_SOURCES
    ]
    if others:
        errors.append(
            f"雑音だけの一式の入力 {', '.join(others)} は source を "
            f"{', '.join(NOISE_SOURCES)} にしてください (データの入力は"
            "持てません)"
        )
    if (info.get("conclusion") or {}) or info.get("decision"):
        errors.append("雑音だけの一式は結論の決め方と意思決定の既定値を"
                      "持てません")
    if (info.get("model") or {}).get("target_scale") is not None:
        errors.append("雑音だけの一式は model.target_scale を持てません")
    held = [k for k in ("check", "uncertainty", "target")
            if info.get(k) is not None]
    if held:
        errors.append(
            f"雑音だけの一式の {', '.join(held)} は null にしてください "
            "(比べる正解が無いため)"
        )

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
        if form == "table":
            errors.extend(check_table_output(output))
        elif output.get("columns") is not None:
            errors.append("output.columns は出力の形 table だけに使えます")

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


def check_table_output(output: dict) -> list[str]:
    """
    表の出力の形の列の符号化と 1 件の形を確かめる関数

    Parameters
    ----------
    output: dict
        inference.yaml の output (form が table)

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 列の符号化と, 1 件の形が符号化した行の幅か
    errors = check_table_columns(output.get("columns"))
    if errors:
        return errors
    width = table_width(output["columns"])
    if output.get("shape") != [width]:
        errors.append(
            f"表の出力の output.shape は符号化した行の幅 [{width}] にして"
            "ください"
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
    if info.get("model") is None:
        # 距離に基づく度合いの一式はモデル無し
        spec: dict = {"name": None, "inputs": {}, "outputs": {}}
    else:
        spec = load_spec(inside(root, str(info["model"]["spec"])))
        check_units(spec)
    if info.get("dataset") is None:
        # 雑音だけの一式と既製のモデルの一式は説明とレシピが空の
        # データセット
        dataset = Dataset(path=root, info={}, frames={}, recipe={})
    else:
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
    if info.get("model") is not None and not bundle.weights_path.is_file():
        raise FileNotFoundError(
            f"重み {bundle.weights_path} がありません (実験のディレクトリを"
            "動かしたか消した場合は, 一式を作り直してください)"
        )
    kind = (info.get("model") or {}).get("kind")
    form = bundle.form
    if info.get("output") is None:
        classify = bundle.task == "classification"
        if kind is not None and (kind == "value") == classify:
            raise ValueError(
                f"model.kind {kind} は {bundle.task} に使えません (分類は "
                "logits か probabilities, 回帰と予測は value)"
            )
    elif form in FORM_TASKS and not bundle.from_anomaly and (
        bundle.task not in FORM_TASKS[form]
    ):
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
    check_ready_spec(bundle)
    if bundle.from_rank:
        from ml_inference.ranking import check_rank_bundle

        check_rank_bundle(bundle)
    if bundle.from_group:
        from ml_inference.grouping import check_group_bundle

        check_group_bundle(bundle)
    if bundle.from_anomaly:
        from ml_inference.anomaly import check_anomaly_bundle

        check_anomaly_bundle(bundle)
    if bundle.from_uplift:
        from ml_inference.uplift import check_uplift_bundle

        check_uplift_bundle(bundle)
    if info.get("calibrate") is not None:
        classes = bundle.classes if (
            bundle.form == "class" and not bundle.from_ready
        ) else None
        errors = check_calibrate(info["calibrate"], classes)
        if errors:
            raise ValueError("inference.yaml の誤り: " + " / ".join(errors))
    names = set(node_items(spec))
    for record in (info.get("model") or {}).get("pretrained") or []:
        if record["node"] not in names:
            raise ValueError(
                f"model.pretrained の {record['node']} はモデルの定義の"
                "ノードにありません"
            )

    return bundle


def check_ready_spec(bundle: Bundle) -> None:
    """
    既製のモデルの一式の ready と定義のノードが合うかを確かめる関数

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
        定義に model のノードが無い場合と, ノードの params の name と
        weights が ready の model と weights と違う場合
    """
    # 記録した出どころと, 組み立てて読む重みを同じに
    if not bundle.from_ready:
        return
    ready = bundle.info["ready"]
    node = (bundle.spec.get("nodes") or {}).get("model")
    params = (node.get("params") or {}) if isinstance(node, dict) else {}
    if not isinstance(node, dict):
        raise ValueError("既製のモデルの一式の定義に model のノードが"
                         "ありません")
    for key, name in (("model", "name"), ("weights", "weights")):
        if params.get(name) != ready[key]:
            raise ValueError(
                f"ready.{key} の {ready[key]} が定義の model のノードの "
                f"{name} ({params.get(name)}) と違います"
            )


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
    if items is None and info.get("format_version") == ITEMS_BUNDLE_VERSION:
        raise ValueError(
            f"版 {ITEMS_BUNDLE_VERSION} の一式には items (版 2 のデータ"
            "セットの 1 件の扱い) が要ります"
        )
    if items is None:
        return
    if info.get("format_version") != ITEMS_BUNDLE_VERSION:
        raise ValueError(
            f"items を持つ一式は版 {ITEMS_BUNDLE_VERSION} にしてください"
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
    dataset: Union[str, Path, None], weights: Union[str, Path],
    inputs: Optional[dict] = None, output: Optional[str] = None,
    kind: Optional[str] = None,
    target_scale: Union[dict, str, None] = None,
    conclusion: Optional[dict] = None, decision: Optional[dict] = None,
    name: Optional[str] = None, copy_weights: bool = True,
    overwrite: bool = False, role: Optional[str] = None,
    form: Optional[str] = None, output_options: Optional[dict] = None,
    target: Optional[dict] = None, procedure: Optional[dict] = None,
    pretrained: Optional[dict] = None, items: Optional[dict] = None,
    calibrate: Optional[str] = None
) -> dict:
    """
    推論の一式を作り, 見本を推論できることを確かめる関数

    Parameters
    ----------
    path: Union[str, Path]
        作るディレクトリ
    spec: Union[str, Path, dict]
        モデルの定義 (ML-ModelControl の形) かそのファイル
    dataset: Union[str, Path, None]
        ML-EDA で書き出したデータセットのディレクトリ (版 1 か, 版 2 の
        ファイルの組). None は雑音だけの一式 (データセットを持たず, 件数
        と種で推論する. 版 5. 表の出力は版 7)
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
        predict. 雑音だけの一式は generate)
    form: Optional[str] = None
        出力の形 (class, value, distribution, mask, point_cloud, vector,
        text, image, table. 省略するとタスクから. 雑音だけの一式は要る.
        table は雑音だけの一式だけ)
    output_options: Optional[dict] = None
        出力の形ごとの欄 (distribution の quantiles, 雑音から始まる手順の
        shape, mask の classes, table の columns (列の符号化の並びか, その
        JSON のファイル. shape は省くと符号化した行の幅))
    target: Optional[dict] = None
        正解の種類 (kind) と出どころ (origin. ML-EDA の版 2 の
        target.origin の形). 省略すると列の出力は kind: column, 他は null
    procedure: Optional[dict] = None
        繰り返しの手順 (name, steps, seed, options)
    pretrained: Optional[dict] = None
        ノードの名前 (部分モデルの中のノードは <外のノード>.<中のノード>)
        ごとの重みのライセンスの情報 (ML-ModelControl の pretrained_info
        の形). 省略すると pretrained_info で読む
    items: Optional[dict] = None
        版 2 のデータセットの 1 件の扱い (input, target, prepare,
        options, values, restore. 省いた欄はデータセットから. 点を選ぶ
        扱い方は options.points が要る)
    calibrate: Optional[str] = None
        分類の確率の補正の選び方 (platt か isotonic. 検証用の分割で当て
        はめる). 省くと補正しない

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

    # 確率の補正の選び方
    if calibrate is not None and calibrate not in CALIBRATE_METHODS:
        raise ValueError(
            f"calibrate は {', '.join(CALIBRATE_METHODS)} のどれかにして"
            f"ください (今は {calibrate})"
        )

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
    spec_dict = load_spec(spec)
    check_units(spec_dict)
    item_info = None
    data_dir = None
    if dataset is None:
        # 雑音だけの一式 (データセットを持たず, 説明とレシピが空)
        full: Union[Dataset, ItemDataset] = Dataset(
            path=final, info={}, frames={}, recipe={}
        )
        kinds, inputs = noise_fields(
            spec_dict, inputs, role, form, output_options, target,
            procedure, target_scale, items
        )
    else:
        data_dir = Path(dataset).expanduser().resolve()
        if not data_dir.is_dir():
            raise ValueError(
                f"データセット {data_dir} はディレクトリではありません"
            )
        full = read_dataset(data_dir, frames=True)
    if isinstance(full, ItemDataset):
        if target_scale is not None:
            raise ValueError("版 2 のデータセットの一式に target_scale は"
                             "使えません")
        kinds, item_info = item_fields(
            full, role, form, output_options, target, items
        )
        check_point_count(spec_dict, inputs, item_info)
    elif data_dir is not None:
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
        if data_dir is not None:
            (writing / "dataset").mkdir()
            shutil.copy(data_dir / "dataset.yaml", writing / "dataset")
        if data_dir is not None and item_info is None:
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
            "format_version": bundle_version(
                data_dir, item_info, kinds["output"]["form"]
            ),
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
            "dataset": None if data_dir is None else "dataset",
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
        if calibrate is not None and not (
            bundle.column_output and bundle.form == "class"
        ):
            raise ValueError(
                "確率の補正 (calibrate) は, 版 1 のデータセットの分類の一式"
                f"だけに使えます (今は出力の形 {bundle.form})"
            )
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
        elif bundle.from_noise:
            check_noise_sample(bundle)
            info["check"] = None
            info["uncertainty"] = None
        elif bundle.column_output:
            check_sample(bundle, full)
            measure_column(bundle, full, run, info, calibrate)
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


def plan_weights(
    spec: dict, source: Path, pretrained: Optional[dict], copy_weights: bool
) -> tuple[dict, list, Optional[dict], list[str]]:
    """
    学習済みの重みを読むノードの置き方と, 一式に写す重みを決める関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    source: Path
        重みのファイル
    pretrained: Optional[dict]
        ノードの名前ごとの重みのライセンスの情報 (省略すると
        pretrained_info で読む)
    copy_weights: bool
        重みを一式に写すか

    Returns
    ----------
    plan: tuple[dict, list, Optional[dict], list[str]]
        版を固定した定義, model.pretrained の記録, 一式に写す重み (丸ごと
        写すなら None), 注意の文
    """
    # 学習済みの重みを読むノードの置き方 (無ければ丸ごとの写し)
    infos, notes = lookup_pretrained(spec, pretrained)
    records: list = []
    kept = None
    if infos:
        spec = pin_spec(spec, infos)
        state = torch.load(source, map_location="cpu", weights_only=True)
        fresh = build_model(spec, seed=0, load_weights=False).state_dict()
        records, kept, placed = place_weights(
            infos, state, fresh, copy_weights, spec
        )
        notes.extend(placed)

    # ライセンスを確かめずに一式に写す重みの注意 (写す場合だけ. 置き方の
    # 注意でライセンスを確かめるよう書いたノードは除く)
    if copy_weights:
        noted = {
            str(r["node"]) for r in records if not r["redistributable"]
        }
        looked = None if pretrained is None else {
            str(node) for node in pretrained
        }
        notes.extend(unchecked_notes(spec, noted, looked))
    plan = (spec, records, kept, notes)

    return plan


def make_ready_bundle(
    path: Union[str, Path], model: str, weights: str,
    library: str = "torchvision", sha256: Optional[str] = None,
    name: Optional[str] = None, license: Optional[dict] = None,
    conclusion: Optional[dict] = None, overwrite: bool = False,
    form: str = "class"
) -> dict:
    """
    学習しない既製のモデルの一式を, 重みの版と hash を固定して作る関数

    重みは一式に写さず, 使うときに出どころ (ライブラリ) から読む (利用者
    が置いたファイルか, 初回に取りに行ったもの)

    Parameters
    ----------
    path: Union[str, Path]
        作るディレクトリ
    model: str
        モデルの名前 (resnet18 など)
    weights: str
        重みの名前 (IMAGENET1K_V1 など. DEFAULT はその時の名前に直す)
    library: str = "torchvision"
        出どころのライブラリ (READY_LIBRARIES の名前)
    sha256: Optional[str] = None
        決めておいた重みの hash (渡すと, 読んだ重みと違えば作らない)
    name: Optional[str] = None
        一式の名前 (省略するとディレクトリの名前)
    license: Optional[dict] = None
        重みのライセンスの情報 (ML-ModelControl の pretrained_info の
        形). 省略すると pretrained_info で読む
    conclusion: Optional[dict] = None
        結論の決め方の既定値 (種類の予測の一式だけ)
    overwrite: bool = False
        既にある一式を置き換えるか
    form: str = "class"
        出力の形 (class は種類の予測, vector は分類の最後の層の手前の
        出力の埋め込み)

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
        出どころ, モデル, 重みの名前, 出力の形が無い場合と, 重みを読めない
        場合, hash が sha256 と違う場合, 埋め込みを取り出せない場合
    """
    # 置き換えられるのは, 前に作った一式だけ
    final = Path(path).expanduser().resolve()
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"{final} は既にあります")
        if not (final / "inference.yaml").is_file():
            raise FileExistsError(f"{final} は推論の一式ではないので置き換えません")
    if library not in READY_LIBRARIES:
        raise ValueError(
            f"既製のモデルの出どころ {library} はありません (出どころ: "
            f"{', '.join(READY_LIBRARIES)})"
        )
    if form not in READY_FORMS:
        raise ValueError(
            f"既製のモデルの一式の出力の形 {form} はありません (形: "
            f"{', '.join(READY_FORMS)})"
        )
    if form == "vector" and conclusion:
        raise ValueError(
            "既製のモデルの埋め込みの一式は結論の決め方を持てません"
        )

    # 重みの説明 (種類の名前の表と前処理) と, 1 つのノードの定義
    found = READY_LIBRARIES[library](model, weights)
    crop = int(found["preprocess"]["crop"])
    spec = ready_spec(found, model)

    # 出どころの重みを読んで hash を求める (初回は取りに行く)
    record, ready, notes, built = ready_record(
        spec, found, library, model, license
    )
    if sha256 is not None and sha256 != record["sha256"]:
        raise ValueError(
            f"{library} の {model} の重み {record['pinned_revision']} の "
            f"hash が決めておいたものと違います (決めたもの "
            f"{sha256[:12]}…, 読んだもの {record['sha256'][:12]}…)"
        )

    # 隣の一時ディレクトリに作り, 確かめてから入れ替える (重みは空)
    writing = final.with_name(f".{final.name}.writing")
    if writing.exists():
        shutil.rmtree(writing)
    writing.mkdir(parents=True)
    try:
        dump_spec(spec, writing / "model_spec.yaml")
        torch.save({}, writing / "final_model.pt")
        info = {
            "format_version": READY_BUNDLE_VERSION,
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "role": "predict",
            "output": ready_output(form, found, built, crop),
            "target": None,
            "procedure": None,
            "model": {
                "spec": "model_spec.yaml",
                "weights": "final_model.pt",
                "output": None,
                "kind": "logits" if form == "class" else "value",
                "target_scale": None,
                "pretrained": [record],
            },
            "inputs": {},
            "dataset": None,
            "run": None,
            "conclusion": plain(conclusion or {}),
            "decision": None,
            "check": None,
            "uncertainty": None,
            "notes": notes,
            "ready": ready,
        }
        (writing / "inference.yaml").write_text(
            yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
            encoding="utf-8"
        )
        bundle = load_bundle(writing)
        if conclusion:
            errors = check_rules(conclusion, bundle.task, bundle.classes)
            if errors:
                raise ValueError("結論の決め方の誤り: " + " / ".join(errors))
        check_ready_sample(bundle)
    except Exception:
        shutil.rmtree(writing, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(final)
    writing.rename(final)

    return info


def ready_spec(found: dict, model: str) -> dict:
    """
    既製のモデルの 1 つのノード (model) の定義を作る関数

    Parameters
    ----------
    found: dict
        出どころの重みの説明 (READY_LIBRARIES の関数の結果)
    model: str
        モデルの名前

    Returns
    ----------
    spec: dict
        前処理の後の画像を入力にし, 出どころの部品の出力を logits にする
        定義

    Raises
    ----------
    ValueError
        部品が登録されていない場合と, 定義が誤りの場合
    """
    # 入力は前処理の後の (3, crop, crop) の画像
    crop = int(found["preprocess"]["crop"])
    spec = {
        "format_version": 1,
        "name": model,
        "inputs": {"image": {"shape": [1, 3, crop, crop]}},
        "nodes": {
            "model": {
                "unit": found["unit"], "params": dict(found["params"]),
                "inputs": "image",
            }
        },
        "outputs": {"logits": "model"},
    }
    spec = load_spec(spec)
    check_units(spec)

    return spec


def ready_output(form: str, found: dict, built: Any, crop: int) -> dict:
    """
    既製のモデルの一式の output を作る関数

    Parameters
    ----------
    form: str
        出力の形 (class か vector)
    found: dict
        出どころの重みの説明 (READY_LIBRARIES の関数の結果)
    built: Any
        出どころの重みを読んで組み立てたモデル
    crop: int
        前処理の後の画像の辺

    Returns
    ----------
    output: dict
        種類の予測は種類の名前の表, 埋め込みは 1 件の形 ([幅])

    Raises
    ----------
    ValueError
        埋め込みを取り出せない場合
    """
    # 埋め込みの幅は, 最後の層の入力を確かめた幅 (確かめた後に hook は
    # 外す)
    if form == "class":
        output = {"form": "class", "classes": list(found["classes"])}
        return output
    embedder = ReadyEmbedding(built.eval())
    try:
        width = check_head(embedder, crop)
    finally:
        embedder.release()
    output = {"form": "vector", "shape": [width]}

    return output


def ready_record(
    spec: dict, found: dict, library: str, model: str,
    license: Optional[dict]
) -> tuple[dict, dict, list[str]]:
    """
    既製のモデルの重みを出どころから読み, 置き方の記録と ready を作る関数

    Parameters
    ----------
    spec: dict
        1 つのノード (model) の定義
    found: dict
        出どころの重みの説明 (READY_LIBRARIES の関数の結果)
    library: str
        出どころのライブラリ
    model: str
        モデルの名前
    license: Optional[dict]
        重みのライセンスの情報 (省略すると pretrained_info で読む)

    Returns
    ----------
    result: tuple[dict, dict, list[str], Any]
        model.pretrained の記録, ready, 注意の文, 組み立てたモデル

    Raises
    ----------
    ValueError
        出どころの重みを読めない場合
    """
    # 組み立てると部品が出どころの重みを読む (外部の重みのファイルは無い)
    version = str(found["params"].get("weights") or "")
    try:
        built = build_model(spec, seed=0, load_weights=False)
        state = built.state_dict()
    except Exception as error:
        raise ValueError(
            f"{library} の {model} の重み {version} を読めません "
            f"({type(error).__name__}: {error}). 初回はネットワークから"
            f"取りに行きます. ネットワークを使えない場合は "
            f"{found.get('file') or '重みのファイル'} を PyTorch の置き場 "
            "(torch.hub.get_dir() の checkpoints) に置いてください"
        ) from error
    keys = node_keys(state, "model")
    infos, notes = lookup_pretrained(
        spec, None if license is None else {"model": license}
    )
    info = infos.get("model") or {
        "unit": found["unit"], "status": "unknown", "redistributable": False,
        "source": "読めなかった (ライセンスの情報を読めない)",
    }

    # 版は重みの名前 (名前ごとに重みのファイルが決まる). 重みは写さない
    record: dict[str, Any] = {"node": "model"}
    record.update({k: info.get(k) for k in INFO_KEYS})
    record["model"] = model
    record["revision"] = version
    record["pinned_revision"] = info.get("pinned_revision") or version
    record["redistributable"] = bool(info.get("redistributable"))
    record["place"] = "source"
    record["sha256"] = state_hash(state, keys)
    record["keys"] = len(keys)
    ready = {
        "library": library,
        "model": model,
        "weights": version,
        "input": found["input"],
        "preprocess": dict(found["preprocess"]),
        "url": found.get("url"),
        "file": found.get("file"),
        "license": ready_license(info),
    }
    ready["license"]["redistributable"] = record["redistributable"]
    notes = list(notes) + [
        f"重みは一式に写さず, 使うときに {library} の {model} の "
        f"{version} を出どころから読み, hash "
        f"({record['sha256'][:12]}…) で確かめます"
    ]
    if info.get("status") != "commercial":
        notes.append(
            f"重みのライセンスは {info.get('license') or '分からない'} "
            f"({info.get('status') or 'unknown'}) です. 商用で使う前に配布"
            "元で確かめてください"
        )
    if info.get("gated"):
        notes.append("重みは申請制です. 使う人ごとに配布元で申請してください")

    return record, ready, notes, built


def check_ready_sample(bundle: Bundle) -> None:
    """
    既製のモデルの一式で, 前処理の後の大きさの画像を推論できるかを
    確かめる関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        推論できない場合と, 出力が種類の名前の表か埋め込みの幅と合わない
        場合
    """
    # 予測の関数はこのモジュールを読むので, 使うときに読み込む
    from ml_inference.predict import shape_inputs

    # 前処理の後の (3, crop, crop) の 0 の画像を数件 (埋め込みは読んだ
    # 一式でも最後の層を確かめる)
    crop = int(bundle.info["ready"]["preprocess"]["crop"])
    images = np.zeros((NOISE_SAMPLE_ROWS, 3, crop, crop), dtype=np.float32)
    tensors = shape_inputs(bundle, {"items": images}, NOISE_SAMPLE_ROWS)
    sample_inference(bundle, tensors)
    if bundle.form == "vector":
        check_head(bundle.model(), crop)


def bundle_version(
    data_dir: Optional[Path], item_info: Optional[dict], form: str
) -> int:
    """
    書く一式の形の版を決める関数

    Parameters
    ----------
    data_dir: Optional[Path]
        データセットのディレクトリ (雑音だけの一式は None)
    item_info: Optional[dict]
        版 2 のデータセットの 1 件の扱い (版 1 のデータセットは None)
    form: str
        出力の形

    Returns
    ----------
    version: int
        表の出力は 7, 他の雑音だけの一式は 5, 版 2 のデータセットの一式は
        4, 他は 3
    """
    # 読む側が知る古い版で書けるものは古い版で (表の出力の形を持つ版は
    # 雑音だけの一式でなくても 7 にし, 読むときの誤りにする)
    if form == "table":
        version = TABLE_BUNDLE_VERSION
    elif data_dir is None:
        version = NOISE_BUNDLE_VERSION
    elif item_info is not None:
        version = ITEMS_BUNDLE_VERSION
    else:
        version = COLUMN_BUNDLE_VERSION

    return version


def noise_fields(
    spec: dict, inputs: Optional[dict], role: Optional[str],
    form: Optional[str], options: Optional[dict], target: Optional[dict],
    procedure: Optional[dict], target_scale: Union[dict, str, None],
    items: Optional[dict]
) -> tuple[dict, dict]:
    """
    雑音だけの一式の受け持ち, 出力の形, 正解の項目と入力を決める関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    inputs: Optional[dict]
        モデルの入力ごとの出どころ (入力が 1 つなら省略可)
    role: Optional[str]
        受け持ち (省略すると generate)
    form: Optional[str]
        出力の形
    options: Optional[dict]
        出力の形ごとの欄 (shape が要る)
    target: Optional[dict]
        正解の種類と出どころ (使えない. 正解は null)
    procedure: Optional[dict]
        繰り返しの手順
    target_scale: Union[dict, str, None]
        目的の値の標準化 (使えない)
    items: Optional[dict]
        版 2 のデータセットの 1 件の扱い (使えない)

    Returns
    ----------
    fields: dict
        role, output, target
    inputs: dict
        入力ごとの出どころ (省いた 1 つの入力は state)

    Raises
    ----------
    ValueError
        データセットの要る項目を渡した場合と, 手順か出力の形が無い場合
    """
    # データセットの要る項目は使えない
    if target_scale is not None or items is not None or target is not None:
        raise ValueError(
            "雑音だけの一式 (データセット無し) に target_scale, items, "
            "target は使えません"
        )
    if not procedure:
        raise ValueError(
            "雑音だけの一式 (データセット無し) には雑音から始まる手順 "
            "(procedure) が要ります"
        )
    if form is None:
        raise ValueError(
            "雑音だけの一式 (データセット無し) では出力の形 (form) を選んで"
            "ください"
        )
    if form == "ranking":
        raise ValueError(RANKING_ONLY)
    if form == "cluster":
        raise ValueError(GROUP_ONLY)
    if form == "effect":
        raise ValueError(EFFECT_ONLY)

    # 出力の形 (受け持ちは省くと生成する. 正解は無し. 表の列の符号化は
    # 一式に写し, 1 件の形は省くと符号化した行の幅)
    output = {"form": form, **plain(dict(options or {}))}
    if "form" in (options or {}):
        raise ValueError("output_options に form は書けません (form で選ぶ)")
    if form == "table":
        if output.get("columns") is None:
            raise ValueError(
                "表の出力の形 (table) には output_options の columns (列の"
                "符号化の並びか, その JSON のファイル) が要ります"
            )
        output["columns"] = plain(read_table_columns(output["columns"]))
        errors = check_table_columns(output["columns"])
        if errors:
            raise ValueError("列の符号化の誤り: " + " / ".join(errors))
        output.setdefault("shape", [table_width(output["columns"])])
    fields = {
        "role": role or "generate",
        "output": output,
        "target": None,
    }
    errors = check_kinds(fields)
    if errors:
        raise ValueError("出力の形の誤り: " + " / ".join(errors))

    # 入力が 1 つで省いたら手順の中身
    plan = dict(inputs or {})
    if not plan and len(spec["inputs"]) == 1:
        plan = {next(iter(spec["inputs"])): {"source": "state"}}

    return fields, plan


def check_noise_sample(bundle: Bundle) -> None:
    """
    雑音だけの一式の手順で数件を作り, 出力の形を確かめる関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        手順を動かせない場合と, 出力の形が合わない場合
    """
    # 予測の関数はこのモジュールを読むので, 使うときに読み込む
    from ml_inference.predict import shape_inputs

    # 種は手順の種で数件
    tensors = shape_inputs(bundle, {}, NOISE_SAMPLE_ROWS)
    sample_inference(bundle, tensors)


def sample_inference(bundle: Bundle, tensors: list) -> None:
    """
    見本の入力で推論して読み, 失敗を分かりやすい誤りにする関数

    Parameters
    ----------
    bundle: Bundle
        作っている一式
    tensors: list
        モデルの入力の順のテンソル

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        推論か出力の読み方に失敗した場合
    """
    # 予測の関数はこのモジュールを読むので, 使うときに読み込む
    from ml_inference.predict import decode, run_model

    # 形の誤りは分かりやすい文に (つなぎ方の誤りの文はそのまま. 長い誤り
    # の文は始めだけ)
    try:
        outputs = run_model(bundle, tensors, 64, "cpu")
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


def measure_column(
    bundle: Bundle, dataset: Dataset, run: Optional[dict], info: dict,
    calibrate: Optional[str] = None
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
        書き足す inference.yaml の中身 (check, uncertainty, notes.
        補正すれば calibrate)
    calibrate: Optional[str] = None
        確率の補正の選び方 (省くと補正しない)

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        補正するのに, 検証用の分割と目的の列が無い場合
    """
    # 記録との比べは注意のためなので, 失敗しても作るのは止めない (記録は
    # 補正の前のモデルの指標なので, 補正の前に比べる)
    try:
        info["check"] = check_against_run(bundle, dataset, run)
    except Exception as error:
        info["check"] = None
        info["notes"].append(
            f"検証用の分割を推論して記録と比べられませんでした ({error})"
        )
    if info["check"] and info["check"]["matches"] is False:
        info["notes"].append(check_note(info["check"]))

    # 確率の補正 (頼まれたものなので, できなければ止める). 外れ方は補正
    # した確率で測る
    from ml_inference.uncertainty import (
        fit_column_calibration, measure_uncertainty
    )

    if calibrate is not None:
        info["calibrate"] = fit_column_calibration(
            bundle, dataset, calibrate
        )

    # 意思決定のための外れ方 (失敗しても作るのは止めない)
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
    if form == "table":
        raise ValueError(
            "表の出力の形 (table) は雑音だけの一式 (データセット無し) だけに"
            "使えます"
        )
    if form == "ranking":
        raise ValueError(RANKING_ONLY)
    if form == "cluster":
        raise ValueError(GROUP_ONLY)
    if form == "effect":
        raise ValueError(EFFECT_ONLY)
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
    from ml_inference.predict import shape_inputs

    # 前処理を済ませた学習用の最初の数件 (分割の名前は train を優先)
    split = "train" if "train" in dataset.frames else next(
        iter(dataset.frames)
    )
    sources = split_sources(bundle, dataset, split, 0, SAMPLE_ROWS)
    sample_inference(bundle, shape_inputs(bundle, sources))


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
    from ml_inference.items import item_form, resolve_items

    described = dataset.info.get("target") or {}
    kind = item_form(dataset)
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
