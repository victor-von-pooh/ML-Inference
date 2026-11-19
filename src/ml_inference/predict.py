from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from ml_eda import Data, load_data

from ml_inference.bundle import Bundle, load_bundle
from ml_inference.procedures import (
    get_procedure, make_generator, run_procedure, start_noise
)

# 予測しなかった行の印
OK, SKIPPED = "ok", "skipped"

# 予測の値が数でない行の理由
NOT_FINITE = "予測の値が数ではありません (入力に無限大などが無いか)"

# 表として前処理をかける種類
TABLE_KINDS = ("table", "timeseries")

# 1 件の出力を配列のまま持つ出力の形 (書き出しは artifacts/)
ARRAY_FORMS = ("mask", "point_cloud", "vector", "image")

# 繰り返しの手順の中身と時刻の入力 (データからは作らない)
STEP_SOURCES = ("state", "time")


def lag_settings(bundle: Bundle) -> dict:
    """
    レシピの過去の値の手順が読む列, 系列を分ける列, 時刻の列を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    settings: dict
        columns (ずらす列), by (系列を分ける列), time (時刻でずらす場合の
        時刻の列) の集合
    """
    # lag の手順ごとの列を集める
    settings: dict = {"columns": set(), "by": set(), "time": set()}
    for spec in bundle.dataset.recipe.get("steps", []):
        if spec["op"] != "lag":
            continue
        settings["columns"].update(spec.get("state", {}).get("columns", []))
        by = spec.get("by") or []
        settings["by"].update([by] if isinstance(by, str) else by)
        if spec.get("time"):
            settings["time"].add(spec["time"])

    return settings


def series_keys(frame: pd.DataFrame, settings: dict) -> pd.Series:
    """
    行ごとの系列の名前を返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        渡したデータ
    settings: dict
        lag_settings の結果

    Returns
    ----------
    keys: pd.Series
        系列を分ける列の値をつないだ文字 (列が無ければ 1 つの系列)
    """
    # 系列を分ける列の値のまま (表記のゆれはそろえて渡す)
    by = [c for c in settings["by"] if c in frame.columns]
    keys = frame[by].astype(str).agg("\u0000".join, axis=1) if by else (
        pd.Series("", index=frame.index)
    )

    return keys


def read_source(
    bundle: Bundle, source: Union[str, Path, pd.DataFrame, Data],
    options: Optional[dict] = None
) -> Data:
    """
    推論するデータを, 学習のときと同じ読み込みで読む関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    source: Union[str, Path, pd.DataFrame, Data]
        ファイルかディレクトリ, 表, ML-EDA で読んだデータ
    options: Optional[dict] = None
        読み込みの設定 (学習のときの設定に上書きする. 場所を渡した場合
        だけ使う)

    Returns
    ----------
    data: Data
        読み込んだデータ

    Raises
    ----------
    ValueError
        表で渡せない種類の場合と, 文の列が無い場合
    """
    # 読んだデータと表はそのまま (文は文の列を持つ表)
    kind = bundle.kind
    if isinstance(source, Data):
        return source
    if isinstance(source, pd.DataFrame):
        frame = source.reset_index(drop=True)
        if kind in TABLE_KINDS:
            return Data(kind=kind, frame=frame)
        column = bundle.dataset.info.get("text_column")
        if kind != "text" or not column:
            raise ValueError(
                f"{kind} のデータは表では渡せません. ファイルかディレクトリの"
                "場所を渡してください"
            )
        if column not in frame.columns:
            raise ValueError(f"文の列 {column} がありません")
        # 欠損の文は空の文 (ML-EDA の読み込みと同じ. 空の文は予測しない)
        texts = frame[column].fillna("").astype(str)
        return Data(
            kind="text", frame=frame, roles={"text": column},
            item_loader=lambda i: texts.iloc[i]
        )

    # 場所は学習のときの読み込みと設定で. 読めなければ形式を判定し直す
    # (学習は csv で, 推論は parquet など)
    path = Path(source).expanduser()
    if not path.exists():
        raise ValueError(f"{path} がありません")
    if path.is_dir() and kind in TABLE_KINDS:
        raise ValueError(
            f"{path} はディレクトリです. {kind} の一式にはファイルを選んで"
            "ください"
        )
    recorded = bundle.dataset.info.get("source") or {}
    merged = {**(recorded.get("options") or {}), **(options or {})}
    try:
        data = load_data(path, loader=recorded.get("loader"), **merged)
    except Exception as first:
        try:
            data = load_data(path, **merged)
        except Exception:
            raise ValueError(
                f"{path.name} を読めません ({first})"
            ) from first

    return data


def features_of(
    bundle: Bundle, data: Data
) -> tuple[np.ndarray, list[str]]:
    """
    表の列に前処理をかけ, 学習に使う列の配列と行ごとの問題を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    data: Data
        読み込んだデータ

    Returns
    ----------
    x: np.ndarray
        (件数, 列の数) の float32 の配列
    reasons: list[str]
        行ごとの予測できない理由 (予測できる行は空の文字)
    """
    # 当てはめた値のままの前処理 (行は増減しない). 一部の行の値で全体が
    # 失敗した場合は, 行ごとにかけて失敗した行を飛ばす (過去の値の列は
    # 前の行が要るので行ごとにはかけない)
    try:
        frame = bundle.dataset.transform(data.frame)
        failed = [""] * len(frame)
        # 行を落とす手順で落ちた行 (ML-EDA が手順の名前を返す場合)
        for row, op in (frame.attrs.get("dropped") or {}).items():
            failed[int(row)] = (
                f"前処理の手順 {op} で除く行です (学習でも使わない行)"
            )
    except ValueError as error:
        steps = [s["op"] for s in bundle.dataset.recipe.get("steps", [])]
        if "lag" in steps or "前処理の後に, 学習に使う列が足りません" in str(
            error
        ) or "にかける列がありません" in str(error):
            raise
        frame, failed = transform_rows(bundle, data.frame)
    try:
        x = frame.to_numpy(dtype=np.float32, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"前処理の後に数でない列があります ({error}). レシピで数に"
            "してください"
        ) from error
    reasons = []
    for row in frame.itertuples(index=False):
        lost = [name for name, value in zip(frame.columns, row)
                if pd.isna(value)]
        reasons.append(
            "前処理の後に欠損: " + ", ".join(lost[:5])
            + (" など" if len(lost) > 5 else "") if lost else ""
        )
    reasons = [a or b for a, b in zip(failed, reasons)]

    return x, reasons


def transform_rows(
    bundle: Bundle, frame: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """
    1 行ずつ前処理をかけ, 失敗した行を欠損と理由にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    frame: pd.DataFrame
        元の表

    Returns
    ----------
    result: pd.DataFrame
        学習に使う列の表 (失敗した行は欠損)
    failed: list[str]
        行ごとの失敗の理由 (成功した行は空の文字)
    """
    # 行ごとの前処理 (遅いので, 全体で失敗したときだけ使う)
    parts = []
    failed = []
    columns = bundle.dataset.features
    for index in range(len(frame)):
        try:
            part = bundle.dataset.transform(frame.iloc[[index]])
            failed.append("")
        except ValueError as error:
            part = pd.DataFrame(
                [[np.nan] * len(columns)], columns=columns,
                index=frame.index[[index]]
            )
            failed.append(f"前処理できません ({error})")
        parts.append(part)
    result = pd.concat(parts) if parts else pd.DataFrame(columns=columns)

    return result, failed


def items_of(
    bundle: Bundle, data: Data
) -> tuple[Optional[np.ndarray], list[str]]:
    """
    1 件ずつ中身の手順をかけて並べ, 行ごとの問題を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    data: Data
        読み込んだデータ

    Returns
    ----------
    items: Optional[np.ndarray]
        (件数, ...) の配列 (読めなかった行は 0. 1 件も読めなければ None)
    reasons: list[str]
        行ごとの予測できない理由 (予測できる行は空の文字)
    """
    # 読めない件と形の違う件は理由を付けて飛ばす
    arrays: list = []
    reasons = []
    shape = None
    for index in range(len(data.frame)):
        try:
            item = data.load_item(index)
            if isinstance(item, str) and not item.strip():
                arrays.append(None)
                reasons.append("文が空です")
                continue
            array = bundle.dataset.transform_item(item)
        except Exception as error:
            arrays.append(None)
            reasons.append(f"中身を読めません ({error})")
            continue
        if shape is not None and array.shape != shape:
            arrays.append(None)
            reasons.append(f"形が {array.shape} で, 他の件の {shape} と違います")
            continue
        shape = array.shape
        arrays.append(array)
        reasons.append("")
    if shape is None:
        return None, reasons
    first = next(a for a in arrays if a is not None)
    blank = np.zeros(shape, dtype=first.dtype)
    items = np.stack([blank if a is None else a for a in arrays])

    return items, reasons


def shape_inputs(
    bundle: Bundle, sources: dict, count: Optional[int] = None
) -> list:
    """
    中身の出どころごとの配列を, モデルの入力の順のテンソルにする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    sources: dict
        features と items の配列 (あるものだけ)
    count: Optional[int] = None
        件数 (省略すると配列の件数. 繰り返しの手順の中身の入力の形に使う)

    Returns
    ----------
    tensors: list
        モデルの入力の順のテンソル (繰り返しの手順の中身は始まりの中身,
        時刻は件数の 0)

    Raises
    ----------
    ValueError
        データに無い出どころを使う入力がある場合
    """
    # 件数 (手順の雑音の形と時刻の置き場に)
    if count is None:
        count = len(next(iter(sources.values()))) if sources else 0
    procedure = bundle.procedure

    # 文の番号の並びは整数, 他は小数. 先頭に足す番号は文の並びだけ
    tensors = []
    for name, item in bundle.input_plan().items():
        if item["source"] in STEP_SOURCES:
            tensors.append(step_placeholder(
                bundle, item, sources, count, procedure
            ))
            continue
        if item["source"] not in sources:
            raise ValueError(
                f"入力 {name} の {item['source']} が {bundle.kind} のデータに"
                "ありません"
            )
        array = np.asarray(sources[item["source"]])
        integer = item["source"] == "items" and bundle.kind == "text"
        tensor = torch.from_numpy(
            np.array(array, dtype=np.int64 if integer else np.float32)
        )
        if item["prepend"]:
            if not integer:
                raise ValueError(
                    f"入力 {name} の prepend は文の番号の並びにだけ使えます"
                )
            head = torch.tensor(item["prepend"], dtype=torch.long)
            tensor = torch.cat([head.expand(len(tensor), -1), tensor], dim=1)
        tensors.append(tensor)

    return tensors


def step_placeholder(
    bundle: Bundle, item: dict, sources: dict, count: int,
    procedure: Optional[dict]
) -> torch.Tensor:
    """
    繰り返しの手順の中身と時刻の入力の, 始まりのテンソルを作る関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    item: dict
        入力の出どころ (source と prepend)
    sources: dict
        features と items の配列 (あるものだけ)
    count: int
        件数
    procedure: Optional[dict]
        繰り返しの手順の設定

    Returns
    ----------
    tensor: torch.Tensor
        時刻は (件数,) の 0. 中身は, 雑音から始まる手順なら 1 件の形の 0
        (雑音は推論のときに引く), 渡した中身から始まる手順なら items の
        番号の並び (prepend を足したもの)

    Raises
    ----------
    ValueError
        渡した中身から始まる手順で items が無い場合
    """
    # 時刻は手順が渡すので置き場だけ
    if item["source"] == "time" or procedure is None:
        tensor = torch.zeros(count)
        return tensor
    if get_procedure(str(procedure["name"])).start == "noise":
        shape = [int(v) for v in bundle.output_info["shape"]]
        tensor = torch.zeros((count, *shape))
        return tensor

    # 文の手順は渡した番号の並びから
    if "items" not in sources:
        raise ValueError(
            f"手順 {procedure['name']} は始まりの番号の並び (items) が要ります"
        )
    tensor = torch.from_numpy(np.array(sources["items"], dtype=np.int64))
    if item["prepend"]:
        head = torch.tensor(item["prepend"], dtype=torch.long)
        tensor = torch.cat([head.expand(len(tensor), -1), tensor], dim=1)

    return tensor


def pick_output(out: Any, chosen: Optional[str]) -> torch.Tensor:
    """
    モデルの出力が辞書なら, 使う出力を選ぶ関数

    Parameters
    ----------
    out: Any
        モデルの出力
    chosen: Optional[str]
        使う出力の名前 (1 つだけなら None で可)

    Returns
    ----------
    tensor: torch.Tensor
        使う出力

    Raises
    ----------
    ValueError
        使う出力が決まらない場合
    """
    # 書いた名前か 1 つだけの出力
    if not isinstance(out, dict):
        return out
    if chosen is None and len(out) != 1:
        raise ValueError(
            f"モデルの出力が {', '.join(out)} の {len(out)} つ "
            "あります. 使う出力を model.output に書いてください"
        )
    if chosen is not None and chosen not in out:
        raise ValueError(
            f"出力 {chosen} がありません (出力: {', '.join(out)})"
        )
    tensor = out[chosen] if chosen is not None else next(iter(out.values()))

    return tensor


class StepModel:
    def __init__(
        self, model: Any, tensors: list, sources: list[str],
        chosen: Optional[str]
    ):
        # モデル, 条件の入力, 入力ごとの出どころ, 使う出力を保存
        self.model = model
        self.tensors = tensors
        self.sources = sources
        self.chosen = chosen

    def __call__(
        self, state: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        手順の中身と時刻を入力の順に置いてモデルを呼ぶ関数

        Parameters
        ----------
        state: torch.Tensor
            手順の中身
        t: Optional[torch.Tensor] = None
            時刻 (時刻を渡さない手順は None)

        Returns
        ----------
        output: torch.Tensor
            使う出力
        """
        # 中身と時刻の入力を差し替え, 他は条件のまま
        inputs = []
        for source, tensor in zip(self.sources, self.tensors):
            if source == "state":
                inputs.append(state)
            elif source == "time":
                inputs.append(t)
            else:
                inputs.append(tensor)
        output = pick_output(self.model(*inputs), self.chosen)

        return output


def run_steps(
    bundle: Bundle, tensors: list, batch_size: int, device: str
) -> np.ndarray:
    """
    繰り返しの手順で, 決まった件数ずつ出力を作る関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    tensors: list
        モデルの入力の順のテンソル (shape_inputs の結果)
    batch_size: int
        1 回に渡す件数
    device: str
        使うデバイス

    Returns
    ----------
    outputs: np.ndarray
        手順の出した中身を並べた配列
    """
    # 1 つの生成器から, 件数の順に雑音と乱数を引く (同じ入力と種なら同じ)
    config = bundle.procedure or {}
    name = str(config["name"])
    procedure = get_procedure(name)
    generator = make_generator(int(config["seed"]))
    sources = [item["source"] for item in bundle.input_plan().values()]
    index = sources.index("state")
    model = bundle.model().to(device)
    chosen = bundle.info["model"].get("output")
    parts = []
    for start in range(0, len(tensors[index]), batch_size):
        batch = [t[start:start + batch_size].to(device) for t in tensors]
        state = batch[index]
        if procedure.start == "noise":
            state = start_noise(tuple(state.shape), generator).to(device)
        step_model = StepModel(model, batch, sources, chosen)
        out = run_procedure(
            name, step_model, state, int(config["steps"]), generator,
            **(config.get("options") or {})
        )
        parts.append(out.detach().cpu().numpy())
    model.to("cpu")
    outputs = np.concatenate(parts) if parts else np.zeros((0,))

    return outputs


def run_model(
    bundle: Bundle, tensors: list, batch_size: int, device: str
) -> np.ndarray:
    """
    モデルを推論のモードで決まった件数ずつ動かす関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    tensors: list
        モデルの入力の順のテンソル
    batch_size: int
        1 回に渡す件数
    device: str
        使うデバイス

    Returns
    ----------
    outputs: np.ndarray
        使う出力を並べた配列

    Raises
    ----------
    ValueError
        出力が辞書で, 使う出力が決まらない場合
    """
    # 繰り返しの手順の一式は手順で
    if bundle.procedure:
        outputs = run_steps(bundle, tensors, batch_size, device)
        return outputs

    # 出力が辞書なら, 書いた名前か 1 つだけの出力
    model = bundle.model().to(device)
    chosen = bundle.info["model"].get("output")
    parts = []
    with torch.no_grad():
        for start in range(0, len(tensors[0]), batch_size):
            batch = [t[start:start + batch_size].to(device) for t in tensors]
            out = pick_output(model(*batch), chosen)
            parts.append(out.detach().float().cpu().numpy())
    model.to("cpu")
    outputs = np.concatenate(parts) if parts else np.zeros((0,))

    return outputs


def inverse_target(bundle: Bundle, values: np.ndarray) -> np.ndarray:
    """
    レシピで変換した目的の列の値を, 元の単位に戻す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    values: np.ndarray
        モデルの出力の値

    Returns
    ----------
    result: np.ndarray
        元の単位の値
    """
    # 目的の列を変換した手順を逆の順に戻す
    target = bundle.dataset.info.get("target")
    result = values.astype(np.float64)
    for spec in reversed(bundle.dataset.recipe.get("steps", [])):
        state = spec.get("state") or {}
        if spec["op"] in ("standardize", "minmax"):
            pair = (state.get("values") or {}).get(target)
            if pair is not None:
                center, scale = pair
                result = result * scale + center
        elif spec["op"] == "log":
            columns = spec.get("columns")
            columns = [columns] if isinstance(columns, str) else columns
            if columns and target in columns:
                result = np.expm1(result)

    return result


def decode(bundle: Bundle, outputs: np.ndarray) -> dict:
    """
    モデルの出力を, 出力の形に合わせて元の形の列にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    outputs: np.ndarray
        モデルの出力

    Returns
    ----------
    columns: dict
        出力の形ごとの列 (分類は prediction, confidence, p:<クラス>, 値は
        prediction, 分布は prediction と q:<水準>, 配列の形は output, 文は
        prediction)

    Raises
    ----------
    ValueError
        出力の形が合わない場合
    """
    # 出力の形ごとの読み方
    columns = DECODERS[bundle.form](bundle, np.asarray(outputs))

    return columns


def scale_back(bundle: Bundle, outputs: np.ndarray) -> np.ndarray:
    """
    目的の値の標準化とレシピの変換を戻して, 元の単位にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    outputs: np.ndarray
        モデルの出力の値 (どの形でも)

    Returns
    ----------
    values: np.ndarray
        元の単位の値
    """
    # 前処理の外の標準化を戻してから, レシピの変換を逆の順に
    scale = bundle.info["model"].get("target_scale")
    if scale:
        outputs = outputs * float(scale["scale"]) + float(scale["center"])
    values = inverse_target(bundle, outputs)

    return values


def decode_value(bundle: Bundle, outputs: np.ndarray) -> dict:
    """
    値の出力を元の単位の列にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    outputs: np.ndarray
        (件数,) か (件数, 1) の出力

    Returns
    ----------
    columns: dict
        prediction の列

    Raises
    ----------
    ValueError
        出力の形が合わない場合
    """
    # 回帰と予測は 1 つの値
    if outputs.ndim == 2 and outputs.shape[1] == 1:
        outputs = outputs[:, 0]
    if outputs.ndim != 1:
        raise ValueError(
            f"{bundle.task} の出力は (件数,) か (件数, 1) にしてください "
            f"(今は {tuple(outputs.shape)})"
        )
    columns = {"prediction": scale_back(bundle, outputs)}

    return columns


def decode_distribution(bundle: Bundle, outputs: np.ndarray) -> dict:
    """
    分位点の出力を, 元の単位の分位点の列と真ん中の予測にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    outputs: np.ndarray
        (件数, 分位点の数) の出力

    Returns
    ----------
    columns: dict
        prediction (0.5 の点. 無ければ真ん中の点) と q:<水準> の列

    Raises
    ----------
    ValueError
        出力の形が分位点の数と合わない場合
    """
    # 元の単位に戻し, 行ごとに小さい順に並べ直す (分位点の交差を直す)
    levels = [float(q) for q in bundle.output_info["quantiles"]]
    if outputs.ndim != 2 or outputs.shape[1] != len(levels):
        raise ValueError(
            f"分布の出力は (件数, {len(levels)}) にしてください "
            f"(今は {tuple(outputs.shape)})"
        )
    values = np.sort(scale_back(bundle, outputs), axis=1)
    middle = levels.index(0.5) if 0.5 in levels else len(levels) // 2
    columns: dict = {"prediction": values[:, middle]}
    for index, level in enumerate(levels):
        columns[f"q:{level:g}"] = values[:, index]

    return columns


def decode_mask(bundle: Bundle, outputs: np.ndarray) -> dict:
    """
    領域の出力を, 1 件ごとの画素のクラスの番号の配列にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    outputs: np.ndarray
        (件数, クラスの数, 高さ, 幅) か (件数, [1,] 高さ, 幅) の出力

    Returns
    ----------
    columns: dict
        output (1 件ごとの (高さ, 幅) の整数の配列) の列

    Raises
    ----------
    ValueError
        出力の次元が合わない場合
    """
    # クラスの次元があれば最も大きいクラス, 1 チャンネルは 2 値
    if outputs.ndim == 4 and outputs.shape[1] > 1:
        masks = outputs.argmax(axis=1)
    elif outputs.ndim in (3, 4):
        single = outputs[:, 0] if outputs.ndim == 4 else outputs
        cut = 0.0 if bundle.output_kind == "logits" else 0.5
        masks = (single > cut) if cut == 0.0 else (single >= cut)
    else:
        raise ValueError(
            "領域の出力は (件数, クラスの数, 高さ, 幅) か (件数, 高さ, 幅) "
            f"にしてください (今は {tuple(outputs.shape)})"
        )
    columns = {"output": [m.astype(np.int64) for m in masks]}

    return columns


def decode_array(bundle: Bundle, outputs: np.ndarray) -> dict:
    """
    点群, ベクトル, 画像の出力を, 1 件ごとの配列にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    outputs: np.ndarray
        (件数, ...) の出力

    Returns
    ----------
    columns: dict
        output (1 件ごとの配列) の列

    Raises
    ----------
    ValueError
        出力の次元が出力の形と合わない場合
    """
    # 形ごとの 1 件の次元の数 (画像は問わない)
    dims = {"vector": 1, "point_cloud": 2}.get(bundle.form)
    if outputs.ndim < 2 or (dims is not None and outputs.ndim != dims + 1):
        expected = {"vector": "(件数, 次元)",
                    "point_cloud": "(件数, 点の数, 次元)"}.get(
            bundle.form, "(件数, ...)"
        )
        raise ValueError(
            f"{bundle.form} の出力は {expected} にしてください "
            f"(今は {tuple(outputs.shape)})"
        )
    columns = {"output": [np.asarray(o, dtype=np.float32) for o in outputs]}

    return columns


def decode_text(bundle: Bundle, outputs: np.ndarray) -> dict:
    """
    文の出力を, 番号を空白でつないだ文字にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    outputs: np.ndarray
        (件数, 長さ, 語彙) の logits か (件数, 長さ) の番号の並び

    Returns
    ----------
    columns: dict
        prediction の列

    Raises
    ----------
    ValueError
        出力の次元が合わない場合
    """
    # logits は位置ごとに最も大きい番号
    if outputs.ndim == 3:
        outputs = outputs.argmax(axis=2)
    if outputs.ndim != 2:
        raise ValueError(
            "文の出力は (件数, 長さ, 語彙) か (件数, 長さ) にしてください "
            f"(今は {tuple(outputs.shape)})"
        )
    columns = {
        "prediction": [
            " ".join(str(int(v)) for v in row) for row in outputs
        ]
    }

    return columns


def decode_class(bundle: Bundle, outputs: np.ndarray) -> dict:
    """
    分類の出力を, クラスの名前と確率の列にする関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    outputs: np.ndarray
        (件数, クラスの数) か 2 クラスで 1 列の出力

    Returns
    ----------
    columns: dict
        prediction, confidence, p:<クラス> の列

    Raises
    ----------
    ValueError
        出力の形がクラスの数と合わない場合
    """
    # 分類はクラスごとの確率 (2 クラスで 1 列なら 2 つ目のクラス)
    count = len(outputs)
    classes = bundle.classes
    kind = bundle.output_kind
    single = outputs.ndim == 1 or (
        outputs.ndim == 2 and outputs.shape[1] == 1
    )
    if single and len(classes) == 2:
        column = outputs.reshape(count).astype(np.float64)
        positive = 1 / (1 + np.exp(-column)) if kind == "logits" else column
        probability = np.stack([1 - positive, positive], axis=1)
    elif outputs.ndim == 2 and outputs.shape[1] == len(classes):
        probability = outputs.astype(np.float64)
        if kind == "logits":
            shifted = np.exp(probability - probability.max(axis=1,
                                                           keepdims=True))
            probability = shifted / shifted.sum(axis=1, keepdims=True)
    else:
        raise ValueError(
            f"分類の出力の形 {tuple(outputs.shape)} がクラスの数 "
            f"{len(classes)} と合いません"
        )
    best = probability.argmax(axis=1) if count else np.zeros(0, dtype=int)
    columns = {
        "prediction": [classes[i] for i in best],
        "confidence": probability[np.arange(count), best],
    }
    for index, name in enumerate(classes):
        columns[f"p:{name}"] = probability[:, index]

    return columns


# 出力の形ごとの読み方
DECODERS = {
    "class": decode_class,
    "value": decode_value,
    "distribution": decode_distribution,
    "mask": decode_mask,
    "point_cloud": decode_array,
    "vector": decode_array,
    "image": decode_array,
    "text": decode_text,
}


def predict(
    bundle: Union[str, Path, Bundle],
    source: Union[str, Path, pd.DataFrame, Data],
    options: Optional[dict] = None, batch_size: int = 256,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    新しいデータを前処理して推論し, 行ごとの予測を返す関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ
    source: Union[str, Path, pd.DataFrame, Data]
        推論するデータ
    options: Optional[dict] = None
        読み込みの設定 (学習のときの設定に上書きする)
    batch_size: int = 256
        1 回に渡す件数
    device: str = "cpu"
        使うデバイス

    Returns
    ----------
    result: pd.DataFrame
        元の列と, status, reason, prediction (分類は confidence と
        p:<クラス> も) の列

    Raises
    ----------
    ValueError
        前処理できない場合と, 出力の形が合わない場合
    """
    # データの読み込みと, 種類に合わせた中身
    if batch_size < 1:
        raise ValueError("batch_size は 1 以上にしてください")
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    data = read_source(bundle, source, options)
    count = len(data.frame)
    sources: dict = {}
    reasons = [""] * count
    plan = {item["source"] for item in bundle.input_plan().values()}
    if "state" in plan and bundle.procedure and get_procedure(
        str(bundle.procedure["name"])
    ).start == "given":
        # 文の手順は渡した番号の並びから始める
        plan.add("items")
    if "features" in plan:
        sources["features"], found = features_of(bundle, data)
        reasons = [a or b for a, b in zip(reasons, found)]
    if "items" in plan:
        items, found = items_of(bundle, data)
        if items is not None:
            sources["items"] = items
        reasons = [a or b for a, b in zip(reasons, found)]

    # 予測できる行だけ推論
    keep = np.array([not r for r in reasons], dtype=bool)
    if keep.any():
        tensors = shape_inputs(
            bundle, {k: np.asarray(v)[keep] for k, v in sources.items()},
            int(keep.sum())
        )
        columns = decode(
            bundle, run_model(bundle, tensors, batch_size, device)
        )
        # 数でない予測 (入力の無限大など) の行は予測しなかった行に
        finite = finite_rows(bundle, columns)
        if not finite.all():
            rows = np.flatnonzero(keep)[~finite]
            for row in rows:
                reasons[row] = NOT_FINITE
            keep[rows] = False
            columns = {
                name: [v for v, ok in zip(values, finite) if ok]
                for name, values in columns.items()
            }
    else:
        columns = decode(bundle, empty_output(bundle))

    # 元の列に予測の列を並べる (予測しなかった行は空). 予測の列と同じ
    # 名前の元の列は「(入力)」を付けて残す
    result = data.frame.copy().reset_index(drop=True)
    result.columns = [str(c) for c in result.columns]
    added = {"status", "reason", "decision", *columns}
    result = result.rename(
        columns={c: f"{c} (入力)" for c in result.columns if c in added}
    )
    result["status"] = np.where(keep, OK, SKIPPED)
    result["reason"] = reasons
    texts = ("output",) + (
        ("prediction",) if bundle.form in ("class", "text") else ()
    )
    for name, values in columns.items():
        full = pd.Series([None] * count, dtype=object)
        full[keep] = pd.Series(list(values), dtype=object).to_numpy()
        result[name] = full if name in texts else pd.to_numeric(full)

    return result


def finite_rows(bundle: Bundle, columns: dict) -> np.ndarray:
    """
    予測の値が数 (無限大や欠損でない) の行を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    columns: dict
        decode の列

    Returns
    ----------
    finite: np.ndarray
        行ごとの真偽値
    """
    # 配列の出力は全ての値, 文は番号なので常に数, 他は確信か値
    if bundle.form in ARRAY_FORMS:
        finite = np.array(
            [bool(np.isfinite(a).all()) for a in columns["output"]],
            dtype=bool
        )
    elif bundle.form == "text":
        finite = np.ones(len(columns["prediction"]), dtype=bool)
    else:
        check = "confidence" if bundle.form == "class" else "prediction"
        finite = np.isfinite(np.asarray(columns[check], dtype=float))

    return finite


def empty_output(bundle: Bundle) -> np.ndarray:
    """
    予測する行が無い場合の, 0 件の出力を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    outputs: np.ndarray
        0 件の出力
    """
    # 分類はクラスの数, 分布は分位点の数の列. 配列の形は次元だけ合わせる
    form = bundle.form
    if form == "class":
        shape: tuple = (0, len(bundle.classes))
    elif form == "distribution":
        shape = (0, len(bundle.output_info["quantiles"]))
    elif form in ("point_cloud", "mask"):
        shape = (0, 1, 1)
    else:
        shape = (0, 1)
    outputs = np.zeros(shape, dtype=np.float32)

    return outputs


def options_of(bundle: Bundle) -> dict:
    """
    学習のときの読み込みの名前と設定を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    reading: dict
        loader と options
    """
    # データセットの説明の source から
    recorded = bundle.dataset.info.get("source") or {}
    reading = {
        "loader": recorded.get("loader"),
        "options": dict(recorded.get("options") or {}),
    }

    return reading


def uncertainty_summary(uncertainty: Optional[dict]) -> Optional[dict]:
    """
    外れ方の記録から, 分位点の並びを除いた要約を返す関数

    Parameters
    ----------
    uncertainty: Optional[dict]
        一式の uncertainty

    Returns
    ----------
    summary: Optional[dict]
        split, rows, kind, measured_at, coverage, ece, restorable
        (作ったときの記録に戻せるか), horizon と steps_measured (段ごとの
        外れ方を測った段の数と, 測れた最も先の段). 記録が無ければ None
    """
    # 画面に出す値だけ (分位点と区間ごとの表は大きいので除く)
    if not uncertainty:
        return None
    calibration = uncertainty.get("calibration") or {}
    summary = {
        "split": uncertainty.get("split"),
        "rows": uncertainty.get("rows"),
        "kind": uncertainty.get("kind"),
        "measured_at": uncertainty.get("measured_at"),
        "coverage": uncertainty.get("coverage"),
        "ece": uncertainty.get("test_ece", calibration.get("ece")),
        "restorable": bool(uncertainty.get("original")),
        "horizon": uncertainty.get("horizon"),
        "steps_measured": max(
            (int(k) for k in (uncertainty.get("steps") or {})
             if k != "rows"), default=0
        ),
    }

    return summary


def describe_bundle(bundle: Union[str, Path, Bundle]) -> dict:
    """
    推論の一式の説明を JSON にできる辞書で返す関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ

    Returns
    ----------
    description: dict
        名前, 種類, タスク, 目的の列, クラス, 学習に使う列, 読み込み,
        前処理の手順, モデルの入力, 実験, 結論の既定値, 何段も先まで
        予測できるか, 実験の記録との比べ, 外れ方の記録の要約, 注意
    """
    # 画面が一式を選ぶときに要ること
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    info = bundle.info
    dataset = bundle.dataset.info
    description: dict[str, Any] = {
        "path": str(bundle.path),
        "name": info.get("name"),
        "created_at": info.get("created_at"),
        "kind": bundle.kind,
        "task": bundle.task,
        "target": dataset.get("target"),
        "classes": bundle.classes,
        "features": [item["name"] for item in dataset.get("features", [])],
        "text_column": dataset.get("text_column"),
        "reading": options_of(bundle),
        "steps": [s["op"] for s in bundle.dataset.recipe.get("steps", [])],
        "item_steps": [
            s["op"] for s in bundle.dataset.recipe.get("item_steps", [])
        ],
        "model": {
            "name": bundle.spec.get("name"),
            "inputs": bundle.input_plan(),
            "output": info["model"].get("output"),
            "kind": bundle.output_kind,
        },
        "run": info.get("run"),
        "conclusion": info.get("conclusion") or {},
        "decision": info.get("decision"),
        "forecastable": bundle.column_output and (
            bundle.task != "classification"
        ) and (
            bundle.dataset.info.get("target") in lag_settings(bundle)[
                "columns"
            ]
        ),
        "check": info.get("check"),
        "uncertainty": uncertainty_summary(info.get("uncertainty")),
        "notes": list(info.get("notes") or []),
        "format_version": info.get("format_version"),
        "role": bundle.role,
        "output": bundle.output_info,
        "column_output": bundle.column_output,
        "target": target_of(bundle),
        "procedure": bundle.procedure,
        "pretrained": list(info["model"].get("pretrained") or []),
        "pretrained_warnings": list(
            bundle.cache.get("pretrained_warnings") or []
        ),
    }

    return description


def target_of(bundle: Bundle) -> Optional[dict]:
    """
    一式の正解の種類と出どころを返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式

    Returns
    ----------
    target: Optional[dict]
        kind と origin (版 2 までの一式は kind: column. 分からなければ
        None)
    """
    # 版 3 は書いたまま, 版 2 までは列の予測
    if "target" in bundle.info:
        target = bundle.info.get("target")
        return target
    target = {"kind": "column"}

    return target
