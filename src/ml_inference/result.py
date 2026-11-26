import datetime as dt
import math
from pathlib import Path
import re
import shutil
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import yaml
from ml_eda import Data

from ml_inference.anomaly import conclude_anomaly
from ml_inference.bundle import Bundle, inside, load_bundle
from ml_inference.conclude import conclude, conclude_outputs, origin_sentence
from ml_inference.decision import decide
from ml_inference.forecast import forecast as forecast_rows
from ml_inference.predict import (
    ARRAY_FORMS, OK, noise_request, predict, target_of
)
from ml_inference.grouping import conclude_groups
from ml_inference.ranking import conclude_ranking, request_users
from ml_inference.price import conclude_price
from ml_inference.uplift import conclude_uplift

# 結果のファイル (result.yaml) の形の版 (混ざる列を残せば 4, 列の型を
# 残せば 3, 表に収まらない中身があれば 2, どれも無ければ 1)
RESULT_FORMAT_VERSION = 4

# 列の型を残す結果の版 (混ざる列の無い結果)
COLUMN_TYPES_FORMAT_VERSION = 3

# 表に収まらない中身を持つ結果の版
ARTIFACTS_FORMAT_VERSION = 2

# 列の型の記録の項目と, 型の名前ごとの読み込みの型
COLUMN_TYPES_KEY = "column_types"
COLUMN_DTYPES = {"text": str, "integer": "Int64"}

# 数と文字の混ざる列の型の名前と, 数に見える文字の行の位置の記録の項目
MIXED = "mixed"
TEXT_ROWS_KEY = "text_rows"

# 混ざる列の値の書き方 (整数は符号と数字だけ, 小数は小数点か指数を持つ数)
INTEGER_TEXT = re.compile(r"[+-]?[0-9]+")
FLOAT_TEXT = re.compile(
    r"[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+(?=[eE]))(?:[eE][+-]?[0-9]+)?"
)

# 欠損を持てる整数 (Int64) の範囲
INT64 = np.iinfo(np.int64)

# 表に収まらない中身の置き場と, 表の列
ARTIFACTS_DIR = "artifacts"
ARTIFACT_COLUMN = "artifact"

# 文の出力の番号の並びの列 (この列のある表は文の出力)
TOKENS_COLUMN = "tokens"

# 書き出すときに result.yaml へ写さない結果の項目 (表, 形, 読んだ場所,
# 読んだ版と中身と列の型と数に見える文字の行の記録)
RESULT_OWN_KEYS = ("frame", "form", "directory", "format_version",
                   "artifacts", COLUMN_TYPES_KEY, TEXT_ROWS_KEY)


def source_text(bundle: Bundle, source: Any) -> Optional[str]:
    """
    結果に残す入力の説明を返す関数

    Parameters
    ----------
    bundle: Bundle
        推論の一式
    source: Any
        推論したデータ

    Returns
    ----------
    text: Optional[str]
        パスはそのまま, 雑音だけの一式は「件数 n, 種 s」, 並べるの一式は
        「利用者 n 人」, 他は None
    """
    # 雑音だけの一式は件数と種, 並べるの一式は利用者の人数
    if bundle.from_noise:
        count, seed = noise_request(bundle, source)
        text: Optional[str] = f"件数 {count}, 種 {seed}"
        return text
    if bundle.from_rank:
        text = f"利用者 {len(request_users(bundle, source)):,} 人"
        return text
    text = str(source) if isinstance(source, (str, Path)) else None

    return text


def infer(
    bundle: Union[str, Path, Bundle],
    source: Union[str, Path, pd.DataFrame, Data, dict],
    rules: Optional[dict] = None, options: Optional[dict] = None,
    batch_size: int = 256, device: str = "cpu", forecast: bool = False,
    decision: Optional[dict] = None, k: Optional[int] = None,
    seen: Union[str, Path, pd.DataFrame, None] = None
) -> dict:
    """
    新しいデータを推論し, 結論まで出す関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ
    source: Union[str, Path, pd.DataFrame, Data, dict, list]
        推論するデータ (雑音だけの一式は件数と種の辞書. 並べるの一式は
        利用者の値の並びか利用者の列を持つ表)
    rules: Optional[dict] = None
        結論の決め方 (一式の既定値に上書きする. 値を None にすると消す)
    options: Optional[dict] = None
        読み込みの設定 (学習のときの設定に上書きする)
    batch_size: int = 256
        1 回に渡す件数
    device: str = "cpu"
        使うデバイス
    forecast: bool = False
        目的の列が空の先の行を順に予測し, 先の行だけの結論を出すか
        (何段も先の予測)
    decision: Optional[dict] = None
        意思決定の決め方 (渡すと, 行ごとの行動と見込みも出す. 省略すると
        一式の既定値. 空の辞書で意思決定しない)
    k: Optional[int] = None
        並べるの一式の, 利用者ごとに並べる件数 (省くと一式の rank.k)
    seen: Union[str, Path, pd.DataFrame, None] = None
        並べるの一式の, 前に反応した品物の表 (渡すと除いて並べる)

    Returns
    ----------
    result: dict
        bundle (一式の名前と場所), source (入力), created_at, frame
        (予測と判断の表. 何段も先の予測は先の行だけで step 付き),
        summary, sentences, rules, (decision を渡すと) decision
        (summary, sentences, notes, decision. 表には action などの列)
    """
    # 一式の既定の決め方に, 渡した決め方を重ねる
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    # (陽性のクラスを消した場合は, しきい値も消す. 一緒に使えない陽性の
    # クラスと確信の下限は, 渡した方を使う)
    given = dict(rules or {})
    defaults = dict(bundle.info.get("conclusion") or {})
    if given.get("positive") is not None:
        defaults.pop("min_confidence", None)
    if given.get("min_confidence") is not None:
        defaults.pop("positive", None)
        defaults.pop("threshold", None)
    merged = {**defaults, **given}
    if merged.get("positive") is None:
        merged.pop("threshold", None)
    merged = {k: v for k, v in merged.items() if v is not None}

    # 異常の度合いの一式は度合いの要約と度合いの上位から回す振り分け,
    # 効果の一式は効果の要約と効果の上位から配る判断, 値段の一式は売れた数
    # と弾力性の要約と値段の判断
    if bundle.from_anomaly or bundle.from_uplift or bundle.from_price:
        result = infer_ranked(bundle, source, given, options, batch_size,
                              device, forecast, decision)
        return result

    # 列の出力でない一式は, 推論と件数だけ (並べるの一式は利用者の人数,
    # 群分けの一式は群の要約も)
    if not bundle.column_output:
        if given or decision or forecast:
            bundle.require_column("結論の決め方, 意思決定, 何段も先の予測")
        prediction = predict(bundle, source, options, batch_size, device,
                             k=k, seen=seen)
        result = {
            "bundle": {"name": bundle.info.get("name"),
                       "path": str(bundle.path)},
            "source": source_text(bundle, source),
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "form": bundle.form,
            **(conclude_ranking(prediction) if bundle.from_rank
               else conclude_groups(bundle, prediction) if bundle.from_group
               else conclude_outputs(prediction)),
        }
        return result
    if k is not None or seen is not None:
        raise ValueError(
            "並べる件数 (k) と前に反応した品物の表 (seen) は, 並べるの一式"
            "だけに渡せます"
        )

    # 予測 (何段も先の予測は先の行だけ) と結論
    if forecast:
        prediction = forecast_rows(bundle, source, options, batch_size,
                                   device)
        ahead = prediction.attrs.get("ahead")
        if ahead is None or not ahead.any():
            raise ValueError(
                "先の行 (系列ごとに最後の分かる値より後の, 目的の列が空の行)"
                "がありません. 予測したい行の目的の列を空にして渡してください"
            )
        prediction = prediction[ahead].reset_index(drop=True)
    else:
        prediction = predict(bundle, source, options, batch_size, device)
    conclusion = conclude(prediction, bundle.task, bundle.classes, merged)
    sentence = origin_sentence(target_of(bundle))
    if sentence:
        conclusion["sentences"].append(sentence)
    if decision is None:
        decision = bundle.info.get("decision")
    if decision:
        decided = decide(bundle, conclusion["frame"], decision)
        conclusion["frame"] = decided.pop("frame")
        conclusion["decision"] = decided
    result = {
        "bundle": {"name": bundle.info.get("name"), "path": str(bundle.path)},
        "source": str(source) if isinstance(source, (str, Path)) else None,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        **conclusion,
    }

    return result


def infer_ranked(
    bundle: Bundle, source: Any, given: dict, options: Optional[dict],
    batch_size: int, device: str, forecast: bool, decision: Optional[dict]
) -> dict:
    """
    異常の度合いか効果か売れた数と, 要約と判断 (上位から上限まで, 値段)
    を出す関数

    Parameters
    ----------
    bundle: Bundle
        異常の度合いの一式か効果の一式か値段の一式
    source: Any
        表のファイル, 表, 表の Data
    given: dict
        渡した結論の決め方 (使えないので空だけ)
    options: Optional[dict]
        読み込みの設定
    batch_size: int
        1 回に渡す行の数
    device: str
        使うデバイス
    forecast: bool
        何段も先の予測 (使えないので False だけ)
    decision: Optional[dict]
        振り分けか配る判断か値段の判断の決め方 (省略すると一式の既定.
        空の辞書で決めない)

    Returns
    ----------
    result: dict
        bundle, source, created_at, form, frame (決めると action などの
        列も), summary, sentences, rules (空), (決めると) decision

    Raises
    ----------
    ValueError
        結論の決め方か何段も先の予測を渡した場合と, 判断の決め方に誤りが
        ある場合
    """
    # 結論の決め方と何段も先の予測は誤り
    if given or forecast:
        bundle.require_column("結論の決め方と何段も先の予測")
    prediction = predict(bundle, source, options, batch_size, device)
    conclusion = conclude_uplift(bundle, prediction) if (
        bundle.from_uplift
    ) else conclude_price(bundle, prediction) if (
        bundle.from_price
    ) else conclude_anomaly(bundle, prediction)
    if decision is None:
        decision = bundle.info.get("decision")
    if decision:
        decided = decide(bundle, conclusion["frame"], decision)
        conclusion["frame"] = decided.pop("frame")
        conclusion["decision"] = decided
    result = {
        "bundle": {"name": bundle.info.get("name"), "path": str(bundle.path)},
        "source": source_text(bundle, source),
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "form": bundle.form,
        **conclusion,
    }

    return result


def write_result(result: dict, path: Union[str, Path]) -> Path:
    """
    予測と結論をディレクトリに書き出す関数

    Parameters
    ----------
    result: dict
        infer の結果
    path: Union[str, Path]
        書き出すディレクトリ (無ければ作る)

    Returns
    ----------
    root: Path
        書き出したディレクトリ

    Raises
    ----------
    FileExistsError
        書き出し先に前の結果のものでない artifacts/ がある場合
    """
    # 出力の形 (infer の結果か, read_result で読んだ結果の記録から)
    root = Path(path).expanduser()
    form = result.get("form") or (result.get("artifacts") or {}).get("form")
    frame, artifacts = result["frame"], None
    if form in ARRAY_FORMS:
        frame, artifacts = place_artifacts(result, frame, root)
        artifacts["form"] = form
    else:
        root.mkdir(parents=True, exist_ok=True)

    # 表は表計算のアプリで開ける BOM 付きの UTF-8 (読み直すと値の変わる列
    # の型を残す)
    frame.to_csv(root / "predictions.csv", index=False, encoding="utf-8-sig")
    types, rows = column_types(frame, root / "predictions.csv")

    # 表以外の説明 (形の版を先頭に. 混ざる列があれば版 4, 型も中身も無け
    # れば版 1 のまま. 読んだ結果の版と中身と型の記録は書き直し)
    if MIXED in types.values():
        version = RESULT_FORMAT_VERSION
    elif types:
        version = COLUMN_TYPES_FORMAT_VERSION
    else:
        version = ARTIFACTS_FORMAT_VERSION if artifacts else 1
    meta: dict[str, Any] = {"format_version": version}
    meta.update({
        k: v for k, v in result.items() if k not in RESULT_OWN_KEYS
    })
    if artifacts:
        meta["artifacts"] = artifacts
    if types:
        meta[COLUMN_TYPES_KEY] = types
    if rows:
        meta[TEXT_ROWS_KEY] = rows
    (root / "result.yaml").write_text(
        yaml.safe_dump(meta, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )

    return root


def clear_artifacts(root: Path) -> None:
    """
    前に書き出した結果の artifacts/ だけを消す関数

    Parameters
    ----------
    root: Path
        書き出すディレクトリ

    Returns
    ----------
    None

    Raises
    ----------
    FileExistsError
        artifacts/ があり, 前の結果 (artifacts を記録した result.yaml)
        のものでない場合
    """
    # 結果のものと確かめられない artifacts/ は利用者のデータとして残す
    folder = root / ARTIFACTS_DIR
    if not folder.exists():
        return
    meta = {}
    file = root / "result.yaml"
    if file.is_file():
        meta = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(meta, dict) or not meta.get("artifacts"):
        raise FileExistsError(
            f"{folder} は前の結果の artifacts/ ではないので消しません. "
            "別の書き出し先を選んでください"
        )
    shutil.rmtree(folder)


def place_artifacts(
    result: dict, frame: pd.DataFrame, root: Path
) -> tuple[pd.DataFrame, dict]:
    """
    配列の出力の結果の artifacts/ を書き出し先にそろえる関数

    Parameters
    ----------
    result: dict
        infer か read_result の結果
    frame: pd.DataFrame
        予測と判断の表
    root: Path
        書き出すディレクトリ

    Returns
    ----------
    frame: pd.DataFrame
        artifact の列 (相対パス) を持つ表
    artifacts: dict
        directory, column, format, count

    Raises
    ----------
    ValueError
        表に output の列も artifact の列も無い場合
    """
    # 推論したばかりの配列は書き出し, 読んだ結果は配列を持ち越す
    source = result.get("directory")
    same = source is not None and Path(source).expanduser().resolve() == (
        root.resolve()
    )
    if "output" in frame.columns:
        clear_artifacts(root)
        root.mkdir(parents=True, exist_ok=True)
        frame, artifacts = write_artifacts(frame, root)
        return frame, artifacts
    if ARTIFACT_COLUMN not in frame.columns or source is None:
        raise ValueError(
            "配列の出力の結果に output の列も, 読んだ結果の artifact の列も"
            "ありません"
        )

    # 別の書き出し先には, 表が指す配列を写す (同じ場所はそのまま)
    paths = [str(p) for p in frame[ARTIFACT_COLUMN].fillna("") if p]
    if not same:
        clear_artifacts(root)
        (root / ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
        for relative in paths:
            shutil.copy(artifact_path(source, relative),
                        artifact_path(root, relative))
    artifacts = {
        "directory": ARTIFACTS_DIR,
        "column": ARTIFACT_COLUMN,
        "format": "npy",
        "count": len(paths),
    }

    return frame, artifacts


def write_artifacts(
    frame: pd.DataFrame, root: Path
) -> tuple[pd.DataFrame, dict]:
    """
    表の output の列の配列を artifacts/ に書き, 相対パスの列にする関数

    Parameters
    ----------
    frame: pd.DataFrame
        予測と判断の表
    root: Path
        結果のディレクトリ

    Returns
    ----------
    frame: pd.DataFrame
        output の列を artifact (相対パス. 無い行は空) に替えた表
    artifacts: dict
        directory, column, format, count
    """
    # 配列の列が無ければ, 中身の無い記録
    if "output" not in frame.columns:
        frame = frame.copy()
        frame["output"] = None
    folder = root / ARTIFACTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for row, value in enumerate(frame["output"]):
        if value is None or (np.isscalar(value) and pd.isna(value)):
            paths.append("")
            continue
        relative = f"{ARTIFACTS_DIR}/{row:06d}.npy"
        np.save(root / relative, np.asarray(value), allow_pickle=False)
        paths.append(relative)
    position = list(frame.columns).index("output")
    frame = frame.drop(columns=["output"])
    frame.insert(position, ARTIFACT_COLUMN, paths)
    artifacts = {
        "directory": ARTIFACTS_DIR,
        "column": ARTIFACT_COLUMN,
        "format": "npy",
        "count": sum(1 for p in paths if p),
    }

    return frame, artifacts


def artifact_path(root: Union[str, Path], relative: str) -> Path:
    """
    表の artifact の列の相対パスを, 結果のディレクトリの中に解決する関数

    Parameters
    ----------
    root: Union[str, Path]
        結果のディレクトリ
    relative: str
        表の相対パス

    Returns
    ----------
    path: Path
        配列のファイル

    Raises
    ----------
    ValueError
        結果のディレクトリの外を指す場合
    """
    # 一式と同じく外を指すパスの禁止
    path = inside(Path(root).expanduser(), str(relative))

    return path


def column_kind(values: pd.Series) -> Optional[str]:
    """
    書き出す表の列の, 残す候補の型を返す関数

    Parameters
    ----------
    values: pd.Series
        書き出す表の列

    Returns
    ----------
    kind: Optional[str]
        欠けを除いた値が全て文字なら text, 全て Int64 に収まる整数 (真偽値
        を除く) なら integer, 文字と整数と有限の小数の 2 つ以上の種類だけ
        なら mixed, 他 (値の無い列, 小数, 真偽値などの混ざる列) は None
    """
    # 欠けを除いた値の型で見る (値の無い列と真偽値の列は残さない. 整数の
    # 型の列も Int64 に収まる値だけ. 値の型は列ごとにまとめて見る)
    kept = values.dropna()
    if not len(kept) or pd.api.types.is_bool_dtype(values.dtype):
        return None
    if pd.api.types.is_integer_dtype(values.dtype):
        kind = "integer"
    elif isinstance(values.dtype, pd.StringDtype):
        kind = "text"
    else:
        kept = np.asarray(kept, dtype=object)
        inferred = pd.api.types.infer_dtype(kept, skipna=False)
        kind = {"string": "text", "integer": "integer"}.get(inferred)
        if inferred.startswith("mixed"):
            kind = mixed_kind(kept)
    if kind == "integer" and not (
        INT64.min <= int(kept.min()) and int(kept.max()) <= INT64.max
    ):
        kind = None

    return kind


def value_kind(value: Any) -> Optional[str]:
    """
    混ざる列の 1 つの値の種類を返す関数

    Parameters
    ----------
    value: Any
        欠けでない値

    Returns
    ----------
    kind: Optional[str]
        文字 (空の文字も) は text, 整数 (真偽値を除く) は integer, 有限の
        小数は float, 他 (真偽値, 無限の小数, 配列など) は None
    """
    # 真偽値は整数の子なので先に除く
    if isinstance(value, str):
        return "text"
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return "integer"
    if isinstance(value, (float, np.floating)) and math.isfinite(value):
        return "float"

    return None


def mixed_kind(kept: np.ndarray) -> Optional[str]:
    """
    値の種類の混ざる列が, 値ごとの型で戻せる混ざる列かを返す関数

    Parameters
    ----------
    kept: np.ndarray
        欠けを除いた列の値 (object)

    Returns
    ----------
    kind: Optional[str]
        文字と整数と有限の小数のうち 2 つ以上の種類だけなら mixed, 他は
        None
    """
    # 空の文字も文字の種類に数える (空の文字と整数の列の整数を, 小数に
    # 読まないため)
    kinds = set()
    for value in kept:
        kind = value_kind(value)
        if kind is None:
            return None
        kinds.add(kind)

    return MIXED if len(kinds) >= 2 else None


def mixed_values(values: pd.Series, text: set) -> pd.Series:
    """
    文字で読んだ混ざる列を, 値ごとの型に戻す関数

    Parameters
    ----------
    values: pd.Series
        文字で読んだ列 (欠けは欠損)
    text: set
        数に見えても文字のままにする行の位置

    Returns
    ----------
    values: pd.Series
        整数の書き方の値は整数, 小数の書き方の値は小数, 他は文字の列
        (object)
    """
    # 書き方で数と分けられない文字は, 残した行の位置で見分ける
    back = []
    for position, value in enumerate(values.tolist()):
        if not isinstance(value, str) or position in text:
            back.append(value)
        elif INTEGER_TEXT.fullmatch(value):
            back.append(int(value))
        elif FLOAT_TEXT.fullmatch(value):
            back.append(float(value))
        else:
            back.append(value)

    return pd.Series(back, index=values.index, name=values.name,
                     dtype=object)


def same_mixed(mine: list, back: list) -> bool:
    """
    混ざる列の書いた値と読み直した値が, 同じ種類と値かを返す関数

    Parameters
    ----------
    mine: list
        書き出した列の値
    back: list
        値ごとの型に戻して読み直した列の値

    Returns
    ----------
    same: bool
        全ての値の種類と値が同じ (欠けと空の文字は欠損で読めれば同じ)
        なら True
    """
    # 欠けと空の文字は欠損に戻る (空の文字は CSV で欠損と分けられない).
    # NumPy の値は Python の値に直して比べる (float32 と Python の小数を
    # 比べると float32 の精度で比べるため)
    if len(mine) != len(back):
        return False
    cast = {"text": str, "integer": int, "float": float}
    for a, b in zip(mine, back):
        if pd.isna(a) or a == "":
            if not pd.isna(b):
                return False
            continue
        kind = value_kind(a)
        if pd.isna(b) or kind != value_kind(b) or cast[kind](a) != b:
            return False

    return True


def mixed_text_rows(mine: list, written: pd.Series) -> Optional[list]:
    """
    混ざる列の数に見える文字の行の位置を返す関数

    Parameters
    ----------
    mine: list
        書き出した列の値
    written: pd.Series
        書いた predictions.csv から文字で読んだ同じ位置の列

    Returns
    ----------
    rows: Optional[list]
        数に見える文字の値の行の位置 (0 から) の並び. 値ごとの型に戻して
        書いた値と同じにならない列は None
    """
    # 数に見える文字の行を残した上で値ごとの型に戻し, 書いた値と比べる
    rows = [
        i for i, (a, b) in enumerate(zip(mine, written.tolist()))
        if isinstance(a, str) and isinstance(b, str)
        and (INTEGER_TEXT.fullmatch(b) or FLOAT_TEXT.fullmatch(b))
    ]
    back = mixed_values(written, set(rows)).tolist()
    if not same_mixed(mine, back):
        return None

    return rows


def text_positions(rows: Any, name: str) -> set:
    """
    result.yaml の数に見える文字の行の記録から, 列の行の位置を返す関数

    Parameters
    ----------
    rows: Any
        result.yaml の text_rows (列の名前から位置の並びへの対応)
    name: str
        列の名前

    Returns
    ----------
    positions: set
        行の位置 (並びでない記録と整数でない位置は無視)
    """
    # 壊れた記録は無視する (知らない型の名前と同じ扱い)
    values = rows.get(name) if isinstance(rows, dict) else None
    if not isinstance(values, list):
        return set()
    positions = {
        v for v in values if isinstance(v, int) and not isinstance(v, bool)
    }

    return positions


def column_changed(mine: pd.Series, back: pd.Series, kind: str) -> bool:
    """
    書いた列の値が, 型を知らずに読み直すと変わるかを返す関数

    Parameters
    ----------
    mine: pd.Series
        書き出した表の列 (column_kind が kind を返した列)
    back: pd.Series
        型を知らずに読み直した同じ位置の列
    kind: str
        残す候補の型 (text か integer)

    Returns
    ----------
    changed: bool
        integer は読み直した列が整数の型でないか値が違う場合, text は空で
        ない文字の値が文字のまま同じ値で読めない場合に True
    """
    # 整数は読み直した列の型と値をまとめて比べる (読み直した列が整数の型
    # なら欠損は無い. int64 でない整数の型と長さの違う列は 1 件ずつ)
    if kind == "integer":
        if not pd.api.types.is_integer_dtype(back.dtype):
            return True
        if back.dtype != np.int64 or len(mine) != len(back):
            return any(int(a) != int(b) for a, b in zip(mine, back))
        return bool(
            (mine.to_numpy(dtype=np.int64) != back.to_numpy()).any()
        )

    # 文字は空でない文字の値だけを, 読み直した値が文字で同じかで比べる
    # (空の文字は CSV で欠損と分けられない. 長さの違う列は 1 件ずつ)
    if len(mine) != len(back):
        return any(
            not (isinstance(b, str) and a == b)
            for a, b in zip(mine.tolist(), back.tolist())
            if isinstance(a, str) and a != ""
        )
    if not isinstance(mine.dtype, pd.StringDtype):
        mine = mine.astype(object)
    mine = mine.reset_index(drop=True)
    back = back.reset_index(drop=True)
    keep = (mine.notna() & mine.ne("")).to_numpy(dtype=bool)
    if not keep.any():
        return False
    mine, back = mine[keep], back[keep]
    if isinstance(back.dtype, pd.StringDtype):
        if back.isna().any():
            return True
    elif pd.api.types.infer_dtype(
        np.asarray(back, dtype=object), skipna=False
    ) != "string":
        return True

    return bool(mine.ne(back).any())


def column_types(
    frame: pd.DataFrame, path: Path
) -> tuple[dict[str, str], dict[str, list]]:
    """
    書いた予測の表を型を知らずに読み直すと値の変わる列の型と, 混ざる列の
    数に見える文字の行の位置を返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        書き出した表
    path: Path
        書いた predictions.csv

    Returns
    ----------
    types: dict[str, str]
        列の名前から型 (text か integer か mixed) への対応 (値の変わらない
        列と, 値ごとの型に戻らない混ざる列は除く)
    rows: dict[str, list]
        混ざる列の名前から, 数に見える文字の値の行の位置 (0 から) の並び
        への対応 (その行の無い列は除く)
    """
    # 候補の列だけを, 型を知らずに読んだ値と列ごとに比べる (列は位置で
    # 引く. 混ざる列は値の種類が 2 つ以上あり 1 つの型で読むとどれかの
    # 種類が必ず変わるので, 同じ 1 回の読み込みで文字で読み, 値ごとの型に
    # 戻せるかを比べる)
    kinds = [column_kind(frame.iloc[:, i]) for i in range(frame.shape[1])]
    if not any(kinds):
        return {}, {}
    mixed = {str(frame.columns[i]): "text"
             for i, kind in enumerate(kinds) if kind == MIXED}
    plain = read_predictions(path, mixed)
    types, rows = {}, {}
    for position, kind in enumerate(kinds):
        name = str(frame.columns[position])
        mine, back = frame.iloc[:, position], plain.iloc[:, position]
        if kind == MIXED:
            text = mixed_text_rows(mine.tolist(), back)
            if text is not None:
                types[name] = kind
            if text:
                rows[name] = text
        elif kind is not None and column_changed(mine, back, kind):
            types[name] = kind

    return types, rows


def given_types(types: Optional[dict], name: str) -> dict[str, Any]:
    """
    呼び手の渡した列の型を, 読み込みの型に直す関数

    Parameters
    ----------
    types: Optional[dict]
        列の名前から型 (text か integer) への対応
    name: str
        誤りの文に出す引数の名前

    Returns
    ----------
    dtypes: dict[str, Any]
        列の名前から読み込みの型への対応

    Raises
    ----------
    ValueError
        辞書でない場合, 知らない型の名前がある場合
    """
    # 渡した型の名前の確かめ (column_types と同じ名前だけ)
    if types is None:
        return {}
    if not isinstance(types, dict):
        raise ValueError(f"{name} は列の名前から型への辞書で渡してください")
    dtypes = {}
    for column, kind in types.items():
        if not isinstance(kind, str) or kind not in COLUMN_DTYPES:
            known = ", ".join(COLUMN_DTYPES)
            raise ValueError(
                f"{name} の列 {column} の型 {kind!r} は使えません "
                f"({known} のどれか)"
            )
        dtypes[str(column)] = COLUMN_DTYPES[kind]

    return dtypes


def read_predictions(
    path: Path, types: Optional[dict] = None,
    override_types: Optional[dict] = None,
    fallback_types: Optional[dict] = None,
    text_rows: Optional[dict] = None
) -> pd.DataFrame:
    """
    予測の表を読む関数 (文の出力の文と番号は文字のまま)

    Parameters
    ----------
    path: Path
        predictions.csv
    types: Optional[dict] = None
        result.yaml の列の型 (列の名前から text か integer か mixed)
    override_types: Optional[dict] = None
        呼び手の列の型 (types と文の出力の文と番号より強く当てる)
    fallback_types: Optional[dict] = None
        呼び手の列の型 (types に無い列だけに, 文の出力の文と番号より
        弱く当てる)
    text_rows: Optional[dict] = None
        result.yaml の混ざる列の数に見える文字の行の位置

    Returns
    ----------
    frame: pd.DataFrame
        予測の表 (空の欄は欠損. 文の出力の予測した行の空の文は空の文字.
        text の列は文字, integer の列は欠損を持てる整数, mixed の列は
        値ごとの型)

    Raises
    ----------
    ValueError
        呼び手の列の型が辞書でない場合, 知らない型の名前がある場合
    """
    # 弱い順に重ねる (呼び手の補いの型, 文の出力の文と番号, 残した列の型,
    # 呼び手の強い型. 表に無い列と知らない残した型は無視)
    strong = given_types(override_types, "override_types")
    weak = given_types(fallback_types, "fallback_types")
    known = {**COLUMN_DTYPES, MIXED: str}
    kept = {str(name): known[kind]
            for name, kind in (types or {}).items() if kind in known}
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    columns = set(head.columns)
    dtype: dict[str, Any] = {
        name: kind for name, kind in weak.items() if name not in kept
    }

    # tokens の列のある表は文の出力なので, 文と番号を文字で読む (番号が 1
    # つの並びや数に見える文を数にしない)
    text = TOKENS_COLUMN in columns
    if text:
        dtype.update({"prediction": str, TOKENS_COLUMN: str})
    dtype.update(kept)
    dtype.update(strong)
    dtype = {name: kind for name, kind in dtype.items() if name in columns}
    frame = pd.read_csv(
        path, encoding="utf-8-sig", keep_default_na=False, na_values=[""],
        dtype=dtype or None
    )

    # 混ざる列は文字で読んでから値ごとの型に戻す (呼び手の強い型を当てた
    # 列は除く)
    for name, kind in (types or {}).items():
        name = str(name)
        if kind == MIXED and name in columns and name not in strong:
            frame[name] = mixed_values(frame[name],
                                       text_positions(text_rows, name))

    # 予測した行の空の文 (戻すと何も残らない並び) は欠損と分ける (呼び手
    # が文字でない型を強く当てた列は除く)
    if text and "status" in frame.columns:
        done = frame["status"] == OK
        for name in ("prediction", TOKENS_COLUMN):
            if name in frame.columns and strong.get(name, str) is str:
                frame[name] = frame[name].astype(object)
                frame.loc[done & frame[name].isna(), name] = ""

    return frame


def read_result(
    path: Union[str, Path], override_types: Optional[dict] = None,
    fallback_types: Optional[dict] = None
) -> dict:
    """
    書き出した結果を読む関数

    Parameters
    ----------
    path: Union[str, Path]
        結果のディレクトリ
    override_types: Optional[dict] = None
        呼び手の都合の列の型 (列の名前から text か integer.
        column_types と文の出力の文と番号より強く当てる)
    fallback_types: Optional[dict] = None
        呼び手の補いの列の型 (列の名前から text か integer.
        column_types に無い列だけに, 文の出力の文と番号より弱く当てる.
        column_types の無い版 1 と 2 の結果に呼び手が残した型など)

    Returns
    ----------
    result: dict
        result.yaml の中身と, frame (予測と判断の表. column_types の列は
        その型. mixed の列は値ごとの型. 配列は artifact の列の相対パスのまま. 読むのは
        artifact_path と numpy.load), form
        (配列の出力の形. 無ければ None), directory (読んだ場所. 書き直す
        ときに配列を持ち越すため)

    Raises
    ----------
    FileNotFoundError
        result.yaml が無い場合
    ValueError
        新しい版の場合, 呼び手の列の型が辞書でない場合, 知らない型の
        名前がある場合
    """
    # 知らない新しい版の読み込みの停止
    root = Path(path).expanduser()
    file = root / "result.yaml"
    if not file.is_file():
        raise FileNotFoundError(f"{root} に result.yaml がありません")
    meta = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    version = meta.get("format_version")
    if not isinstance(version, int) or version > RESULT_FORMAT_VERSION:
        raise ValueError(
            f"format_version {version} の結果は, このライブラリ (版 "
            f"{RESULT_FORMAT_VERSION} まで) では読めません"
        )
    types = meta.get(COLUMN_TYPES_KEY)
    frame = read_predictions(
        root / "predictions.csv", types if isinstance(types, dict) else None,
        override_types, fallback_types, meta.get(TEXT_ROWS_KEY)
    )
    artifacts = meta.get("artifacts") or {}
    if ARTIFACT_COLUMN in frame.columns and artifacts:
        frame[ARTIFACT_COLUMN] = frame[ARTIFACT_COLUMN].fillna("")
    result = {
        **meta, "frame": frame, "form": artifacts.get("form"),
        "directory": str(root.resolve()),
    }

    return result
