import datetime as dt
from pathlib import Path
import shutil
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import yaml
from ml_eda import Data

from ml_inference.bundle import Bundle, inside, load_bundle
from ml_inference.conclude import conclude, conclude_outputs, origin_sentence
from ml_inference.decision import decide
from ml_inference.forecast import forecast as forecast_rows
from ml_inference.predict import (
    ARRAY_FORMS, OK, noise_request, predict, target_of
)
from ml_inference.grouping import conclude_groups
from ml_inference.ranking import conclude_ranking, request_users

# 結果のファイル (result.yaml) の形の版 (表に収まらない中身が無ければ 1)
RESULT_FORMAT_VERSION = 2

# 表に収まらない中身の置き場と, 表の列
ARTIFACTS_DIR = "artifacts"
ARTIFACT_COLUMN = "artifact"

# 文の出力の番号の並びの列 (この列のある表は文の出力)
TOKENS_COLUMN = "tokens"

# 書き出すときに result.yaml へ写さない結果の項目 (表, 形, 読んだ場所,
# 読んだ版と中身の記録)
RESULT_OWN_KEYS = ("frame", "form", "directory", "format_version",
                   "artifacts")


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

    # 表は表計算のアプリで開ける BOM 付きの UTF-8
    frame.to_csv(root / "predictions.csv", index=False, encoding="utf-8-sig")

    # 表以外の説明 (形の版を先頭に. 中身が無ければ版 1 のまま. 読んだ結果
    # の版と中身の記録は書き直し)
    meta: dict[str, Any] = {
        "format_version": RESULT_FORMAT_VERSION if artifacts else 1
    }
    meta.update({
        k: v for k, v in result.items() if k not in RESULT_OWN_KEYS
    })
    if artifacts:
        meta["artifacts"] = artifacts
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


def read_predictions(path: Path) -> pd.DataFrame:
    """
    予測の表を読む関数 (文の出力の文と番号は文字のまま)

    Parameters
    ----------
    path: Path
        predictions.csv

    Returns
    ----------
    frame: pd.DataFrame
        予測の表 (空の欄は欠損. 文の出力の予測した行の空の文は空の文字)
    """
    # tokens の列のある表は文の出力なので, 文と番号を文字で読む (番号が 1
    # つの並びや数に見える文を数にしない)
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    text = TOKENS_COLUMN in head.columns
    frame = pd.read_csv(
        path, encoding="utf-8-sig", keep_default_na=False, na_values=[""],
        dtype={"prediction": str, TOKENS_COLUMN: str} if text else None
    )

    # 予測した行の空の文 (戻すと何も残らない並び) は欠損と分ける
    if text and "status" in frame.columns:
        done = frame["status"] == OK
        for name in ("prediction", TOKENS_COLUMN):
            if name in frame.columns:
                frame[name] = frame[name].astype(object)
                frame.loc[done & frame[name].isna(), name] = ""

    return frame


def read_result(path: Union[str, Path]) -> dict:
    """
    書き出した結果を読む関数

    Parameters
    ----------
    path: Union[str, Path]
        結果のディレクトリ

    Returns
    ----------
    result: dict
        result.yaml の中身と, frame (予測と判断の表. 配列は artifact の
        列の相対パスのまま. 読むのは artifact_path と numpy.load), form
        (配列の出力の形. 無ければ None), directory (読んだ場所. 書き直す
        ときに配列を持ち越すため)

    Raises
    ----------
    FileNotFoundError
        result.yaml が無い場合
    ValueError
        新しい版の場合
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
    frame = read_predictions(root / "predictions.csv")
    artifacts = meta.get("artifacts") or {}
    if ARTIFACT_COLUMN in frame.columns and artifacts:
        frame[ARTIFACT_COLUMN] = frame[ARTIFACT_COLUMN].fillna("")
    result = {
        **meta, "frame": frame, "form": artifacts.get("form"),
        "directory": str(root.resolve()),
    }

    return result
