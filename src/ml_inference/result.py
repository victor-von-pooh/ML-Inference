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
from ml_inference.predict import ARRAY_FORMS, predict, target_of

# 結果のファイル (result.yaml) の形の版 (表に収まらない中身が無ければ 1)
RESULT_FORMAT_VERSION = 2

# 表に収まらない中身の置き場と, 表の列
ARTIFACTS_DIR = "artifacts"
ARTIFACT_COLUMN = "artifact"


def infer(
    bundle: Union[str, Path, Bundle],
    source: Union[str, Path, pd.DataFrame, Data],
    rules: Optional[dict] = None, options: Optional[dict] = None,
    batch_size: int = 256, device: str = "cpu", forecast: bool = False,
    decision: Optional[dict] = None
) -> dict:
    """
    新しいデータを推論し, 結論まで出す関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ
    source: Union[str, Path, pd.DataFrame, Data]
        推論するデータ
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

    # 列の出力でない一式は, 推論と件数だけ
    if not bundle.column_output:
        if given or decision or forecast:
            bundle.require_column("結論の決め方, 意思決定, 何段も先の予測")
        prediction = predict(bundle, source, options, batch_size, device)
        result = {
            "bundle": {"name": bundle.info.get("name"),
                       "path": str(bundle.path)},
            "source": str(source) if isinstance(source, (str, Path)) else None,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "form": bundle.form,
            **conclude_outputs(prediction),
        }
        return result

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
    """
    # 表に収まらない中身の artifacts/ への書き出し (前の結果の中身は削除)
    root = Path(path).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if (root / ARTIFACTS_DIR).is_dir():
        shutil.rmtree(root / ARTIFACTS_DIR)
    frame, artifacts = result["frame"], None
    if result.get("form") in ARRAY_FORMS:
        frame, artifacts = write_artifacts(frame, root)
        artifacts["form"] = result["form"]

    # 表は表計算のアプリで開ける BOM 付きの UTF-8
    frame.to_csv(root / "predictions.csv", index=False, encoding="utf-8-sig")

    # 表以外の説明 (形の版を先頭に. 中身が無ければ版 1 のまま)
    meta: dict[str, Any] = {
        "format_version": RESULT_FORMAT_VERSION if artifacts else 1
    }
    meta.update({
        k: v for k, v in result.items() if k not in ("frame", "form")
    })
    if artifacts:
        meta["artifacts"] = artifacts
    (root / "result.yaml").write_text(
        yaml.safe_dump(meta, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )

    return root


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
        列の相対パスのまま. 読むのは artifact_path と numpy.load)

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
    frame = pd.read_csv(
        root / "predictions.csv", encoding="utf-8-sig",
        keep_default_na=False, na_values=[""]
    )
    if ARTIFACT_COLUMN in frame.columns and meta.get("artifacts"):
        frame[ARTIFACT_COLUMN] = frame[ARTIFACT_COLUMN].fillna("")
    result = {**meta, "frame": frame}

    return result
