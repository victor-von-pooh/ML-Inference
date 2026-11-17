import datetime as dt
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml
from ml_eda import Data

from ml_inference.bundle import Bundle, load_bundle
from ml_inference.conclude import conclude
from ml_inference.decision import decide
from ml_inference.forecast import forecast as forecast_rows
from ml_inference.predict import predict

# 結果のファイル (result.yaml) の形の版
RESULT_FORMAT_VERSION = 1


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
    # 表は表計算のアプリで開ける BOM 付きの UTF-8
    root = Path(path).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    result["frame"].to_csv(
        root / "predictions.csv", index=False, encoding="utf-8-sig"
    )

    # 表以外の説明 (形の版を先頭に)
    meta = {"format_version": RESULT_FORMAT_VERSION}
    meta.update({k: v for k, v in result.items() if k != "frame"})
    (root / "result.yaml").write_text(
        yaml.safe_dump(meta, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )

    return root
