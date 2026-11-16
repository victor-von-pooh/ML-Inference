import datetime as dt
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml

from ml_eda import Data
from ml_inference.bundle import Bundle, load_bundle
from ml_inference.conclude import conclude
from ml_inference.predict import predict


def infer(
    bundle: Union[str, Path, Bundle],
    source: Union[str, Path, pd.DataFrame, Data],
    rules: Optional[dict] = None, options: Optional[dict] = None,
    batch_size: int = 256, device: str = "cpu"
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

    Returns
    ----------
    result: dict
        bundle (一式の名前と場所), source (入力), created_at, frame
        (予測と判断の表), summary, sentences, rules
    """
    # 一式の既定の決め方に, 渡した決め方を重ねる
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    merged = {**(bundle.info.get("conclusion") or {}), **(rules or {})}
    merged = {k: v for k, v in merged.items() if v is not None}

    # 予測と結論
    prediction = predict(bundle, source, options, batch_size, device)
    conclusion = conclude(prediction, bundle.task, bundle.classes, merged)
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

    # 表以外の説明
    meta = {k: v for k, v in result.items() if k != "frame"}
    (root / "result.yaml").write_text(
        yaml.safe_dump(meta, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )

    return root
