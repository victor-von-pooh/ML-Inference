from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from ml_eda import Data

from ml_inference.bundle import Bundle, load_bundle
from ml_inference.predict import TABLE_KINDS, predict, read_source


def explain(
    bundle: Union[str, Path, Bundle],
    source: Union[str, Path, pd.DataFrame, Data],
    options: Optional[dict] = None, repeats: int = 3,
    max_rows: int = 2000, seed: int = 0
) -> dict:
    """
    渡したデータの元の列ごとに, 予測を動かす大きさ (並べ替えの重要度) を
    求める関数

    Parameters
    ----------
    bundle: Union[str, Path, Bundle]
        推論の一式かそのディレクトリ
    source: Union[str, Path, pd.DataFrame, Data]
        推論するデータ (表か時系列)
    options: Optional[dict] = None
        読み込みの設定 (学習のときの設定に上書きする)
    repeats: int = 3
        列ごとに並べ替える回数
    max_rows: int = 2000
        求めるのに使う行の上限 (多い場合は無作為に選ぶ. 過去の値の列を
        作るレシピは, 前後関係を残すために全ての行を使う)
    seed: int = 0
        乱数のシード

    Returns
    ----------
    result: dict
        measure (分類は probability, 回帰と予測は value), rows (使った
        行の数), columns (列ごとの column, effect, spread, rows の並び.
        効きの大きい順)

    Raises
    ----------
    ValueError
        表と時系列でない一式の場合と, 予測できる行が無い場合
    """
    # 表と時系列の元の列だけ
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    if bundle.kind not in TABLE_KINDS:
        raise ValueError(
            f"{bundle.kind} の一式は, 1 件の中身が列ではないので列の効きを"
            "求めません"
        )
    data = read_source(bundle, source, options)
    frame = data.frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    target = bundle.dataset.info.get("target")
    lagged = [
        spec.get("state", {}).get("columns", [])
        for spec in bundle.dataset.recipe.get("steps", [])
        if spec["op"] == "lag"
    ]
    # 過去の値の列は前の行が要るので, 行を選ぶのは過去の値の列が無い場合だけ
    if len(frame) > max_rows and not lagged:
        chosen = np.sort(rng.choice(len(frame), max_rows, replace=False))
        frame = frame.iloc[chosen].reset_index(drop=True)

    # 並べ替える前の予測
    classify = bundle.task == "classification"
    base = predict(bundle, frame)
    ok = (base["status"] == "ok").to_numpy()
    if not ok.any():
        raise ValueError("予測できる行が無いので, 列の効きを求められません")
    chosen_class = base["prediction"].to_numpy()

    def measure(table: pd.DataFrame) -> np.ndarray:
        """
        予測の表から, 並べ替える前に予測したクラスの確率か予測の値を返す
        関数

        Parameters
        ----------
        table: pd.DataFrame
            predict の表

        Returns
        ----------
        values: np.ndarray
            行ごとの値 (予測しなかった行は欠損)
        """
        # 回帰と予測は値, 分類は行ごとに並べ替える前に予測したクラスの確率
        if not classify:
            return pd.to_numeric(
                table["prediction"], errors="coerce"
            ).to_numpy(dtype=float)
        values = np.full(len(table), np.nan)
        for name in bundle.classes:
            hit = chosen_class == name
            values[hit] = pd.to_numeric(
                table[f"p:{name}"], errors="coerce"
            ).to_numpy(dtype=float)[hit]

        return values

    reference = measure(base)

    # 列ごとに並べ替えて, 予測の変わり方の平均. 目的の列は, 過去の値の
    # 列を作る場合だけ「過去の値」として求める
    reads_target = any(target in names for names in lagged)
    columns = []
    for column in frame.columns:
        if column == target and not reads_target:
            continue
        effects = []
        used = 0
        for _ in range(repeats):
            shuffled = frame.copy()
            shuffled[column] = rng.permutation(frame[column].to_numpy())
            changed = measure(predict(bundle, shuffled))
            both = ok & ~np.isnan(changed) & ~np.isnan(reference)
            used = int(both.sum())
            if used:
                effects.append(
                    float(np.mean(np.abs(changed[both] - reference[both])))
                )
        columns.append({
            "column": f"{column} (過去の値)" if column == target else str(
                column
            ),
            "effect": float(np.mean(effects)) if effects else None,
            "spread": float(np.std(effects)) if effects else None,
            "rows": used,
        })
    columns.sort(key=lambda item: -(item["effect"] or 0.0))
    result = {
        "measure": "probability" if classify else "value",
        "rows": int(ok.sum()),
        "columns": columns,
    }

    return result
