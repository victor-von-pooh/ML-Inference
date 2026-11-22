from typing import Any, Optional

import numpy as np

# 確率の補正の選び方
CALIBRATE_METHODS = ("platt", "isotonic")

# 補正で確率を logit にするときの端の値と, Platt の当てはめの回数
EPSILON = 1e-6
PLATT_STEPS = 100

# calibrate の欄
CALIBRATE_KEYS = ("method", "split", "rows", "classes", "before")


def sigmoid(values: np.ndarray) -> np.ndarray:
    """
    桁あふれしない sigmoid の関数

    Parameters
    ----------
    values: np.ndarray
        logit

    Returns
    ----------
    probability: np.ndarray
        1 / (1 + e^-x)
    """
    # tanh で書くと大きな値でも桁あふれしない
    probability = 0.5 * (1 + np.tanh(values / 2))

    return probability


def logit(probability: np.ndarray) -> np.ndarray:
    """
    確率を端で切ってから logit にする関数

    Parameters
    ----------
    probability: np.ndarray
        確率

    Returns
    ----------
    values: np.ndarray
        log(p / (1 - p))
    """
    # 0 と 1 は無限大になるので端で切る
    clipped = np.clip(probability.astype(np.float64), EPSILON, 1 - EPSILON)
    values = np.log(clipped / (1 - clipped))

    return values


def fit_platt(probability: np.ndarray, hit: np.ndarray) -> dict:
    """
    確率の logit に 1 次の式を当てる補正 (Platt) を当てはめる関数

    補正した確率は sigmoid(a × logit(p) + b). 当たりと外れの的は Platt の
    決まりで少しずらす (当たり (N+ + 1) / (N+ + 2), 外れ 1 / (N- + 2).
    分けきれる組でも a が無限に大きくならないため)

    Parameters
    ----------
    probability: np.ndarray
        補正の前の確率
    hit: np.ndarray
        当たったか

    Returns
    ----------
    params: dict
        a と b
    """
    # 的と, ニュートン法で負の対数尤度を最も小さくする a, b (損が減るまで
    # 歩幅を半分にする)
    z = logit(probability)
    hit = hit.astype(bool)
    positive, negative = int(hit.sum()), int((~hit).sum())
    goal = np.where(hit, (positive + 1) / (positive + 2),
                    1 / (negative + 2))

    def loss(a: float, b: float) -> float:
        # 負の対数尤度 (log(1 + e^s) を桁あふれせずに)
        s = a * z + b
        return float((np.logaddexp(0, s) - goal * s).sum())

    a, b = 1.0, 0.0
    current = loss(a, b)
    for _ in range(PLATT_STEPS):
        p = sigmoid(a * z + b)
        gradient = np.array([((p - goal) * z).sum(), (p - goal).sum()])
        weight = p * (1 - p)
        hessian = np.array([
            [(weight * z * z).sum() + 1e-9, (weight * z).sum()],
            [(weight * z).sum(), weight.sum() + 1e-9],
        ])
        step = np.linalg.solve(hessian, gradient)
        size = 1.0
        while size > 1e-8 and loss(a - size * step[0],
                                   b - size * step[1]) > current:
            size /= 2
        if size <= 1e-8:
            break
        a, b = a - size * float(step[0]), b - size * float(step[1])
        previous, current = current, loss(a, b)
        if previous - current < 1e-12 * max(1.0, abs(previous)):
            break
    params = {"a": float(a), "b": float(b)}

    return params


def fit_isotonic(probability: np.ndarray, hit: np.ndarray) -> dict:
    """
    確率の順を保つ階段の補正 (isotonic) を当てはめる関数

    確率の小さい順に, 当たった割合が下がる隣どうしを 1 つの塊にまとめる
    (pool adjacent violators). 塊ごとの確率の範囲の両端と当たった割合を
    残し, 間は直線でつなぐ. 範囲の外は端の値

    Parameters
    ----------
    probability: np.ndarray
        補正の前の確率
    hit: np.ndarray
        当たったか

    Returns
    ----------
    params: dict
        x (補正の前の確率の並び) と y (補正した確率の並び)
    """
    # 小さい順の塊 (確率の下端, 上端, 当たりの数, 行の数)
    order = np.argsort(probability, kind="stable")
    xs = probability[order].astype(np.float64)
    ys = hit[order].astype(np.float64)
    blocks: list[list[float]] = []
    for x, y in zip(xs, ys):
        blocks.append([x, x, y, 1.0])
        while len(blocks) > 1 and (
            blocks[-2][2] / blocks[-2][3] >= blocks[-1][2] / blocks[-1][3]
        ):
            low, _, hits, count = blocks.pop(-2)
            blocks[-1] = [low, blocks[-1][1], blocks[-1][2] + hits,
                          blocks[-1][3] + count]

    # 塊の両端の点 (幅の無い塊は 1 つ)
    x_points: list[float] = []
    y_points: list[float] = []
    for low, high, hits, count in blocks:
        rate = hits / count
        for x in ((low,) if low == high else (low, high)):
            x_points.append(float(x))
            y_points.append(float(rate))
    params = {"x": x_points, "y": y_points}

    return params


def calibrated(
    params: dict, method: str, probability: np.ndarray
) -> np.ndarray:
    """
    1 つのクラスの確率に補正をかける関数

    Parameters
    ----------
    params: dict
        当てはめた値 (platt は a と b, isotonic は x と y)
    method: str
        補正の選び方
    probability: np.ndarray
        補正の前の確率

    Returns
    ----------
    values: np.ndarray
        補正した確率
    """
    # Platt は logit の 1 次の式, isotonic は点の間を直線で
    if method == "platt":
        values = sigmoid(
            float(params["a"]) * logit(probability) + float(params["b"])
        )
    else:
        values = np.interp(
            probability.astype(np.float64),
            np.asarray(params["x"], dtype=np.float64),
            np.asarray(params["y"], dtype=np.float64),
        )

    return values


def fit_calibration(
    method: str, classes: list, probability: np.ndarray,
    answers: np.ndarray
) -> dict:
    """
    クラスごとの確率に補正を当てはめる関数

    2 クラスは 2 つ目のクラスの確率だけ (1 つ目は残り), 3 クラス以上は
    クラスごとに「そのクラスか」の確率を補正する

    Parameters
    ----------
    method: str
        補正の選び方 (platt か isotonic)
    classes: list
        クラスの名前
    probability: np.ndarray
        (行, クラス) の補正の前の確率
    answers: np.ndarray
        正しいクラスの番号

    Returns
    ----------
    fitted: dict
        クラスの名前ごとの当てはめた値
    """
    # 補正するクラスの番号
    fit = fit_platt if method == "platt" else fit_isotonic
    answers = answers.astype(int)
    indices = [1] if len(classes) == 2 else range(len(classes))
    fitted = {
        str(classes[index]): fit(probability[:, index], answers == index)
        for index in indices
    }

    return fitted


def apply_calibration(
    calibrate: dict, classes: list, probability: np.ndarray
) -> np.ndarray:
    """
    一式の calibrate で, (行, クラス) の確率を補正する関数

    3 クラス以上は, クラスごとに補正した確率を行の和で割って足して 1 に
    する (和が 0 の行は補正の前のまま)

    Parameters
    ----------
    calibrate: dict
        一式の calibrate
    classes: list
        クラスの名前
    probability: np.ndarray
        (行, クラス) の補正の前の確率

    Returns
    ----------
    values: np.ndarray
        (行, クラス) の補正した確率
    """
    # 2 クラスは 2 つ目のクラスの確率から
    method = calibrate["method"]
    fitted = calibrate["classes"]
    if len(classes) == 2:
        positive = calibrated(fitted[str(classes[1])], method,
                              probability[:, 1])
        values = np.stack([1 - positive, positive], axis=1)
        return values

    # 3 クラス以上はクラスごとに補正して和で割る
    parts = np.stack([
        calibrated(fitted[str(name)], method, probability[:, index])
        for index, name in enumerate(classes)
    ], axis=1) if len(probability) else probability.astype(np.float64)
    total = parts.sum(axis=1, keepdims=True)
    values = np.where(total > 0, parts / np.where(total > 0, total, 1),
                      probability)

    return values


def check_calibrate(calibrate: Any, classes: Optional[list]) -> list[str]:
    """
    inference.yaml の calibrate の形を確かめる関数

    Parameters
    ----------
    calibrate: Any
        calibrate の欄
    classes: Optional[list]
        一式のクラスの名前 (分類でなければ None)

    Returns
    ----------
    errors: list[str]
        誤りの一覧 (無ければ空)
    """
    # 欄の名前, 選び方, 補正するクラス
    if not isinstance(calibrate, dict):
        return ["calibrate は辞書にしてください"]
    errors = []
    unknown = [str(k) for k in calibrate if k not in CALIBRATE_KEYS]
    if unknown:
        errors.append(
            f"calibrate に知らない欄があります: {', '.join(unknown)}"
        )
    method = calibrate.get("method")
    if method not in CALIBRATE_METHODS:
        errors.append(
            f"calibrate.method は {', '.join(CALIBRATE_METHODS)} のどれか"
            "にしてください"
        )
    if not classes:
        errors.append("確率の補正 (calibrate) は分類の一式だけに使えます")
        return errors
    fitted = calibrate.get("classes")
    names = [str(classes[1])] if len(classes) == 2 else [
        str(c) for c in classes
    ]
    if not isinstance(fitted, dict) or sorted(map(str, fitted)) != sorted(
        names
    ):
        errors.append(
            f"calibrate.classes はクラス {', '.join(names)} ごとの値にして"
            "ください"
        )
        return errors
    wanted = {"platt": ("a", "b"), "isotonic": ("x", "y")}.get(method)
    for name, params in fitted.items():
        if wanted is None:
            break
        if not isinstance(params, dict) or any(k not in params
                                               for k in wanted):
            errors.append(
                f"calibrate.classes.{name} には {', '.join(wanted)} が要り"
                "ます"
            )
        elif method == "isotonic" and not (
            isinstance(params["x"], list) and isinstance(params["y"], list)
            and params["x"] and len(params["x"]) == len(params["y"])
        ):
            errors.append(
                f"calibrate.classes.{name} の x と y は同じ長さの並びにして"
                "ください"
            )

    return errors
