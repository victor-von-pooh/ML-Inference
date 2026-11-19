"""
繰り返しの手順 (拡散のサンプリング, 文の生成) の登録簿

PyTorch だけを使う純粋な関数で, 推論の一式, ML-EDA, ML-Provide に
依らない. 学習の途中の見本 (ML-Found) からも同じ名前で使える
"""
from dataclasses import dataclass
import inspect
import math
from typing import Any, Callable

import torch

# 始まりの中身の種類 (雑音から, 渡した中身から)
STARTS = ("noise", "given")


@dataclass
class Procedure:
    # 名前と説明
    name: str
    description: str
    # 手順の関数 function(model, state, steps, generator, **options)
    function: Callable[..., torch.Tensor]
    # 始まりの中身 (noise か given)
    start: str
    # モデルに時刻を渡すか
    time: bool
    # 組み込みの手順か
    builtin: bool = False


def make_generator(seed: int) -> torch.Generator:
    """
    乱数の種から, 手順が乱数を引く生成器を作る関数

    Parameters
    ----------
    seed: int
        乱数の種

    Returns
    ----------
    generator: torch.Generator
        cpu の生成器
    """
    # 手順の乱数は全てこの生成器から
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    return generator


def start_noise(
    shape: tuple, generator: torch.Generator
) -> torch.Tensor:
    """
    雑音から始まる手順の, 始まりの雑音を引く関数

    Parameters
    ----------
    shape: tuple
        雑音の形 (件数を含む)
    generator: torch.Generator
        乱数の生成器

    Returns
    ----------
    noise: torch.Tensor
        標準正規分布の雑音 (float32, cpu)
    """
    # 標準正規分布
    noise = torch.randn(
        tuple(int(v) for v in shape), generator=generator,
        dtype=torch.float32
    )

    return noise


def draw_noise(
    like: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """
    中身と同じ形とデバイスの雑音を引く関数

    Parameters
    ----------
    like: torch.Tensor
        形とデバイスを合わせる中身
    generator: torch.Generator
        乱数の生成器 (cpu)

    Returns
    ----------
    noise: torch.Tensor
        標準正規分布の雑音
    """
    # cpu で引いてから移す (デバイスに依らず同じ値)
    noise = start_noise(tuple(like.shape), generator).to(
        device=like.device, dtype=like.dtype
    )

    return noise


def time_tensor(
    value: Any, count: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    件数の分だけ同じ時刻を並べる関数

    Parameters
    ----------
    value: Any
        時刻
    count: int
        件数
    dtype: torch.dtype
        型
    device: torch.device
        デバイス

    Returns
    ----------
    times: torch.Tensor
        (件数,) の時刻
    """
    # 件ごとに同じ時刻
    times = torch.full((count,), value, dtype=dtype, device=device)

    return times


def ddim(
    model: Callable[..., torch.Tensor], state: torch.Tensor, steps: int,
    generator: torch.Generator, train_steps: int = 1000,
    beta_start: float = 1e-4, beta_end: float = 0.02, eta: float = 0.0
) -> torch.Tensor:
    """
    拡散のモデルで, 雑音から DDIM のサンプリングをする関数

    Parameters
    ----------
    model: Callable[..., torch.Tensor]
        model(x, t) で加えた雑音を予測するもの (t は (件数,) の整数)
    state: torch.Tensor
        始まりの雑音
    steps: int
        繰り返しの回数 (train_steps 以下)
    generator: torch.Generator
        乱数の生成器 (eta が 0 より大きいときだけ使う)
    train_steps: int = 1000
        学習の時刻の数
    beta_start: float = 1e-4
        線形の雑音の予定の始まり
    beta_end: float = 0.02
        線形の雑音の予定の終わり
    eta: float = 0.0
        各回に足す雑音の大きさ (0 で決まった道)

    Returns
    ----------
    x: torch.Tensor
        作った中身
    """
    # 線形の雑音の予定と, 間引いた時刻 (大きい順)
    if not 1 <= steps <= train_steps:
        raise ValueError(
            f"ddim の回数は 1 から train_steps ({train_steps}) にしてください"
        )
    betas = torch.linspace(
        beta_start, beta_end, train_steps, dtype=torch.float64
    )
    alphas = torch.cumprod(1 - betas, dim=0)
    times = torch.linspace(train_steps - 1, 0, steps).round().long()
    times = torch.unique_consecutive(times).tolist()

    # 予測した雑音を引いて元のデータを見積もり, 1 つ前の時刻へ
    x = state
    for index, t in enumerate(times):
        now = float(alphas[t])
        before = float(alphas[times[index + 1]]) if (
            index + 1 < len(times)
        ) else 1.0
        eps = model(x, time_tensor(t, len(x), torch.long, x.device))
        x0 = (x - math.sqrt(1 - now) * eps) / math.sqrt(now)
        sigma = eta * math.sqrt(
            max((1 - before) / (1 - now) * (1 - now / before), 0.0)
        )
        x = math.sqrt(before) * x0 + math.sqrt(
            max(1 - before - sigma ** 2, 0.0)
        ) * eps
        if sigma > 0:
            x = x + sigma * draw_noise(x, generator)

    return x


def euler(
    model: Callable[..., torch.Tensor], state: torch.Tensor, steps: int,
    generator: torch.Generator, t_start: float = 0.0, t_end: float = 1.0
) -> torch.Tensor:
    """
    フローのモデルで, 速度を Euler 法で足していく関数

    Parameters
    ----------
    model: Callable[..., torch.Tensor]
        model(x, t) で速度を返すもの (t は (件数,) の小数)
    state: torch.Tensor
        時刻 t_start の中身 (雑音)
    steps: int
        繰り返しの回数
    generator: torch.Generator
        乱数の生成器 (使わない. 手順の関数の形をそろえるため)
    t_start: float = 0.0
        始まりの時刻
    t_end: float = 1.0
        終わりの時刻

    Returns
    ----------
    x: torch.Tensor
        時刻 t_end の中身
    """
    # 等しい刻みで速度を足す
    if steps < 1:
        raise ValueError("euler の回数は 1 以上にしてください")
    x = state
    width = (t_end - t_start) / steps
    for index in range(steps):
        t = t_start + index * width
        velocity = model(x, time_tensor(t, len(x), x.dtype, x.device))
        x = x + width * velocity

    return x


def last_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    文のモデルの出力から, 次の番号の logits を取り出す関数

    Parameters
    ----------
    logits: torch.Tensor
        (件数, 長さ, 語彙) か (件数, 語彙)

    Returns
    ----------
    last: torch.Tensor
        (件数, 語彙) の logits
    """
    # 3 次元は最後の位置
    if logits.ndim == 3:
        last = logits[:, -1]
    elif logits.ndim == 2:
        last = logits
    else:
        raise ValueError(
            "文のモデルの出力は (件数, 長さ, 語彙) か (件数, 語彙) にして"
            f"ください (今は {tuple(logits.shape)})"
        )

    return last


def extend_tokens(
    state: torch.Tensor, chosen: torch.Tensor, done: torch.Tensor,
    eos: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    選んだ番号を並びの後ろに足し, 終わった件を記録する関数

    Parameters
    ----------
    state: torch.Tensor
        (件数, 長さ) の番号の並び
    chosen: torch.Tensor
        (件数,) の選んだ番号
    done: torch.Tensor
        (件数,) の終わったか
    eos: Any
        終わりの番号 (None なら終わらない)

    Returns
    ----------
    state: torch.Tensor
        (件数, 長さ + 1) の並び
    done: torch.Tensor
        終わったか
    """
    # 終わった件は終わりの番号で埋める
    if eos is not None:
        chosen = torch.where(done, torch.full_like(chosen, int(eos)), chosen)
        done = done | (chosen == int(eos))
    state = torch.cat([state, chosen.to(state.dtype)[:, None]], dim=1)

    return state, done


def greedy(
    model: Callable[..., torch.Tensor], state: torch.Tensor, steps: int,
    generator: torch.Generator, eos: Any = None
) -> torch.Tensor:
    """
    毎回最も確率の高い番号を足して, 文の番号の並びを作る関数

    Parameters
    ----------
    model: Callable[..., torch.Tensor]
        model(ids) で次の番号の logits を返すもの
    state: torch.Tensor
        (件数, 長さ) の始まりの番号の並び
    steps: int
        足す番号の数の上限
    generator: torch.Generator
        乱数の生成器 (使わない. 手順の関数の形をそろえるため)
    eos: Any = None
        終わりの番号 (全ての件が出したら止める)

    Returns
    ----------
    state: torch.Tensor
        番号を足した並び
    """
    # 最も大きい logits の番号
    if steps < 1:
        raise ValueError("greedy の回数は 1 以上にしてください")
    done = torch.zeros(len(state), dtype=torch.bool, device=state.device)
    for _ in range(steps):
        chosen = last_logits(model(state)).argmax(dim=-1)
        state, done = extend_tokens(state, chosen, done, eos)
        if eos is not None and bool(done.all()):
            break

    return state


def sample(
    model: Callable[..., torch.Tensor], state: torch.Tensor, steps: int,
    generator: torch.Generator, temperature: float = 1.0, eos: Any = None
) -> torch.Tensor:
    """
    温度で割った確率から番号を引いて, 文の番号の並びを作る関数

    Parameters
    ----------
    model: Callable[..., torch.Tensor]
        model(ids) で次の番号の logits を返すもの
    state: torch.Tensor
        (件数, 長さ) の始まりの番号の並び
    steps: int
        足す番号の数の上限
    generator: torch.Generator
        乱数の生成器
    temperature: float = 1.0
        温度 (0 より大きい. 小さいほど確率の高い番号に寄る)
    eos: Any = None
        終わりの番号 (全ての件が出したら止める)

    Returns
    ----------
    state: torch.Tensor
        番号を足した並び
    """
    # 確率は cpu で求めて引く (デバイスに依らず同じ値)
    if steps < 1:
        raise ValueError("sample の回数は 1 以上にしてください")
    if not temperature > 0:
        raise ValueError("sample の temperature は 0 より大きくしてください")
    done = torch.zeros(len(state), dtype=torch.bool, device=state.device)
    for _ in range(steps):
        logits = last_logits(model(state)).detach().double().cpu()
        probability = torch.softmax(logits / float(temperature), dim=-1)
        chosen = torch.multinomial(
            probability, 1, generator=generator
        )[:, 0].to(state.device)
        state, done = extend_tokens(state, chosen, done, eos)
        if eos is not None and bool(done.all()):
            break

    return state


# 名前から手順を引く登録簿
PROCEDURES: dict[str, Procedure] = {
    "ddim": Procedure(
        "ddim", "拡散のモデルの DDIM のサンプリング", ddim, "noise", True,
        True
    ),
    "euler": Procedure(
        "euler", "フローのモデルの Euler 法", euler, "noise", True, True
    ),
    "greedy": Procedure(
        "greedy", "最も確率の高い番号を足す文の生成", greedy, "given",
        False, True
    ),
    "sample": Procedure(
        "sample", "温度で割った確率から引く文の生成", sample, "given",
        False, True
    ),
}


def register_procedure(
    name: str, function: Callable[..., torch.Tensor], start: str,
    time: bool, description: str = "", overwrite: bool = False
) -> Procedure:
    """
    自作の繰り返しの手順を登録簿に足す関数

    Parameters
    ----------
    name: str
        手順の名前
    function: Callable[..., torch.Tensor]
        function(model, state, steps, generator, **options)
    start: str
        始まりの中身 (noise か given)
    time: bool
        モデルに時刻を渡すか
    description: str = ""
        説明
    overwrite: bool = False
        同じ名前の自作の手順を置き換えるか (組み込みは置き換えない)

    Returns
    ----------
    procedure: Procedure
        登録した手順

    Raises
    ----------
    ValueError
        名前が空, 組み込みの名前, 既にある名前, 知らない始まりの場合
    """
    # 組み込みの手順は置き換えない
    if not name:
        raise ValueError("手順の名前が空です")
    if start not in STARTS:
        raise ValueError(f"start は {', '.join(STARTS)} のどちらかにしてください")
    known = PROCEDURES.get(name)
    if known is not None and (known.builtin or not overwrite):
        raise ValueError(f"手順 {name} は既にあります")
    procedure = Procedure(name, description, function, start, bool(time))
    PROCEDURES[name] = procedure

    return procedure


def get_procedure(name: str) -> Procedure:
    """
    名前から繰り返しの手順を引く関数

    Parameters
    ----------
    name: str
        手順の名前

    Returns
    ----------
    procedure: Procedure
        手順

    Raises
    ----------
    KeyError
        登録簿に無い名前の場合
    """
    # 無い名前は登録簿の名前を添える
    if name not in PROCEDURES:
        raise KeyError(
            f"手順 {name} は登録簿にありません (今の手順: "
            f"{', '.join(sorted(PROCEDURES))})"
        )
    procedure = PROCEDURES[name]

    return procedure


def option_names(name: str) -> list[str]:
    """
    手順の設定 (関数の 5 つ目からの引数) の名前を返す関数

    Parameters
    ----------
    name: str
        手順の名前

    Returns
    ----------
    names: list[str]
        設定の名前 (**kwargs を持つ関数は空で, 何でも受け取る)
    """
    # model, state, steps, generator の後の引数
    signature = inspect.signature(get_procedure(name).function)
    params = list(signature.parameters.values())[4:]
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return []
    names = [p.name for p in params]

    return names


def run_procedure(
    name: str, model: Callable[..., torch.Tensor], state: torch.Tensor,
    steps: int, generator: torch.Generator, **options: Any
) -> torch.Tensor:
    """
    名前で選んだ繰り返しの手順を, 勾配を計算せずに動かす関数

    Parameters
    ----------
    name: str
        手順の名前
    model: Callable[..., torch.Tensor]
        呼べるもの (nn.Module に限らない)
    state: torch.Tensor
        始まりの中身 (雑音か番号の並び)
    steps: int
        繰り返しの回数
    generator: torch.Generator
        乱数の生成器 (make_generator で作る)
    **options: Any
        手順ごとの設定

    Returns
    ----------
    result: torch.Tensor
        手順の出した中身
    """
    # 同じ入力と同じ生成器の状態なら同じ出力
    procedure = get_procedure(name)
    with torch.no_grad():
        result = procedure.function(
            model, state, int(steps), generator, **options
        )

    return result
