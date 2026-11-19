"""
自作の部品の登録と, 一式のモデルの定義の部品の確かめ

研究の自作の部品は, アプリの起動の設定で明示したモジュールからだけ登録する
(ML_tester の決めたこと 14). 一式やモデルの定義に書かれた名前からは何も
読み込まない
"""
from typing import Any

from ml_modelcontrol import get_unit

# 定義の部分モデルを指す部品の名前の頭 (登録簿を見ない)
SUBGRAPH_PREFIX = "sub."

# 登録されていない部品の足し方の案内
UNIT_HINT = (
    "自作の部品なら, 部品を宣言したモジュールの名前を起動の設定に足して"
    "起動し直すか, 一式を読む前に register_units([\"モジュールの名前\"]) "
    "を呼んでください (一式やモデルの定義に書かれた名前からは何も読み込み"
    "ません)"
)


class UnregisteredUnitError(ValueError):
    def __init__(self, units: list[str], nodes: list[str]) -> None:
        # 親クラスのコンストラクタを呼び出す
        super(UnregisteredUnitError, self).__init__(
            unit_message(units, nodes)
        )

        # 登録されていない部品の名前と, 使うノードを保存
        self.units = list(units)
        self.nodes = list(nodes)


def register_units(module_names: list[str]) -> dict[str, list[str]]:
    """
    起動の設定で明示したモジュールから自作の部品を登録する関数

    ML-ModelControl の register_units_from を呼ぶ. 読み込んだモジュールは
    任意のコードとして動くので, 信頼できるものだけを渡す

    Parameters
    ----------
    module_names: list[str]
        モジュールの名前の並び (my_lab.units など)

    Returns
    ----------
    loaded: dict[str, list[str]]
        モジュールの名前ごとの, 登録した部品の名前

    Raises
    ----------
    TypeError
        module_names がリストかタプルでない場合
    ValueError
        読み込めない, 部品を宣言していない, 登録できないモジュールがある
        場合
    ImportError
        入っている ML-ModelControl に register_units_from が無い場合
    """
    # 入っている ML-ModelControl での登録 (古いものには関数が無い)
    try:
        from ml_modelcontrol import register_units_from
    except ImportError as error:
        raise ImportError(
            "入っている ML-ModelControl はモジュールから部品を登録できません "
            "(register_units_from が無い). ML-ModelControl を新しくして"
            "ください"
        ) from error
    loaded = register_units_from(module_names)

    return loaded


def spec_units(spec: dict) -> list[tuple[str, str]]:
    """
    モデルの定義の本体と部分モデルの, ノードと部品の名前の組を返す関数

    Parameters
    ----------
    spec: dict
        モデルの定義 (ML-ModelControl の形)

    Returns
    ----------
    pairs: list[tuple[str, str]]
        (ノードの名前, 部品の名前). 部分モデルのノードは「部分モデル/
        ノード」
    """
    # 本体と, 名前を付けた部分モデル
    graphs: list[tuple[str, Any]] = [("", spec)]
    subgraphs = spec.get("subgraphs")
    if isinstance(subgraphs, dict):
        graphs += [(f"{name}/", graph) for name, graph in subgraphs.items()]

    # ノードごとの部品の名前 (形の誤りは ML-ModelControl の確かめに任せる)
    pairs: list[tuple[str, str]] = []
    for prefix, graph in graphs:
        nodes = graph.get("nodes") if isinstance(graph, dict) else None
        if not isinstance(nodes, dict):
            continue
        for name, node in nodes.items():
            if isinstance(node, dict) and "unit" in node:
                pairs.append((f"{prefix}{name}", str(node["unit"])))

    return pairs


def check_units(spec: dict) -> None:
    """
    モデルの定義の部品が全て登録簿にあるかを確かめる関数

    登録簿を引くだけで, 定義に書かれた名前から何も読み込まない

    Parameters
    ----------
    spec: dict
        モデルの定義 (ML-ModelControl の形)

    Returns
    ----------
    None

    Raises
    ----------
    UnregisteredUnitError
        登録されていない部品がある場合
    """
    # 部分モデル (sub.) を除いた部品の登録簿での検索
    units: list[str] = []
    nodes: list[str] = []
    for node, unit in spec_units(spec):
        if unit.startswith(SUBGRAPH_PREFIX):
            continue
        try:
            get_unit(unit)
        except KeyError:
            if unit not in units:
                units.append(unit)
            nodes.append(node)
    if units:
        raise UnregisteredUnitError(units, nodes)


def unit_message(units: list[str], nodes: list[str]) -> str:
    """
    登録されていない部品の誤りの文を作る関数

    Parameters
    ----------
    units: list[str]
        登録されていない部品の名前
    nodes: list[str]
        その部品を使うノードの名前

    Returns
    ----------
    message: str
        部品の名前, ノード, 足し方の案内
    """
    names = ", ".join(repr(unit) for unit in units)
    message = (
        f"この部品 ({names}) は登録されていません (モデルの定義のノード "
        f"{', '.join(nodes)}). {UNIT_HINT}"
    )

    return message
