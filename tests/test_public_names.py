import ast
from pathlib import Path

import ml_eda
import ml_modelcontrol

# ライブラリのコードの場所 (テストのコードは対象の外)
SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "ml_inference"

# 他の段階のライブラリと, その公開の名前
PUBLIC = {
    "ml_eda": set(ml_eda.__all__),
    "ml_modelcontrol": set(ml_modelcontrol.__all__),
}


def private_names(tree: ast.AST) -> set[str]:
    """
    構文の木から, 他の段階のライブラリの公開されていない名前を集める関数

    Parameters
    ----------
    tree: ast.AST
        1 つのファイルの構文の木

    Returns
    ----------
    names: set[str]
        モジュールから続けた名前 (ml_eda.items.PLACEMENTS など)
    """
    # from の import と, ライブラリの名前から続く属性
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top not in PUBLIC:
                continue
            for alias in node.names:
                if node.module != top or alias.name not in PUBLIC[top]:
                    names.add(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Import):
            # 中のモジュールの import と, 別名 (後の属性を追えない) は誤り
            for alias in node.names:
                if alias.name.split(".")[0] not in PUBLIC:
                    continue
                if alias.asname:
                    names.add(f"import {alias.name} as {alias.asname}")
                elif "." in alias.name:
                    names.add(alias.name)
        elif isinstance(node, ast.Attribute):
            chain = []
            value: ast.expr = node
            while isinstance(value, ast.Attribute):
                chain.append(value.attr)
                value = value.value
            if not isinstance(value, ast.Name) or value.id not in PUBLIC:
                continue
            chain.reverse()
            if chain[0] not in PUBLIC[value.id]:
                names.add(".".join([value.id, *chain]))

    return names


def test_only_public_names() -> None:
    """
    ライブラリのコードが ML-EDA と ML-ModelControl の公開の名前
    (__all__) だけを使うことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 全てのファイルの公開されていない名前
    found: dict[str, set[str]] = {}
    for path in sorted(SOURCE_DIR.glob("**/*.py")):
        names = private_names(ast.parse(path.read_text(encoding="utf-8")))
        if names:
            found[str(path.relative_to(SOURCE_DIR))] = names

    assert found == {}


def test_private_names_found() -> None:
    """
    公開されていない名前を, from の import と属性の両方で見つけ, 別名を
    付ける import も見つけることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 公開の名前は数えず, 公開されていない名前だけ
    code = (
        "import ml_eda\n"
        "from ml_eda import PLACEMENTS, load_items\n"
        "from ml_eda.items import PLACEMENTS\n"
        "from ml_eda.pipeline import DATASET_FORMAT_VERSION\n"
        "ml_eda.load_items(x)\n"
        "ml_eda.media.read_image(x)\n"
        "from ml_modelcontrol import build_model\n"
        "from ml_modelcontrol.spec import load_spec\n"
    )
    names = private_names(ast.parse(code))
    assert names == {
        "ml_eda.items.PLACEMENTS", "ml_eda.pipeline.DATASET_FORMAT_VERSION",
        "ml_eda.media.read_image", "ml_eda.media",
        "ml_modelcontrol.spec.load_spec",
    }

    # 別名を付ける import は, 公開の名前だけを使う場合も誤り
    aliased = private_names(ast.parse(
        "import ml_eda as e\nimport ml_eda.items as m\ne.load_items(x)\n"
        "from ml_eda import load_items as load\n"
    ))
    assert aliased == {"import ml_eda as e", "import ml_eda.items as m"}
