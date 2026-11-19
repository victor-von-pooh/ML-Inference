from pathlib import Path
import sys
import uuid

import pytest
import torch
import yaml
from conftest import save_weights
from ml_eda import load_dataset

from ml_inference import (
    UnregisteredUnitError, load_bundle, make_bundle, predict, register_units
)
from ml_inference.units import check_units, spec_units

# 自作の部品を宣言するモジュールの中身 ({unit} は部品の名前)
UNIT_MODULE = '''
from torch import nn


class Scale(nn.Module):
    def __init__(self, features: int = 2):
        super(Scale, self).__init__()
        self.fc = nn.Linear(features, features)

    def forward(self, x):
        out = self.fc(x) * 2.0
        return out


UNITS = [{{"cls": Scale, "name": "{unit}"}}]
'''

# 読み込まれたら印のファイルを作るモジュールの中身
MARK_MODULE = '''
from pathlib import Path

Path(r"{mark}").write_text("imported", encoding="utf-8")
'''


def unique(prefix: str) -> str:
    """
    テストごとに重ならない名前を返す関数

    Parameters
    ----------
    prefix: str
        名前の頭

    Returns
    ----------
    name: str
        頭と乱数の文字をつないだ名前
    """
    # 登録簿と読み込み済みのモジュールはプロセスで共通
    name = f"{prefix}_{uuid.uuid4().hex[:8]}"

    return name


def unit_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str]:
    """
    自作の部品を宣言するモジュールを書き, 読める場所に置く関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        sys.path の書き換え

    Returns
    ----------
    names: tuple[str, str]
        モジュールの名前と部品の名前
    """
    # モジュールのファイルと読み込む場所
    module = unique("lab_units")
    unit = f"{module}.Scale"
    place = tmp_path / "modules"
    place.mkdir(exist_ok=True)
    (place / f"{module}.py").write_text(
        UNIT_MODULE.format(unit=unit), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(place))
    names = (module, unit)

    return names


def unit_spec(width: int, unit: str) -> dict:
    """
    自作の部品と全結合の層のモデルの定義を返す関数

    Parameters
    ----------
    width: int
        入力の列の数
    unit: str
        自作の部品の名前

    Returns
    ----------
    spec: dict
        ML-ModelControl の定義
    """
    # 自作の部品から 3 クラスへ
    spec = {
        "format_version": 1,
        "inputs": {"x": {"shape": [4, width]}},
        "nodes": {
            "lab": {"unit": unit, "params": {"features": width},
                    "inputs": "x"},
            "head": {"unit": "nn.Linear",
                     "params": {"in_features": width, "out_features": 3},
                     "inputs": "lab"},
        },
        "outputs": {"logits": "head"},
    }

    return spec


def custom_bundle(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str]:
    """
    自作の部品を登録し, その部品を使う一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    dataset: Path
        データセット
    monkeypatch: pytest.MonkeyPatch
        sys.path の書き換え

    Returns
    ----------
    made: tuple[Path, str]
        一式のディレクトリと部品の名前
    """
    # 起動の設定のモジュールからの登録
    module, unit = unit_module(tmp_path, monkeypatch)
    loaded = register_units([module])
    assert loaded == {module: [unit]}

    # 自作の部品の一式
    width = len(load_dataset(dataset).features)
    spec = unit_spec(width, unit)
    weights = save_weights(spec, tmp_path / "custom.pt")
    path = tmp_path / "custom"
    make_bundle(path, spec, dataset, weights)
    made = (path, unit)

    return made


def rename_unit(path: Path, old: str, new: str) -> None:
    """
    一式のモデルの定義の部品の名前を書き換える関数

    Parameters
    ----------
    path: Path
        一式のディレクトリ
    old: str
        元の部品の名前
    new: str
        新しい部品の名前

    Returns
    ----------
    None
    """
    # 定義のファイルの書き換え
    file = path / "model_spec.yaml"
    spec = yaml.safe_load(file.read_text(encoding="utf-8"))
    for node in spec["nodes"].values():
        if node["unit"] == old:
            node["unit"] = new
    file.write_text(yaml.safe_dump(spec), encoding="utf-8")


def test_registered_unit_bundle_predicts(
    tmp_path: Path, table_dataset: Path, table_csv: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # 登録した自作の部品の一式は読めて推論できる
    path, unit = custom_bundle(tmp_path, table_dataset, monkeypatch)
    bundle = load_bundle(path)
    assert bundle.spec["nodes"]["lab"]["unit"] == unit
    result = predict(bundle, table_csv)
    assert len(result) > 0
    assert isinstance(bundle.model(), torch.nn.Module)


def test_unregistered_unit_names_unit_and_setting(
    tmp_path: Path, table_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 登録されていない部品は, 名前とノードと起動の設定の案内で止まる
    path, unit = custom_bundle(tmp_path, table_dataset, monkeypatch)
    missing = unique("other_lab") + ".Block"
    rename_unit(path, unit, missing)
    with pytest.raises(UnregisteredUnitError) as caught:
        load_bundle(path)
    message = str(caught.value)
    assert repr(missing) in message
    assert "lab" in message
    assert "起動の設定" in message
    assert "register_units" in message
    assert caught.value.units == [missing]
    assert caught.value.nodes == ["lab"]
    assert isinstance(caught.value, ValueError)


def test_bundle_names_are_never_imported(
    tmp_path: Path, table_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 定義に書かれた名前のモジュールが読める場所にあっても読み込まない
    path, unit = custom_bundle(tmp_path, table_dataset, monkeypatch)
    module = unique("planted")
    mark = tmp_path / "imported.txt"
    (tmp_path / "modules" / f"{module}.py").write_text(
        MARK_MODULE.format(mark=mark), encoding="utf-8"
    )
    rename_unit(path, unit, f"{module}.Block")
    (path / f"{module}.py").write_text(
        MARK_MODULE.format(mark=mark), encoding="utf-8"
    )
    with pytest.raises(UnregisteredUnitError):
        load_bundle(path)
    assert module not in sys.modules
    assert not mark.exists()


def test_make_bundle_stops_before_writing(
    tmp_path: Path, table_dataset: Path
) -> None:
    # 登録されていない部品の定義からは一式を作らない
    width = len(load_dataset(table_dataset).features)
    missing = unique("absent") + ".Block"
    spec = unit_spec(width, missing)
    weights = tmp_path / "unused.pt"
    torch.save({}, weights)
    path = tmp_path / "never"
    with pytest.raises(UnregisteredUnitError, match="起動の設定"):
        make_bundle(path, spec, table_dataset, weights)
    assert not path.exists()


def test_subgraph_units_are_checked() -> None:
    # 部分モデルの中の部品も確かめ, sub. の名前は登録簿を見ない
    missing = unique("inner") + ".Block"
    spec = {
        "inputs": {"x": {}},
        "nodes": {"b": {"unit": "sub.block", "inputs": "x"}},
        "outputs": {"out": "b"},
        "subgraphs": {
            "block": {
                "inputs": {"x": {}},
                "nodes": {"in": {"unit": missing, "inputs": "x"},
                          "act": {"unit": "nn.ReLU", "inputs": "in"}},
                "outputs": {"out": "act"},
            }
        },
    }
    assert ("block/in", missing) in spec_units(spec)
    with pytest.raises(UnregisteredUnitError) as caught:
        check_units(spec)
    assert caught.value.units == [missing]
    assert caught.value.nodes == ["block/in"]


def test_register_units_errors() -> None:
    # 文字 1 つの取り違えと, 読めないモジュール
    with pytest.raises(TypeError):
        register_units("lab.units")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        register_units([unique("no_such_module")])
