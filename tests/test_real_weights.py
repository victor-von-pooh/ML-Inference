from pathlib import Path

import numpy as np
import pytest

from ml_inference import load_bundle, make_ready_bundle, predict

# 本物の重みの名前と, 記録した重みの hash (問題の設定の見本と同じ)
MODEL = "resnet18"
WEIGHTS = "IMAGENET1K_V1"
SHA256 = "8ec1484cdbb88ded96d21c3a5936faf3cdcc665aef92f331f675d0c63038f0ba"

# 重みを取りに行くので, 印を選んだ実行だけで動かす
pytestmark = pytest.mark.network


def test_real_resnet18_embedding(tmp_path: Path) -> None:
    """
    本物の torchvision の resnet18 の重みで埋め込みの一式を作り, 幅, hash,
    最後の層の手前の出力が torchvision のモデルと合うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 重みを取りに行き (2 回目からは置き場から), 記録した hash で固定
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("torchvision.models")
    image = pytest.importorskip("PIL.Image")
    bundle = tmp_path / "embedder"
    info = make_ready_bundle(bundle, MODEL, WEIGHTS, library="torchvision",
                             sha256=SHA256, form="vector")
    assert info["output"] == {"form": "vector", "shape": [512]}
    assert info["model"]["pretrained"][0]["sha256"] == SHA256
    assert info["ready"]["weights"] == WEIGHTS
    assert load_bundle(bundle).embedding_width == 512

    # 色の違う写真の埋め込みと, torchvision のモデルの平均の層の出力
    photos = tmp_path / "photos"
    photos.mkdir()
    rng = np.random.default_rng(0)
    for index in range(2):
        array = rng.integers(0, 255, (300, 260, 3), dtype=np.uint8)
        image.fromarray(array).save(photos / f"{index}.png")
    frame = predict(bundle, photos)
    assert (frame["status"] == "ok").all()
    points = np.stack(frame["output"].to_list())
    assert points.shape == (2, 512)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights).eval()
    model.fc = torch.nn.Identity()
    batch = torch.stack([
        weights.transforms()(image.open(photos / name).convert("RGB"))
        for name in frame["path"]
    ])
    with torch.no_grad():
        expected = model(batch).numpy()
    assert np.allclose(points, expected, rtol=1e-4, atol=1e-4)
