import ml_eda
import numpy as np

from ml_inference.items import restore_size


def test_restore_size_nearest() -> None:
    """
    領域を元の大きさに戻すとき, 戻した画素の中心に最も近い出力の画素の
    値だけを使い, ML-EDA の書き出しが領域をそろえる手順 (resize_mask) と
    同じ結果になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 倍は画素の繰り返し, 同じ大きさはそのまま
    mask = np.array([[0, 1], [2, 3]], dtype=np.int64)
    expected = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    assert np.array_equal(restore_size(mask, (4, 4)), expected)
    assert np.array_equal(restore_size(mask, (2, 2)), mask)

    # 広げる, 縮める, 割り切れない大きさ, 縦横で向きの違う大きさで,
    # ML-EDA の resize_mask と同じ値と型 (同じ式を持つので, ずれに気付く
    # ため)
    resize_mask = ml_eda.ITEM_STEPS["resize_mask"].apply
    rng = np.random.default_rng(0)
    for shape in [(32, 32), (5, 7), (480, 640), (1, 3)]:
        grid = rng.integers(0, 3, size=shape).astype(np.int64)
        for height, width in [(40, 48), (32, 24), (7, 5), (33, 31),
                              (1, 1), (481, 639), (3, 1000)]:
            ours = restore_size(grid, (height, width))
            theirs = resize_mask(grid, {"width": width, "height": height}, {})
            assert ours.shape == (height, width) and ours.dtype == np.int64
            assert np.array_equal(ours, theirs)
            assert ours.dtype == theirs.dtype
