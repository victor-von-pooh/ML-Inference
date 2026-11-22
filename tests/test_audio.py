import wave
from pathlib import Path

import numpy as np
import torch
from conftest import flat_spec, save_weights
from ml_eda import export_dataset, load_data, load_dataset, prepare
from ml_modelcontrol import build_model

from ml_inference import describe_bundle, make_bundle, predict

# 学習に使う標本化の周波数と長さ (秒)
RATE = 8000
SECONDS = 0.1


def write_wave(path: Path, samples: np.ndarray, rate: int) -> None:
    """
    (標本の数, チャンネル) の振幅 (-1 から 1) を 16 bit の WAVE に書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    samples: np.ndarray
        振幅
    rate: int
        標本化の周波数

    Returns
    ----------
    None
    """
    # 16 bit の整数にして, 標準ライブラリの wave で書く
    path.parent.mkdir(parents=True, exist_ok=True)
    numbers = np.clip(np.round(samples * 32768), -32768, 32767)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(samples.shape[1])
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(numbers.astype("<i2").tobytes())


def tone(
    pitch: float, seconds: float, rate: int, channels: int = 1
) -> np.ndarray:
    """
    決まった高さの音の振幅を作る関数

    Parameters
    ----------
    pitch: float
        音の高さ (Hz)
    seconds: float
        長さ (秒)
    rate: int
        標本化の周波数
    channels: int = 1
        チャンネルの数 (どのチャンネルも同じ音)

    Returns
    ----------
    samples: np.ndarray
        (標本の数, チャンネル) の振幅
    """
    # 正弦波を同じ値でチャンネルに並べる
    times = np.arange(int(seconds * rate)) / rate
    wave_ = 0.5 * np.sin(2 * np.pi * pitch * times)
    samples = np.repeat(wave_[:, None], channels, axis=1)

    return samples


def test_audio(tmp_path: Path) -> None:
    """
    音声の一式で, 周波数, チャンネル, 長さの違う新しい音声のディレクトリ
    と 1 つのファイルを推論でき, 学習と同じ配列になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 低い音と高い音 (周波数, チャンネル, 長さが混ざる)
    root = tmp_path / "sounds"
    for index in range(4):
        write_wave(root / "low" / f"l{index}.wav",
                   tone(200, 0.12, RATE), RATE)
        write_wave(root / "high" / f"h{index}.wav",
                   tone(1500, 0.08, 16000, 2), 16000)
    data = load_data(root)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "item_steps": [{"op": "resample", "rate": RATE},
                       {"op": "mix_channels", "channels": 1},
                       {"op": "fix_length", "seconds": SECONDS}],
        "split": {"method": "stratified", "valid": 0.25, "seed": 0},
    }
    assert len(prepare(data, recipe).frames["train"]) > 0
    export_dataset(data, recipe, tmp_path / "ds")
    width = int(RATE * SECONDS)
    spec = flat_spec([1, width], 2)
    weights = save_weights(spec, tmp_path / "w.pt")
    make_bundle(tmp_path / "bundle", spec, tmp_path / "ds", weights)
    assert describe_bundle(tmp_path / "bundle")["kind"] == "audio"

    # 学習の配列と, 推論の変換 (ファイルの場所と読んだ辞書) が同じ
    dataset = load_dataset(tmp_path / "ds")
    x, _ = dataset.arrays("train")
    frame = dataset.frames["train"]
    first = root / frame["path"].iloc[0]
    assert x.shape[1:] == (1, width)
    assert np.array_equal(dataset.transform_item(first), x[0])
    item = load_data(first).load_item(0)
    assert np.array_equal(dataset.transform_item(item), x[0])

    # 学習に使った 1 つのファイルの確率は, 学習の配列をモデルに通した値
    model = build_model(spec)
    model.load_state_dict(torch.load(weights))
    with torch.no_grad():
        expected = model(torch.from_numpy(np.array(x[:1]))).softmax(dim=1)
    single = predict(tmp_path / "bundle", first)
    assert list(single["status"]) == ["ok"]
    got = single[["p:high", "p:low"]].to_numpy()[0]
    assert np.allclose(got, expected[0].numpy(), atol=1e-5)

    # クラスのフォルダの無い新しい音声 (周波数と長さの違うものと読めない
    # もの)
    fresh = tmp_path / "fresh"
    write_wave(fresh / "a.wav", tone(200, 0.3, 22050, 2), 22050)
    write_wave(fresh / "b.wav", tone(1500, 0.05, RATE), RATE)
    (fresh / "c.wav").write_bytes(b"RIFF....WAVEjunk")
    result = predict(tmp_path / "bundle", fresh)
    names = result["path"].map(lambda value: Path(value).name)
    assert list(names) == ["a.wav", "b.wav", "c.wav"]
    assert list(result["status"]) == ["ok", "ok", "skipped"]
    assert result["reason"][2]

    # 新しい音声の確率は, 変換した配列をモデルに通した値
    fresh_x = torch.from_numpy(np.stack([
        dataset.transform_item(fresh / name) for name in ("a.wav", "b.wav")
    ]))
    with torch.no_grad():
        expected = model(fresh_x).softmax(dim=1).numpy()
    got = result[["p:high", "p:low"]].to_numpy()[:2]
    assert np.allclose(got, expected, atol=1e-5)
    assert set(result["prediction"][:2]) <= {"low", "high"}


def test_audio_shape(tmp_path: Path) -> None:
    """
    中身の手順の無い音声の一式で, 学習と長さかチャンネルの違う件を理由を
    付けて飛ばすことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同じ周波数と長さの音声だけで, 手順の無い一式
    root = tmp_path / "sounds"
    for index in range(4):
        write_wave(root / "low" / f"l{index}.wav",
                   tone(200, SECONDS, RATE), RATE)
        write_wave(root / "high" / f"h{index}.wav",
                   tone(1500, SECONDS, RATE), RATE)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "item_steps": [],
        "split": {"method": "stratified", "valid": 0.25, "seed": 0},
    }
    export_dataset(load_data(root), recipe, tmp_path / "ds")
    spec = flat_spec([1, int(RATE * SECONDS)], 2)
    weights = save_weights(spec, tmp_path / "w.pt")
    make_bundle(tmp_path / "bundle", spec, tmp_path / "ds", weights)

    # 先頭が長い件でも, 学習の形と比べて飛ばす (周波数の違う件は長さが違う)
    fresh = tmp_path / "fresh"
    write_wave(fresh / "a.wav", tone(200, 2 * SECONDS, RATE), RATE)
    write_wave(fresh / "b.wav", tone(200, SECONDS, RATE), RATE)
    write_wave(fresh / "c.wav", tone(200, SECONDS, 2 * RATE), 2 * RATE)
    write_wave(fresh / "d.wav", tone(200, SECONDS, RATE, 2), RATE)
    result = predict(tmp_path / "bundle", fresh)
    assert list(result["status"]) == ["skipped", "ok", "skipped", "skipped"]
    for row in (0, 2, 3):
        assert "学習の (1, 800) と違います" in result["reason"][row]

    # 1 つのファイルだけでも同じ
    single = predict(tmp_path / "bundle", fresh / "a.wav")
    assert list(single["status"]) == ["skipped"]
