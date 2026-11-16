# ML-Inference

学習したモデルで新しいデータを推論し, 予測から結論を出して価値として提供するライブラリ.

- **推論の一式**: モデルの定義 (ML-ModelControl), 学習した重み (ML-Found の実験), 前処理 (ML-EDA のデータセット) を 1 つのディレクトリにまとめ, 見本を推論できることを確かめてから作る
- **予測**: 新しいデータに学習のときと同じ前処理をかけて推論し, クラスの名前と確率, 元の単位の値で返す. 予測できない行 (前処理の後に欠損がある, 中身を読めない) は理由を付けて飛ばす
- **結論**: 確信の低いものを保留にする, 決めた確率で陽性を決める, 値を帯に分ける, 列の値ごとにまとめる. 全体のまとめを日本語の文でも返す

前処理は [ML-EDA](https://github.com/victor-von-pooh/ML-EDA), モデルの組み立ては [ML-ModelControl](https://github.com/victor-von-pooh/ML-ModelControl) を使い, [ML-Found](https://github.com/victor-von-pooh/ML-Found) で学習した重みを読む. 画面は [ML-Provide](https://github.com/victor-von-pooh/ML-Provide). 設計は [docs/DESIGN.md](docs/DESIGN.md).

## 導入

Python 3.10 以上.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # GPU を使う場合は PyTorch の案内に従う
pip install "ml-inference @ git+https://github.com/victor-von-pooh/ML-Inference.git"
```

画像, Excel, parquet, 部分語の文を読む場合は `pip install "ml-inference[all] @ git+https://github.com/victor-von-pooh/ML-Inference.git"` で ML-EDA の任意の依存も入れる. ML-EDA と ML-ModelControl は各リポジトリの `main` から入る.

## 使い方

### 推論の一式を作る

学習の後に, モデルの定義, ML-EDA で書き出したデータセット, ML-Found の実験のディレクトリ (か重みのファイル) から作る.

```python
from ml_inference import make_bundle

make_bundle(
    path="bundles/reviews",                            # 作る場所
    spec="models/classifier.yaml",                     # モデルの定義 (ファイルか辞書)
    dataset="outputs/dataset",                         # ML-EDA で書き出したデータセット
    weights="outputs/classifier/2026-11-16/10-00-00",  # 実験のディレクトリか重みのファイル
    conclusion={"min_confidence": 0.7},                # 結論の決め方の既定値 (任意)
)
```

- モデルの定義は ML-ModelControl の形 (書き方は ML-ModelControl の README). 入力の幅はデータセットの学習に使う列の数 (表) か中身の配列の形 (画像, 文, 点群), 出力の幅はクラスの数 (2 クラスは 1 列でもよい) か 1 (回帰, 予測)
- 重みは `state_dict` (ML-Found の `final_model.pt`). 実験のディレクトリを渡すと `final_model.pt` を使い, `run.json` の状態 (ベストの値など) を記録する
- 作るときに, データセットの学習用の見本を推論して, 入力の形と出力の幅を確かめる. 合わない場合は作らずに理由を返す
- さらに検証用の分割を推論し, 正解率か平均絶対誤差を実験の記録 (`valid/acc`, `valid/mae` のベストの値) と比べる. 合わない場合は, 前処理の外でしたことの書き忘れを疑う注意を `notes` に残す

- 入力が 1 つのモデルは, 表なら学習に使う列, 画像と文と点群なら中身の配列をそのまま渡す. 入力が複数のモデルや, 学習のスクリプトで並びの先頭に [CLS] などの番号を足した場合は `inputs={"ids": {"source": "items", "prepend": [2001]}}` のように書く
- 出力が複数のモデルは `output="logits"` のように使う出力を書く
- 学習のスクリプトで目的の値を `(y - center) / scale` にして学習した場合は `target_scale={"center": ..., "scale": ...}` を書く (予測を元の単位に戻す)

### 学習のスクリプトの最後に作る (勧める形)

学習のスクリプトが前処理の外でしたこと (並びの先頭に足した番号, 目的の値の標準化など) は, レシピとモデルの定義からは分からない. それを知っている学習のスクリプトが, 学習の最後に実験のディレクトリの中に一式を作ると, 書き忘れが起きない (実験のディレクトリには, プロジェクトが自分のファイルを足してよい. ML-Found の約束).

```python
# ML-Found の学習スクリプトの最後 (fit の後)
from pathlib import Path

from ml_inference import make_bundle

info = make_bundle(
    Path(experiment.run_dir) / "inference",
    spec="models/regressor.yaml",
    dataset="outputs/dataset",
    weights=experiment.run_dir,                        # final_model.pt と run.json
    target_scale={"center": center, "scale": scale},   # 目的の値を標準化して学習した場合
)
experiment.logger.info(f"推論の一式の検証用の評価: {info['check']}")
```

`info["check"]` の `matches` が `False` の場合は, 前処理の外でしたことの書き忘れか, 指標の決め方の違いを疑う.

### 推論して結論を出す

推論するデータは, 学習のときと同じ列を持つファイル (目的の列は有っても無くてもよい), 表 (`DataFrame`), 画像や点群のディレクトリ.

```python
from ml_inference import infer, write_result

result = infer("bundles/reviews", "new_reviews.csv")   # 学習のときと同じ読み込みで読む
print("\n".join(result["sentences"]))
# 1,000 件のうち 1,000 件を予測
# 良い 612 件 (61.2 %), 悪い 321 件 (32.1 %), 保留 67 件 (6.7 %)

write_result(result, "results/reviews")   # predictions.csv と result.yaml
```

`result["frame"]` は元の列に `status`, `reason`, `prediction`, `confidence`, `p:<クラス>`, `decision` を足した `DataFrame`. 予測だけが要る場合は `predict(bundle, source)` (読み込みの設定は `options`, 件数は `batch_size`, デバイスは `device`), 一式の説明は `describe_bundle(bundle)`, 結論の決め方を変える場合は `infer(..., rules={...})` (下の表. 一式の既定値に重なる. 値を `None` にすると既定値を消す. `positive` と `min_confidence` は一緒に使えないので, 渡した方を使う).

| 決め方 | タスク | 意味 |
| --- | --- | --- |
| `min_confidence` | 分類 | 予測したクラスの確率がこれより低い行を「保留」に |
| `positive`, `threshold` | 2 クラスの分類 | `positive` のクラスの確率が `threshold` 以上なら `positive` に |
| `bands` | 回帰, 予測 | `[{label: 少ない, upper: 100}, {label: 多い}]` のような値の帯 |
| `group_by` | 全て | 列の値ごとに件数, 判断の内訳, 値の合計などをまとめる |

### 何段も先を予測する

目的の列の過去の値の列 (lag) を作る一式では, 目的の列を空にした先の行を足して `forecast` に渡すと, 系列ごとに予測した値を次の入力に使いながら順に埋める.

```python
from ml_inference import forecast

# 過去の行 (売上あり) の後ろに, 先の 7 日の行 (売上は空, 気温と販促の予定は書く)
result = forecast("bundles/sales", "sales_with_next_week.csv")
print(result[result["step"].notna()][["date", "store", "step", "prediction"]])
```

`step` は何段目の予測か (過去の行は空). 先の段ほど誤りが積み重なる. 先の行だけの結論が要る場合は `infer(..., forecast=True)`.

### どの列が予測を動かしているか

```python
from ml_inference import explain

result = explain("bundles/sales", "new_sales.csv")
for item in result["columns"]:
    print(item["column"], item["effect"])   # 効きの大きい順
# sales (過去の値) 123262, store 117509, date 99698, temperature 40959, promo 33882
```

列を 1 つずつ行の間で並べ替えて推論し直し, 予測の変わり方の平均を効きとする (分類は予測したクラスの確率, 回帰と予測は元の単位の値). 正解の無い新しいデータでも求められる. 表と時系列の一式だけ. 系列を分ける列は系列の名前を入れ替え, 過去の値の列を作るレシピは系列ごとの最後の行を使う (詳しくは DESIGN の「列の効き」).

## 開発

```bash
git clone https://github.com/victor-von-pooh/ML-Inference.git
cd ML-Inference
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
```

## ライセンス

MIT
