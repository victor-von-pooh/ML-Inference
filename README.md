# ML-Inference

学習したモデルで新しいデータを推論し, 予測から結論を出して価値として提供するライブラリ.

- **推論の一式**: モデルの定義 (ML-ModelControl), 学習した重み (ML-Found の実験), 前処理 (ML-EDA のデータセット) を 1 つのディレクトリにまとめ, 見本を推論できることを確かめてから作る
- **予測**: 新しいデータに学習のときと同じ前処理をかけて推論し, クラスの名前と確率, 元の単位の値で返す. 予測できない行 (前処理の後に欠損がある, 中身を読めない) は理由を付けて飛ばす
- **結論**: 確信の低いものを保留にする, 決めた確率で陽性を決める, 値を帯に分ける, 列の値ごとにまとめる. 全体のまとめを日本語の文でも返す

画面は ML-Provide. 設計は [docs/DESIGN.md](docs/DESIGN.md).

## 導入

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # GPU を使う場合は PyTorch の案内に従う
pip install "ml-inference @ git+https://github.com/victor-von-pooh/ML-Inference.git"
```

画像, Excel, parquet, 部分語の文を読む場合は `ml-inference[all]` で ML-EDA の任意の依存も入れる.

## 使い方

### 推論の一式を作る

学習の後に, モデルの定義, ML-EDA で書き出したデータセット, ML-Found の実験のディレクトリ (か重みのファイル) から作る.

```python
from ml_inference import make_bundle

make_bundle(
    "bundles/reviews",                         # 作る場所
    "models/classifier.yaml",                  # モデルの定義
    "outputs/dataset",                         # ML-EDA で書き出したデータセット
    "outputs/classifier/2026-11-16/10-00-00",  # 実験 (final_model.pt と run.json を使う)
    conclusion={"min_confidence": 0.7},        # 結論の決め方の既定値 (任意)
)
```

- 入力が 1 つのモデルは, 表なら学習に使う列, 画像と文と点群なら中身の配列をそのまま渡す. 入力が複数のモデルや, 学習のスクリプトで並びの先頭に [CLS] などの番号を足した場合は `inputs={"ids": {"source": "items", "prepend": [2001]}}` のように書く
- 出力が複数のモデルは `output="logits"` のように使う出力を書く
- 学習のスクリプトで目的の値を `(y - center) / scale` にして学習した場合は `target_scale={"center": ..., "scale": ...}` を書く (予測を元の単位に戻す)

### 推論して結論を出す

```python
from ml_inference import infer, write_result

result = infer("bundles/reviews", "new_reviews.csv")   # 学習のときと同じ読み込みで読む
print("\n".join(result["sentences"]))
# 1,000 件のうち 1,000 件を予測
# 良い 612 件 (61.2 %), 悪い 321 件 (32.1 %), 保留 67 件 (6.7 %)

write_result(result, "results/reviews")   # predictions.csv と result.yaml
```

`result["frame"]` は元の列に `status`, `reason`, `prediction`, `confidence`, `p:<クラス>`, `decision` を足した `DataFrame`. 予測だけが要る場合は `predict(bundle, source)`, 結論の決め方を変える場合は `infer(..., rules={...})` (下の表).

| 決め方 | タスク | 意味 |
| --- | --- | --- |
| `min_confidence` | 分類 | 予測したクラスの確率がこれより低い行を「保留」に |
| `positive`, `threshold` | 2 クラスの分類 | `positive` のクラスの確率が `threshold` 以上なら `positive` に |
| `bands` | 回帰, 予測 | `[{label: 少ない, upper: 100}, {label: 多い}]` のような値の帯 |
| `group_by` | 全て | 列の値ごとに件数, 判断の内訳, 値の合計などをまとめる |

## 開発

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
```

## ライセンス

MIT
