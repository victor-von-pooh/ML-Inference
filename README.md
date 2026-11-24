# ML-Inference

「機械学習を全て GUI 上で行う」うちの「推論」の段階のライブラリ. 学習したモデルで新しいデータを推論し, 予測から結論と意思決定の材料を出す. 画面のアプリは [ML-Provide](https://github.com/victor-von-pooh/ML-Provide).

## できること

- **推論の一式**: モデルの定義 (ML-ModelControl), 学習した重み (ML-Found の実験), 前処理 (ML-EDA のデータセット) を 1 つのディレクトリにまとめ, 見本を推論できることを確かめてから作る
- **予測**: 新しいデータに学習のときと同じ前処理をかけて推論し, クラスの名前と確率, 元の単位の値で返す. 予測できない行 (前処理の後に欠損がある, 中身を読めない) は理由を付けて飛ばす. 領域, 点群, ベクトル, 文, 作った表などの出力と, 拡散のサンプリングのような繰り返しの手順も扱う
- **結論**: 確信の低いものを保留にする, 決めた確率で陽性を決める, 値を帯に分ける, 列の値ごとにまとめる. 全体のまとめを日本語の文でも返す
- **何段も先の予測と列の効き**: 予測した値を次の入力に使って先の行を埋める. どの列が予測を動かしているかを並べ替えで求める
- **意思決定**: 予測から次の行動 (量, 選ぶ対象, 人が確かめる対象) と見込みの得と損を出し, 過去のデータでたどり直して, 素朴なやり方と今のやり方 (問題の設定と同じ決まりでも書ける) と比べる. 新しい実績で外れ方を測り直す
- **方策の一式**: ML-Found が書き出す強化学習の方策を読み, 観測から行動と守り (行動の範囲, 止める条件, 外れたときの動き) の結果を返す. 実機では動かさない
- **既製のモデルの一式**: 学習しない既製のモデル (torchvision の学習済みの画像の分類など) を, 重みの版と hash を固定し, 種類の名前の表と前処理を持たせて一式にする
- **並べるの一式**: 利用者と品物の 2 塔のモデル (ML-Quest の並べるの骨組みの実験) から, 品物の埋め込みを索引に持つ一式を作り, 利用者ごとに上位 K 件の品物と点数を返す. 前に反応した品物を除くかを選べ, 後の反応から Recall@K と NDCG を数える
- **見本の一式**: 表か文のデータから小さなモデルをその場で学習し, 画面で操作を試すための一式を作る

前処理は [ML-EDA](https://github.com/victor-von-pooh/ML-EDA), モデルの組み立ては [ML-ModelControl](https://github.com/victor-von-pooh/ML-ModelControl) を使い, [ML-Found](https://github.com/victor-von-pooh/ML-Found) で学習した重みを読む.

## どのアプリから使われるか

| 相手 | 使うもの |
| --- | --- |
| ML-Provide (推論) | 推論の一式の作成と確かめ (並べるの一式は `make_rank_bundle` と `rank_metrics`), 版 2 のデータセットの説明 (`describe_items`), 予測と同じデータの読み込み (`read_source`), 正解の種類と出どころ (`target_of`, `origin_sentence`), 結論 (決め方の項目 `RULE_KEYS`, 件数だけの結論 `conclude_outputs`, まとめからの文 `summary_sentences`), 撮影の深度から点 (`backproject`), 意思決定, 先の予測と列の効き, 方策の一式. 自作の繰り返しの手順 (`register_procedure`, `PROCEDURES`) と自作の部品 (`register_units`) の登録. 問題の設定 (ML-Quest) の損得と今のやり方は, 画面が引数で渡す |
| ML-Found の学習のスクリプト (ML-Quest の骨組みを含む) | 学習の後に `make_bundle` で実験のディレクトリの中に一式を作る |

## 導入

Python 3.10 以上.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # GPU を使う場合は PyTorch の案内に従う
pip install "ml-inference @ git+https://github.com/victor-von-pooh/ML-Inference.git"
```

ML-EDA が書き出すデータセット (pyarrow があれば parquet) を読めるように, parquet を読む pyarrow は依存に含む. 画像, Excel, 部分語の文を読む場合は `pip install "ml-inference[all] @ git+https://github.com/victor-von-pooh/ML-Inference.git"` で ML-EDA の任意の依存も入れる. ML-EDA と ML-ModelControl は各リポジトリの `main` から入る.

開発する場合は, リポジトリを取ってきて入れ, `pytest` でテストする.

```bash
git clone https://github.com/victor-von-pooh/ML-Inference.git
cd ML-Inference
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
```

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

作るときに, 学習用の見本を推論して入力の形と出力の幅を確かめ, 検証用の分割の評価を実験の記録と比べる (合わなければ `notes` に注意). 入力が複数のモデルや並びの先頭に足した番号 (`inputs`), 使う出力 (`output`), 目的の値の標準化 (`target_scale`) の書き方は [docs/DESIGN.md](docs/DESIGN.md) の「推論の一式の形」.

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

推論するデータは, 学習のときと同じ列を持つファイル (目的の列は有っても無くてもよい), 表 (`DataFrame`), 画像, 点群, 音声のファイルかディレクトリ.

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

### 構造のある出力, ベクトル, 繰り返しの手順 (一式の版 3)

一式は受け持ち (`role`) と出力の形 (`form`: `class`, `value`, `distribution`, `mask`, `point_cloud`, `vector`, `text`, `image`, `table`) を持ち, 出力の形ごとに読み方を変える. 省くとタスクから決める (分類は `class`, 回帰と予測は `value`). 版 1 と 2 の一式もそのまま読める.

```python
from ml_inference import infer, make_bundle, predict, write_result

# 埋め込みのモデル (出力がベクトル). 正解の無いデータセットでも作れる
make_bundle("bundles/embed", spec, "outputs/dataset", weights, form="vector")
result = infer("bundles/embed", "new_rows.csv")
write_result(result, "results/embed")   # 配列は artifacts/000000.npy など, 表の artifact の列から指す

# 拡散のモデル. 手順の名前, 回数, 乱数の種を一式に書く. 条件の無い生成は
# データセットの代わりに None を渡し (雑音だけの一式. 版 5), 件数と種で推論する
make_bundle(
    "bundles/gen", spec, None, weights, form="vector",
    output_options={"shape": [16]},
    inputs={"x": {"source": "state"}, "t": {"source": "time"}},
    procedure={"name": "ddim", "steps": 50, "seed": 0},
)
result = infer("bundles/gen", {"count": 8, "seed": 1})   # 同じ件数と種なら同じ出力

# 表を作る拡散のモデル. 列の符号化 (ML-Quest の骨組みの table_columns.json) を
# 一式に写し (版 7), 件数と種から列の名前と型のある表を返す
make_bundle(
    "bundles/rows", spec, None, run_dir, form="table",
    output_options={"columns": f"{run_dir}/table_columns.json"},
    inputs={"x": {"source": "state"}, "t": {"source": "time"}},
    procedure={"name": "ddim", "steps": 50, "seed": 0},
)
rows = predict("bundles/rows", {"count": 100, "seed": 0})   # sample, seed, status, reason と表の列
```

表の出力 (`table`) は, 符号化した行 (数の列は標準化, 種類の列はワンホット) を, 数の列は値 × `std` + `mean` (整数の列は丸める), 種類の列は区間で最も大きい値の種類に戻す (DESIGN の「表の出力の雑音だけの一式」). 結果は配列を持たないので, 書き出しは `predictions.csv` だけ.

繰り返しの手順 (`ddim`, `euler`, `greedy`, `sample`) と学習済みの重みの置き方は, DESIGN の「繰り返しの手順」「学習済みの重みの置き方」. 学習済みの重みは, 定義の一番上のノードと部分モデル (`subgraphs`) の中のノードのどちらも, 再配布できるものだけ一式に写し, できないものは出どころから読んで hash で確かめる (外部の重みを重ねたもの, 学習で変わったもの, ライセンスか版を読めなかったものは, 注意を付けて一式に残す. DESIGN の「学習済みの重みの置き方」). 部分モデルの中のノードの情報を `pretrained` で渡すときは `block.inner.fc` のように外のノードの名前とつなぐ. 研究の自作の部品を使う一式は, 一式を読む前に `ml_inference.register_units(["my_lab.units"])` で部品を登録する (ML-Provide なら起動の `--units` か `ML_PROVIDE_UNITS`. DESIGN の「自作の部品」).

### 領域と点群 (ML-EDA の版 2 と 3 のデータセット. 一式の版 4)

ML-EDA の版 2 と 3 のデータセット (画像と領域の対, 撮影と点群の対) で学習した重みからも一式を作れる. 版 2 には前処理のレシピが無いので, 学習のスクリプトがした 1 件の扱い (画像を 0 から 1 にする, 深度から点に戻して決まった数を選ぶ) を `items` で一式に書く. 出力は, 領域はデータセットと同じ領域の値, 点群は元の撮影の座標 (正規化の平行移動を引く) で返す. 版 3 のデータセット (書き出しで画像の大きさをそろえるなどの中身の手順をかけたもの) の一式は, 元の画像や撮影にも同じ手順をかけてから推論する (元の画像の領域は, 件ごとに元の画像の大きさに最も近い画素の値で戻して返す. データセットのディレクトリの件は, データセットの大きさのまま).

```python
from ml_inference import infer, make_bundle, write_result

# 画像と領域. 扱い方とクラスはデータセットから決まる
make_bundle("bundles/seg", "models/unet.yaml", "outputs/masks", "outputs/seg/2026-11-21/10-00-00")

# 撮影と点群. 選ぶ点の数と乱数の種は学習のスクリプトと同じものを書く
make_bundle(
    "bundles/shape", "models/points.yaml", "outputs/clouds", run_dir,
    items={"options": {"points": 512, "target_points": 1024, "seed": 1}},
)

# 版 2 か 3 のデータセットか, 元の画像や撮影のディレクトリを推論する
result = infer("bundles/shape", "new_captures/")
write_result(result, "results/shape")   # 点群は artifacts/000000.npy など
```

一式を作る前にデータセットの項目と `items` の既定を見るには `describe_items("outputs/clouds")` (作れないデータセットは `problem` に理由が入る). 点を選ぶ入力の項目の種類は `POINT_KINDS`.

一式を作るときに, 検証用の分割の IoU (領域) か Chamfer 距離 (点群) を求め, 実験の監視した指標 (`valid/iou`, `valid/chamfer`) と比べる (DESIGN の「版 2 のデータセットの一式」).

### 学習しない既製のモデル (一式の版 6)

ライブラリの学習済みの重みを学習せずにそのまま使う一式も作れる. 今は torchvision の画像の分類 (ML-ModelControl の追加の依存 `torchvision` で入れる). 種類の名前の表と前処理 (大きさ, 切り出し, 正規化) は重みの説明から一式に書き, 重みは写さずに名前 (版) と hash で固定する. 重みは初回に取りに行くか, 利用者が PyTorch の置き場 (`torch.hub.get_dir()` の `checkpoints`) に置く. 置くファイルは信頼できるものだけにする (hash の照合の前に torchvision が読み込むため).

```python
from ml_inference import describe_bundle, infer, make_ready_bundle

make_ready_bundle("bundles/resnet18", "resnet18", "IMAGENET1K_V1")
description = describe_bundle("bundles/resnet18")
len(description["classes"])            # 1000 (tench, goldfish, ...)
description["ready"]["license"]        # ライセンスの警告 (torchvision は unknown)
result = infer("bundles/resnet18", "photos/")   # prediction は種類の名前
```

### 利用者ごとに品物を並べる (一式の版 8)

並べるの実験 (重み `final_model.pt`, 定義 `model_spec.yaml`, 利用者と品物の番号の表 `rank_index.json`) から一式を作る. モデルの入力は利用者と品物の番号のワンホット `user` と `item`, 出力は点数 `score` と埋め込み `user_embedding`, `item_embedding`. 作るときに全ての品物の埋め込みを索引にし, 点数が埋め込みの内積の定数倍であることを確かめる.

```python
import pandas as pd
from ml_inference import infer, make_rank_bundle, predict, rank_metrics

# k と列の名前は問題の設定 (ML-Quest の受け渡し) の rank から
make_rank_bundle("bundles/shop", "outputs/shop/run", k=10,
                 columns={"user": "member", "item": "item_id",
                          "reaction": "qty"})

# 利用者ごとの上位 10 件 (user, status, reason, rank, item, score の表)
seen = pd.read_csv("reactions.csv")             # 前に反応した品物を除くとき
frame = predict("bundles/shop", ["m001", "m002", "m999"], seen=seen)
# m999 は番号の表に無いので status が skipped (理由付き)

result = infer("bundles/shop", ["m001", "m002"], k=5)
result["sentences"]   # ['利用者 2 人のうち 2 人に, 上位 5 件の品物を並べた']

# 後の反応で確かめる (前の反応は正解から除く)
rank_metrics("bundles/shop", frame, truth=pd.read_csv("later.csv"),
             seen=seen)  # {'users': ..., 'k': 10, 'recall_at_k': ..., 'ndcg': ...}
```

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

列を 1 つずつ行の間で並べ替えて推論し直し, 予測の変わり方の平均を効きとする (表と時系列の一式だけ. DESIGN の「列の効き」).

1 行ごとの理由は `explain_rows` で, 予測の表に列ごとの効き (`effect:<列>`. 正ならその行の値が予測を上げた) と, 効きの大きい 3 列の文 (`reason`) を足して返す.

```python
from ml_inference import explain_rows

rows = explain_rows("bundles/sales", "new_sales.csv")
print(rows["frame"]["reason"][0])
# 例: promo (+210), temperature (-35.2), store (+12.1)
```

### 予測から意思決定へ

予測を出して終わりにせず, 次の行動と, その行動の見込みの得と損を出す. 損得と制約はデータからは出てこない課題の値 (例えば, 店なら廃棄の損, 研究なら 1 件の処理時間, 遊びなら続けて遊ぶ時間) なので書いて渡す (画面の ML-Provide は, 問題の設定 (ML-Quest) をつなぐとその値を既定にする).

```python
from ml_inference import backtest, infer

# 量を決める: 欠品の損が売れ残りの 3 倍なら, 予測の外れ方の 75 % の点まで置く
decision = {"type": "quantity", "under_cost": 3, "over_cost": 1, "stock": "在庫"}
result = infer("bundles/sales", "next_week.csv", decision=decision)
print("\n".join(result["decision"]["sentences"] + result["decision"]["notes"]))
# 足りない損 3 と余る損 1, 予測の外れ方の記録が正しいとすれば, 損が最も小さいのは sales の外れ方の 75 % の点まで置くこと
# 120 行のうち 85 行に, 合計 3,240 を足す
# ...
# 外れ方は検証用の分割で測った. 75 % の点を, テスト用の分割の実際の値が下回ったのは 66 % (外れ方を小さく見積もっている. 量は少なめ)

# 過去のデータで, この方針なら実際にどれだけの損だったかを比べる
print(backtest("bundles/sales", "last_month.csv", decision)["sentences"])
```

| 型 | 使いどころ | 書くこと |
| --- | --- | --- |
| `quantity` | 補充, 仕入れ, 人員 | `under_cost`, `over_cost` (1 単位の足りない損と余る損), `stock` (今ある量), `step` (刻み) |
| `select` | 引き止め, 営業先, 点検 | `positive`, `value` (当たりの価値), `cost` (行動の費用), `effect` (行動で防げる割合. 仮定), `effect_kind` (`uplift` なら `effect` は 行動したとき − しないときの当たりの確率), `effect_goal` (`uplift` の向き. `reduce` か `raise`), `budget` か `budget_cost` |
| `triage` | 人が確かめる, 再検査 | `positive`, `miss_cost` (見逃しの損), `false_alarm_cost` (空振りの損), `capacity` (回せる件数) |

今のやり方 (`current`) は列の名前か, 問題の設定 (ML-Quest) と同じ決まりで書く. 決まりなら, たどり直しが行ごとの値を作って比べる (`current_from_baseline` で列だけを作ることもできる. DESIGN の「今のやり方を決まりで書く」).

```python
import pandas as pd
from ml_inference import backtest, current_from_baseline

# 今のやり方: 同じ店の 7 日前と同じ量を置く (問題の設定の action.current と同じ形)
rule = {"type": "lag", "periods": 7, "time": "date", "by": ["store"]}
decision = {"type": "quantity", "under_cost": 3, "over_cost": 1, "current": rule}
print(backtest("bundles/sales", "last_month.csv", decision)["sentences"])

frame = pd.read_csv("last_month.csv")
now = current_from_baseline(frame, rule, target="sales")   # 行ごとの今のやり方の量
```

守る指標 (引き換えに悪くしてはいけない数字. 問題の設定の受け渡しの `guard`) は, 型ごとの数えられる量 (`GUARD_COUNTS`) のどれを指すか (`counted`) を足して渡すと, たどり直しがこの方針の行動で数え, 守る線を守れたかと並べる. 指していなければ「過去のデータでは数えられない. 課題の場で数える」と並べる (DESIGN の「守る指標を数える」).

```python
from ml_inference import GUARD_COUNTS, backtest

print(list(GUARD_COUNTS["quantity"]))   # ['shortage_rate', 'shortage', 'surplus']
# decision は上の例の量の決め方
guard = {"name": "欠品の割合", "unit": None, "better": "lower", "line": 0.1,
         "measure": "棚が空いた日を数える", "counted": "shortage_rate"}
result = backtest("bundles/sales", "last_month.csv", decision, guard=guard)
print(result["guard"]["value"], result["guard"]["kept"])
```

損得の値は数か, 予測の表の列の名前. 文は「書いた損得と外れ方の記録が正しいとすれば」の前提を先に述べるので, 実際の差は `backtest` で確かめる. 値段を変える判断は扱わない (DESIGN の「意思決定」).

選ぶと振り分けの型は確率を額面どおりに使う. 分類の確率がずれている (説明の `uncertainty.ece` が大きい) 一式は, 作るときに検証用の分割で確率を補正できる (`make_bundle(..., calibrate="platt")` か `"isotonic"`). 補正の前と後のずれは `calibrate.before` と `calibrate.after` に残り, 予測と意思決定は補正した確率を使う (DESIGN の「確率の補正」).

新しい実績が貯まったら, その実績で外れ方を測り直して一式に書き戻す (`restore_uncertainty` で戻せる. 何段も先の量には `horizon` で段ごとの外れ方も, 店や品ごとに外れ方が違うなら `groups` で行の種類ごとの外れ方も測る. 確率の補正は当て直さない (当て直すには一式を作り直す). DESIGN の「外れ方の測り直し」).

```python
from ml_inference import remeasure

result = remeasure("bundles/sales", "last_month.csv")
print(result["sentences"])
# ['今の記録の 80 % の幅は, 渡した実績の 606 行の 72 % を覆った (90 % の幅は 83 %). 外れ方を小さく見積もっていた', '測り直した外れ方の 80 % の幅は, 予測の 0.87 倍から 1.14 倍 (606 行で測った)']
```

### 強化学習の方策に守りを当てる

ML-Found が書き出した方策の一式 (`policy/`) を読み, 観測から行動と守りの結果 (`ok`, `clipped`, `fallback`, `stop`, `handover`) を返す. モデルは一式に写したモデルの定義 (`metadata.model.spec`) から組み立てるか, 同じ組み立てのモデルを渡す (DESIGN の「方策の一式」).

```python
from ml_inference import load_policy

policy = load_policy("outputs/cartpole/policy")
policy.act([0.01, -0.2, 0.03, 0.1])
# {"action": 1, "status": "ok", "reasons": []}
policy.act([2.5, 0.0, 0.0, 0.0])
# {"action": None, "status": "handover", "reasons": ["観測の要素 0 が範囲の外 (2.5)"]}
```

### 画面で試す見本の一式

表か文のデータのファイルと ML-EDA のレシピから, 小さなモデルをその場で学習して一式を作る (操作を試すためのもので, 課題のモデルではない. DESIGN の「見本の一式」).

```python
from ml_inference import make_sample_bundle

recipe = {"target": "label", "task": "classification", "steps": [],
          "split": {"method": "stratified", "valid": 0.2, "seed": 0}}
make_sample_bundle("bundles/sample", "train.csv", recipe,
                   conclusion={"min_confidence": 0.8})
```

## 文書

- [docs/DESIGN.md](docs/DESIGN.md): 今の設計と決まり, 他のリポジトリとの約束 (推論の一式と結果の形と版)
- [TODO.md](TODO.md): まだやっていないこと (優先度の順)
- [CODING_STYLE.md](CODING_STYLE.md): コードの書き方
- [CLAUDE.md](CLAUDE.md): このリポジトリで作業するときの方針

## ライセンス

MIT
