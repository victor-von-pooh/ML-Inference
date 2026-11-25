# 設計

ML-Inference の今の設計と決まりをまとめる. まだやっていないことは [TODO.md](../TODO.md) にある.

## 1. 役割と位置づけ

学習したモデルで新しいデータを推論し, 予測から結論を出して価値として提供するライブラリ.

- 予測: 学習のときと同じ前処理を新しいデータにかけ, 学習した重みのモデルで推論し, 元の形 (クラスの名前, 目的の列の単位の値) で返す
- 結論: 予測を, 利用者が使える形にまとめる. 確信の低いものを保留にする, 決めた確率を超えたものだけを陽性にする, 値を帯に分ける, 系列やグループごとに足し合わせる, 全体の件数と割合を文で返す, など
- GUI のアプリ (ML-Provide) が必要とすること (推論の一式の説明, 確かめた結果, 予測の表, 結論) は, このライブラリが JSON にできる辞書か `DataFrame` で返す. 画面に固有のものは入れない

「機械学習を全て GUI 上で行う」アプリ (問題の設定, データの分析, モデルの作成・選定, モデルの学習・評価, 推論の段階に分けて作ることを目指している) のうち, 「推論」の段階でライブラリを受け持つ. 画面は ML-Provide.

- **他の段階のライブラリの使い方**: 前処理をかけるのは ML-EDA の `Dataset.transform` と `Dataset.transform_item`, モデルを組み立てるのは ML-ModelControl の `build_model` を使う. 約束の中身 (当てはめた値の意味, 部品の組み立て方) を正しく扱えるのは, 約束を持つライブラリだけなので, 写して作り直さない. ML-EDA と ML-ModelControl は公開の名前 (`ml_eda` と `ml_modelcontrol` の `__all__` にあり `from ml_eda import ...` のように読めるもの) だけを使い, `from ml_eda.<モジュール> import ...` や `ml_eda.<モジュール>.<名前>` のように中の名前を読まない. 公開されていない名前は持ち主が断りなく変えるため. 対象は `src/ml_inference` のコードで, テストのコードは除く. `tests/test_public_names.py` が構文の木で確かめるので, `getattr` や `import_module` に渡す文字の中の名前は確かめの対象の外. 要る名前は持ち主に公開してもらう. ML-Found のコードは使わない (重みは `state_dict` を PyTorch で読むだけ). 学習は扱わない. ただし画面で操作を試す見本の一式 (9 の節) だけは, 小さなモデルをその場で学習する
- **問題の設定との分け方**: 何を結論として得たいか (保留にする確信の下限, 陽性とする確率, 値の帯, まとめる単位) は, 本来は問題の設定の段階が決める. このライブラリは問題の設定のファイルを読まず, 推論の一式の既定値と, 引数で渡した値を使う. 画面 (ML-Provide) は, 一式に問題の設定 (ML-Quest の受け渡し) をつなぐと, その値を引数で渡す
- **外の処理を呼ばない**: 推論の段階は外の処理や本番の仕組みを呼ばず, 外の処理の結果は記録として読む. 機械学習の結果は意思決定の材料に留め, どう使うかは人が決めるため
- **依存**: 外のライブラリや重みは要るまで依存に入れず, 入れるときは追加の依存にする

## 2. 他のリポジトリとの約束

| 約束 | 持つか読むか | 版 |
| --- | --- | --- |
| データの分析の段階のデータセット (ML-EDA の `dataset.yaml` と, 版 1 は当てはめた値を含む `recipe.yaml`) | 読む (版 1 は ML-EDA の `load_dataset`, 版 2 と 3 のファイルの組は `load_items` で読む) | 版 1, 2, 3 (入っている ML-EDA が読める版. 版 2 と 3 は ML-EDA の `ITEMS_FORMAT_VERSIONS`) |
| モデルの作成の段階のモデルの定義 (ML-ModelControl の形) | 読む (組み立ては ML-ModelControl の `build_model`) | 版の確かめは ML-ModelControl に任せる |
| 学習・評価の段階の実験のディレクトリ (ML-Found の `docs/FORMAT.md`. 重み `final_model.pt` と `run.json`) | 読む | `run.json` の版 1. 新しい版は読める項目だけを使い, `notes` で知らせる |
| 推論の一式 (`inference.yaml`) | 持つ | 版 9. 版 1 から 9 を読む. 版 1 のデータセットの一式は版 3, 版 2 のデータセットの一式は版 4, 雑音だけの一式は版 5 (表の出力は版 7), 既製のモデルの一式は版 6, 並べるの一式は版 8, 群分けの一式は版 9 で書く (下の「推論の一式の形」, 「版 2 のデータセットの一式」, 「雑音だけの一式」, 「表の出力の雑音だけの一式」, 「既製のモデルの一式」, 「並べるの一式」, 「群分けの一式」) |
| 学習・評価の段階の並べるの実験の利用者と品物の番号の表 (ML-Quest の並べるの骨組みが実験に書く `rank_index.json`) | 読む (下の「並べるの一式」. 一式に写す) | 版を持たない (`users` と `items` の並び) |
| 学習・評価の段階の群分けの実験の成果物 (ML-Quest のまとめるの骨組みが実験に書く `group_centers.json`, `group_assignments.csv`, `evaluations/group_stability.json`, `evaluations/group_summary.json`) | 読む (下の「群分けの一式」. 一式に写す) | 版を持たない (評価の表は ML-Found の `evaluations` の形) |
| 群の要約の行 (ML-EDA の `describe_groups` の `groups`) | ML-EDA の `describe_groups` で作る (下の「群分けの一式」) | 無し |
| 学習・評価の段階の方策の一式 (ML-Found の `docs/FORMAT.md` の「方策の一式」. `policy.json` と `policy.pt`) | 読む (下の「方策の一式」) | 版 1. 新しい版と版の無い一式は読まずに止まる |
| 推論の結果 (`result.yaml`, `predictions.csv`, `artifacts/`) | 持つ | 版 4. 版 1 から 4 を読む (下の「書き出し」) |
| 受け持ち, 出力の形, 正解の出どころの値の名前 | ML-Quest と ML-EDA の名前を写す | 無し |
| 今のやり方の決まり (ML-Quest の問題の設定の `action.current`. `type` と `periods` か `column`) | 同じ形と決まりで値を作る (下の「今のやり方を決まりで書く」. 問題の設定のファイルは読まない) | 無し |

- **推論の一式は ML-Inference が持つ約束**: 推論に要るもの (モデルの定義, 重み, 前処理, 入力と出力のつなぎ方, 結論の既定値) を 1 つのディレクトリにまとめた形は, このライブラリが `format_version` で版を付けて持つ

### 推論の一式の形 (版 3. 版 1 のデータセットの一式)

```
<一式>/
├── inference.yaml      # 説明 (版, 名前, モデル, 入力と出力のつなぎ方, 結論の既定値)
├── model_spec.yaml     # モデルの定義 (ML-ModelControl の形. 写したもの)
├── dataset/
│   ├── dataset.yaml    # データセットの説明 (ML-EDA の形. 写したもの)
│   └── recipe.yaml     # 当てはめた値を含むレシピ (写したもの. 版 1 のデータセットだけ)
└── final_model.pt      # 重み (state_dict. 写した場合だけ. 写さない場合は実験のディレクトリを指す)
```

```yaml
format_version: 3
name: ja_reviews
created_at: '2026-11-16T22:00:00+09:00'
role: predict                  # 受け持ち (下の「受け持ちと出力の形」)
output:                        # 出力の形
  form: class                  # class, value, distribution, mask, point_cloud, vector, text, image, table
target:                        # 正解の種類と出どころ (分からなければ null)
  kind: column                 # 列の予測は column. 他は ML-EDA の版 2 の target.kind (mask, point_cloud など)
  origin:                      # 正解の出どころ (分かる場合だけ. ML-EDA の版 2 の target.origin と同じ形)
    type: annotation
procedure: null                # 繰り返しの手順 (下の「繰り返しの手順」)
model:
  spec: model_spec.yaml
  weights: final_model.pt      # 一式からの相対パスか, 絶対パス
  output: logits               # 出力が辞書のモデルで使う出力の名前 (1 つなら省略可)
  kind: logits                 # logits, probabilities, value (省略すると出力の形から)
  target_scale: null           # 学習のスクリプトが目的の値を (y - center) / scale にした場合の {center, scale}
  pretrained: []               # 学習済みの重みを読むノードの置き方 (下の「学習済みの重みの置き方」)
inputs:                        # モデルの入力ごとの中身 (入力が 1 つなら省略可)
  ids:
    source: items              # features (表の列) か items (画像, 文, 点群, 音声の配列)
    prepend: [2001]            # 並びの先頭に足す番号 ([CLS] など. 文の番号の並びだけ)
dataset: dataset
run:                           # 重みを取った実験 (記録のため. 無くても推論できる)
  path: /…/outputs/ja_bert/2026-11-16/10-00-00
  status: completed
  monitor: valid/acc
  best_value: 0.945
conclusion:                    # 結論の決め方の既定値 (下の「結論」)
  min_confidence: 0.6
decision:                      # 意思決定の決め方の既定値 (下の「意思決定」. 無ければ null)
  type: triage
  positive: 悪い
  miss_cost: 5000
  false_alarm_cost: 500
check:                         # 検証用の分割を一式で推論した評価と, 実験の記録との比べ
  split: valid
  rows: 9592
  metric: acc                  # 分類は正解率 (acc), 回帰と予測は元の単位の平均絶対誤差 (mae)
  value: 0.9416
  recorded: 0.9416             # 実験の監視した指標のベストの値 (比べられなければ null)
  matches: true                # 差が 2 % 以内か (比べられなければ null)
uncertainty:                   # 予測の外れ方 (下の「予測の外れ方」)
  split: valid
  rows: 1272
  kind: relative               # 回帰: 使う誤りの形 (relative か absolute)
  absolute: [...]              # 差の 1 % から 99 % の分位点
  relative: [...]              # 比の対数の分位点 (正の値だけのとき)
  coverage: {"80": 0.74, "90": 0.82}   # 検証用で求めた幅が, テスト用で覆った割合
calibrate:                     # 分類の確率の補正 (補正しなければ項目ごと無い. 下の「確率の補正」)
  method: platt                # platt か isotonic
  split: valid
  rows: 1272
  classes: {悪い: {a: 0.41, b: -0.08}}   # クラスごとの当てはめた値 (2 クラスは 2 つ目のクラスだけ)
  before: {ece: 0.081, test_ece: 0.092}  # 補正の前の検証用とテスト用のずれ (ECE)
  after: {ece: 0.012, test_ece: 0.031}   # 補正の後の同じずれ
notes: []                      # 作るときの注意 (戻せない変換, 学習を終えていない実験など)
```

- 一式の中のパス (`model.spec`, `dataset`. 雑音だけの一式の `dataset` は null) は一式からの相対パスで, 一式の外を指せない. 重み (`model.weights`) だけは, 大きな重みを写さずに使えるように, 絶対パスも書ける
- `make_bundle` が一式を作り, `load_bundle` が読む (版と形, パスの確認. モデルは使うときに組み立てる). `describe_bundle` は画面のための説明 (場所, 名前, 作った時刻, 種類, タスク, 目的の列, クラス, 学習に使う列, 文の列, 読み込み, 前処理の手順と中身の手順, モデルの入力, 実験, 結論の既定値, 何段も先まで予測できるか (`forecastable`), 実験の記録との比べ, 外れ方の記録の要約 (`uncertainty`. 分割, 行の数, 形, 測った時刻, テスト用で覆った割合, 確率のずれ. 分位点の並びは除く), 注意, 版 (`format_version`), 受け持ち (`role`), 出力の形 (`output`), 列の出力か (`column_output`. 結論と意思決定を使えるか), 正解の種類と出どころ (`target_info`. `target` は目的の列の名前), 繰り返しの手順 (`procedure`), 学習済みの重みの置き方 (`pretrained`) と読んだときの警告 (`pretrained_warnings`. 組み立てる前は空), 既製のモデルの出どころと前処理 (`ready`. 既製のモデルの一式でなければ `None`), 並べるの欄 (`rank`) と群分けの欄 (`group`. それぞれの一式でなければ `None`), 確率の補正の要約 (`calibrate`. 下の「確率の補正」)) を辞書で返す
- 実験のディレクトリから作る場合は `run.json` の状態 (版, 状態, 監視した指標, ベストの値とエポック, 最終の重みの種類) を `run` に写す. 知らない新しい版の `run.json` は読める項目だけを使い, 学習を終えていない実験 (`completed` 以外) や途中の重み (`final_weights: partial`) とともに `notes` で知らせる (ML-Found の読む側の決まり). `run.json` の無い実験は場所だけを記録する
- 写したデータセットの説明 (`dataset.yaml`) も ML-EDA の版を確かめ, レシピのファイルは一式の外を指せない
- `model.kind` は出力の形に合うものだけ (分類 (`class`) と領域 (`mask`) と文 (`text`) は `logits` か `probabilities` (文は `logits` だけ), 他は `value`)
- 一式は, 学習のスクリプトが学習の後に実験のディレクトリの中 (例: `<実験>/inference/`) に作るか, 画面 (ML-Provide) が自分の置き場に作る. 実験のディレクトリには ML-Found と学習のスクリプトだけが書く (ML-Found の約束) ので, 画面は実験のディレクトリに書かない
- **定義の外部の重み (ノードの `weights`)**: 一式づくりと一式の読み込みは, 定義を ML-ModelControl の `build_model(..., load_weights=False)` で組み立て, 外部の重みのファイルを読まない. 外部の重みを読んで学習した後の重みは全て一式の重み (`final_model.pt`) にあり, それを strict に読むため. そこで, 外部の重みを相対パス (ML-ModelControl の決まりでは定義のファイルのディレクトリから辿る) で持つ定義も, 定義を辞書で渡すか今のディレクトリがどこかに依らずに一式を作れ, 一式を別の場所へ移しても読める. 一式の `model_spec.yaml` の `weights` は渡した定義のまま写し (パスを直さず, 外部の重みのファイルも一式に写さない), 定義の記録として残す. そのため `weights.path` が絶対パスなら, 手元の場所がそのまま一式に入る. 学習済みの重みを取得する部品 (ノードの `unit` と引数. `pretrained: true` など) は `load_weights` に依らず組み立てのときに重みを読むので, 下の「学習済みの重みの置き方」はそのまま当てる
- 一式を作るときに, 定義から組み立てて重みを読み込み, データセットの見本 (学習用の最初の数件) を推論できるかを確かめる (入力の形, 出力の幅とクラスの数). 確かめられない場合は作らずに誤りを返す
- 形が合っても, 学習のスクリプトが前処理の外でしたこと (`prepend`, `target_scale`) を書き忘れると, 予測が黙って誤る. そこで, 検証用の分割 (無ければテスト用, 学習用) を一式で推論し, 分類は正解率, 回帰と予測は元の単位の平均絶対誤差を `check` に残す. 実験の監視した指標が同じ分割の同じ指標 (`valid/acc`, `valid/mae` など) で, 最終の重みがベストの場合は記録と比べ, 2 % より離れていれば `notes` で知らせる (作るのは止めない. 学習のスクリプトが指標を元の単位でない値や違う決まりで求めている場合があるため). 比べるのは 5 万行までの分割で, 256 行ずつ読んで推論する. 比べられない場合は 256 行だけで値を残す. 推論に失敗した場合も, 理由を `notes` に残して作る
- **学習のスクリプトとの指標の名前の約束**: 実験の監視した指標 (`run.json` の `monitor`) は `<分割>/<名前>` の形で, 名前が `acc` か `accuracy` (分類の正解率), `mae` (回帰と予測の元の単位の平均絶対誤差), `iou` (領域), `chamfer` (点群) のときだけ記録と比べる (`METRIC_NAMES`. 領域と点群は下の「版 2 のデータセットの一式」). 損失や RMSE などを監視した実験は比べず, 値だけを残す. 学習のスクリプトはこの名前で step の指標を返すと, 一式の確かめで前処理の外の処理の書き忘れに気付ける (ML-Quest の書き出す骨組みはこの名前を使う)
- 版の決まりは ML-EDA, ML-ModelControl と同じ. 項目を足す, 消す, 変える場合は版を上げ, 読む側は知っている版より新しい一式を「この版は読めない」とする. 今の版は 9 で, 版 1 から 9 の一式を読む. 版 4 は版 3 に `items` (下の「版 2 のデータセットの一式」) を足した版で, 版 2 のデータセットの一式だけを版 4 で書き, 版 1 のデータセットの一式は版 3 で書く (版 3 までしか知らない読む側も, 今までの一式を読めるように. 古い版で足りるものは古い版で書くのは結果の形の版と同じ決まり. 結果の形の版が何を表すかは下の「書き出し」). 版 5 は版 4 の `dataset` を null にできる版 (データセットを持たない. 下の「雑音だけの一式」) で, 画像とベクトルなどの配列の出力の雑音だけの一式を版 5 で書く (表の出力の雑音だけの一式は版 7). 他の一式は今までと同じ版 (3 か 4) で書くので, 版 4 までしか知らない読む側 (前の ML-Inference を submodule で使う ML-Provide など) は今までの一式をそのまま読み, 版 5 の一式だけを「format_version 5 の一式は … 読めません」で止める. 版 5 の一式が `dataset` か `items` を持つ, `dataset` が null の一式が版 5, 6, 7, 8, 9 のどれでもない, は誤り. 版 6 は版 5 に `ready` (下の「既製のモデルの一式」) を足した版で, 既製のモデルの一式だけを版 6 で書く (他の一式は版 3 から 5 のまま. 版 5 までしか知らない読む側は, 版 6 の一式だけを「format_version 6 の一式は … 読めません」で止める). 版 6 の一式が `ready` を持たない, `ready` を持つ一式が版 6 でない, は誤り. 版 7 は版 6 に出力の形 `table` と列の符号化 (`output.columns`. 下の「表の出力の雑音だけの一式」) を足した版で, 表の出力の雑音だけの一式だけを版 7 で書く (他の雑音だけの一式は版 5 のまま. 版 6 までしか知らない読む側は, 版 7 の一式だけを「format_version 7 の一式は … 読めません」で止める). 出力の形が `table` の一式が版 7 でない, 版 7 の一式の出力の形が `table` でない, 版 7 の一式が `dataset` か `items` を持つ, は誤り. 版 1 は `uncertainty` と `decision` を持たない (外れ方が要る意思決定は, 一式を作り直すよう案内する). 版 1 と 2 は `role`, `output`, `target`, `procedure`, `model.pretrained` を持たず (列の予測の一式は, 版 2 と版 3 でこの 5 つの項目だけが違う), 受け持ちは `predict`, 出力の形はタスクから (分類は `class`, 回帰と予測は `value`), 正解の種類は `column` として読む. 版 2 までしか知らない読む側は, 版 3 の一式を「format_version 3 の一式は … 読めません」で止める. 版 3 の一式が版 2 のデータセットを指す, 版 4 の一式が `items` を持たない, などの食い違いは誤り. 版 8 は版 7 に出力の形 `ranking` と `rank` (下の「並べるの一式」) を足した版で, 並べるの一式だけを版 8 で書く (他の一式は版 3 から 7 のまま. 版 7 までしか知らない読む側は, 版 8 の一式だけを「format_version 8 の一式は … 読めません」で止める). `rank` を持つ一式が版 8 でない, 出力の形が `ranking` の一式が版 8 でない, 版 8 の一式の出力の形が `ranking` でないか `rank` を持たない, は誤り. 版 9 は版 8 に出力の形 `cluster` と `group` (下の「群分けの一式」) を足した版で, 群分けの一式だけを版 9 で書く (他の一式は版 3 から 8 のまま. 版 8 までしか知らない読む側は, 版 9 の一式だけを「format_version 9 の一式は … 読めません」で止める). `group` を持つ一式が版 9 でない, 出力の形が `cluster` の一式が版 9 でない, 版 9 の一式の出力の形が `cluster` でないか `group` を持たない, は誤り
- `load_bundle` は重みのファイルがあるかも確かめる (写さない重みの実験を動かすと読めない)

#### 受け持ちと出力の形 (版 3)

推論の一式は受け持ち (`role`) と出力の形 (`output.form`) を持ち, モデルの出力の読み方を出力の形で分ける. 列の予測 (クラスか値) の他に, 構造のある出力 (領域, 点群, 画像) とベクトル (埋め込み) と文を読む.

- 受け持ち: `predict` (予測する), `generate` (作る), `rank` (並べる), `group` (まとめる), `effect` (効果を見積もる). 値は ML-Quest の受け持ちと同じ. 表す (埋め込み) は受け持ちにせず段の部品なので, 埋め込みのモデルの一式は, その段が属する受け持ち (`rank`, `group` など) か `predict` で書く
- 出力の形: ML-EDA の版 2 の項目の種類と同じ名前を使う (ML-Quest の出力の形とは `mask` が `region`, `point_cloud` が `shape` に当たる). 並べるの `ranking` と群分けの `cluster` は ML-Quest の出力の形 (順位と群) と同じ名前

| 形 | モデルの出力 | 予測の表 | `model.kind` | 結論と意思決定 |
| --- | --- | --- | --- | --- |
| `class` | (件数, クラスの数) か 2 クラスで 1 列 | `prediction`, `confidence`, `p:<クラス>` | `logits`, `probabilities` | 使える |
| `value` | (件数,) か (件数, 1) | `prediction` (元の単位) | `value` | 使える |
| `distribution` | (件数, 分位点の数). 分位点の水準は `output.quantiles` (0 から 1 の小さい順) | `prediction` (0.5 の点. 無ければ前後の水準の間を線形に補う. 0.5 が水準の外なら最も近い端の点), `q:<水準>` (元の単位. 行ごとに小さい順に並べ直す. 分位点の交差を直すため) | `value` | 値と同じ (`prediction` を使う) |
| `mask` | (件数, クラスの数, 高さ, 幅) か (件数, 高さ, 幅) | `output` (1 件の (高さ, 幅) の整数の配列. 版 2 のデータセットの一式は領域の値. 書き出しは `artifacts/`) | `logits`, `probabilities` | 件数だけ |
| `point_cloud` | (件数, 点の数, 次元) | `output` ((点の数, 次元)) | `value` | 件数だけ |
| `vector` | (件数, 次元) | `output` ((次元,)) | `value` | 件数だけ |
| `image` | (件数, ...) | `output` (1 件の配列のまま) | `value` | 件数だけ |
| `table` | (件数, 幅) の符号化した行 (幅は `output.columns` から) | 列の符号化の列の名前ごとの値 (数の列は小数, 整数の列は `Int64`, 種類の列は種類の値) | `value` | 件数だけ |
| `ranking` | 点数 `score` と利用者と品物の埋め込み (下の「並べるの一式」) | 利用者と順位ごとの `user`, `rank`, `item`, `score` | `value` | 利用者の人数だけ |
| `cluster` | 行の埋め込み `embedding` (下の「群分けの一式」) | `group` (最も近い群の中心の番号), `distance` | `value` | 群ごとの件数と群の要約と代表の行 |
| `text` | (件数, 長さ, 語彙の数) の logits か, 繰り返しの手順の出した番号の並び | `prediction` (レシピの語彙で戻した文. 語彙が無ければ `tokens` と同じ), `tokens` (番号を空白でつないだ文字) | `logits` | 件数だけ |

- 領域は, クラスの次元があれば最も大きいクラスの番号, 1 チャンネルなら `logits` は 0 より大きいか, `probabilities` は 0.5 以上かで 1 と 0 にする. `output.classes` (番号ごとの名前) は説明に写すだけ
- 列の出力 (`class`, `value`, `distribution`) だけが, 結論の決め方, 意思決定, たどり直し, 外れ方の測り直し, 列の効き, 何段も先の予測を使える. 他の形の一式でこれらを呼ぶと誤り. 結論は件数 (全体, 推論した, しなかった) と文だけを返す. 外れ方 (`uncertainty`) も列の出力だけ (他の形は `null`). 一式を作るときの記録との比べ (`check`) は列の出力と, 版 2 のデータセットの一式 (領域と点群) だけ
- 出力の形ごとの読み方は関数の表 (`DECODERS`) にして, 形を足すときは表に足す
- **文の戻し方** (`ml_inference.texts`): 番号から文字への戻し方は, 一式の写しのレシピの最後の `vocabulary` の手順 (前処理がかける手順) の語彙 (`state.vocabulary`. 番号の順の語) と分け方 (`unit`) と並びの長さ (`length`. 既定 128) から読む (ML-EDA が番号を語に戻す使い手のために残す語彙. 戻す関数は ML-EDA に無いので, 語彙の並びを引くだけをこのライブラリが持つ). 部分語で `state.vocabulary` の無いレシピは, 区切りの JSON (`state.tokenizer`) の `model.vocab` を番号の順に並べて語彙にする (tokenizers は使わない). 最後の手順に語彙が無い (当てはめていない, 区切りの JSON を読めない) 場合は, 前の手順の語彙は使わず, `prediction` も番号のままにする. 1 件の番号の並びから, 繰り返しの手順の一式では `state` の入力の `prepend` (前処理の外で足した語彙の外の番号) が先頭にあれば除き, 手順の `eos` があれば, 渡した文の並び (`prepend` の後の `length` 個) より後ろの最初の `eos` から後を除き (ML-EDA の語彙には終わりの語が無く `eos` に「。」や埋め草の 0 を選ぶことがあるので, 渡した文と埋め草では切らない), 埋め草 (0) を除いてから, 番号を語にする. 語彙の外の番号は `<番号>`, 語彙に無い語 (1) は `<unk>` のまま. 語のつなぎ方は分け方ごとに, 文字 (`char`) はそのままつなぐ, 語 (`word`) は, 文字と数字の並び (かなと漢字を除く. ML-EDA の語の分け方と同じ文字の組) と `<…>` (まとめて語) どうしの間, 区切りの記号 (`, . ! ? ; : )`) と後ろの語か `(` の間, 語と後ろの `(` の間だけに空白を入れる (英語のような書き方の空白を戻す程度の決まり. ただし前後が数字の区切りの記号 (`3.5`, `10:30`, `1,000`) の後ろには入れない. 「Mr.Smith」「e.g.」(`e. g.` になる) のような文字の間の記号と, 引用符 `"` のまわりの空白 (引用符は前後の語に付ける) は戻らない), 部分語 (`subword`) はそのままつないで空白の印 (`▁`. ML-EDA の部分語の区切りの Metaspace) を空白にし, 前後の空白を除く. 前処理で落とした小文字と大文字の違い, 文字の分け方の空白, 語の分け方のかなと漢字のまわりの空白, 長さで切った後ろは戻らない. 繰り返しの手順の一式は渡した文 (始まりの番号の並び) に足した番号まで続けた文を返す. 除くと何も残らない並び (埋め草と `eos` だけなど) の行は, 予測した行 (`status` が `ok`) のまま `prediction` を空の文字にする. 書き出しの表では空の欄になり, 予測しなかった行の欠損と同じに見えるので, `read_result` は `tokens` の列のある表 (文の出力. 下の「書き出し」の「文の出力の見分け」. `tokens` は予測の列の名前に取ってあり, 入力の同じ名前の列は出力の形に依らず「tokens (入力)」になる. 文の列の名前が `tokens` のデータセットでは, 予測の表の文の列は「tokens (入力)」なので, データセットの `text_column` で表の列を引く呼び手 (ML-Provide の画面など) は名前を読み替える) の `prediction` と `tokens` を文字のまま読み (番号 1 つの並びや数に見える文を数にしない), `ok` の行の空の欄を空の文字に, 他の行の空の欄を欠損にする
- 予測できなかった行 (前処理の後の欠損, 数でない出力) の扱いは形に依らず同じ. 配列の出力は, 1 つでも数でない値があれば予測しなかった行にする (領域のように整数に変える形も, 変える前のモデルの出力で確かめる. `sample` は数でない logits があると引けずに誤りになる)
- 作るときに `role` と `form` を省くと, タスクから決める (分類は `predict` と `class`, 回帰と予測は `predict` と `value`. タスクが `none` のデータセットは `form` が要る). `class` は分類のデータセット, `value` と `distribution` は回帰か予測のデータセットだけに使える
- ベクトルの出力の一式 (埋め込みのモデル): `make_bundle(..., form="vector")`. 正解の無いデータセット (タスクが `none`) でも作れる. 見本の推論で (件数, 次元) の形を確かめる
- `target`: 正解の種類 (`kind`) と出どころ (`origin`). 列の予測では `kind: column`. 出どころは ML-EDA の版 2 の `target.origin` と同じ形 (`type`: `process`, `annotation`, `measurement`, `synthetic`, `unknown`. `name` と `settings`) で, `make_bundle(target=...)` で渡した場合だけ書く (版 1 のデータセットは出どころを持たないため). `type: process` の一式は, 結論の文に「正解は … の出力なので, その処理との一致までしか分からない」を添える (評価と結論で, その処理を超えられないことを示すため). `target_of(bundle)` は一式の `target` の写しを返し (版 2 までの一式は `kind: column`), `origin_sentence(target)` はその文を返す (処理の出力でなければ `None`. 画面が結果の説明に同じ文を出すため)

#### 版 2 のデータセットの一式 (版 4. `items`)

ML-EDA の版 2 と 3 のデータセット (ファイルの組. `load_items` で読む) から作った一式は `items` を持ち, 版 4 で書く. この節の「版 2 のデータセット」は版 3 も含む (版 3 は版 2 に項目ごとの中身の手順を足したもの. 下の「中身の手順」). 版 2 のデータセットは前処理のレシピを持たず, 1 件の扱い (画像を 0 から 1 にする, 深度から点に戻して決まった数を選ぶ) は学習のスクリプトがする. そこで一式は, 学習のスクリプトがした 1 件の扱いを名前と設定で記録し, 推論で同じ扱いをかける. この扱いは ML-EDA に無いので, ML-Quest の骨組みの学習のスクリプトと同じ扱いを `ml_inference.items` が持つ (骨組みの扱いを変えたら, こちらも合わせる. 一式の確かめは領域の値の取り違えには気付けるが, 点の数や種の書き違いは, 点の並びに依らないモデルでは記録との差が小さく気付けないことがある).

```yaml
format_version: 4
role: predict
output: {form: mask, classes: [background, circle, square]}
target: {kind: mask, origin: {type: synthetic}}
items:
  input: image                 # モデルに渡す入力の項目 (データセットの entries の名前)
  target: mask                 # 正解の項目 (一式の確かめで使う)
  prepare: image               # 入力の 1 件の扱い方 (下の表)
  options: {}                  # 扱い方の設定
  values: [0, 1, 2]            # 領域だけ: 出力のクラスの番号ごとの領域の値
  restore: null                # 点群だけ: shift (出力から正規化の平行移動を引いて元の座標に戻す) か null
dataset: dataset               # dataset.yaml だけを写す (版 2 はレシピと件の表を写さない)
```

| 扱い方 (`prepare`) | 入力の項目の種類 | モデルの入力 | 設定 (`options`) |
| --- | --- | --- | --- |
| `image` | `image` | (件数, チャンネル, 高さ, 幅) の float32. uint8 の画素を 255 で割る | 無し |
| `capture_points` | `capture` | 深度から戻した世界の座標の点から `points` 個を選んだ (件数, `points`, 3) の float32 | `points` (要る), `target_points` (確かめで正解から選ぶ数. 省くと正解の全ての点), `seed` (既定 0) |
| `points` | `point_cloud` | 点群から `points` 個を選んだ (件数, `points`, 3) の float32 | 同上 |

- **深度から点へ**: 撮影の座標の決まり (ML-EDA の `convention`) が OpenGL の軸 (+X 右, +Y 上, -Z 前) で深度が `z` のものだけを扱う (他は理由を付けて飛ばす). 姿勢のある枚の, 深度が 0 より大きい画素 (行 r, 列 c, 深度 d) を, カメラの座標の ((c + 画素の中心 - cx) / fl_x × d, -(r + 画素の中心 - cy) / fl_y × d, -d) にし, 姿勢で世界の座標にする. 深度の大きさが画像 (`w`, `h`) と違えば, カメラの内部の値を同じ比で縮める. この戻し方は `backproject(capture)` (ML-EDA の撮影の辞書から (点の数, 3) の点) で公開する (画面の見本が同じ点を作るため)
- **点の選び方**: 件ごとに `numpy.random.default_rng([seed, 位置])` (位置は分割か渡したデータの中の 0 からの番号) で, 入力の点から `points` 個を選び, 確かめでは続けて正解から `target_points` 個を選ぶ (足りなければ重ねて選ぶ). 同じ一式と同じデータなら同じ点を選ぶ. ML-Quest の骨組みの検証用の分割と同じ選び方なので, 骨組みは `seed` に検証用の分割の種を渡し, 一式の確かめが学習の記録と合う
- **省いた欄**: `make_bundle(items=...)` で渡す. 省くと, `input` は役が `input` の最初の項目, `target` は `target.entries` の最初, `prepare` は入力の項目の種類から, 領域の `values` は `classes` の値 (無ければ `values`) の小さい順 (名前の無い値 0 は背景 `background` として足す. ML-Quest の骨組みと同じ決まり), `output.classes` は `values` の名前の並び, `restore` はデータセットが `normalization` を持ち, 入力が撮影なら `shift`. `capture_points` と `points` の `points` は省けない (選ぶ数が学習と違うと, 黙って違う予測になり, 一式の確かめでも見つからないことがあるため. 学習のスクリプトと同じ値を書く)
- **点の数と定義**: 点を選ぶ一式では, `items` を渡すモデルの入力の定義の形 (件, 点の数, 3) か (件, 3, 点の数) の点の数と `points` を比べ, 違えば一式を作る時点で誤りにする (誤りの文は両方の数を挙げ, `points` か定義の `shape` のどちらかを直すよう案内する). 件の後ろの 2 つの軸のうち 3 の軸を座標とし (後ろの軸を先に見る), 他方を点の数とする. 点の数の軸が `None` (可変) の形と, 軸を決められない形 (2 つの軸でない, どちらも 3 でない, `shape` が無い) は比べない. 点の並びに依らないモデルは点の数が違っても動き, 一式の確かめでも見つからないことがあるため
- **作る前の説明** (`describe_items(path)`): 版 2 のデータセットを ML-EDA の `load_items` で読み (読めないデータセットと `dataset.yaml` の無い場所は ML-EDA の誤りのまま `ValueError`), 一式を作る画面のための説明を返す. 欄は `format_version`, `name`, `role`, `target` (`kind`, `entries`, `origin`), `entries` (書き出した順の項目ごとの `name`, `kind`, `role`, `shape`, `points`, `steps`), `splits` (分割ごとの件数), `classes` (正解の領域の項目に書いた値と名前. 名前の無い値は `None`. 領域でなければ空), `normalized` (撮影の正規化があるか), `items` (上の「省いた欄」の決まりで埋めた `items` の既定. `make_bundle` が `items` を省いたときに使うものと同じ), `problem` (`items` を決められない理由. 下の「作れる組」でないデータセットなど). 点を選ぶ扱い方の `points` は学習と同じ値を書くものなので, 既定の `options` は空のまま. 項目の `points` は, その項目の種類が点を選ぶ扱い方の入力の種類 (`POINT_KINDS`. `capture` と `point_cloud`) で, 選ぶ点の数が要るか
- **作れる組**: 予測する (データセットの `role: predict`) で, 正解の種類 (`target.kind`) が `mask` か `point_cloud` のもの. 出力の形は正解の種類と同じ (省くと正解の種類). 他の組 (`boxes` など) と, 版 2 のデータセットに無い項目の名前は誤り
- **出力の戻し方**: 領域は, クラスの番号を `values` で領域の値にする (データセットの領域のファイルと同じ値). 点群は, `restore: shift` なら件ごとの shift を引いて元の座標 (撮影の元の世界. メートル) にする (正規化した座標 = 元の座標 + shift)
- **推論するデータ** (`predict` の `source`): 版 2 のデータセットのディレクトリ (全ての分割の件. 撮影は正規化した後なので, 表の `normalization.columns` の列 (`shift_x` など) の shift で戻す) か, 入力の項目と同じ種類の元のファイル (画像は画像のファイルかディレクトリ, 撮影は撮影のディレクトリかそれを並べたディレクトリ, 点群はファイルかディレクトリ. ML-EDA の読み込みで読む). 元の撮影には学習と同じ正規化 (ML-EDA の `normalize_capture` で一時の場所に写し, `read_capture` で読む) をかけ, 足した量を表の `shift_x`, `shift_y`, `shift_z` に残す. 表と ML-EDA の `Data` は渡せない
- **中身の手順** (版 3 のデータセット): ML-EDA の書き出しが項目にかけた手順 (画像の大きさをそろえる, 灰色にする, 点群の点を選ぶなど) は, データセットの説明の項目の `steps` にあり, 一式の写しの `dataset.yaml` にも残る. 元の画像と点群には ML-EDA の `transform_entry` で, 元の撮影には `normalize_capture(..., steps=...)` で, 書き出しと同じ手順をかけてから 1 件の扱いをかける (大きさの違う元の画像も, 学習と同じ大きさにそろう). 元の画像の領域の出力は, 件ごとに元の画像の大きさ (高さ, 幅) に戻す. 中身の手順のある一式で, 読んだときの元の画像の大きさを件ごとに覚えておき, 出力の大きさが違えば最も近い画素の値で広げるか縮める (戻した画素の行 r は出力の行 floor((r + 0.5) × 出力の高さ / 元の高さ) の値. 列も同じ. 値を混ぜないので, 出力は `values` の値だけのまま. 組み込みの `resize` のように画像の全体を引き伸ばす手順を前提にする). この選び方は ML-EDA の書き出しが領域をそろえる手順 (`resize_mask`) と同じ式を写して持ち, 同じ結果になることをテスト (`tests/test_restore_size.py`) が ML-EDA と比べて確かめる. 手順は ML-EDA が持つので, 写して作り直さない. データセットのディレクトリの件は書き出しで手順をかけた後なので, 手順をかけずに読み, 領域の出力もデータセットの大きさのまま返す. 入力の項目の手順が一式と違うデータセットは, 黙って違う予測にならないよう誤りにする (元のファイルを渡すよう案内する). 正規化しない撮影の項目に手順があれば誤り (ML-EDA の撮影の書き出しは必ず正規化する)
- 読めない件, 画像のチャンネルの数が学習の入力 (データセットの `shape`) と違う件は, 理由を付けて飛ばす. 大きさの違う件は, 大きさごとにまとめて推論する
- **確かめ** (`check`): 見本の推論 (学習用の最初の数件) と, 検証用の分割 (無ければテスト用, 学習用) の評価. 領域は `iou` (分割の全ての画素で数えたクラスごとの重なりと和から, 背景 (値 0) を除くクラスの IoU の平均. 予測にも正解にも無いクラスは 1. 正解の値が `values` に無い画素は数えない), 点群は `chamfer` (件ごとの, 両向きの最も近い点までの距離の平均の和の半分を, 件で平均. 正規化した座標で, 単位はデータセットのまま). どちらも ML-Quest の骨組みの指標と同じ求め方. 実験の監視した指標が同じ分割の同じ名前 (`valid/iou`, `valid/chamfer`) なら, 列の一式と同じ決まり (最終の重みがベスト, 5 万件まで, 差が 2 % 以内) で記録と比べ, 合わなければ `notes` で知らせる. 比べられない場合は 256 件だけで値を残す
- 説明 (`describe_bundle`) の `kind` は `items`, `items` は `items` の欄に入力の項目の種類 (`input_kind`) を足したもの

#### 繰り返しの手順 (版 3. `ml_inference.procedures`)

拡散やフローのモデルのサンプリング, 文の番号を 1 つずつ足す生成のように, モデルを何度も呼んで出力を作る手順を, 名前で選ぶ登録簿に置く. 一式には手順の名前, 回数, 乱数の種を書く. 学習の途中の見本 (ML-Found) も同じ登録簿を使えるように, `ml_inference.procedures` は PyTorch だけを使う純粋な関数にし, 推論の一式, ML-EDA, ML-Provide に依らない.

```yaml
procedure:
  name: ddim                   # 登録簿の名前
  steps: 50                    # 繰り返しの回数 (文の生成は足す番号の数の上限)
  seed: 0                      # 乱数の種
  options: {eta: 0.0}          # 手順ごとの設定 (省略可)
```

| 名前 | 何をするか | モデルの呼び方 | 始まり | 設定 |
| --- | --- | --- | --- | --- |
| `ddim` | 拡散のモデルの DDIM のサンプリング. 雑音から, 予測した雑音を引いて戻す | `model(x, t)` → 予測した雑音. `t` は学習の時刻 (整数) | 雑音 | `train_steps` (学習の時刻の数. 整数だけ. 既定 1000), `beta_start`, `beta_end` (線形の雑音の予定. 既定 1e-4, 0.02), `eta` (0 で決まった道. 既定 0) |
| `euler` | フローのモデルの Euler 法. 時刻 0 の雑音から時刻 1 のデータへ速度を足していく | `model(x, t)` → 速度. `t` は 0 から 1 の小数 | 雑音 | `t_start`, `t_end` (既定 0 と 1) |
| `greedy` | 文の番号の生成. 毎回最も確率の高い番号を足す | `model(ids)` → logits ((件数, 長さ, 語彙) か (件数, 語彙). 3 次元は最後の位置) | 渡した番号の並び | `eos` (終わりの番号. 全ての件が出したら止め, 出した件はその後も `eos` で埋める) |
| `sample` | 文の番号の生成. 温度で割った確率から引く | 同上 | 同上 | `temperature` (0 より大きい. 既定 1), `eos` |

- 入口: `PROCEDURES` (名前から手順), `get_procedure(name)`, `register_procedure(name, function, start, time, description, overwrite=False)` (自作の手順. 組み込みの名前は上書きできない), `make_generator(seed)`, `start_noise(shape, generator)`, `run_procedure(name, model, state, steps, generator, **options)`, `procedure_options(name)` (手順の設定の名前と既定値. 画面が設定の欄を作るため). 手順の関数は `function(model, state, steps, generator, **options)` で, `model` は呼べるものなら何でもよい (`nn.Module` に限らない). 乱数は渡した `torch.Generator` (cpu) からだけ引くので, 同じ入力と同じ種なら同じ出力になる. 勾配は計算しない
- 設定は手順の関数の 5 つ目からの引数で, `procedure_options(name)` は `{"names": [...], "defaults": {...}}` を返す. `names` は設定の名前 (設定の無い手順は空. `**kwargs` を持つ関数と, 引数を読めない関数は `None` で, 何でも受け取る. `*args` は除く). `defaults` は名前ごとの既定値で, 有限の数, 真偽値, 文字の既定値だけを入れる (既定値の無い設定と, `None`, inf, NaN, 並びなどの既定値は入れない)
- ML-Found からの使い方の例: `g = make_generator(0); x = start_noise((8, 16), g); samples = run_procedure("ddim", denoiser, x, 50, g)`
- 一式で使うとき: モデルの入力の出どころに `state` (繰り返しで変わる中身) と `time` (手順の渡す時刻. `ddim` と `euler` だけ) を足す. 他の入力 (`features`, `items`) は条件として毎回同じものを渡す. 雑音から始まる手順は `output.shape` (1 件の形) が要り, 予測する行の数だけ種から雑音を引く (手順を動かす前に, 全ての行の分を行の順に 1 行ずつ引く). 文の手順は `items` の番号の並び (`state` の入力の `prepend` を足したもの) から始める. 同じ ML-Inference と PyTorch の版で, 同じ一式, 同じ入力, 同じ `batch_size` なら同じ出力 (版が違うと雑音の引き方が変わりうる. 雑音と乱数は 1 つの生成器から予測する行の順に引くので, 予測できない行 (飛ばす行) が変わると, 他の行の出力も変わる. 手順の中で引く乱数は件数ずつ引くので `batch_size` に依る. 雑音から始まる手順の始まりの雑音は `batch_size` に依らない). `eos` で件数ずつ止まる長さが違う文の手順は, 最も長いものに `eos` で埋めて並べる
- 一式を読むときに, 手順の名前が登録簿にあるか, 回数が 1 以上の整数か, 種が整数か, 設定の名前が手順の引数にあるか (`procedure_options` の `names` と同じ決まり. 設定の無い手順は設定を受け取らず, `names` が `None` の手順は何でも受け取る), 組み込みの設定の値の型と範囲, 手順の始まりと出力の形の組 (雑音からは `class`, `value`, `distribution`, `text` 以外, 番号の並びからは `text` だけ. 番号の並びから始まる手順は文のデータセットだけ) を確かめる. 知らない名前の一式は「登録簿に無い」誤りで止める (自作の手順は, 一式を読む前に `register_procedure` で登録する. 一式に書かれたコードは読まない)

#### 雑音だけの一式 (版 5. `dataset: null`)

条件の無い生成のモデル (雑音から始まる手順の中身と時刻だけを入力にし, 他の入力を持たないモデル. 拡散やフローのモデル) の一式は, データセットを持たず, データの行の代わりに件数と種を入力にして推論する. 版 5 で書く.

```yaml
format_version: 5
role: generate
output: {form: image, shape: [1, 32, 32]}
target: null
procedure: {name: ddim, steps: 50, seed: 0, options: {eta: 0.0}}
model: {spec: model_spec.yaml, weights: final_model.pt, kind: null, ...}
inputs:
  x: {source: state}
  t: {source: time}
dataset: null                  # データセットを持たない (dataset/ も作らない)
conclusion: {}
decision: null
check: null
uncertainty: null
```

- **作り方**: `make_bundle(path, spec, None, weights, procedure=..., inputs=..., form=..., output_options={"shape": [...]})` (データセットに `None` を渡す). 要るのは, 雑音から始まる手順 (登録簿の `start` が `noise`), `output.shape` (1 件の形), 雑音から始まる手順に使える出力の形 (`mask`, `point_cloud`, `vector`, `image`, `table`. `table` は下の「表の出力の雑音だけの一式」). モデルの入力の出どころは `state` と `time` だけ (`features` と `items` は誤り. 入力が 1 つで `inputs` を省けば `state`). 受け持ちは省くと `generate`. 正解 (`target`), 結論の決め方と意思決定の既定値, `target_scale`, `items` は誤り (列の出力でなく, データセットも正解も無いため)
- **確かめ**: 作るときに, 一式の手順で 2 件を作って出力の形と読み方を確かめる (確かめられなければ作らずに誤り). 比べる正解が無いので `check`, `uncertainty`, `target` は null (null でない版 5 の一式は誤り). 実験の記録 (`run`) と注意 (`notes`. 学習を終えていない実験など) は他の一式と同じ
- **推論**: `predict(bundle, {"count": 件数, "seed": 種})` (`infer` も同じ). 件数は 1 以上の整数, 種は -2^63 以上 2^64 未満の整数 (PyTorch の生成器の種の範囲) で, 省くと `procedure.seed`. 他の入力 (パス, 表, ML-EDA の `Data`) と読み込みの設定 (`options`) は誤り, 雑音だけの一式でない一式に件数と種を渡すのも誤り. 表は件ごとの `sample` (0 からの番号), `seed` (使った種), `status`, `reason`, `output` (1 件の配列. 書き出しは `artifacts/`). 結果の `source` は「件数 n, 種 s」の文字. 件数に上限は無く, 始まりの雑音は全ての件の分を一度に作ってから手順を動かすので, 大きな件数は件数 × 1 件の形の大きさの記憶が要る (画面で受け付ける件数の上限は画面 (ML-Provide) が決める)
- **同じ出力**: 始まりの雑音は, 種から作った 1 つの生成器から, 件の順に 1 件ずつ, 手順を動かす前に全ての件の分を引く. そこで同じ一式と同じ種なら, 始まりの雑音は `batch_size` に依らず, 件数を増やしても前の件の雑音は変わらない. 手順の中で引く乱数 (`ddim` の `eta` が 0 より大きいときなど) は, 同じ生成器から件数ずつ引くので `batch_size` に依り, 始まりの雑音の後に引くので件数にも依る (そうした手順の出力は, 件数を変えると前の件も変わる). 同じ ML-Inference と PyTorch の版で, 同じ一式, 同じ件数, 同じ種, 同じ `batch_size` なら, 必ず同じ出力になる (版が違うと雑音の引き方が変わりうる)
- **説明**: `describe_bundle` の `kind` は `noise`, `task` は空で, データセットから読む欄 (`target`, `classes`, `features`, `steps` など) は空. `Bundle.dataset` は説明とレシピが空のデータセット (ML-EDA の `Dataset`) なので, データセットの説明を見る呼び手 (`bundle.dataset.info.get(...)`) もそのまま動く
- 列の出力でないので, 結論は件数だけで, 列の効き, 何段も先の予測, たどり直し, 外れ方の測り直しは使えない (上の「受け持ちと出力の形」の決まりのまま)

#### 表の出力の雑音だけの一式 (版 7. `output.form: table`)

表を作る生成のモデル (列ごとに符号化した行を拡散で学んだモデル. ML-Quest の表の生成の骨組み) の雑音だけの一式は, 列の符号化を一式に写し, 件数と種から列の名前と型のある表を返す. 版 7 で書く. 上の「雑音だけの一式」の決まり (作り方, 確かめ, 推論の件数と種, 同じ出力, 説明) はそのままで, 違うところを下に書く.

```yaml
format_version: 7
role: generate
output:
  form: table
  shape: [16]                  # 符号化した行の幅 (省くと列の符号化から)
  columns:                     # 列の順の符号化 (ML-Quest の骨組みの table_columns.json の columns)
  - {name: x, type: number, mean: 10.0, std: 2.0, integer: false}
  - {name: n, type: number, mean: 3.0, std: 1.5, integer: true}
  - {name: color, type: category, categories: [blue, green, red]}
dataset: null
# 他の項目は雑音だけの一式 (版 5) と同じ
```

- **列の符号化** (`output.columns`): 列の順の辞書の並び. `name` (空でない文字. 重ならない) と `type` (`number` か `category`). 数の列は `mean` (有限の数), `std` (正の数), `integer` (真偽値), 種類の列は `categories` (文字, 数, 真偽値の 1 つ以上の並び. 文字にして重ならない. 符号化のワンホットの順). 他の欄は誤り. 符号化した行の幅は, 数の列の数と種類の列の値の種類の数の和で, `output.shape` はこの幅だけの `[幅]` (違えば誤り)
- **作り方**: `make_bundle(path, spec, None, weights, form="table", output_options={"columns": ...}, procedure=..., inputs=...)`. `columns` は列の符号化の並び, `columns` を持つ辞書, そのどちらかの JSON のファイル (ML-Quest の骨組みが実験に書く `table_columns.json`) のどれかで, 読んで一式の `output.columns` に写す (一式だけで表に戻せるように. 実験のファイルは読み続けない). `shape` は省くと幅から. データセットを渡した一式に `table` は使えない
- **列の名前**: 予測の表の列 (`sample`, `seed`, `status`, `reason`, `decision`, `output`, `artifact`, `tokens`, `prediction`, `confidence`), `p:` と `q:` で始まる名前, 「 (入力)」で終わる名前は, 予測の表で列が重なるか取り違えるので誤り (データセットの列の名前を変える)
- **戻し方** (`ml_inference.tables`): 1 件の符号化した行を, 列の順に区間に切り, 数の列は値 × `std` + `mean` (整数の列は偶数への丸めで整数に), 種類の列はその列の区間で最も大きい値の種類 (`categories` の値のまま. 同じ大きさなら前の種類) に戻す. 学習のスクリプトの符号化 (ML-Quest の骨組みの `decode_table`) と同じ式. 戻せない値のある行は, 予測しなかった行にする (`status` は `skipped`, `reason` は「予測の値を表に戻せません …」). 戻せない値は, 数の列の数でない値, 整数の列の丸めた値が `Int64` の範囲 (-2^63 から 2^63 - 1) の外, 種類の列の区間の数でない値. 例外では止めない
- **推論**: `predict(bundle, {"count": 件数, "seed": 種})` の表は, 件ごとの `sample`, `seed`, `status`, `reason` と, 列の符号化の順の列 (1 件が表の 1 行). 型は, 数の列が `float64`, 整数の列が欠損を持てる `Int64`, 種類の列が種類の値の `object` (予測しなかった行は欠損). `output` の列は無い. 同じ一式と同じ種なら, 同じ重みのベクトル (`vector`) の一式の行を列の符号化で戻したものと同じ
- **結論と書き出し**: 列の出力でないので, 結論は件数だけ (`conclude_outputs`). 結果は配列を持たないので, 書き出しは `predictions.csv` と `result.yaml` だけ (下の「書き出し」. 数に見える種類の値か欠損のある整数の列があれば, 列の型を残す版 3. 数と文字の混ざる列があれば版 4). `read_result` は種類の値を文字, 整数の列を欠損を持てる整数で読む
- **説明**: `describe_bundle` の `output` に `columns` (列の符号化) が入り, `kind` は `noise`

#### 既製のモデルの一式 (版 6. `ready`)

学習せずにそのまま使う既製のモデル (ライブラリの学習済みの重み) の一式は, データセットと実験を持たず, モデルの出どころ (ライブラリ, モデルの名前, 重みの名前), 種類の名前の表, 前処理を一式に書く. 版 6 で書く. 今は画像の分類のモデルだけで, 出どころは torchvision だけ. 出力の形 (`READY_FORMS`) は, 種類の予測 (`class`) と, 分類の最後の層の手前の出力の埋め込み (`vector`) の 2 つ.

```yaml
format_version: 6
role: predict
output: {form: class, classes: [tench, goldfish, ...]}   # 重みの説明の種類の名前の表 (ImageNet は 1000)
target: null
procedure: null
model:
  spec: model_spec.yaml        # tv.Model の 1 つのノード (model) の定義
  weights: final_model.pt      # 空の state_dict (重みは写さない)
  kind: logits
  pretrained:                  # 出どころから読むノードの記録 (上の「学習済みの重みの置き方」と同じ形)
    - {node: model, unit: tv.Model, model: resnet18, revision: IMAGENET1K_V1, pinned_revision: IMAGENET1K_V1, place: source, sha256: 8ec1…, keys: 122, ...}
inputs: {}
dataset: null
conclusion: {}
decision: null
check: null
uncertainty: null
ready:
  library: torchvision         # 出どころ (READY_LIBRARIES の名前)
  model: resnet18
  weights: IMAGENET1K_V1       # 重みの名前 (版. DEFAULT は作るときの名前に直す)
  input: image                 # 入力の項目の種類
  preprocess:                  # 前処理 (重みの説明から)
    resize: 256                # 短い辺をそろえる大きさ
    crop: 224                  # 真ん中を切り出す正方形の辺
    interpolation: bilinear    # bilinear か bicubic
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  url: https://download.pytorch.org/models/resnet18-f37072fd.pth
  file: resnet18-f37072fd.pth  # 出どころの重みのファイルの名前
  license: {license: null, status: unknown, redistributable: false, gated: false, alert: true, message: …, source: …, url: null}
```

埋め込みの一式は, 出力と `model.kind` だけが違う (定義, 重み, `ready` は種類の予測と同じ).

```yaml
output: {form: vector, shape: [512]}   # 1 件の形 ([埋め込みの幅]. 分類の最後の層の入力の幅)
model: {spec: model_spec.yaml, weights: final_model.pt, kind: value, pretrained: [...]}
conclusion: {}
```

- **作り方**: `make_ready_bundle(path, "resnet18", "IMAGENET1K_V1", library="torchvision", sha256=None, name=None, license=None, conclusion=None, overwrite=False, form="class")`. `form` は `READY_FORMS` のどれか (`vector` で埋め込みの一式. 結論の決め方は渡せない). 出どころの読み方 (`READY_LIBRARIES` の関数) が重みの説明から部品 (`unit`, `params`), 種類の名前の表, 前処理, 重みの URL とファイルの名前を読み, 1 つのノードの定義を作る. torchvision は `torchvision.models.get_model_weights(model)[weights]` の `meta["categories"]` と `transforms()` の大きさ, 切り出し, 補い方, 正規化を読む (画像の分類の重みでないか, 補い方が `bilinear` と `bicubic` でなければ作る前に誤り). torchvision は ML-ModelControl の追加の依存 (`ml-modelcontrol[torchvision]`) で入れる
- **重みの版と hash**: 版は重みの名前 (torchvision は名前ごとに重みのファイルが決まり, 取りに行くときにファイルの名前の hash の頭で確かめる). 作るときに定義を組み立てて出どころの重みを読み, ノードの重みの hash (`sha256`. 「学習済みの重みの置き方」と同じ求め方) を記録する. `sha256` を渡すと (問題の設定や前に作った一式で決めておいた値), 読んだ重みと違えば作らない
- **重みを写さない**: 既製のモデルの一式は, ライセンスに依らず重みを一式に写さず (`place: source`. `final_model.pt` は空), 使うときに出どころから読む. 重みは小さくなく, 版と hash で固定できるため. 読むときに hash が記録と違えば, 警告でなく誤りにする (学習した重みが無く, 違う重みで推論しても気付けないため)
- **重みの置き場**: 出どころの重みは, 初回に組み立てたときに部品が取りに行き, PyTorch の置き場 (`torch.hub.get_dir()` の `checkpoints`) に置いて次から読む. ネットワークを使えない場所では, 利用者が `ready.file` の名前のファイルをその場所に置く. 置くファイルは信頼できるものだけにする (sha256 の照合の前に torchvision がファイルを読み込むため). `ready.file` は区切りと `..` を含まないファイルの名前だけで, 置き場は名前の部分だけから作る. `ready.url` は文字. `ready.library` は `READY_LIBRARIES` にあり, `ready.model` と `ready.weights` は定義の `model` のノードの `params` の `name` と `weights` と同じ (違えば読むときに誤り). CI とテストは作り物の出どころ (`READY_LIBRARIES` に足した関数と登録した部品) を使い, ネットワークを使わない
- **ライセンス**: 重みのライセンスの情報は `license` で `pretrained_info` の形の辞書を渡せる (画面が既に読んだ情報. テスト). 省くと ML-ModelControl の `pretrained_info` で読む (torchvision は機械で読める情報が無いので `unknown`). `ready.license` に `license`, `status`, `redistributable`, `gated`, `alert`, `message`, `source`, `url` を写す. 読めなかった情報 (`pretrained_info` が誤りになった, 無い, 結果が無い) と, 渡した情報で `alert` や `message` が無いものも, `alert` を必ず真偽値にする. `status` が `commercial`, `noncommercial`, `unknown` のどれでもなければ `unknown`, `alert` が真偽値でなければ `commercial` 以外で真, 警告で `message` が空なら「ライセンス: <ライセンスか不明> (<分類>). 配布元のページで条件を確かめる」にする (分類の言い方は ML-ModelControl の `pretrained_info` と同じ). 読めなかった訳は `source` に残る. 画面は `alert` が真なら `message` と `source` を並べて警告する. `notes` には, 出どころから読むことと hash の頭, 商用 (`commercial`) でないライセンスなら確かめる注意, 申請制 (`gated`) なら使う人ごとに申請する注意を残す
- **前処理** (`ml_inference.ready.prepare_ready`): 画像を ML-EDA の画像の読み込みで読み, RGB にし (灰色は 3 つに重ね, 透明度は除く), 短い辺を `resize` にそろえ (長い辺は比で切り捨て. Pillow で `interpolation` の補い方), 真ん中の `crop` の正方形を切り出し (上と左は `round((大きさ - crop) / 2)`), 255 で割って (値 - `mean`) / `std` にした (3, `crop`, `crop`) の float32. torchvision の画像の分類の前処理に Pillow の画像を渡したときと同じ値になる
- **埋め込み** (`ml_inference.ready.ReadyEmbedding`): 定義は種類の予測と同じ 1 つのノードのままにし, 組み立てて重みを読んだモデルの中で登録の順が最後の全結合の層 (`torch.nn.Linear`. torchvision の画像の分類のモデルの分類の層. ResNet の `fc`, ViT の `heads.head`, EfficientNet の `classifier` の最後など) の入力を, 1 回の推論ごとに受け取って埋め込みにする (モデルの出力は使わない). 定義の部品はライブラリのモデル 1 つで, 中の層をノードとして指せないため. 全結合の層が無いモデル (SqueezeNet など), 1 回の推論で最後の層を 1 度だけ通らない, 入力が (件数, 幅) でない, は誤り. 作るときに, 決まった種の乱数の画像 2 枚で, 埋め込みを最後の層に通した値がモデルの出力と合う (相対と絶対の差 1e-4 まで. 絶対の差は出力の大きさに比べる) ことを確かめ, 合わなければ (最後の層の後にも処理がある) 作らない. 幅は `output.shape` に書き, 読んで組み立てるときに最後の層の入力の幅と違えば誤り
- **確かめ**: 作るときに, 前処理の後の大きさの 0 の画像で推論し, 出力の幅が種類の名前の数と合うか (埋め込みは (件数, 幅) の形と上の最後の層) を確かめる (合わなければ作らない). 比べる正解が無いので `check`, `uncertainty`, `target` は null
- **推論**: `predict(bundle, 画像のファイルかディレクトリ)` (`infer` も同じ). 表は読み込みの表 (`path` など) と, 分類の列 (`prediction` は種類の名前, `confidence`, `p:<種類>`. 埋め込みは `output` に 1 件の (幅,) の float32. 書き出しは `artifacts/`). 読めない画像は理由を付けて飛ばす. 読み込みの設定 (`options`), 表, ML-EDA の `Data` は誤り. 結論の決め方 (確信の下限など) は種類の予測だけで使える (埋め込みの結論は件数だけ). 意思決定, たどり直し, 外れ方の測り直し, 列の効き, 何段も先の予測は, データセットと外れ方の記録を持たないので誤り
- **説明**: `describe_bundle` の `kind` は `ready`, `task` は `classification` (埋め込みは空), `classes` は種類の名前の表 (埋め込みは空). `ready` は `ready` の欄に, 重みのファイルを置く場所 (`weights_file`) とそこにあるか (`weights_cached`. 無ければ初回に取りに行く) を足したもの
- **画面と骨組みが使う値**: 画面 (ML-Provide の登録の画面) は, 作る前に `READY_LIBRARIES[library](model, weights)` で種類の名前の表と前処理と重みのファイルの名前を読める (重みは取りに行かない). 作るときは `make_ready_bundle` に出どころ, モデルの名前, 重みの名前 (と読んだライセンスの情報) を渡し, 作った後は説明の `classes`, `ready` (`license` の `alert` と `message`, `weights_cached`), `pretrained` (`sha256`, `place`), `notes` を出す. 問題の設定の作り方「既製のモデルをそのまま」の骨組み (ML-Quest) は, 学習の代わりに `library`, `model`, `weights` と, 決めておくなら `sha256` を一式づくりに渡す値として残す. 種類の名前の表は重みの説明から読むので渡さない

#### 並べるの一式 (版 8. `rank`)

利用者と品物を別の塔で埋め込みにし, 内積を点数にする 2 塔のモデル (ML-Quest の並べるの骨組みが学ぶモデル. ML-ModelControl の同梱の例 `two_tower_retrieval`) の一式は, 利用者と品物の番号の表と, 全ての品物の埋め込み (索引) を一式に持ち, 利用者ごとに上位 K 件の品物と点数を返す. データセットを持たず, 版 8 で書く.

```
<一式>/
├── inference.yaml
├── model_spec.yaml       # モデルの定義 (写したもの)
├── final_model.pt        # 重み
├── rank_index.json       # 利用者と品物の番号の表 (実験のものを写したもの)
└── item_embeddings.npy   # 品物の埋め込みの索引 ((品物の数, 幅) の float32. 番号の順)
```

```yaml
format_version: 8
role: rank
output: {form: ranking}
target: null
procedure: null
model: {spec: model_spec.yaml, weights: final_model.pt, output: null, kind: value, target_scale: null, pretrained: []}
inputs: {}
dataset: null
conclusion: {}
decision: null
check: null
uncertainty: null
rank:
  index: rank_index.json         # 番号の表 (一式の中のファイルの名前)
  embeddings: item_embeddings.npy
  users: 120                     # 利用者の数 (番号の表の users の長さ)
  items: 80                      # 品物の数
  width: 16                      # 埋め込みの幅
  scale: 20.0                    # 点数 = scale × 利用者と品物の埋め込みの内積
  k: 10                          # 並べる件数の既定 (問題の設定の rank.k)
  columns: {user: member, item: item_id, reaction: qty}   # 反応の表の列 (問題の設定の rank の列)
```

- **ワンホットの入力の約束** (ML-Quest の並べるの骨組みと同じ): モデルの入力は利用者の番号のワンホット `user` ((件数, 利用者の数) の float32) と品物の番号のワンホット `item` ((件数, 品物の数)) の 2 つ. 出力は名前の辞書で, 全ての組の点数の表 `score` ((利用者の件数, 品物の件数)) と, 利用者の埋め込み `user_embedding` ((利用者の件数, 幅)) と品物の埋め込み `item_embedding` ((品物の件数, 幅)). 番号は番号の表の順. 入力の名前が違う, 3 つの出力のどれかが無い, 入力の形の最後の大きさが番号の表の数と違う (書いてある場合) は, 作る前と読むときに誤り
- **番号の表** (`rank_index.json`): `users` と `items` の, 番号の順の元の値 (文字か数) の空でない並び. 利用者と品物は文字にして引くので (数の利用者 103 は `"103"` でも引ける), 文字にすると重なる値のある表は誤り
- **作り方**: `make_rank_bundle(path, weights, spec=None, index=None, k=10, columns=None, name=None, copy_weights=True, overwrite=False, pretrained=None, batch_size=256)`. `weights` は ML-Found の実験のディレクトリ (`final_model.pt`, `model_spec.yaml`, `rank_index.json` を使い, `run.json` の状態を記録する) か重みのファイル (定義と番号の表も渡す). 実験に定義が無い (仮のモデルで学習した) 場合は作らずに誤り. `k` は 1 以上の整数で, 受け渡しの `rank.k` を渡す. `columns` は前に反応した品物の表と正解の反応の表の列の名前 (`user`, `item` と任意の `reaction`. 受け渡しの `rank` の `user`, `item`, `reaction` を渡す. 省くと `user` と `item`). 学習済みの重みの置き方と注意は `make_bundle` と同じ
- **索引と確かめ**: 作るときに, 利用者を 1 人 (番号 0) だけ渡して品物のまとまり (`batch_size` 件) ごとに `item_embedding` を集め, 全ての品物の埋め込みを索引にする. 続けて, 見本 (利用者は最初の 8 人と全体から等間隔の 8 人, 品物は最初の 256 件と全体から等間隔の 256 件. 重なりは 1 つにし, 数より少なければ全て) の全ての組の `score` と埋め込みの内積を比べ, 最小二乗で求めた倍率が正で, 全ての組が倍率 × 内積と合う (相対と絶対の差 1e-4 まで. 絶対の差は点数の大きさに比べる) ときだけ倍率を `scale` に残す. 点数が内積の正の定数倍でない (内積の後に sigmoid を通すなど), 品物の埋め込みが一緒に渡した利用者で変わる, `score` が (利用者の件数, 品物の件数) でない, 埋め込みに数でない値がある, はどれも作らずに誤り (索引の内積で並べた順と点数が, モデルの点数と違ってしまうため). 正解を持たないので `check`, `uncertainty`, `target` は null
- **推論**: `predict(bundle, source, options=None, batch_size=256, device="cpu", k=None, seen=None)` (`infer` も同じ `k` と `seen` を持つ). `source` は利用者の値の並び (list, tuple, numpy の配列, pandas の Series) か, 利用者の列 (`rank.columns.user`) を持つ表. `k` は省くと `rank.k`. 利用者のまとまり (`batch_size` 人) ごとに, 品物を 1 つ (番号 0) だけ渡して `user_embedding` を求め, 索引との内積に `scale` を掛けた点数で, 点数の高い順に上位 K 件を選ぶ (同じ点数なら番号の小さい品物が先. 品物が K 件より少なければ全て). 全ての品物は並べ替えず, 利用者ごとに K 番目の点数を `numpy.partition` で求め, その点数以上の品物 (K 番目と同じ点数の品物は全て) だけを並べ替える. 利用者と全ての品物の点数の表は, まとまりの分だけを持つ. 読み込みの設定 (`options`) は誤り. 他の一式に `k` か `seen` を渡すのも誤り
- **前に反応した品物を除く**: 除くかは推論の引数で選び, 除くときは前に反応した品物の表 (`seen`. 表か, ML-EDA の `load_data` で読むファイルの場所) を渡す (省くと除かない). 表は `rank.columns` の列を持ち (無ければ誤り), `reaction` の列があれば, 数にして 0 より大きい行だけを反応とする (ML-Quest の骨組みの反応の決まりと同じ). 番号の表に無い利用者と品物の行, 欠けた値の行は使わない. 除くと品物が残らない利用者は並べない
- **表**: 利用者と順位ごとに 1 行で, `user` (渡した値のまま), `status`, `reason`, `rank` (1 からの順位. `Int64`), `item` (番号の表の元の値), `score` (点数). 渡した順の利用者ごとに, 順位の順に並べる. 並べられない利用者は 1 行で, `status` は `skipped`, `rank`, `item`, `score` は欠損, `reason` は「番号の表に無い利用者です (学習用の反応に無い利用者は並べられません)」(学習用の反応に無い利用者はモデルの番号を持たない), 「利用者が空です」, 「前に反応した品物を除くと, 並べる品物が残りません」のどれか. 同じ利用者を何度渡してもよい (同じ並びを繰り返す). 表の `attrs` に `k` と `excluded` (除いたか) を残す
- **結論と書き出し**: 結論は利用者の人数だけ (`summary` は `rows` (渡した利用者の数), `predicted` (並べた利用者の数), `skipped`, `k`, `excluded`. 文は「利用者 n 人のうち m 人に, 上位 K 件の品物を並べた」(並べられない利用者がいれば, 後ろに「 (s 人は並べられず)」を付ける) と, 除いたときは「前に反応した品物は除いて並べた」. 表の `decision` は並べた行が空, 他は「予測なし」). 結論の決め方, 意思決定, 何段も先の予測, 列の効き, たどり直し, 外れ方の測り直しは使えない. 結果の `source` は「利用者 n 人」(ファイルの場所は渡せない). 結果は配列を持たないので, 書き出しは結果の形の版 1 か, 列の型を残す版 3 (並べられない利用者の行があり順位の列に欠損がある場合など. 下の「書き出し」). `read_result` は値が全て文字の利用者と品物の列を文字, 順位を欠損を持てる整数で読む. 文字と数の混ざる利用者と品物の列は混ざる列 (`mixed`. 結果の形の版 4) として値ごとの型で読む (呼び手が番号の表で引くときは, `override_types` で文字にして引く)
- **Recall@K と NDCG** (`rank_metrics(bundle, prediction, truth, seen=None, k=None)`): 並べた表と, 並べた後の正解の反応の表 (`truth`. 列と反応の強さの決まりは `seen` と同じ) から数える. 正解は番号の表にある品物で, `seen` を渡すとその品物を正解から除く. 数えるのは, 表で並べた利用者と, 前に反応した品物を除くと残りが無かった利用者 (空の上位として数える. 同じ利用者が何度あっても 1 人) のうち, 正解のある利用者. Recall@K は上位 K 件に入った正解の数を正解の全数で割った値, NDCG は上位 K 件の正解の位置 r (1 から) ごとの 1 / log2(r + 1) の和を, 正解を上から min(K, 正解の数) 件並べたときの和で割った値で, どちらも利用者で平均する (ML-Quest の並べるの骨組みの評価と同じ数え方. 前の分割の反応を `seen`, 後の分割の反応を `truth` にし, 学習用にいる正解のある全ての利用者を並べれば, 骨組みの値と同じになる). `k` は渡した値, 省くと表の `attrs` の `k`, 無ければ (`read_result` で読んだ表など) 並べた行の最大の順位, 並べた行も無ければ `rank.k`. 返すのは `users` (数えた利用者の数), `k`, `recall_at_k`, `ndcg` (数える利用者がいなければ, 骨組みと同じく 0.0)
- **説明**: `describe_bundle` の `kind` は `rank`, `task` は空で, `rank` に `rank` の欄の写しが入る (他の一式は `None`). `rank.k` は推論の件数の既定, `rank.columns` は前に反応した品物の表と正解の反応の表の列の名前

#### 群分けの一式 (版 9. `group`)

表の行を埋め込みにする符号化器と, 埋め込みの空間の群の中心 (ML-Quest のまとめるの骨組みが対照学習で学び, k-means で分けたもの. ML-ModelControl の同梱の例 `contrastive_encoder`) の一式は, 列の符号化と群の中心と, 実験の群の安定性と群の要約を一式に持ち, 新しい行を最も近い群の中心に割り当てて, 群の要約と群の代表の行を返す. データセットを持たず, 版 9 で書く.

```
<一式>/
├── inference.yaml
├── model_spec.yaml       # モデルの定義 (写したもの)
├── final_model.pt        # 重み
└── group_centers.json    # 列の符号化と群の中心 (実験のものを写したもの)
```

```yaml
format_version: 9
role: group
output: {form: cluster}
target: null
procedure: null
model: {spec: model_spec.yaml, weights: final_model.pt, output: null, kind: value, target_scale: null, pretrained: []}
inputs: {}
dataset: null
conclusion: {}
decision: null
check: null
uncertainty: null
group:
  centers: group_centers.json    # 列の符号化と群の中心 (一式の中のファイルの名前)
  k: 3                           # 群の数
  columns: [spend, visits, region]   # 群を作る列 (群の要約の列の順)
  features: 5                    # 符号化した行の幅
  width: 16                      # 埋め込みの幅
  id: customer                   # 実験の群の割り当ての表の行を見分ける列 (行の番号なら null)
  rows: 240                      # 学習で群に分けた行の数
  stability:                     # 群の安定性 (実験の group_stability の行の写しと, 比べる相手の名前とエポック)
    k: 3
    method: seed                 # seed (種を変えて作り直す) か sample (標本を変えて作り直す)
    repeats: 5
    ari: 0.97
    ari_baseline: 0.99
    ari_diff: -0.02
    ari_min: 0.95
    ari_min_baseline: 0.98
    ari_min_diff: -0.03
    silhouette: 0.62
    silhouette_baseline: 0.55
    silhouette_diff: 0.07
    baseline: 列の特徴をそのまま k-means で分けた群
    epoch: 12
  summary: [...]                 # 学習のときの群の要約 (実験の group_summary の行の写し. 最後の行は「全体」)
  check: {rows: 240, agreement: 1.0}   # 学習の表を一式で割り当て直した一致 (表を渡した場合だけ. 無ければ null)
```

- **実験の成果物** (ML-Quest のまとめるの骨組みが実験のディレクトリに書くもの. 名前と形は骨組みと同じ): `group_centers.json` は `k` (群の数), `encoding` (列の符号化. `numeric` は数の列ごとの `mean` と `std`, `categories` は種類の列ごとの値の文字の並び), `centers` ((群の数, 埋め込みの幅) の群の中心) の辞書. `group_assignments.csv` は行ごとの群の割り当ての表で, 列は行を見分ける列 (骨組みの `group.id`. 無ければ `row` で行の番号), `group` (群の番号), `split` (`train` か `valid`). `evaluations/group_stability.json` と `evaluations/group_summary.json` は ML-Found の `evaluations` の形 (`rows` が表の行). 群の安定性の行は `k`, `method`, `repeats` と, `ari` (作り直した群との調整ランド指数の平均), `ari_min` (最も低い値), `silhouette` (シルエット係数) と, それぞれの `_baseline` (比べる相手. 列の特徴をそのまま k-means で分けた群) と `_diff` (モデル − 比べる相手). 群の要約の行は群の番号の順の群ごとの行と, 最後の `group` が `全体` の行 (下の「群の要約」と同じ形)
- **入力と出力の約束** (ML-Quest のまとめるの骨組みと同じ): モデルの入力は符号化した行の特徴 `x` ((件数, 符号化した行の幅) の float32) の 1 つ. 出力は名前の辞書で, 埋め込み `embedding` ((件数, 埋め込みの幅)) を使う (学習で使う類似度の表 `similarity` は使わない). 入力が `x` の 1 つでない, `embedding` の出力が無い, `x` の形の最後の大きさが符号化した行の幅と違う (書いてある場合) は, 作る前と読むときに誤り
- **列の符号化** (骨組みの `encode_rows` と同じ式): 数の列は (値 − `mean`) / `std` で, 欠けた値は 0. 種類の列は値を文字にし (欠けた値は `(欠損)`), `categories` の値ごとに等しければ 1, 違えば 0 の列 (学習用に無い値は全て 0). 列の順は数の列, 種類の列の順 (`encoding` の順). 数の列で数にできない値 (欠けでない文字など) の行は割り当てない. この符号化は ML-EDA の公開の名前に無いので, 骨組みと同じ式をこのライブラリが持つ (骨組みの式を変えたら, こちらも合わせる)
- **作り方**: `make_group_bundle(path, run, name=None, data=None, options=None, copy_weights=True, overwrite=False, pretrained=None, batch_size=256)`. `run` は ML-Found の実験のディレクトリで, `final_model.pt`, `model_spec.yaml`, 上の 4 つの成果物を使い, `run.json` の状態を記録する. 実験に定義が無い (仮のモデルで学習した) 場合と, 成果物が無いか形が違う場合 (群の数が 2 以上の整数でない, 中心が (群の数, 幅) の数でない, 符号化の列が空か重なる, 割り当ての表の列が違うか群の番号が範囲の外, 安定性の行に指標が無い, 群の数が成果物の間で違う, 群の要約の列と符号化の列が違う) は作らずに誤り. 学習済みの重みの置き方と注意は `make_bundle` と同じ
- **確かめ**: 作るときに, 0 の行を 2 件渡して埋め込みの形が (2, 中心の幅) で数の値かを確かめる (違えば作らない). `data` に学習に使った表 (骨組みの `data.dataset`. ファイルか表か ML-EDA の `Data`. ファイルは `load_data(data, **options)` で読む) を渡すと, 全ての行を一式で割り当て直し, 実験の割り当ての表と同じ位置の行の群が同じ割合を `group.check` (`rows`, `agreement`) に残す. 99 % より低いと `notes` で知らせる (作るのは止めない). 行の数が違う場合と, 行を見分ける列の値の並びが違う場合は比べず, 理由を `notes` に残す. 渡さなければ `group.check` は null. 正解を持たないので `check`, `uncertainty`, `target` は null
- **推論**: `predict(bundle, source, options=None, batch_size=256, device="cpu")` (`infer` も同じ). `source` は表のファイル (`load_data(source, **options)` で読む. `options` に `loader` も書ける), 表, 表の ML-EDA の `Data` のどれか (表でない `Data` と, 表と `Data` に `options` を渡すのは誤り). 群を作る列が無ければ誤り (無い列を挙げる). 表は元の列と, `status`, `reason`, `group` (最も近い群の中心の番号. `Int64`), `distance` (その中心までの埋め込みの空間の距離). 元の列が予測の列と同じ名前なら「(入力)」を付けて残す. 割り当てない行は `status` が `skipped` で, `reason` は「列 … の値 … を数にできません」か「埋め込みに数でない値があります」. 埋め込みは `batch_size` 行ずつ求める. 二乗距離は骨組みの `assign` と同じく内積で求め, 同じ距離なら番号の小さい群
- **群の要約** (`group_summary(bundle, prediction)`): 割り当てた行の群の要約の行の並び. 行の形は ML-EDA の `describe_groups` の `groups` の行と同じで, ML-EDA の `describe_groups` (群を作る列と `labels` に割り当てた群) で作る (群の要約の行の形の持ち主は ML-EDA なので, 写して作り直さない). 群の番号の順に `group`, `rows`, `share` (割り当てた行に対する割合), `stability`, 群を作る列ごとの要約 (数の列は平均, 種類の列は最も多い値と割合の文) の行を並べ, 最後に `group` が `全体` の行を置く. `stability` は, 新しいデータでは群を作り直さないので, 一式の学習のときの群の要約の群ごとの値 (作り直した群との Jaccard 係数の平均) を写す (`全体` は None). 割り当てた行の無い群も, `rows` が 0, `share` が 0.0, 列の要約が None の行で並べる. 割り当てた行が無ければ空の並び. ML-EDA に渡すとき, 列の名前は一時の名前に置き換えて戻す (ML-EDA の群の要約の鍵と同じ名前の列でも作れるように), 数の列は数に, 種類の列は文字に (欠けは欠けのまま) する. 新しいデータの種類の値が多く, 符号化した行の幅が ML-EDA の上限を超えると ML-EDA の誤りのまま止まる. `read_result` で読んだ表も渡せる
- **群の代表の行** (`group_representatives(bundle, prediction, count=3)`): 群ごとに, 群の中心に近い順の割り当てた行 (代表の行) を `count` 件まで (群の行が少なければその数だけ) 並べる. 群の要約の行の形は ML-EDA が持つので, 群の要約の行には足さず, 別の並びで返す. 並びは群の番号の順の群ごとの辞書 (`group` (群の番号), `rows` (代表の行の並び)) で, 群の要約と同じく行の無い群も (`rows` が空) 並べ, 割り当てた行が無ければ空の並び (`全体` の行は無い). 代表の行は `row` (予測の表の行の位置. 0 から), `distance` (予測の表の `distance`. 群の中心までの埋め込みの空間の距離), `values` (群を作る列の名前から元の値. 予測の列と同じ名前の列は「(入力)」付きの列の値. 欠けは None, 真偽値と数はそのまま, 他は文字. `result.yaml` に書けるように) の辞書. 同じ距離なら行の位置の小さい行を先にする. 割り当てない行 (`status` が `skipped`) は除く. `count` が 1 以上の整数でない場合, 予測の表に `status`, `group`, `distance` か群を作る列が無い場合は誤り. `read_result` で読んだ表も渡せる
- **結論と書き出し**: 結論は件数と群の要約 (`summary` は `rows`, `predicted` (割り当てた行の数), `skipped`, `k`, `groups` (上の群の要約), `representatives` (上の群の代表の行. 件数は既定の 3), `stability` (一式の群の安定性の写し). 文は「n 行のうち m 行を k 個の群に割り当てた」(割り当てない行があれば後ろに「 (s 行は割り当てられず)」), 割り当てた行があれば「群 0 a 行 (x %), 群 1 …」, 学習の群の安定性「学習の群の安定性 (種を変えて作り直す, 5 回): 調整ランド指数 0.970 (最も低い 0.950), シルエット係数 0.620. 列の特徴をそのまま k-means で分けた群は 0.990 (0.980), 0.550」. 表の `decision` は割り当てた行が空, 他は「予測なし」). 結論の決め方, 意思決定, 何段も先の予測, 列の効き, たどり直し, 外れ方の測り直しは使えない. 結果は配列を持たないので, 書き出しは結果の形の版 1 か, 列の型を残す版 3 (割り当てない行があり群の番号の列に欠損がある場合, 数に見える種類の値がある場合など) か 4 (数の列に数にできない値が混ざり割り当てない行がある場合など. 下の「書き出し」. `representatives` はまとめの中身なので版は変わらない). `read_result` で読み直した表の群を作る列と群の番号は書き出す前と同じ値になり, 読み直した表から作る群の要約と代表の行は書き出す前と同じになる
- **説明**: `describe_bundle` の `kind` は `group`, `task` は空で, `group` に `group` の欄の写しが入る (他の一式は `None`)

#### 自作の部品 (起動の設定のモジュール. `ml_inference.units`)

研究の自作の部品 (ML-ModelControl の `register_unit` で登録する `nn.Module`) を使うモデルの一式は, 起動の設定で明示したモジュールから部品を登録してから読む. 一式やモデルの定義に書かれた名前からは何も読み込まない.

- 登録: `register_units(module_names)` は ML-ModelControl の `register_units_from` をそのまま呼び, モジュールの名前ごとの登録した部品の名前 (`{"my_lab.units": ["my.Block"]}`) を返す. アプリ (ML-Provide など) は起動のときに, 起動の設定に書いたモジュールの名前だけを渡す. モジュールの書き方 (`UNITS` か `register_units(register)`) は ML-ModelControl の README の「自作の部品」. 読めないモジュールは ML-ModelControl の誤り (`ValueError`) で止まる. 名前の文字 1 つを渡した取り違えは `TypeError`. 入っている ML-ModelControl に `register_units_from` が無ければ, 新しくするよう言う `ImportError`
- 確かめ: `load_bundle` と `make_bundle` は, モデルの定義の本体と部分モデル (`subgraphs`) の全てのノードの部品の名前 (`unit`), 外部の重みを読む部品の名前 (`weights.unit`), 引数に書いた部品 (ML-ModelControl の版 3. ノードの `params` の値の, `unit` を文字で持つ辞書. 中の部品の `params` も同じくたどる) の名前が登録簿にあるかを, 組み立てる前に確かめる (`sub.` で始まる名前は定義の部分モデルなので登録簿を見ない). 引数に書いた部品の場所は `enc.params.encoder_layer` (中の部品は `enc.params.encoder_layer.params.norm`) のように挙げる. たどる入れ子は ML-ModelControl と同じく 10 段まで (ノードの引数に書いた部品が 1 段目) で, それより深い部品と, 外側の部品を中に含む循環はたどらない (ML-ModelControl の確かめが `invalid_value` にする). 無ければ `UnregisteredUnitError` (`ValueError` の子. `units` に登録されていない部品の名前の並び) で止める. 文は部品の名前とノードを挙げ, 「自作の部品なら, 部品を宣言したモジュールの名前を起動の設定に足して起動し直すか, 一式を読む前に `register_units(["モジュールの名前"])` を呼ぶ」と案内する. アプリは, この例外のときに自分の起動の設定の書き方 (引数や環境変数の名前) を添える
- 読み込まないこと: 確かめは登録簿を引くだけで, `importlib` などで定義に書かれた名前を読み込まない. 一式の中の Python のファイルも読まない

#### 学習済みの重みの置き方 (版 3. `model.pretrained`)

学習済みの重みを読むノード (ML-ModelControl の `pretrained_info` が `None` でないノード) の重みを, 一式に写すか, 出どころから読むかを決めて記録する. 再配布できる重みは写して固定し, できないものと不明なものは出どころの名前と版の識別子から読み, 読んだ重みが同じかを確かめる.

- 見るノード: 定義の一番上のノードと, 本体から `sub.<名前>` で辿れる部分モデル (`subgraphs`) の中のノード. 部分モデルの中のノードの名前は, 使う外のノードの名前と `.` でつなぐ (`block.inner.fc`. ML-ModelControl の外部の重みの報告と同じ). 同じ部分モデルを複数のノードが使えば, ノードごとに別のノードとして見る. 部分モデルを部品にしたノード (部品の名前が `sub.` で始まる) そのものと, どのノードも使わない部分モデルと, 外側の部分モデルを中に含む循環は見ない. 部分モデルの中のノードの重みのキーは, 外のノードの `nodes` の下 (`nodes.block.nodes.inner.nodes.fc.`) にある
- 重みのライセンスの情報は `make_bundle(pretrained=...)` で, ノードの名前 (部分モデルの中のノードは上の名前) ごとの `pretrained_info` の形の辞書で渡せる (画面が既に読んだ情報を使う. テストもこれでネットワークを使わない). 渡した情報に無いノードは学習済みの重みを読まないものとする. 定義に無い名前は誤り. 同じ部分モデルを使う複数のノードで, 部分モデルの中の同じノードに違う情報 (`redistributable`, `pinned_revision` など) を渡すのも誤り (版の固定は共有の部分モデルの定義を書き換え, 使う全てのノードに当たるため). 省くと, 見るノードごとに `ml_modelcontrol.pretrained_info` を呼ぶ (ネットワークを使う. 入っている ML-ModelControl にこの関数が無ければ読まずに注意を残す. 読めなかったノードは `unknown`). ノードの `weights` の `hub` や `unit` から読んだ外部の重みのライセンスは一式に記録しない (その重みは学習した重みとして一式の重みに含まれる)
- 確かめずに写す重みの注意: 重みを一式に写す場合は, ノードの `weights` の `hub` か `unit` から読んだ外部の重みと, 情報を渡したときに渡さなかった部分モデルの中の学習済みの部品 (`pretrained_info` が `None` でないもの. ネットワークを使わずに重みを読む部品かだけを見る. 見られなかったものも含める) があれば, ノードを挙げて「ライセンスを確かめずに一式の重みに含めたので, 確かめてから一式を渡す」注意を種類ごとに 1 文 `notes` に残す (外部の重みを重ねた写せない学習済みのノードは, 置き方の注意がライセンスを確かめるよう書くので除く). 重みを写さない一式 (`copy_weights=False`) には残さない
- 写せるか: `redistributable` が真のものだけ写す. 偽のもののうち, ライセンス (`license`) と版の識別子 (`pinned_revision`) を読めたものだけを写さない. 読めなかったもの (ネットワークが無い, 配布元に書いていない) は, 出どころから同じ重みを読めるか分からないので一式の重みに残し, 理由を `notes` に書く. 写さなかったノードも, 出どころの名前と版を `notes` に必ず書く
- 写さない重みのノードは, 置き方に依らず, その部品に `revision` の引数があれば一式の定義のその値を読んだ版 (`pinned_revision`) に替えて固定する (比べる重みも固定した版で組み立てる. 部分モデルの中のノードは部分モデルの定義を替えるので, その部分モデルを使う全てのノードに当たる). そのうえで, 学習で変わっていない (外部の重みを読まずに定義から組み立てた直後の重みと, 渡した重みのノードの部分が全て等しい) 場合だけ, 一式の重みのファイルからそのノードの重みを除き, 使うときに出どころ (部品の名前と引数, 版) から読む (`place: source`)
- 定義で外部の重み (`weights`) を重ねた学習済みのノード (部分モデルの中のノードは, 外のノードが重ねたものも) は, 一式が外部の重みを読まずに組み立てるので出どころから読み直すと重みが違う. そこで学習で変わったかに依らず一式の重みに残し (`place: bundle`), 「外部の重みを重ねたので出どころと違う」ことと「元の重みのライセンスを確かめてから一式を渡す」注意を `notes` に残す
- 学習で変わったノード (微調整した) は出どころから読み直せないので, 写さないライセンスでも一式の重みに残し (`place: bundle`), 「元の重みのライセンスを確かめてから一式を渡す」注意を `notes` に残す
- 重みを写さない一式 (`copy_weights=False`) は, ファイルを作らないので除かずに記録だけ残す
- 記録 (ノードごと. 一番上のノード, 部分モデルの中のノードの順): `node` (部分モデルの中のノードは上の名前), `unit`, `model`, `hub_id`, `revision`, `pinned_revision`, `license`, `status`, `redistributable`, `source` (どこから読んだか), `place` (`bundle` か `source`), `sha256` (ノードの重みの hash. キーの名前, 型, 形, 中身から), `keys` (キーの数)
- 読むとき (`Bundle.model`): 出どころから読むノードがあれば, 組み立てた後に重みを `strict=False` で読み, 足りないキーがそのノードのものだけかを確かめ (違えば誤り), ノードの重みの hash を記録と比べる. 違えば警告 (`warnings.warn`) し, 説明の `pretrained_warnings` にも出す (出どころの重みが入れ替わった, 版を固定できない部品など. 推論は止めない). ただし hash の違うノードが定義で外部の重み (`weights`) を重ねる場合は, 一式を作ったときの重み (外部の重みを重ねたもの) にならないので, 警告でなく誤りにして一式を作り直すよう案内する (外部の重みを相対パスで持つ一式は, 定義の元の場所が分からず読み直せないため). 出どころから読むノードを持つモデルを組み立てられない場合は, そのノードと出どころと版を添えた誤りにする
- 学習済みのノードの無い一式は, 学習した `state_dict` を丸ごと `final_model.pt` に写す (版 2 までと同じ)

#### 入力のつなぎ方

- 表と時系列 (`features`): `Dataset.transform` で前処理をかけた学習に使う列を, `float32` の (件数, 列の数) にする
- 画像, 文, 点群, 音声 (`items`): 1 件ずつ `Dataset.transform_item` をかけて並べる. 文は番号の並びなので `int64`, 他は `float32`. 音声は ML-EDA の `read_audio` で読んだ辞書に, 書き出しと同じ中身の手順 (周波数, チャンネル, 長さをそろえる) をかけた (チャンネル, 標本の数) の配列. 手順でそろえない周波数, チャンネル, 長さの違う件は形が違うので, 理由を付けて飛ばす
- 変換した 1 件の形が学習のデータセットの 1 件の形 (分割の `shape` から件数を除いたもの) と違う件は, 理由を付けて飛ばす (モデルに通すと落ちるか, 誤った値になるため). 説明に形が無ければ, 最初に変換できた件の形と比べる
- `prepend`: 文の番号の並びの先頭に決まった番号を足す. 分類の印 ([CLS]) を前処理の外で足して学習したモデルのため. 学習のスクリプトで足した番号と同じものを書く
- モデルの入力が 1 つで `inputs` を省いた場合は, データの種類から `features` か `items` を選ぶ
- 入力が複数のモデルで `source` を書かなかった入力は, データの種類から選ぶ (同じ中身を複数の入力に渡せる). ML-EDA は画像, 文, 点群, 音声のデータセットに表の列を学習に使う列として残さないので, 表の列と中身を別々の入力に渡すことはできない

#### 出力の戻し方

- 分類: (件数, クラスの数) の出力を, `kind: logits` なら softmax で確率にする (`probabilities` ならそのまま). クラスの名前はデータセットの `classes` の順. 2 クラスで出力が 1 列 ((件数,) か (件数, 1)) の場合は, 2 つ目のクラスの logit (`probabilities` なら確率) とする
- 回帰と予測: (件数,) か (件数, 1) の出力を値にする. 学習のスクリプトが目的の値を前処理の外で `(y - center) / scale` にして学習した場合は, `target_scale` で戻す (`make_bundle` に `"train"` を渡すと, よくある決め方である学習用の分割の目的の値の平均と不偏の標準偏差を求めて入れる). 前処理の外の標準化の戻し方はレシピに入れず, 標準化した学習のスクリプトが `make_bundle` の `target_scale` に渡して一式に書く. その後, レシピで目的の列を変換していた場合 (`standardize`, `minmax`, `log`) は, 逆の順に戻して元の単位にする. `clip` は戻せないので, 一式を作るときの注意 (`notes`) にする
- `predict` の予測は 1 つ先だけ. 時系列の過去の値の列 (lag) は, 渡したデータの目的の列の過去の値から作る. 過去の値が足りない行は予測せず, 理由を付ける. 何段も先の予測は `forecast` (下の「何段も先の予測」)

### 書き出し (結果の形の版 4)

`write_result(result, path)` は, 予測と結論をディレクトリに書く.

```
<結果>/
├── predictions.csv     # 予測と判断の表 (UTF-8 の BOM 付き. 表計算のアプリで開けるように)
├── result.yaml         # 形の版 (format_version), 一式の名前, 入力, 時刻, 結論の決め方, 列の型, まとめ, 文
└── artifacts/          # 表に収まらない中身 (版 2. 列の型も残す結果は版 3 か 4. 配列の出力の一式だけ)
    ├── 000000.npy      # 行の位置 (6 桁の 0 埋め) ごとの配列 (numpy の .npy)
    └── ...
```

- 表に収まらない中身 (領域, 点群, 画像, ベクトル) は `artifacts/` に 1 行 1 つの `.npy` で書き, 表の `artifact` の列に結果のディレクトリからの相対パス (`artifacts/000000.npy`) を入れる. 予測しなかった行は空. 表から `output` の列は除く
- `result.yaml` の `artifacts` に `directory` (`artifacts`), `column` (`artifact`), `format` (`npy`), `form` (出力の形), `count` (書いた数) を書く. 書き出すのは配列の出力の形 (`mask`, `point_cloud`, `vector`, `image`) の結果だけで, 入力に `output` や `artifact` の列があっても, 予測の表では「(入力)」付きの名前になるので混ざらない
- 結果の形の版が表すもの: ファイルの並び (`predictions.csv`, `result.yaml`, `artifacts/`) と, `result.yaml` の一式の種類に依らない項目 (`format_version`, 一式の名前, 入力, 時刻, 結論の決め方, `artifacts`, `column_types`, `text_rows`) の形. これを足す, 消す, 変える場合は版を上げる. まとめ (`summary`) と文の中身は一式の種類ごとに違い, 読む側は一式の種類で読み分けて知らない項目を無視するので, 版に含めない (群分けの一式の群の要約と代表の行も版 1 のまとめに入る)
- 版: 表に収まらない中身が無い結果 (列の出力と, 表の出力の雑音だけの一式と, 並べるの一式と, 群分けの一式の結果) は版 1 で書く (版 1 だけを知る読む側も読めるように. ML-EDA と同じ決まり). ある場合だけ版 2. 列の型 (下) を残す結果は, 表に収まらない中身の有無に依らず版 3 (版 3 は版 2 に `column_types` を足した版. 型を知らずに読むと値が変わる表なので, 版 2 までしか知らない読む側は読み違えずに「読めません」で止める). 混ざる列 (下) を残す結果は版 4 (版 4 は版 3 の `column_types` に型 `mixed` を, `result.yaml` に `text_rows` を足した版. 混ざる列の無い結果は版 3 までのまま. 版 3 までしか知らない読む側は `mixed` を知らずに数の値を文字で読み違えるので, 版 4 の結果だけを「読めません」で止める)
- **列の型** (`column_types`): CSV は値の型を持たないので, そのまま読み直すと数に見える文字 (`"001"`, `"1e3"`) と真偽値に見える文字 (`"True"`) は数と真偽値に, 欠損のある整数の列は小数になる. 書き出す表の列のうち, 欠けを除いた値が全て文字の列 (`text`) と, 全て整数 (真偽値を除く) の列 (`integer`. 整数の型か, 値が全て整数の列. どちらも値が `Int64` に収まる列だけ) を候補にし (値が 1 つも無い列と 0 行の表の列は除く), 書いた `predictions.csv` を型を知らずに読み直して値が変わる列だけを, 列の名前から型への対応で `result.yaml` に残す (変わらない列は残さず, 残す列が無ければ版 1 か 2 のまま). `read_result` は `text` の列を文字, `integer` の列を欠損を持てる整数 (`Int64`) で読む. 型は一式の列の情報ではなく書き出す表から決める (入力の元の列は一式の列の情報に無く, 一式を消した後も読めるように. 全ての一式の種類の結果で同じ決まり). 小数の列は残さない (小数の書き方の値は型を知らずに読んでも小数に戻る). 空の文字は CSV で欠損と分けられないので欠損に戻る (文の出力の予測した行の空の文は上の「文の戻し方」の決まりで空の文字). `column_types` の無い結果 (版 1 と 2) は今までどおり読む (空の欄だけを欠損にし, `NA` などは文字のまま)
- **混ざる列** (`mixed`. 版 4): 欠けを除いた値に, 文字 (空の文字も), 整数 (真偽値を除く), 有限の小数 (Python と NumPy の値) のうち 2 つ以上の種類があり, 他の種類 (真偽値, 無限の小数, 配列など) の値が無い列 (数と文字の混ざる利用者の値, 数の列に数にできない値が混ざり割り当てない行を持つ群分けの入力の列, 整数と小数の混ざる列) は, 1 つの型では値が戻らないので, `column_types` に `mixed` を残す. 数の値は CSV の書き方 (`962.8`, `3`, `1e+20`) から戻せるので, 書き方では数と分けられない数に見える文字の値 (`"001"`, `"1.5"`, `"-0"`) の行の位置 (0 から) だけを, 列の名前から位置の並びへの対応で `text_rows` に残す (その行の無い列は挙げない. `predictions.csv` の書き方は他の結果と同じ). `read_result` は `mixed` の列を文字で読み, `text_rows` の行は文字のまま, 他の値は整数の書き方 (符号と数字だけ) なら整数, 小数の書き方 (小数点か指数を持つ数) なら小数, 他は文字に戻す. 書いた後にこの読み方で読み直し, 全ての値が書き出す前と同じ種類と値になる列だけを残す (戻らない値を持つ列, 例えば小数の書き方で値の変わる `float32` の値を持つ列は残さず, 今までどおり文字で読む). 空の文字は欠損に戻る (上の「列の型」と同じ. 空の文字と整数だけの列も整数は整数で戻る). `NA`, `nan`, `inf` の文字は数の書き方でないので文字で戻る. 混ざる列の整数は `Int64` の外の値 (`2 ** 70`) も整数で戻る. 無限の小数は `inf` の文字と書き方が同じで, 真偽値は `True` の文字と同じなので, それを含む列は混ざる列にせず, 全ての値を文字で読む (`[True, 1]` は `"True"`, `"1"`). `read_result` は, 知らない型の名前と, 並びでない `text_rows` の項目と整数でない位置を無視する
- 扱いの限り: 整数だけで `Int64` の外の値を持つ列 (`2 ** 63` など) は列の型を残さないので, 読み直した値は pandas の読み方のまま (符号なしの整数, 文字, 小数のどれか) になり, 書き出す前と型か値が違うことがある
- `read_result(path, override_types=None, fallback_types=None)` は結果を読み, 表 (`frame`) と説明を返す. 知らない新しい版は「読めません」で止める. `artifact_path(root, relative)` は表の相対パスを結果のディレクトリの中に解決する (外を指せば誤り)
- **呼び手の列の型** (`override_types`, `fallback_types`): 呼び手 (ML-Provide の画面など) の都合の列の型を, 列の名前から型 (`text` か `integer`. 混ざる列は `text_rows` と組なので渡せない) への対応で渡し, `predictions.csv` を 1 回で読む. `override_types` は `column_types` と文の出力の文と番号の型より強く当て (画面で文字として引く状態や利用者の列など), `fallback_types` は `column_types` に無い列 (`column_types` の無い版 1 と 2 の結果では全ての列) だけに, 文の出力の文と番号の型より弱く当てる (呼び手が古い版の結果に残した型の補い). 強さの順は `override_types`, `column_types`, 文の出力の文と番号 (文字), `fallback_types` (`override_types` を当てた混ざる列は値ごとの型に戻さない). 表に無い列の型は無視し, 辞書でない対応と知らない型の名前は誤り. 文の出力の見分け (下の「文の出力の見分け」), 予測した行の空の文 (`prediction` と `tokens` の `ok` の行の空の欄は空の文字. `override_types` で文字でない型を当てた列は除く), 欠損の決まり (空の欄だけを欠損), `artifact` の列の欠け (空の文字) は渡した型に依らず同じ. 型を渡さないときは `column_types` と文の出力の決まりだけで読む
- **文の出力の見分け**: `read_result` は `tokens` の列の有無だけで文の出力を見分ける (結果の形の版の項目に出力の形が無いため). `tokens` の列の無い古い文の結果は, 一式の出力の形を知る呼び手が `override_types` で `prediction` を文字で読み, 予測した行の空の文を空の文字に戻す
- 書き出す先に `artifacts/` がある場合, 前の結果のもの (`artifacts` を記録した `result.yaml` がある) だけを消してから書く (別の結果の配列が混ざらないように). 前の結果のものと確かめられない `artifacts/` は利用者のデータとして消さず, 誤りで止まる. 列の結果の書き出しは `artifacts/` に触れない
- 読んで書き直す (`write_result(read_result(p), q)`) と配列は残る. `read_result` は `form` (配列の出力の形) と `directory` (読んだ場所) を返し, `write_result` は表の `artifact` の列を持つ結果の配列を, 同じ場所ならそのまま使い, 別の場所なら写す. 読んだ版と中身の記録, 列の型, 読んだ場所は書き直すときに作り直す (読んだ表を変えて書き直しても, 型は書く表から決め直す)
- 配列の出力の表 (`prediction` の列が無い) を `conclude` に渡すと, 件数だけの結論 (`conclude_outputs(prediction)`. 表に `decision` を足し, まとめは `rows`, `predicted`, `skipped`) を使うよう言って止まる

## 3. 用語

| 用語 | 意味 |
| --- | --- |
| 推論の一式 (bundle) | 推論に要るものをまとめたディレクトリ. `inference.yaml` を持つ |
| 予測 (prediction) | 新しいデータの 1 行 (1 件) ごとの, モデルの出力を元の形に戻したもの. 分類はクラスと確率, 回帰と予測は値 |
| 結論の決め方 (rules) | 予測から結論を出す決まり. 保留, 陽性のしきい値, 値の帯, まとめる列 |
| 結論 (conclusion) | 行ごとの判断 (`decision`) と, 全体のまとめ (件数, 割合, 合計など) と, それを述べる文 |

## 4. 予測

`predict(bundle, source)` は, 渡したデータの行ごとの予測を `DataFrame` で返す.

- `source` はファイルかディレクトリのパス, `DataFrame` (表, 時系列, 文の列を持つ表), ML-EDA の `Data` のどれか. 雑音だけの一式は件数と種の辞書 `{"count": n, "seed": s}` だけ (上の「雑音だけの一式」). 並べるの一式は利用者の値の並びか利用者の列を持つ表だけ (上の「並べるの一式」). 群分けの一式は表のファイル, 表, 表の `Data` だけ (上の「群分けの一式」). パスは, 学習のときと同じ読み込みと設定 (データセットの `source.loader` と `source.options`) で読み, 読めなければ形式を判定し直す (学習は csv, 推論は parquet など). 読み込みの設定は引数で変えられる. この読み込みは `read_source(bundle, source, options)` で公開する (画面が結果の入力を推論と同じに読み直すため). 表と時系列の一式にディレクトリを渡すと誤り. 画像, 点群, 音声の一式には, 1 つのファイルもディレクトリ (クラスのフォルダは無くてよい) も渡せる. 版 2 のデータセットの一式は, 版 2 のデータセットか入力の項目と同じ種類の元のファイルの場所 (上の「版 2 のデータセットの一式」)
- 列は, 渡したデータの元の列 (画像, 点群, 音声はファイルの場所など, ML-EDA の読み込みの表の列) と, 次の列
    - `status`: `ok` (予測した), `skipped` (前処理の後に欠損がある, 中身を読めない, 予測の値が数でない (入力に無限大があるなど) など, 予測しなかった)
    - `reason`: 予測しなかった理由 (`ok` は空)
    - `prediction`: 分類はクラスの名前, 回帰と予測は値, 文の出力は文 (番号は `tokens`. 他の形は上の「受け持ちと出力の形」)
    - `confidence`: 分類の, 予測したクラスの確率
    - `p:<クラス>`: 分類の, クラスごとの確率
- 行の数と順は渡したデータと同じ. 行を落とす手順 (`drop_duplicates`, `drop_missing`, `filter_rows`) は, ML-EDA の `transform` が過去の値の列の前後関係を学習とそろえるためにかけ, 入力の並びに戻す. 重複した行は残した行と同じ予測になり, 落ちた行は「前処理の手順 … で除く行」の理由で飛ばす
- 一部の行の値 (数にできない文字, 対数を取れない負の値など) で前処理の全体が失敗した場合は, 1 行ずつかけ直し, 失敗した行を理由付きで飛ばす. ただし過去の値の列 (lag) を作るレシピは前の行が要るので, 行ごとにはかけずに誤りにする
- 文の空と欠損は予測せずに飛ばす. 渡したデータに予測の列と同じ名前の列 (`status` など. `output`, `artifact`, `tokens` は出力の形に依らず) がある場合は, 元の列を「`<名前> (入力)`」にして残す
- 予測は `torch.no_grad` の推論のモードで, 決まった件数ずつ (既定 256 件) 行う. デバイスは既定で cpu
- 前処理の誤り (学習に使う列が足りないなど) は, ML-EDA の誤りの文をそのまま返す

## 5. 結論

`conclude(prediction, task, classes, rules)` は, 予測に判断の列 `decision` を足した表と, まとめと文を返す.

| 決め方 | タスク | 意味 |
| --- | --- | --- |
| `min_confidence` | 分類 | 予測したクラスの確率がこれより低い行を「保留」にする |
| `positive`, `threshold` | 2 クラスの分類 | `positive` のクラスの確率が `threshold` 以上なら `positive`, それ以外はもう 1 つのクラス (確率の最大で選ぶより, 見逃しや誤りの重さに合わせて決めたい場合) |
| `bands` | 回帰, 予測 | 値の帯. `[{label: 少ない, upper: 100}, {label: 多い}]` のように, 上限の小さい順に名前を付ける (最後の帯は上限無し). 値が上限より小さければその帯 |
| `group_by` | 全て | 渡したデータの列の値ごとに, 件数と判断の内訳 (分類) か, 合計, 平均, 最小, 最大 (回帰, 予測) をまとめる |

- 判断 `decision` は, 分類ではクラスの名前か「保留」, 回帰と予測では帯の名前 (帯が無ければ空). 予測しなかった行は「予測なし」
- まとめは, 全体の件数, 予測した件数, 予測しなかった件数, 判断ごとの件数と割合, 回帰と予測では値の合計, 平均, 最小, 最大
- 文は, まとめを日本語で述べた文の並び (例: 「1,000 件のうち 980 件を予測 (20 件は予測できず)」「良い 612 件 (62.4 %), 悪い 301 件 (30.7 %), 保留 67 件 (6.8 %)」). 画面とファイルにそのまま出せる. `summary_sentences(summary, task)` はまとめから同じ文を作る (画面がまとめの一部から文を作り直すため)
- `positive` と `min_confidence` は一緒に使えない (しきい値で決めた陽性を確信の下限で保留にしてしまうため)
- `infer(bundle, source, rules)` は予測と結論をまとめて行う. `forecast=True` では何段も先の予測をし, 先の行 (系列ごとに最後の分かる値より後の空の行. 予測した行と, 先へ進めなかった行) だけの表と結論を返す. 先の行が無ければ誤り. 説明 (`describe_bundle`) の `forecastable` は, 目的の列の過去の値を使う回帰と予測の一式か (何段も先まで予測できるか). 決め方は一式の既定値に重ね, 値を `None` にした項目は既定値を消す (`positive` を消すと `threshold` も消す. 一緒に使えない `positive` と `min_confidence` は, 渡した方を使い, 既定値のもう一方を消す. numpy の数は Python の数にしてから確かめる)
- 決め方の項目の名前の並びは `RULE_KEYS`. `check_rules(rules, task, classes, columns)` は, 決め方の誤り (知らない項目, 範囲の外の確率, 無いクラス, 2 クラスでない `positive`, 並びの崩れた帯) を一覧で返す

## 6. 何段も先の予測

`forecast(bundle, source)` は, 目的の列の過去の値の列 (lag) を作る一式で, 目的の列が欠けた行 (先の行) を順に予測して埋める.

- 渡すデータは, 目的の列が分かる過去の行と, 目的の列が空の先の行 (他の列, たとえば気温や販促の予定は書く). 先の行は, 系列ごと (lag の `by`) に最後の分かる値より後の空の行 (過去の途中の空の行は先の行にしない. 学習のときと同じく前後関係から除かれる). 行でずらす lag は渡した並びのまま, 時刻でずらす lag は時刻 (文字なら日時にして) の順に進め, 最後に渡した並びに戻す. 系列は渡したデータの値で分けるので, 表記のゆれはそろえて渡す
- 目的の列は数にする. 空でない値で数にできないもの (「738,984円」など, 読み込む前の形) があれば誤りにする. 分類の一式は誤り
- 予測できる先の行 (過去の値が揃う行) のうち, 系列ごとに最も前の行を予測し, その値を目的の列に入れる. これを予測できる先の行が無くなるまで繰り返す (系列ごとに 1 段ずつ進むので, 繰り返しの回数は先の行の数の最大)
- 返す表は `predict` の表に, `step` (何段目の予測か. 過去の行は空) を足したもの. `prediction` は先の行では予測を, 過去の行では 1 つ先の予測を持つ. 先の行の目的の列には予測した値が入る
- 予測した値を次の入力に使うので, 先の段ほど誤りが積み重なる. 何段目かで誤差を見るのがよい
- 予測の値は元の単位で目的の列に入れる (レシピの目的の列の変換は, 前処理がかけ直す)
- 予測できない先の行 (入力の欠損など) があると, その系列はそこで止まり, 後ろの先の行は理由付きで予測しない. 予測の値が数でない (入力に無限大があるなど) 場合も止める
- 繰り返しの回数は系列ごとの先の行の数の最大で, 毎回全ての行を推論し直す. 長い過去を渡すと遅いので, 過去の値の列に要る分 (最大の lag と移動平均の幅) より前の行は除いて渡すとよい
- 過去の値の列を作らない一式では, 先の行も 1 回の推論で予測できるので `predict` と同じ (全ての行が 1 段目)
- 先の行の重複 (同じ系列の同じ時刻の行が 2 つ) は別々の行として順に進む (1 つ目を埋めると重複でなくなり, 2 つ目は 1 つ目を前の値として使う). 先の行は重複を除いてから渡す

## 7. 列の効き

`explain(bundle, source)` は, 渡したデータの元の列ごとに, その列が予測をどれだけ動かすかを返す (並べ替えの重要度). 正解の無い新しいデータでも求められる.

- 列を 1 つずつ, 行の間で無作為に並べ替えてから前処理と推論をし直し, 並べ替える前の予測からの変わり方を平均する. 分類は並べ替える前に予測したクラスの確率の変わり方, 回帰と予測は値の変わり方 (元の単位) の絶対値
- 並べ替えは決まった回数 (既定 3 回) 行い, 平均 (`effect`) と, 回数の間の標準偏差 (`spread`. 並べ替えの乱数による揺れで, 行の選び方の揺れは含まない. 1 回なら無し) を返す. 乱数のシードは固定
- 行が多い場合は決まった数 (既定 2,000 行) を選んで求める. 過去の値の列を作るレシピは前の行が要るので, 全ての系列の最後の続いた行を, 系列ごとに同じ数 (30 行以上) だけ選ぶ (系列を分ける列が無ければ 1 つの系列)
- 並べ替えた後に予測できなくなった行は平均から除き, 数を `lost` で返す (平均に使った行の数は `rows`). 並べ替えると前処理や推論ができなくなる列は, 理由 (`error`) を付けて先頭に並べる (最も頼っている列のため). 求めない時刻の列 (`kind` が `time`) は理由を付けて最後
- 目的の列は除く. ただし過去の値の列が目的の列を読む場合は「`<目的の列> (過去の値)`」として, 目的の列が分かる行の中だけで並べ替えて求める (先の予測の行の欠損は動かさない)
- 系列を分ける列 (lag の `by`) は, 行ごとに並べ替えると過去の値の系列が壊れて別の効きが混ざるので, 系列の名前だけを入れ替える (系列の中の前後関係は残す). 時刻でずらす lag の時刻の列は求めない
- 前処理で落とす列 (ID など) は効きが 0 になる. 過去の値の列を作る時系列では, 元の列を並べ替えると過去の値の列も変わるので, その列の過去の値の効きも含む
- 文, 画像, 点群, 音声は 1 件の中身が列ではないので求めない
- 効きは「この一式が, このデータで, その列にどれだけ頼っているか」で, 原因の強さではない. 似た列があると, 並べ替えで実際には無い組み合わせができるので, 効きが分かれることも大きく出ることもある

### 1 行ごとの理由 (`explain_rows`)

`explain_rows(bundle, source)` は, 渡したデータの行ごとに, 元の列ごとにその行の値が予測をどちらへどれだけ動かしたかを求め, 予測の表 (`predict` の表) に足して返す. 画面が 1 件ごとの理由を見せるため.

- 行ごとの効きは「並べ替える前の値 − その列だけを並べ替えた後の値の平均」. その列の値が他の行の値に替わったら予測がどれだけ変わるかで, 正ならその行の値が予測を上げた (回帰と予測は元の単位の値, 分類は並べ替える前に予測したクラスの確率). 並べ替えは決まった回数 (既定 10 回. 推論の回数は列の数 × 回数) で, 乱数のシードは固定. 回数が少ないと揺れが大きい
- 並べ替え方, 求める列, 目的の列の過去の値, 系列を分ける列, 時刻の列の扱いは `explain` と同じ. 行は選ばず, 渡した全ての行で求める (理由を見せたい行と, 過去の値に要る前の行だけを渡す)
- 足す列: 列ごとの `effect:<名前>` (目的の列の過去の値は `effect:<目的の列> (過去の値)`) と, 効きの絶対値の大きい 3 列を「名前 (+値)」で並べた `reason` (0 と欠損は除く. 無ければ空). 予測しなかった行と, 並べ替えた後に一度も予測できなかった行の効きは空. 並べ替えると推論できない列は効きを空にし, 求めない時刻の列と一緒に理由を `errors` (名前から理由) で返す
- 返すもの: `measure` (分類は `probability`, 回帰と予測は `value`), `frame`, `columns` (足した効きの列の名前), `errors`
- 効きは列ごとに 1 つずつ替えて求めるので, 足しても予測の値と平均の差にはならない (列どうしが効き合う場合). 意味は列の効きと同じく「この一式がこの行でその列の値にどれだけ頼っているか」で, 原因の強さではない
- 文, 画像, 点群, 音声は求めない

## 8. 意思決定 (予測から行動へ)

機械学習の価値は, 予測そのものではなく, 予測から「だから何をするか」を決められることにある. 推論の段階は, 予測を出して終わりにせず, 次の行動 (いくつ補充するか, 誰に連絡するか, どれを人が確かめるか) と, その行動を取るとどれだけ得か (今のやり方と比べて) を, 意思決定の材料として届けるところまでを受け持つ.

- 行動を決めるには, 予測の他に, 取れる行動, 行動ごとの得と損, 制約 (予算, 人手, 在庫) が要る. これはデータからは出てこない課題の値 (店なら廃棄の損, 研究なら再構成の誤差や 1 件の処理時間, 遊びなら続けて遊ぶ時間) で, 本来は問題の設定の段階が決める. このライブラリでは, 一式の既定値か引数で渡す (結論の決め方と同じ扱い. 画面は問題の設定をつなぐとその値を渡す)
- 予測には外れ方がある. 量を決めるには点の予測ではなく, どれだけ外れうるかが要る. 確率で選ぶには, 確率が当たっていること (70 % と出た行の 7 割が本当にそうなる) が要る. どちらも一式を作るときに測って残し, 意思決定の文に添える
- 行動が結果を変える判断 (値段を変える, 引き止めの連絡をする) で要るのは「行動したらどう変わるか」(因果の効果) で, 過去の予測からは分からない. 効果を推定できる記録 (過去に行動を変えた記録や試験) が無い場合は, 効果を仮定として利用者に書いてもらい, 結果の文でも仮定であることを明示する. 値段を変える型は, 価格を変えた記録が要るので扱わない (下の「扱わないこと」)
- 計算の正しさ (理論の最適な答えに合うか) は, 答えの分かるデータ (誤りの分布や確率を決めて作ったデータ) で確かめる (`tests/test_known_answers.py`). 量は本当の分布の分位点に, 振り分けのしきい値は実際の損の最も小さいしきい値に, 費用の上限のある選び方の得の合計は, 全ての組み合わせを調べた最適の半分以上 (テストの 200 個の平均で最適の 95 % 以上) になる. 測り直しは, 比で ±40 % の一様な誤りの 10 % と 90 % の点 (0.68 倍と 1.32 倍) を取り戻し, 段ごとの外れ方は, ランダムウォークで k 段先の幅が 1 段先の √k 倍になる. 同じ日の行どうしが似ているデータでは, 差の 90 % の幅は, 行ごとに選び直すと本当の差を 6 割も覆わず, 日ごとに選び直すと 9 割近く覆う (行が独立なら行ごとでも 9 割近く)
- 手法が狙いどおり働くかは, 店ごとの水準, 曜日, 季節, 気温, 販促から作った**合成のデータ** (店ごとの日々の売上) で学習した一式で確かめる (例: ML_tester の課題 20). 見るのは, 選んだ外れ方の形 (差か比), 検証用の期間で求めた幅がテスト用の期間の実際の値を覆った割合, 予測の値どおりに置く場合と比べた実際の損の差とその 90 % の幅 (損の比ごと), 1 週間分をまとめて置く場面での段ごとの外れ方の効き. 実際の課題で効く根拠ではない. 損の比 (9 : 1 など), 比べる「今のやり方」(前の週の同じ曜日と同じ量), 「7 日おきに 1 週間分を置く」場面は, 確かめのために置いた**想定**で, 課題から決めた値ではない. 実際の課題で効くかは, 問題の設定で決めた損得で, その課題のデータでたどり直して確かめる

### 予測の外れ方 (一式の `uncertainty`)

`make_bundle` が, データセットの検証用の分割 (無ければ学習用) を一式で推論して測り, 推論の一式に残す.

- 回帰と予測: 誤り (実際の値 − 予測) の 1 % から 99 % の分位点を, `absolute` (同じ単位の差) と `relative` (実際の値と予測の比の対数. 予測と実際の値が全て正のときだけ) の 2 つの形で残す. 行の少ない場合に幅が狭くなりすぎないよう, 分位点の水準を外へずらす (共形予測の補正. 上側は (n + 1) q / n). 使う形 (`kind`) は, 行を半分に分けて片方で求めた 90 % の幅が, もう片方を予測の水準の 4 つの区間ごとにどれだけ均等に覆うかで選ぶ (外れ方が水準に比例するなら比, そうでなければ差)
- テスト用の分割があれば, 検証用で求めた 80 % と 90 % の幅が実際の値を覆った割合 (`coverage`) と, 1 % から 99 % の点ごとに実際の値がそれ以下だった割合 (`test_below`) を残す. 量の判断は 1 つの点 (τ の分位点) を使うので, その点の片側の当たり具合を文に添える (期間がずれると外れ方は大きくなる)
- 分類: 確率を 10 の区間に分け, 区間ごとの確率の平均と実際に当たった割合, そのずれの平均 (ECE) を残す. 2 クラスは 2 つ目のクラスの確率で測る (判断に使う当たりの確率. 予測したクラスの確率では, 当たりが少ない場合の低い確率の区間のずれが折り返して見えない). テスト用の分割のずれ (`test_ece`) も残す. 確率を補正した一式 (下の「確率の補正」) は, 補正した確率で測る
- 何段も先の予測の外れ方は 1 つ先より大きい. 一式を作るときに残すのは 1 つ先の外れ方だけ (学習の分割は前処理の後の表で, 先の行を空にして何段も先まで予測し直せないため). 段ごとの外れ方は, 実績から `remeasure(..., horizon=h)` で測る (下の「外れ方の測り直し」). 測っていなければ, 何段も先の行の量には 1 つ先の外れ方を当て, 文に注意を添える
- 一式を作るときの外れ方は全ての行で 1 つ. 行の種類 (店や品など) ごとの外れ方は, 実績から `remeasure(..., groups=列)` で測る (下の「外れ方の測り直し」)

### 確率の補正 (一式の `calibrate`)

分類の確率が当たっていない (ずれが大きい) 一式は, `make_bundle(..., calibrate="platt")` か `calibrate="isotonic"` で, 検証用の分割で確率を補正する. 選ぶと振り分けの型は確率を額面どおりに使うため. 省くと補正しない.

- **使える一式**: 版 1 のデータセットの分類の一式 (出力の形 `class`) で, データセットが目的の列のある検証用の分割 (`valid`) を持つもの. 分類以外の一式と, 検証用の分割の無いデータセットは誤り (学習用の分割は学習に使ったので, 確率が当たりすぎて見える). 検証用の分割に行の無いクラスがある場合 (2 クラスで 1 クラスだけ, 3 クラス以上で出ないクラスがある) も誤り (そのクラスかの確率を当てはめられないため). 頼まれた補正なので, できなければ一式を作らずに止める
- **当てはめ方**: 検証用の分割 (5 万行まで) の補正の前の確率に当てはめる. 2 クラスは 2 つ目のクラスの確率だけを補正し, 1 つ目は残り. 3 クラス以上はクラスごとに「そのクラスか」の確率を補正し, 行の和で割って足して 1 にする (和が 0 の行は補正の前のまま). 予測したクラス (`prediction`) と `confidence` も補正した確率から決める
  - `platt`: 補正した確率 = sigmoid(a × logit(p) + b). logit は確率を 1e-6 と 1 − 1e-6 で切ってから求める. a と b は負の対数尤度を最も小さくする値 (ニュートン法. 損が減るまで歩幅を半分にする). 的は Platt の決まりで当たり (N+ + 1) / (N+ + 2), 外れ 1 / (N- + 2) にずらす (分けきれる組でも a が無限に大きくならないため). 確率の順を保ち, 値は 2 つだけ. 傾き a が 0 以下になったら (確率の高い順が逆になる. モデルの確率が当たりと逆向きか, 行が少なすぎる場合), 補正せずに一式を作るのを止め, 補正しないか `isotonic` を選ぶよう案内する (注意を残して作ると, 補正した確率で黙って逆の判断になるため)
  - `isotonic`: 同じ確率の行を先に 1 つにまとめ (同じ確率に当たった割合の違う塊が並ぶと, 補間が行の順で変わるため), 確率の小さい順に, 当たった割合が下がる隣どうしを行の数で重み付けて 1 つの塊にまとめ (pool adjacent violators), 塊ごとの確率の範囲の両端 (`x`) と当たった割合 (`y`) を残す. 間は直線でつなぎ, 範囲の外は端の値. 確率の順を保つ (同じ値は許す). 行が少ないと階段が粗い
- **確かめ**: 補正の前と後の検証用とテスト用のずれ (確率と実際の当たり方の差の平均, ECE. `before` と `after` の `ece` と `test_ece`. 同じ求め方で, テスト用の分割が無ければ `test_ece` は `None`) を残し, 外れ方の記録 (`uncertainty`) も補正した確率で測る. 補正したのは検証用の分割なので, 検証用のずれは小さく出る. 補正が効いたかは, テスト用の分割のずれの `before.test_ece` と `after.test_ece` で比べる. 実験の記録との比べ (`check`) は記録と同じ補正の前のモデルで求める. 補正すると予測したクラスと確率の並びが変わりうるので, 補正の後の正解率や AUC などの指標は実験の記録とずれうる
- **読むとき**: `calibrate` を持つ一式は, 予測 (`predict` と, それを使う結論, 意思決定, 列の効き, 測り直し) の全てで補正した確率を返す. 形の崩れた `calibrate` (知らない選び方と欄, 一式のクラスと合わないクラス, `platt` の `a` と `b` や `isotonic` の同じ長さの `x` と `y` の無いもの) と, 分類でない一式と既製のモデルの一式の `calibrate` は誤り. 補正を知らない古いライブラリは知らない項目として読まずに止まる (黙って補正の前の確率を返さない)
- **説明と文**: 説明 (`describe_bundle`) の `calibrate` は当てはめた値を除いた要約 (`method`, `split`, `rows`, `before`, `after`. 補正していなければ `None`). 選ぶと振り分けの前提の文は, 確率を補正したこと (分割, 行の数, 選び方の名前 Platt 法か isotonic 回帰) と, テスト用の分割のずれ (確率と実際の当たり方の差の平均, ECE) の補正の前と後を述べる
- 測り直し (`remeasure`) は補正を当て直さない (補正した確率の当たり具合を測る). 補正を当て直すには, 一式を作り直す

### 外れ方の測り直し (`remeasure`)

一式を作った後に新しい実績 (目的の列が分かるデータ) が貯まったら, `remeasure(bundle, source)` でその実績から外れ方を測り直し, 一式に書き戻す. 期間がずれると外れ方は変わるため.

- 予測と数える行の選び方は `backtest` と同じ (全ての行で予測し, 予測でき, 目的の値が分かり, `count` で選んだ行). 20 行より少なければ測らない
- 書き戻す前に, 今の記録の当たり具合を返す (`before`). 回帰は記録の 80 % と 90 % の幅が渡した実績を覆った割合 (今の記録が比の形なら, 予測が 0 以下の行は幅を出せないので除き, 除いた行の数を文に添える). 分類は確率を直さない (補正した一式は補正した確率のまま) ので, 渡した実績でのずれは新しい記録の方で, `before` は比べるための記録したときのずれ (ECE). 一式のクラスに無いラベルの行は除いて数を添える. 外れ方が変わったか (予測の傾向のずれ) に気付くための値で, 測り直すかの判断材料にもなる
- 回帰は誤りの分位点 (形の選び方も含めて `make_bundle` と同じ), 分類は確率の当たり具合を測り直す. 確率そのものは直さない
- 書き戻す `uncertainty` の `split` は `recent` (渡した実績), `measured_at` は測った時刻, `previous` は前の記録の分割と行の数と形. 前の記録のテスト用の分割での確かめ (`coverage`, `test_below`, `test_ece`) は新しい外れ方には当てはまらないので, 新しい記録の値としては残さない. 一式を作ったときの記録は `original` に丸ごと残し (何度測り直しても最初の記録), `restore_uncertainty(bundle)` で戻せる (登録した場所のファイルを書き換えるため, 元に戻せるようにする. `original` を持たない一式は戻せない). 説明の要約の `restorable` は戻せるか. 意思決定の前提の文は, 量では「外れ方は渡した実績の n 行で測り直した」, 選ぶと振り分けでは確率のずれの文に「渡した実績の n 行で測り直した値」を添える
- `write=False` で書き戻さずに結果だけを返す (確かめるだけの場合)
- `horizon=h` (目的の列の過去の値を使う回帰と予測の一式だけ): 測る行の時刻を h おきに予測を始める時刻とし (新しい方から 30 個まで), それより前の実績だけを渡して h 段先まで予測し (`forecast`), 段ごとの誤りの分位点を `steps` に残す (1 段目の記録と同じ形. 誤りが 20 より少ない段は除く. 1 段の誤りの数は始める時刻の数 × 系列の数なので, 系列が 1 つなら 20 × h 個以上の時刻の実績が要る). 渡すのは予測を始める時刻から h 個の時刻まで (目的の値は空) と, 過去の値に要る前の時刻 (ずらしと窓の最大と, 移動平均の始まり `gap` の 2 倍と 2) だけ. 時刻の列は `time` (省略は前処理の時刻でずらす列. 過去の値の手順と同じ時刻の列を選ぶ. 整数の列は期の番号として認める. それ以外の数の列と, 半分より多くの値が日時として読めない列は誤り). 段は時刻の距離ではなく, 系列ごとの先の行の順 (`forecast` と同じ数え方). 量の判断は, 予測の表の `step` の 2 段目より先の行にその段の外れ方を当てる (1 段目は全ての行で測った記録のまま. 測った段より先は最も先の段). 目的の列以外に先では分からない列 (客数など) を使う一式では, その列の本当の値で予測するので, 段ごとの外れ方は小さく出る. `horizon` を付けずに測り直すと, 段ごとの記録は残らない. 説明の要約の `horizon` と `steps_measured` で, 段ごとに測ったかが分かる.
- `groups=列` (回帰と予測の一式だけ): 外れ方を, 渡した実績のその列の値 (店や品などの行の種類) ごとにも測り, `groups` に残す (`column`, `values` は種類ごとの誤りの 1 % から 99 % の分位点, `rows` は種類ごとの行の数, `few` は 20 行より少なく測らなかった種類の数). 形 (`kind`) は全ての行で選んだものを種類ごとにも使う. 種類の名前は値の文字で, 空欄で小数に読まれた整数 (1.0) は整数 (1) にそろえる. 目的の列と, 実績に無い列は誤り. 20 行以上の種類が無ければ誤り. 量の判断は, 予測の表のその列の値が記録にある行にその種類の外れ方を当て, 記録に無い種類と欠けた値の行には全ての行の外れ方を当て, 前提の文にその行の数を添える. 記録の分位点が 99 個の有限の数でない種類 (書き換えた一式など) も全ての行の外れ方を当て, その種類を添える. 当てた種類のうち 100 行より少ない行で測った種類は, 粗いと添える. 予測の表に列が無ければ全ての行の外れ方を当て, そう添える. 2 段目より先の行は段ごとの外れ方 (種類で分けない) を当てる. `groups` を付けずに測り直すと, 種類ごとの記録は残らない. 説明の要約の `groups` は分けた列と種類の数 (分けていなければ `None`)
- 測る実績と, 方針をたどり直す実績は分ける (同じ行で測って確かめると良く見えすぎる). 例えばテスト用の期間の前半で測り直し, 後半でたどり直す

### 意思決定の型

`decide(bundle, prediction, decision)` は, 予測の表に行動の列を足し, 見込みの得と損と, それを述べる文と前提を返す. `decision` は辞書で, `type` で型を選ぶ. 損得の値は数か, 予測の表の列の名前 (行ごとに違う価値など). 値の欠けた行は決めずに, 行の数を前提の文で知らせる (0 とみなすと, ただの行動や 1 % の点になるため).

| 型 | 使いどころ | 書くこと | 決め方 |
| --- | --- | --- | --- |
| `quantity` (量を決める) | 在庫の補充, 仕入れ, 人員の配置 | `under_cost` (1 単位足りないときの損. 欠品の利益など), `over_cost` (1 単位余ったときの損. 捨てるなら原価, 持ち越すなら保管の費用), `stock` (今ある量), `step` (量の刻み. 省略は 1) | 足りない損と余る損の比で決まる水準 τ = under / (under + over) の分位点まで置く (新聞売りの問題の答え). 足す量は刻みの上下の見込みの損の小さい方. τ が 0 なら足さない |
| `select` (対象を選ぶ) | 解約の引き止め, 営業先, 点検する設備 | `positive` (当たりのクラス), `value` (当たりの 1 件の価値), `cost` (行動 1 件の費用), `effect` (行動で当たりの損を防げる割合. 仮定. 外で推定した行ごとの効果の列も書ける), `effect_kind` (`share` 防げる割合 (既定) か `uplift` 行動したときの当たりの確率 − しないときの当たりの確率. 効果のモデルの見積もり. -1 から 1), `effect_goal` (`uplift` のときに必ず書き, ほかでは書かない向き. `reduce` は当たりを減らしたい (解約など. 差が負なら得), `raise` は増やしたい (成約など. 差が正なら得)), `budget` (件数の上限) か `budget_cost` (費用の上限) | 行動した場合の見込みの得 = 確率 × 価値 × 効果 − 費用 (`uplift` は 価値 × 確率の差 − 費用. `reduce` では差の符号を反転. 確率を二重に掛けない) が正の行を選ぶ. 費用の上限は, 得の順と費用あたりの得の順の良い方 (詰め込みの問題の近い答え) |
| `triage` (振り分ける) | 人が確かめる, 再検査に回す | `positive`, `miss_cost` (当たりを見逃す損), `false_alarm_cost` (外れを回す損), `capacity` (回せる件数の上限) | 確率が false_alarm / (miss + false_alarm) 以上なら回す (損の見込みが小さい方). 上限があれば, 回して減る損の大きい順に |

- 足す列: 量は `action` (足す量), `level` (置いた後の量), `expected_cost` (見込みの損), `point_expected_cost` (予測の値どおりに置いた場合の見込みの損). 選ぶは `action` (選んだか), `gain_if_acted` (行動した場合の見込みの得. 選ばなかった行にも入る), `expected_gain` (選んだ行の見込みの得. 選ばなかった行は 0). 振り分けは `action` (回すか), `expected_cost`. 決めなかった行は空
- 文は「書いた損得と外れ方 (確率) の記録が正しいとすれば, 損が最も小さいのは …」の形で前提を先に述べ, 行動の件数と量, 同じ前提での素朴なやり方との比べを続ける. 同じ前提から計算した比べは必ずこの方針が良く出るので, 文にそう書き, 実際の差はたどり直しで確かめると添える. 行ごとに損得が違う場合の τ やしきい値は「行ごとに x から y」
- 素朴なやり方: 量は「予測の値どおりに置く」. 選ぶは「確率の高い順に同じ件数」(同じ費用の上限の中で) と, 上限の無い場合だけ「全てに行動」(上限を破る比べは公平でない). 振り分けは「確率 0.5 で分ける」(同じ人手の上限の中で), 上限の無い場合だけ「全てを回す」, と「どれも回さない」
- 前提の文 (`notes`): 値の欠けた行, 外れ方を測った行の少なさ (100 行未満), 量では 1 期間の判断であること (届くまでの日数, 発注の周期, 余りの持ち越しは考えない), τ の点の片側の当たり具合, 差の形の外れ方の偏り, 行の種類ごとの外れ方を当てた行の数, 欠品などで抑えられた過去の値 (打ち切られた需要) で量が少なめになること, 2 段目より先の予測. 選ぶと振り分けでは確率のずれ, 効果とその確率への比例が仮定であること (`uplift` では, 効果は見積もりで, 無作為の記録でしか確かめられないこと. 文も「仮定」や「たどり直す」でなく無作為の試験で確かめると書く. 確率のずれは, 得に使わず比べの順にだけ使うと書く), 人の判断を補う材料であること
- `infer(..., decision=...)` は結論と一緒に意思決定もし, 表に行動の列を足す. 省略すると一式の既定値 (`make_bundle(decision=...)`. 損得と制約は本来は問題の設定の段階が決めるので, それまでの置き場), 空の辞書で意思決定しない

### たどり直し (`backtest`)

`backtest(bundle, source, decision)` は, 目的の列が分かる過去のデータで, 意思決定の方針を取っていたら実際にどれだけの得と損だったかを, 素朴なやり方と比べる.

- 全ての行で予測し (過去の値の列のため), 数える行 (予測し, 目的の値が分かり, `count` で選んだ行) だけで方針を決める (件数や費用の上限を, 数えない行に使わないように)
- 量: 実際の値で, 足りなかった量と余った量の損を数える. `current` (今のやり方で足した量の列か, 下の「今のやり方を決まりで書く」の決まり) を書けば, それとも比べる
- 選ぶ: 当たりの行を選んだ数と, 価値 × 効果 − 費用 の合計. 効果は仮定なので, 結果も仮定のもとの値 (行動しなかった過去の結果しか分からないため, 効果は試験でしか確かめられない). `current` は今のやり方で行動したかの列. `effect_kind: uplift` はたどり直さない (行動しなかった場合の結果が分からないので, 実際の得を数えられない. 効果のモデルは行動を無作為に割り当てた記録で確かめる)
- 振り分け: 見逃しと空振りの損. `current` は今のやり方で回したかの列
- 選ぶと振り分けの `current` の列は, 真偽値, 数 (0 より大きければ行動した. 1.0 のような数の文字も), 決まった言葉 (yes, y, true, はい, 済 と, no, n, false, いいえ, 未. 全角は半角にそろえ, 大文字と小文字は区別しない) で読む. 読めない値や欠けた値があれば誤り (文字の列をそのまま真偽値にすると, no も行動したことになるため). 量の `current` の列も, 欠けた値か数でない値があれば誤り (今のやり方の損の合計が欠けないように). `current` は列の名前 (文字) か決まり (辞書)
- `current` の列は, たどり直しだけで使う. 決めるだけ (`decide`, `infer`) では列が無くても誤りにしない (一式の既定値に今のやり方の列を入れても, その列の無い新しいデータを推論できるように). 一式の既定値 (`make_bundle(decision=...)`) の `current` は列の名前だけ (一式の版 3 の形を変えないため. 決まりはたどり直しで渡す)
- この方針と他のやり方の差には, 重複を許して選び直した (500 回) 90 % の幅を付け, 幅が 0 をまたぐなら「差は揺れの範囲」とする. 選び直す単位は `group` で選ぶ. 既定 (`auto`) は, 時刻でずらす一式なら時刻の列 (同じ日の店どうしのように, 同じ時刻の系列は似ているので, 行ごとに選び直すと幅が狭く出る), 他は行ごと. `None` で行ごと, 列の名前か行ごとの値の並びも書ける. 文には「date ごとに選び直した」のように添える. まとまりが 20 より少なければ幅が狭く出るので行ごとに戻し, そう添える. まとまりの値が欠けた行は 1 行ずつ別のまとまりにする. 日をまたいだ似かた (前の日と次の日が似ている) は扱わない
- 目的の列のラベルは, 空欄で小数に読まれた値 (1.0) も整数のクラスの名前 (1) と比べる
- 外れ方を測った分割の期間をたどり直すと甘くなるので, それより後の期間で比べる (過去の値の列のために前の行も渡し, `count` で後の期間だけを数える)

#### 守る指標を数える (`guard`)

`backtest(..., guard=...)` は, 守る指標 (引き換えに悪くしてはいけない数字) を, この方針の行動で数えて結果に並べる. 守る指標の名前と測り方は自由な文なので, 数えるのは, 型ごとにたどり直しで数えられる量の一覧 (`GUARD_COUNTS`) のうち, 守る指標が指した量だけにする.

- 形は問題の設定の受け渡し (ML-Quest の推論の `guard`) の `name` (名前. 要る), `unit`, `better` (`lower` か `higher`), `line` (守る線. 有限の数), `measure` に, 指す量の名前 `counted` を足したもの. `counted` は型の一覧の名前か `None`. 一覧に無い名前, 崩れた向きと線, 名前の無い守る指標は誤り
- 一覧 (`GUARD_COUNTS`) は型ごとに, 量の名前から説明 (`description`) と向き (`better`. `lower` は小さいほど良い) を持つ. 問題の設定の問いの選択肢に使える公開の一覧. 割合は 0 から 1

| 型 | 名前 | 量 | 向き |
| --- | --- | --- | --- |
| `quantity` | `shortage_rate` | 足りなかった行の割合 | `lower` |
| `quantity` | `shortage` | 足りなかった量の合計 (目的の列の単位) | `lower` |
| `quantity` | `surplus` | 余った量の合計 (目的の列の単位) | `lower` |
| `select` | `selected` | 選んだ件数 | `lower` |
| `select` | `selected_cost` | 選んだ行の費用の合計 | `lower` |
| `select` | `selected_hit_rate` | 選んだ行のうち当たりの割合 | `higher` |
| `triage` | `routed` | 回した件数 | `lower` |
| `triage` | `missed` | 見逃した件数 (回さなかった当たり) | `lower` |
| `triage` | `false_alarms` | 空振りの件数 (回した外れ) | `lower` |

- 結果の `guard` は `name`, `unit`, `counted`, `description`, `value` (数えた値), `line`, `better`, `kept` (守れたか), `note`. 向きは守る指標の `better` で, 無ければ量の向き. 線が無ければ `kept` は `None`. 選ぶ行が無く割合を数えられなければ `value` は `None` で, `note` にそう書く. 文にも「守る指標「…」(量の説明) は, この方針で x (守る線は y 以下. 守れた)」の形で並べる (割合の値と線は「12 %」のように % で書く). 線が無ければ「… は, この方針で x (守る線は無い)」, 数える行が無ければ「… は, 数える行が無いので数えなかった」
- 指していない (`counted` が `None`) 守る指標は数えず, `note` と文に `GUARD_UNCOUNTED` (「過去のデータでは数えられない. 課題の場で数える」) を並べる. `guard` を渡さなければ結果の `guard` は `None`
- 数えるのはこの方針の行動だけで, 得と損の比べ (`policies`, `gaps`) は変えない
- 外れ方の測り直し (`remeasure`) は方針を取らないので, 守る指標を数えない

### 今のやり方を決まりで書く (`current_from_baseline`)

`current_from_baseline(data, baseline, target)` は, 今のやり方をデータの上で書いた決まりから, 行ごとの今のやり方の値の列 (`Series`. 名前は「今のやり方」, 行の並びと index は渡したデータと同じ) を作る. 決まりの形と値の求め方は問題の設定 (ML-Quest の `action.current`) と同じで, 問題の設定で求めた今のやり方の損と, 学習のプロジェクトの骨組みが比べる相手として求める値と, たどり直しの今のやり方が同じ値になるようにする.

| `type` | 書くこと | 行ごとの値 | 使える型 |
| --- | --- | --- | --- |
| `lag` | `periods` (k. 1 以上の整数) | 同じ系列で, 時刻がちょうど k 期前の行の目的の列の値. その時刻の行が無ければ欠損 (休みで日が飛んでも暦の k 期前. 近い行で代えない) | 量を決める |
| `mean` | `periods` (k) | 同じ系列で, k 期前からその時刻の前まで (その時刻は含めない) の行の目的の列の平均. 期間に値が 1 つでもあれば求め, 無ければ欠損 | 量を決める |
| `column` | `column` (列の名前) | その列の値のまま (量なら置いた量, 選ぶと振り分けなら行動したか) | 全て |

- k 期前と平均で, 目的の列を時刻の列か系列を分ける列にも書いた決まりは誤り
- k 期前と平均は, 他に `time` (時刻の列. 日時として読める列), `by` (系列を分ける列の名前か並び. 省略は 1 つの系列), `freq` (1 期の間隔. `86400s` や `1D`. 省略は時刻の列の重ならない差の最頻値. ML-Quest がデータを入れたときに求める間隔と同じ), `target` (目的の列. 省略は引数の `target`) を書ける. 時刻の列が無ければ誤り (行の並びで数えると, 抜けた日で別の日の値になるため)
- 求めるのは ML-EDA の時刻でずらす手順 (`lag` の `time`, `freq`, `match: exact`. 移動平均は `windows`) で, ML-Quest と同じ. 同じ系列の同じ時刻の行が複数あれば後の行の値を使う. 目的の列の数にできない値と欠けた値 (先の行) は使わない
- 問題の設定と同じく, 量の決まりの値は (列の決まりも) **置いた後の量** (k 期前と同じ量を置く). たどり直しの量の今のやり方は, 今ある量 (`stock`) がそれより多ければ足さず, 少なければ足して決まりの量にする (決まりでなく列の名前を書いた `current` は足した量)
- たどり直し (`backtest`) は, 量の `current` が決まりなら全ての行で値を求め (前の期の値のため, 数えない前の行も使う), 値の無い数える行 (前の期の行が無い始めの行, 列の空欄と数でない値) は, どの方針でも数えず, 文にその数と決まりを添える (問題の設定の今のやり方の損と同じ行). 選ぶと振り分けの列の決まりは, 列の名前を書いた場合と同じ (読めない値は誤り). `time` と `by` を省いた決まりは, 一式の過去の値の手順の時刻の列 (1 つだけの場合) と系列を分ける列で補う
- 選ぶと振り分けの今のやり方は行動したかなので, k 期前と平均は誤り (ML-Quest の確かめと同じ)
- 画面 (ML-Provide) は, 一式に問題の設定をつなぐと, 問題の設定の今のやり方を, 時刻の列, 系列を分ける列, 時刻の間隔を足した決まりにして `current` に渡す
- `check_baseline(baseline, columns)` は決まりの誤り (知らない種類と項目, 期の数, 間隔, 渡した列に無い列) を一覧で返す (画面が, つなぐときにデータの列で確かめるため)
- 1 期の間隔を時刻の列から求められない (行が 1 つ, 1 秒より短い間隔) 場合は誤り

### 扱わないこと

- 行動の効果 (アップリフト) のモデルの学習: 学習の段階の仕事. 推論の段階は, 行ごとの効果の列 (`effect`) として受け取るだけ
- 値段を変える判断: 値段を変えたら需要がどう動くかは, 価格を変えた記録か試験が要る. 記録の無いまま予測から値段を勧めると根拠の無い助言になるため
- 量の型は 1 回の期間 (次の 1 つ分) の判断で, 届くまでの日数, 発注の周期, 在庫の持ち越しは考えない. 選ぶ型の上限は件数か費用の 1 つだけ

## 9. 見本の一式

`make_sample_bundle(path, source, recipe, ...)` は, 表か文のデータのファイルと ML-EDA のレシピから, 小さなモデルをその場で学習して推論の一式を作る. 画面 (ML-Provide) で操作を試すための見本で, 課題のモデルではない. 画面が ML-EDA と ML-ModelControl を直接使わずに見本を作れるように, データセットの書き出しから一式づくりまでをこのライブラリが受け持つ.

- 順: データを ML-EDA の `load_data` で読み (`options` は読み込みの設定), `export_dataset` で一時の場所に書き出し, 小さなモデルの定義を作って ML-ModelControl の `build_model` で組み立て, 学習用の分割で学習し, `make_bundle` で一式にする (データセットと重みは一式に写し, 一時の場所は消す). 結論と意思決定の既定値, 名前, 置き換え (`overwrite`) は `make_bundle` と同じ
- モデル: 表と時系列は学習に使う列から全結合 2 層 (幅 16, ReLU), 文は番号の並びの埋め込みの平均 (幅 16. 中身の手順に `vocabulary` が要る) と全結合 1 層. 画像, 点群, 音声のデータは誤り
- 学習: 学習用の全ての行を 1 度に使い, Adam (学習率 0.02) で決まった回数 (既定 300 回). 分類は交差エントロピー, 回帰と予測は目的の値を学習用の平均と標準偏差で標準化した二乗誤差で, その中心と幅を一式の `target_scale` に書く (目的の値が一定なら誤り). 重みの初期値の乱数の種は `seed`. 一式が既にあり `overwrite` でなければ, 学習の前に止まる
- 一式の確かめ (検証用の分割の評価) と外れ方の記録は, 他の一式と同じく `make_bundle` が行う

## 10. 方策の一式

強化学習で学習した方策 (ML-Found が書き出す `policy/`. `policy.json` と `policy.pt`) を読み, 観測から行動と守りの結果を返す. 守りは ML-Found の `apply_guards` と同じ決まりで当て, 同じ結果を返す. ML-Found のコードは使わず, 一式の形と守りの決まりをこのライブラリが持つ (`ml_inference.policy`).

```python
from ml_inference import load_policy

policy = load_policy("outputs/cartpole/policy")
result = policy.act([0.01, -0.2, 0.03, 0.1])
# {"action": 1, "status": "ok", "reasons": []}
policy.act([2.5, 0.0, 0.0, 0.0])
# {"action": None, "status": "handover", "reasons": ["観測の要素 0 が範囲の外 (2.5)"]}
```

- 読み込み: `read_policy_spec(path)` は `policy.json` を読んで確かめ, `load_policy(path, model=None)` は重みも読んで方策 (`GuardedPolicy`) を返す. 返す方策の `act(observation)` は `action` (行動. `stop` と `handover` では None), `status` (`ok`, `clipped`, `fallback`, `stop`, `handover`), `reasons` (守りが働いた理由の文字のリスト) を返す. `apply_guards(spec, observation, output)` は, モデルの出力に守りを当てるだけの純粋な関数
- 版: 読めるのは版 1 (`POLICY_FORMAT_VERSION`). 守りを読み違えると危ないので, 知っている版より新しい一式と版の無い一式は読まずに止まる. 空間, `guards`, `fallback` の知らない項目 (書き損じ) も誤り. 一番上の知らない項目は無視する. JSON の外の `NaN` と `Infinity` は誤り
- `kind`: `policy` でない一式は読まない. ML-Found の読む側は `kind` を確かめないが, このライブラリは推論の一式 (`inference.yaml`) など他の形も読むので, 取り違えた一式に守りを当てて黙って誤った行動を返さないように確かめる
- 重み: `weights` は `policy.pt` に限り, ファイルの sha256 を `weights_sha256` と比べ, 違えば読まない. `torch.load(weights_only=True)` で cpu に読む. `policy.pt` はモデルの `state_dict` 全体で, 連続の行動の方策の `log_std` や価値の関数のような行動に使わない重みも入るので, 余る重みは読まずに無視し, そのキーの並びを方策の `unused_keys` に持つ (書き出したモデルと違うモデルを渡していないかを確かめられるように). モデルの重みが足りない場合と形が違う場合は誤り. `policy.json` と `policy.pt` は一式の外を指すリンクでも読まない
- モデル: 渡したモデル (`model`) か, 一式に写したモデルの定義 (`metadata.model.spec`. 一式からの相対パスで, 一式の外を指せない) を ML-ModelControl の `build_model(..., load_weights=False)` で組み立てたものを使う. 推論の一式と同じく, 定義の部品は登録簿にあるかを先に確かめ (「自作の部品」), 定義やクラスの名前 (`model_class`, `metadata.model.class`) からはコードを読み込まない. そこで `metadata.model.spec` の無い一式は, 書き出したときと同じ組み立てのモデルを渡す. 定義は入力 (観測) と出力を 1 つずつにする
- 出力の読み方 (`head`): forward の出力だけを読む. 番号で選ぶ行動 (`discrete`) は `categorical` で, 行動ごとのスコアの最も高い行動を選ぶ. 連続の行動 (`box`) は `deterministic` で, 出力 (平均) を行動とする. 観測は float32 にして空間の形にし, 1 件の組 (先頭に 1 の次元) にしてモデルに渡す
- 守りを当てる順:
    1. 観測が数の配列でない (float にできない桁の大きな整数も), 形が空間と違う (同じ大きさの平らな配列は可), 有限でない値がある (float32 に丸めて無限大になるものも), 止める条件 (`observation_low`, `observation_high`. 元の値で比べる) の外のどれかなら, モデルを動かさずに外れたときの動き
    2. モデルが誤りになるか, 出力が数にできないか (桁の大きな整数も), 出力に有限でない値があるか, 大きさが違えば, 外れたときの動き
    3. 行動の範囲を当てる. `box` は空間の範囲と `action_low`, `action_high` の狭い方に clip し, `discrete` は `allowed_actions` に無い行動なら, その中の最もスコアの高い行動にする. 変えた場合は `clipped`
    4. `box` の行動は float32 にする. 丸めで範囲の外に出た要素は内側の隣の float32 に寄せ, 無限大になるか範囲に float32 が無い場合は外れたときの動き. 決めた行動も同じく寄せる
- 外れたときの動き (`fallback`): `stop` (行動を返さず止める), `handover` (行動を返さず元の制御に戻す), `action` (決めた行動. `status` は `fallback`)
- 理由の文と行動の型 (`discrete` は整数, `box` は空間の形の float32 の配列) も ML-Found と同じ. 同じことは, ML-Found で書き出した一式と ML-Found の結果をテストに置いて確かめる (モデルを動かした行動の値は計算の仕組みで最後の桁が変わるので, 差を許して比べる)
- 実機: 方策は観測から行動と守りの結果を返すだけで, 環境や機械に触れない. 方策を使ってよいかと守りの範囲は人が決める (「外の処理を呼ばない」)

## 11. 安全

- 重みは `torch.load(weights_only=True)` で読む (任意のコードを動かせる `pickle` を読まない)
- 読み込みは手元のファイルだけ. ただし学習済みの重みを取得する部品 (`pretrained: true` など) は, 組み立てのときに ML-ModelControl が取得する. 既製のモデルの一式の出どころの重みは hash で確かめ, 違えば推論しない
- 一式の中のパスは, 一式の外を指せない (重みの絶対パスを除く)
- 方策の一式は重みの sha256 を確かめてから読み, 守りを読み違えないように新しい版と書き損じを読まない (「方策の一式」)
- 一式やモデルの定義に書かれた名前からコードを読み込まない. 自作の部品と手順は, アプリの起動の設定で明示したモジュールからだけ登録する (「自作の部品」, 「繰り返しの手順」)

## 12. 課題の型への広げ方

推論の一式は受け持ちと出力の形 (版 3) と, 版 2 のデータセットの 1 件の扱い (版 4) を持ち, データセットを持たない雑音だけの一式 (版 5. 表の出力は版 7) と既製のモデルの一式 (版 6) と並べるの一式 (版 8) と群分けの一式 (版 9) も書け, 配列の出力は結果の形の版 2 から 4 で `artifacts/` に置く (上の「受け持ちと出力の形」, 「書き出し」). ただし結論と意思決定 (量, 選ぶ, 振り分け), 外れ方, たどり直しは, 列の出力 (クラス, 値, 分布) だけを前提にしている. 課題の受け持ち (予測する, 作る, 並べる, まとめる, 効果を見積もる. 値の名前は ML-Quest が持つ), 正解の出どころ (列, 対のファイル, 自分自身, 今の処理の出力, 行動の結果, 介入の記録, 指示), 1 件の単位 (行, まとまり, 対, 要求と候補, 窓), 作り方 (規則, 既製のモデルをそのまま, 頭だけ学習, 微調整, 一から) で, データの形と確かめ方が変わる.

- 型に依らず残すもの: 一式の版と読む側の決まり, 一式の外を指さない場所, 理由付きの「出せなかった行」, 前提を先に言う意思決定の文, まとまりごとに選び直す差の幅, 測り直しの書き戻しと戻し
- 型ごとに実績の出どころが変わる (目的の列, 人の判断, 元の処理の出力, 無作為の試験の記録, 傾向付きの行動の記録). たどり直しの共通の部分 (数える行, 素朴なやり方との比べ, 差の幅) は残し, 型ごとの損得の数え方を関数の表にする
