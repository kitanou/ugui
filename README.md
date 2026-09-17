# UGUI Core

Twitter/Xアーカイブから「うぐいらしい応答」を生成する独立Python Core。
[Issue #1](https://github.com/kitanou/ugui/issues/1) のMVP実装です。
NAHOのソース・Persona・Memory・DBは参照しません。パッケージ名は `ugui-core`、コード管理先は指定された `kitanou/ugui` です。

## 起動

Python 3.11以上が必要です。

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# LM Studio、llama.cppなどのOpenAI互換ローカルサーバーを別途起動。
# モデル名はサーバーの /v1/models に表示される実際のIDに置き換える。
export UGUI_LLM_URL=http://127.0.0.1:1234/v1
export UGUI_LLM_MODEL=your-loaded-chat-model
ugui serve
```

既定の待受は `127.0.0.1:8011`。`.env.example` は設定例で、自動読込はしません。
`/health` はCoreの生存確認であり、LLM接続成功を意味しません。LLMが未起動ならチャットは503を返します。

```sh
curl http://127.0.0.1:8011/health
curl http://127.0.0.1:8011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ugui","messages":[{"role":"user","content":"そばは好き？"}]}'
```

`GET /models`、`GET /v1/models`、`POST /v1/chat/completions` に対応。
`stream: true` はOpenAI形式のSSEを返しますが、現状は生成完了後に分割送信する方式です。
テキスト会話のみ対応し、ツール呼出・画像・音声・厳密なトークン使用量計算は未対応です。
NAHO固有のAPI仕様を入手していないため、NAHO GUIとの実接続は未検証です。

## アーカイブからPersonaまで

```sh
# ZIP / 展開済みフォルダー / tweets.js / tweets.json に対応。
ugui import /path/to/twitter-archive.zip

# 単体ファイルなどaccount.jsがない場合は、自分の数値アカウントIDを明示。
ugui import /path/to/tweets.js --owner-id 123456789
ugui stats

# ローカルLLMで根拠付きの特性候補を抽出し、文体統計と一緒に保存。
ugui persona --extract

# 意味検索には埋め込みモデルの設定と索引作成が必要。
export UGUI_EMBEDDING_MODEL=your-local-embedding-model
export UGUI_EMBEDDING_URL=http://127.0.0.1:1234/v1
ugui index
ugui search '冷たい麺は好き？'
ugui serve
```

埋め込み未設定時は日本語文字bigram＋英単語による**語彙検索**です。意味検索と同等とは扱いません。
意味検索を有効にした後に投稿を追加したら、`ugui index` を再実行してください。
埋め込みモデルの中身や次元を変更したら `ugui index --rebuild` を実行します。
新しい投稿の取込後は、古いPersonaでの応答を拒否します。Personaを更新するには `ugui persona --extract` を再実行します。`ugui persona` だけなら保存済みの抽出結果を再集計します。

取込時は元アーカイブを変更せず、JSを実行せずJSONとして解析します。multipartも対応します。
アカウントIDをデータ領域に固定し、他人の投稿・RT・IFTTT/twittbot・URLだけの投稿・同一本文を検索/Personaの対象から除外します。
除外投稿も理由とともに保存します。`--auto-source SourceName` で除外する投稿元を追加できます。
投稿IDが同じデータは再取込しません。ID/日時不正の行は `invalid` として数えます。
本文、日時、返信先、引用先、投稿元、元ファイル名を保持します。アーカイブに存在しない引用本文は取得しません。
本人IDが個々の投稿にない公式アーカイブは `account.js` または `--owner-id` を根拠とします。

## 大量アーカイブの全件分析と再開

数万件以上のアーカイブには、投稿ごとの結果を保存する全件分析を使えます。

```sh
ugui analyze
ugui analysis-status
# 中断・エラー後も同じモデル設定で再実行すると成功済みの投稿を再処理しない。
ugui analyze
```

`analyze` は新しい投稿から順に、全ての分析対象投稿をLLMで処理します。成功済みの投稿・主張なしの結果・失敗を別々に記録します。
複数件の抽出が失敗した場合はバッチを分割し、単体でも失敗した投稿は次回に再試行します。
全件が成功してから本番Personaを置き換えます。処理中のPersonaは、以前の版または文体統計のみの版です。
処理速度と残り時間の推定を `analysis-status` で確認できます。短時間の初期推定は大きく変動します。
分析キーはモデル・接続先・抽出指示に依存し、変更すると別の分析として開始します。
Qwen3の抽出時は `/no_think` とJSON出力指定を使用します。OpenAI互換サーバーのJSON出力対応が必要です。
`--batch-size 1` 〜 `12` で処理単位を変更できます（既定8）。

バックグラウンド実行は任意です。macOSでスリープを防ぎつつログを保存する例：

```sh
mkdir -p data
(umask 077; nohup caffeinate -i ugui analyze > data/analysis.log 2>&1 &)
```

再起動やモデルサーバー停止の後は同じコマンドを再実行してください。同じデータ領域で分析を複数起動しないでください。

## 人格・記憶の扱い

- 原文はEvidence、抽出された主張はClaim、集約結果はPersona/semantic memoryとして分離します。
- 抽出した主張は、実在の投稿IDと原文に存在する引用が必要です。LLMの解釈そのものの正確さは保証されません。
- 少なくとも3投稿・2日以上・支持スコア0.5以上の候補を `supported` とします。単発は `candidate` のままです。
- 相反する値が後から出た場合、古い方を `historical` として保持します。新しい単発も確定情報にはしません。
- confidenceは件数・一貫性・相対的な新しさを使うヒューリスティックで、校正済み確率ではありません。
- 文体統計は文字数・改行・疑問/強調・絵文字・句読点・一人称・語尾・頻出語・話題分布です。皮肉や冗談の理解は限定的です。
- 応答時は特性・文体・関連する意味記憶・投稿Evidence・現在の会話を組み合わせます。会話ログは永続保存しません。

保存先は既定で `./data` (`UGUI_DATA_DIR` または `ugui --data-dir PATH ...` で変更)。
DBは `data/ugui.sqlite3`、人が確認できるPersonaは `data/persona/profile.json` です。

## 評価

評価用の質問は、保持する投稿の答えを含めずに人が状況として記述してください。

```json
[
  {"tweet_id":"1234567890", "situation":"昼ごはんに麺類を選ぶなら？"}
]
```

```sh
ugui evaluate /path/to/cases.json --judge
```

最も古い評価対象の投稿より**前**のデータだけで一時DB・Persona・索引を作り直します。
既存の全件Persona/索引は流用しません。出力は `data/evaluation/report.json`。
語彙重なり・長さ・文体統計に加え、`--judge` で話題/意見/嗜好/文体/語彙/口調の6軸をLLMが採点します。
採点は参考値です。本人による評価が必要で、自動評価の点数を「本人再現度」とはみなしません。

## プライバシーと運用範囲

- 既定では**数値のloopback URLのみ**許可。`localhost` を含むホスト名やLAN/外部URLには `UGUI_ALLOW_REMOTE=true` が必要です。
- 外部モデル利用を明示設定すると、抽出・埋め込み・チャット・評価で投稿本文がその設定先に送られます。
- 環境プロキシとHTTPリダイレクトを使用しません。APIキーは環境変数のみで指定します。
- `UGUI_API_KEY` で受信API認証、`UGUI_LLM_API_KEY` でモデルサーバー認証を設定できます。
- 非loopback待受にはCLIが受信APIキーを要求します。直接ASGIで公開する場合も認証・TLSを設定してください。
- アーカイブ/DB/秘密情報はgitignore対象です。秘密の入力は `data/` または `archives/` 配下に置いてください。
- 通常ログに投稿全文や会話本文を出しません。`ugui search` は明示的な閲覧コマンドなので本文を表示します。
- SQLiteとレポートは所有者のみ読書き可能で作成します。暗号化は行いません。

現段階は単一ユーザー向けMVPです。ベクトル検索は全件cosineのため大規模データでは遅くなります。
1ファイル256MiBを超えるアーカイブは分割してください。語義の揺れやLLMの誤抽出についてはPersona JSONを確認してください。
LLM/Embedder/RetrieverのProtocolを差し替えることで、Qdrant等の索引や別モデルサーバーへ拡張できます。

## 開発・検証

```sh
pip install -e '.[dev]'
ruff check .
pytest -q
```

テストは実際の個人アーカイブや外部APIキーを使いません。Python 3.11/3.12/3.13向けのCI定義は [docs/github-actions-test.yml](docs/github-actions-test.yml) にあります。GitHub認証にworkflow権限がないため未登録です。権限のある環境で `.github/workflows/test.yml` に配置すると有効になります。
[設計と対応範囲](docs/architecture.md)も参照してください。
