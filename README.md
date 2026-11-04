# Kaggriculture — PPO強化学習の設計・実装・評価

Kaggleの対戦型農業シミュレーションを題材に、**PPOの実装、行動設計、並列対戦による学習、未使用条件での評価、後退原因の分析**に取り組んだチーム開発プロジェクトです。作付け・飼育・売買・雇用が長期の収益に影響する環境で、既存戦略の一部をニューラル方策に置き換える実験を行いました。

この公開版では、実験の中心となるコードと評価を読みやすくまとめています。PPO候補と、別系統の提出用エージェント `main.py` を分けて掲載しています。

## 見ていただきたい点

| テーマ | 実装・取り組み |
| --- | --- |
| 強化学習 | PyTorchによるActor–Critic、clipped PPO、行動マスク、KLによる更新停止 |
| 状態・行動設計 | 352次元の観測・文脈、23方策headで生産・市場・農作業・労働を選択 |
| 学習基盤 | CPU並列ロールアウトとGPU更新、重み固定、再開可能なチェックポイント |
| 検証 | NumPy／PyTorch推論の照合、公式環境との一致、seed・両手番を固定した比較 |
| 改善判断 | 未使用条件の評価とアブレーションで、良い結果だけでなく後退も分析 |

**コードを読む入口：** [`ppo/model.py`](ppo/model.py) → [`ppo/update.py`](ppo/update.py) → [PPOの設計](docs/PPO.md)。ゲームへの接続は [`experiments/v25/`](experiments/v25/README.md) に保存しています。

## 実験結果

v25では2学習・合計**45,824試合**でPPOを実行しました。選択したモデルを未使用64seed・両手番で評価し、基準v24との直接対戦は**78勝50敗（60.94%）**でした。一方、公開6相手への比較では敗戦が55から75へ増えたため、PPO候補は採用しませんでした。

勝ち越しだけを成果とせず、後退する判断群を調べた過程まで残しています。[評価条件・信頼区間・原因分析](docs/RESULTS.md)を参照してください。これらは保存済みのローカル実験結果であり、Kaggle順位の改善を示すものではありません。

## まず動かす

Python 3.12を使用します。GPUやKaggleの認証は不要です。

```bash
git clone https://github.com/HagiwaraShigechika/kaggriculture.git
cd kaggriculture
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-ppo.txt
.venv/bin/python -m ppo.smoke
.venv/bin/python -m unittest discover -s tests -v
```

smokeは合成データで実際のPPO更新と有限な重み変化を確認します。ゲームの勝率を測るデモではありません。完全な対戦再学習には当時の基準方策・対戦相手・シミュレータ・データが別途必要です。[再現できる範囲](docs/PPO.md#再現できる範囲)

v33を公式環境で対戦させる手順は [ゲームと実行手順](docs/GAME.md) に分けています。

## 構成

| 場所 | 内容 |
| --- | --- |
| `ppo/` | 実験から抽出したPPOモデル・更新処理と実行可能なsmoke |
| `experiments/v25/` | 実際のゲーム統合・ロールアウト・学習コード（参照用） |
| `tests/` | 損失・マスク・重み更新の検証 |
| `docs/PPO.md` | 状態、行動、報酬、学習の設計 |
| `docs/RESULTS.md` | 保存済み評価と失敗からの知見 |
| `docs/ppo_provenance.json` | 元実装のコミットと抽出範囲 |
| `main.py` | 別系統のv33エージェント。PPO学習済みモデルではありません |

動作確認の環境と結果は [検証記録](docs/VERIFICATION.md) にまとめています。

## 出典と役割

チーム開発と公開エージェントを基盤に、v25のPPO設計・学習・評価を中心に収録しています。既存エージェント由来の実行基盤・経路と、本プロジェクトの学習・評価コードの範囲は [NOTICE.md](NOTICE.md) に記載しています。継承した著作権表示と [Apache-2.0ライセンス](LICENSE.txt) を保持しています。
