# v25のゲーム統合ソース（参照用）

`codex/v25-ppo` のコミット `055d260e8a01f0e2f4f963fb98ef137ad3183bcf` から、実際の実験で使用したコードを保存しています。下記ファイルは原本から変更していません。元ファイルとの対応とSHA256は [`docs/ppo_provenance.json`](../../docs/ppo_provenance.json) を参照してください。

| ファイル | 役割 |
| --- | --- |
| `controller.py` | 観測特徴量、マスク、NumPy推論、生産経路・売買の選択 |
| `farm_controls.py` | 作付け・家畜・施肥・労働の実行可能な判断 |
| `farm_hooks.py` | 既存エージェントの内部制御に判断口を接続 |
| `build_agent.py` | ハッシュ固定のv24へ制御口と重みを埋め込む |
| `rollout.py` | 固定した方策による完走試合の収集、軌跡・報酬の生成 |
| `train.py` | 並列収集、PPO更新、開発評価、保存・再開 |

元の `model.py` は [`ppo/model.py`](../../ppo/model.py) にそのまま収録しています。`train.py` の `stack_rollouts` と `update` は関数本体をそのまま [`ppo/update.py`](../../ppo/update.py) に抽出しました。

これらの参照ソースは元の `strategy_v25/` という配置、v24の基準エージェント、C++シミュレータ `kagsim`、固定した公開相手、実験設定を前提とします。必要な取得物・モデルを全て配布しているわけではないため、公開版では直接実行できません。実行可能な入口はルートからの `python -m ppo.smoke` です。

この参照ソースを現在の `main.py` と組み合わせても、当時の学習条件は再現できません。`main.py` は後続v33であり、PPO候補の基準v24とは異なります。
