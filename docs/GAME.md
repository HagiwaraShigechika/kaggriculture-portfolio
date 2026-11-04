# ゲームと評価の前提

Kaggricultureは2人の農業経営エージェントが収益を競うシミュレーションです。
本プロジェクトでは `kaggle-environments==1.32.7` の標準720ターンを使い、
作物・家畜の生産、資材購入、雇用、運搬、市場での売却を扱います。
終了時の所持金を比較するため、生産量だけでなく売却の順序や資金繰りが重要です。

仕様の正本はインストールした公式環境です。
外部参照: [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)、
[kaggle-environments](https://github.com/Kaggle/kaggle-environments)。

## 公式環境で1試合動かす

Python 3.12、macOS / Linuxを想定しています。PPOコアだけを見る場合、この環境は不要です。
`kaggle-environments` は他のゲーム向け依存も含むため、導入容量は大きくなります。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./scripts/run_experiment.sh --agent main.py --opponent starter --seeds 1 --save
```

`main.py` はv33のルールベース提出版です。PPOの学習済みモデルではありません。
過去の提出時のファイルを出典コメントとともに保持しています。

結果は `results/runs/` の実行別フォルダへ保存されます。
`money` は常に `[先手, 後手]` 順です。`--seat 1` の場合、自分の資金は2番目です。
両者の `status` が `DONE` であることを先に確認し、その後に資金差を比較してください。
`--save` はリプレイも保存します。実行にKaggle認証や提出は必要ありません。

## 比較の考え方

- 動作確認と勝率の検証を分ける。
- 同じ相手・seed・両手番で、候補と基準版を対応させる。
- 開発用seedと最終確認用seedを分離する。
- 平均資金だけでなく勝敗・下振れ・異常終了を確認する。
- 共有系統の相手を完全に独立した相手として数えない。
- ローカル勝率をKaggle順位や最終成績に換算しない。

PPO実験の条件・結果・限界は [RESULTS.md](RESULTS.md) を参照してください。
