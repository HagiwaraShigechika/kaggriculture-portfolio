# Attribution and third-party notices

This project combines locally developed learning/evaluation code with a preserved
Kaggriculture competition agent derived from public implementations. It does not
claim authorship of the entire competition agent, public routes, or game engine.

## Current competition agent (`main.py`)

The preserved v33 artifact has SHA-256
`be5c4026098d24a961d36662b9a751976d8c04744e5ee11315a54562343d1680`.
It retains its upstream source comments and embedded Apache-2.0 license text.
The complete inherited release notice is preserved in
[`licenses/main.NOTICE.txt`](licenses/main.NOTICE.txt); the license is
[`LICENSE.txt`](LICENSE.txt). Historical path references inside the preserved
notice refer to the original development tree and are not an inventory of this
curated release.

Principal upstream sources identified by the retained source and release notices:

| Component / lineage | Upstream attribution and source |
| --- | --- |
| Farming routes and public state-router lineage | Thomas Tschinkel, [public state router](https://www.kaggle.com/code/thomastschinkel/kaggriculture-93-8-win-rate-public-state-router?scriptVersionId=347936183) and [The 2945 Farm](https://www.kaggle.com/code/thomastschinkel/the-2945-farm-96-vs-the-top-10-public-bots) |
| Shop routing and action data | Yusuke Hayashi (yhay81), [Shop0909](https://www.kaggle.com/code/yhay81/shop-router-0909), with aurax7 and other inherited contributors |
| Production and recovery lineage | Nathan Jacob, [Pipe18 Six Layers](https://www.kaggle.com/code/nathanjacob/kaggriculture-pipe18-six-layers); Ahmed Berat Ozer, [V56](https://www.kaggle.com/code/ahmedberatozer/kaggriculture-v56-smarter-seeds-and-fertilizer); Dmitrii Gluzdov and contributors named in the full notice |
| Sale timing and market mechanisms | tetsutani, [Demand-Preserving Turn Sale Timing](https://www.kaggle.com/code/tetsutani/demand-preserving-turn-sale-timing); shiiin9, [Your Market List Is an Order Book](https://www.kaggle.com/code/shiiin9/your-market-list-is-an-order-book); abhinav0370, [cha22](https://www.kaggle.com/code/abhinav0370/cha22-agent) |
| Shadow market-order model | lynnsakurai, [Farmer John and the Idle Seller](https://www.kaggle.com/code/lynnsakurai/farmer-john-and-the-idle-seller) |
| Extracted deterministic game rules | Kaggle and kaggle-environments contributors, [kaggle-environments](https://github.com/Kaggle/kaggle-environments), version 1.32.7 |

This table is a reading guide, not a replacement for the full inherited notices.
Other contributors, including prvsiyan, destbreso, Rayk Kretzschmar, Seyit Kaan
Gunes, and the authors credited by upstream sources, remain attributed there.
No upstream endorsement is implied.

The local v33 additions include production/input safeguards, deterministic
ordering, and adaptive sale-order selection using observed cash feedback.
Those changes are separate from the PPO learning implementation.

## PPO learning core (`ppo/`)

The learning core is extracted from this repository's v25 research branch:
`strategy_v25/model.py` and the rollout-stacking/PPO-update functions in
`strategy_v25/train.py`. The public example isolates network training and portable
inference from the historical game controllers, opponent collection, server
configuration, and replay corpus. Extraction and import-path changes are described
in the PPO documentation. NumPy and PyTorch are external dependencies and retain
their respective licenses; their implementation code is not vendored here.

## Data provenance

The inherited agent includes public route/forecast data within its source. An
upstream historical notice states that the original episode-level manifest and
replay-author identities were not supplied, and that notebook licensing does not
independently establish that underlying data's complete rights chain. That notice
is retained without inventing missing provenance. This release does not claim to
reconstruct the original collection/training process or to have independently
verified every upstream replay's rights. Raw downloaded replay datasets and
third-party notebook/output collections are excluded from this curated release.
