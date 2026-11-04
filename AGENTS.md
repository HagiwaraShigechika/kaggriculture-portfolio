# Development guide

- This is the curated public portfolio edition. Keep the README brief; put algorithm details and experimental limitations in docs/.
- Preserve source attribution and LICENSE/NOTICE files. Do not describe the rule-based main.py as a PPO policy or the inherited agent as wholly original work.
- ppo/ contains the executable learning core; experiments/v25/ preserves historical integration source with documented external dependencies.
- Verify PPO changes with `python -m unittest discover -s tests -v` and `python -m ppo.smoke`. Synthetic smoke tests do not establish game performance.
- Separate past experiment evidence from newly executed tests. Report evaluation seeds, opponents, seats, and scope with performance claims.
- Do not commit credentials, private server configuration, personal paths, raw replays, model checkpoints, or generated submissions.
- Do not submit to Kaggle or send messages without an explicit request. Review the diff before committing or pushing.
