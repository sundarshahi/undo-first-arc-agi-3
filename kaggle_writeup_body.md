**Track:** ARC-AGI-3 · **Code:** the attached Kaggle notebook (runs offline, numpy only; it reproduces every number below) · **Code-competition submission:** none (this is a paper-only entry, see Honest limits)

## What this is

ARC-AGI-3 scores an agent by how many actions it needs to beat each level of an unseen game, squared against a human baseline (RHAE). Every probe costs, and a probe that cannot be taken back can cost a reset and everything since the level started. The agent here makes **reversibility** the organising signal for exploration:

- it builds a transition graph of each level (states are hashes of the frame with a learned UI-counter mask applied; clicks are one per connected colour component);
- it classifies every tested transition as a **no-op**, **fatal** (game over), **reversible** (a known way back exists) or **not yet reversible**;
- it learns, per action type, how often each class occurs, learns inverse pairs (up/down, left/right) and whether `ACTION7` really undoes;
- it probes likely-undoable actions first and chooses the next frontier by travel cost plus the expected cost of getting back if the probe there cannot be undone.

No training, no model, no network, no game-specific code. It does **not** use the null-coordinate crash that Liew (2026) reports the public engine misclassifies as a win.

## Results (25 public games, 10,000 actions per game, 3 seeds)

| mode | score (official formula, 0-100) | levels of 183 (per seed) | games with a level | median actions per solved level | resets (3 seeds) |
|---|---|---|---|---|---|
| random (Kaggle starter) | 0.10 | 7 (7, 6, 7) | 6/25 | 2,088 | - |
| frontier, no reversibility | 0.37 | 19 (18, 19, 21) | 12/25 | 644 | 27,112 |
| **undo-first (ours)** | 0.38 | **22 (21, 22, 24)** | **15/25** | 812 | **20,048** |

- The reversibility terms add about one level in six on every seed and three more games entered (cn04 on every seed and tu93 on two are games the baseline never enters), with 26% fewer deliberate resets (20,048 vs 27,112; game-over resets about equal); by the end of a run 51% of the state-changing transitions tested had a known way back (43% for the baseline).
- They cost actions per solved level (812 vs 644), so the official score does not move. With hundreds of actions per level against human baselines of 6 to 578, RHAE is near zero for any explorer without a model of the goal; levels completed is the informative metric for this class of agent.
- Measured over all levels reached: 40% of tested transitions were no-ops, 30% had a known way back, 29% had none, 1% ended the game. 17 of 25 games yield inverse pairs; `ACTION7` is a verified undo in 5 games (ar25, bp35, lf52, sk48, su15) and did not return to the previous state in sb26.

Figures in the media gallery: (1) levels completed per game, (2) transition classes per game, (3) one step of the agent.

## Prior work in one paragraph

Rudakov et al. (arXiv 2512.24156) showed a training-free graph explorer is a strong ARC-AGI-3 baseline and lost their official run to an unhandled reset-inducing action, which transition classes prevent. Liew (arXiv 2605.25931) prefers `ACTION7` after exploratory moves and shows the public set falls to blind strategies. Rodionov (arXiv 2605.05138) reaches 15 fully solved games with an executable world model and GPT-5.5 at hours per game. Reversibility-aware RL (Grinsztajn et al. 2021; Eysenbach et al. 2018) learns what can be undone to keep a robot safe; this entry brings that idea to a model-free explorer with a transition graph instead of a learned estimator.

## Honest limits

Reversibility is one step deep; the UI mask is a row rule for the top and bottom four rows; clicks are one per colour component (at most 40), so levels needing a precise cell are out of reach; no goal inference or planning. The public set is weak as an intelligence test and no private-set number exists for this entry because it has no Kaggle code submission (the code competitions closed 2026-11-02).

## Reproduce

Notebook cells: install `arc-agi` (Kaggle wheels or PyPI), write the agent and runner, run the three modes, print the tables. The first online run caches the 25 public games (anonymous API key); everything after is offline. ~5 minutes per mode on 8 CPU workers.

Writing and code were produced with AI assistance (Claude Code); every number above was measured by the attached notebook. License: CC BY 4.0.
