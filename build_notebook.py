"""Build notebooks/undo_agent.ipynb: a self-contained, offline-runnable notebook that inlines the agent and
the local runner, plays the public games in all three modes, and prints the official-formula score table.
The last cell mirrors the official ARC-AGI-3 Kaggle starter's competition-rerun hook (not executable here)."""
import nbformat as nbf
from pathlib import Path

root = Path(__file__).parent
agent_src = (root / "agent" / "undo_agent.py").read_text()
runner_src = (root / "run_local.py").read_text()

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []
cells.append(nbf.v4.new_markdown_cell("""# Undo-first exploration for ARC-AGI-3

**Reversibility-aware frontier exploration**: a training-free, LLM-free agent that builds a transition graph
of each level, learns which actions can be undone (inverse pairs, the `ACTION7` undo, a learned UI-counter mask),
tests undoable actions first and routes to unexplored states through edges it knows how to reverse.

This notebook is the code companion of the ARC Prize 2026 paper-track writeup *Undo-first exploration:
reversibility-aware frontier search for ARC-AGI-3*. It runs fully offline on the 25 public games with the
official `arc-agi` toolkit (Python 3.12) and scores with the toolkit's own formula
(`EnvironmentScoreCalculator`: level score = min((human/agent)^2 x 100, 115), level-index-weighted game score).

Three modes are compared with identical code paths:

| mode | what it does |
|---|---|
| `random` | uniform random over the available actions (the Kaggle starter baseline) |
| `frontier` | graph exploration to the nearest untested state-action pair (prior-work style) |
| `undo` | frontier + reversibility ordering, inverse-pair learning, undo verification, irreversible-edge path penalty |

No LLM, no network, no game-specific code, no exploitation of the null-coordinate crash reported by Liew (2026)."""))

cells.append(nbf.v4.new_code_cell("""import os, sys, glob, subprocess, tempfile
KAGGLE_WHEELS = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
KAGGLE_ENVS = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"
try:
    import arc_agi, arcengine  # noqa
    print("arc-agi", arc_agi.__version__ if hasattr(arc_agi, "__version__") else "present")
except ImportError:
    if os.path.isdir(KAGGLE_WHEELS):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", KAGGLE_WHEELS, "arc-agi", "python-dotenv"], check=True)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "arc-agi"], check=True)
# where the 25 public game files live: the Kaggle competition bundle, ../games/environment_files locally,
# or (anywhere with internet) a one-time download through the toolkit's anonymous API key
if os.path.isdir(KAGGLE_ENVS):
    os.environ["ARC_ENV_DIR"] = KAGGLE_ENVS
elif os.path.isdir("../games/environment_files"):
    os.environ["ARC_ENV_DIR"] = os.path.abspath("../games/environment_files")
else:
    os.environ["ARC_ENV_DIR"] = os.path.join(tempfile.gettempdir(), "arc_environment_files")  # outside the notebook's outputs
    if not os.path.isdir(os.environ["ARC_ENV_DIR"]):
        from arc_agi import Arcade, OperationMode
        arc = Arcade(operation_mode=OperationMode.NORMAL, environments_dir=os.environ["ARC_ENV_DIR"])
        for e in arc.get_environments():
            arc.make(e.game_id)          # downloads and caches the game source
games = [d for d in os.listdir(os.environ["ARC_ENV_DIR"]) if not d.startswith(".")]
assert len(games) == 25, games
print("games:", len(games), "in", os.environ["ARC_ENV_DIR"])"""))

cells.append(nbf.v4.new_markdown_cell("## The agent (single file, numpy only)"))
cells.append(nbf.v4.new_code_cell("%%writefile undo_agent.py\n" + agent_src))
cells.append(nbf.v4.new_markdown_cell("## Local runner and the official score formula"))
cells.append(nbf.v4.new_code_cell("%%writefile run_local.py\n" + runner_src.replace(
    'os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent")', 'os.path.dirname(os.path.abspath(__file__))')))

cells.append(nbf.v4.new_markdown_cell("""## Play all 25 public games in the three modes

Budget: 10,000 actions per game, seed 0 (the paper reports the median of seeds 0-2). Each mode takes a few minutes on
8 CPU workers; the engine runs at roughly 500-2000 steps per second."""))
for mode in ("random", "frontier", "undo"):
    cells.append(nbf.v4.new_code_cell(f"!python run_local.py --mode {mode} --seed 0 --max-actions 10000 --workers 8 --out results/{mode}-s0.json | tail -3"))

cells.append(nbf.v4.new_markdown_cell("## Results table (official formula, 25 public games)"))
cells.append(nbf.v4.new_code_cell("""import json
rows = {m: json.load(open(f"results/{m}-s0.json")) for m in ("random", "frontier", "undo")}
games = [g["game"] for g in rows["undo"]["games"]]
print(f"{'game':6s} " + " ".join(f"{m:>16s}" for m in rows))
for i, g in enumerate(games):
    print(f"{g:6s} " + " ".join(f"{r['games'][i]['levels']}/{r['games'][i]['win_levels']} lv {r['games'][i]['score']:6.2f}".rjust(16) for r in rows.values()))
print("-" * 60)
for m, r in rows.items():
    print(f"{m:8s} score {r['total']:6.2f}   levels {sum(g['levels'] for g in r['games'])}/{sum(g['win_levels'] for g in r['games'])}   "
          f"games with >=1 level {sum(g['levels'] > 0 for g in r['games'])}/25")"""))

cells.append(nbf.v4.new_markdown_cell("""## What the agent learned about reversibility

Final classification of every tested transition across all levels reached (undo mode, seed 0)."""))
cells.append(nbf.v4.new_code_cell("""tot = {}
for g in rows["undo"]["games"]:
    for k, v in g["stats"].items():
        tot[k] = tot.get(k, 0) + v
edges = sum(tot[k] for k in ("reversible_edges", "irreversible_edges", "noop_edges", "fatal_edges"))
for k in ("reversible_edges", "irreversible_edges", "noop_edges", "fatal_edges"):
    print(f"{k:20s} {tot[k]:7d}  {100 * tot[k] / edges:5.1f}%")
print("states visited", tot["states"], " game overs", tot["game_overs"], " resets", tot["resets"])
print("games where ACTION7 was verified as an undo:", [g["game"] for g in rows["undo"]["games"] if g["undo_works"]])
print("games with learned inverse pairs:", {g["game"]: g["inverse_pairs"] for g in rows["undo"]["games"] if g["inverse_pairs"]})"""))

cells.append(nbf.v4.new_markdown_cell("""## Kaggle competition rerun hook

This mirrors the official [ARC-AGI-3-Kaggle-Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter) build:
during a competition rerun the framework's gateway sidecar drives the agent and produces `submission.parquet`.
It is a no-op outside Kaggle. The `UndoAgent` class implements the starter's `is_done` / `choose_action` interface."""))
cells.append(nbf.v4.new_code_cell("""import os, shutil, subprocess, sys, pandas as pd
if os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
    fw = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents"
    shutil.copytree(fw, "/kaggle/working/framework", dirs_exist_ok=True)
    shutil.copy("undo_agent.py", "/kaggle/working/framework/agents/undo_agent.py")
    # register: agents/__init__.py must export the class under the name passed to --agent
    with open("/kaggle/working/framework/agents/__init__.py", "a") as f:
        f.write("\\nfrom .undo_agent import UndoAgent as undoagent\\n")
    subprocess.run([sys.executable, "main.py", "--agent", "undoagent"], cwd="/kaggle/working/framework", check=True)
elif not os.path.exists("/kaggle/working/submission.parquet") and os.path.isdir("/kaggle/working"):
    pd.DataFrame(columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet("/kaggle/working/submission.parquet")"""))

nb["cells"] = cells
out = root / "notebooks" / "undo_agent.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
