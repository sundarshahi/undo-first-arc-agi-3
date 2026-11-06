"""Build the paper figures and numbers.tex from results/*.json (median over seeds per game)."""
import glob, json, os, sys, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "paper", "figures"); os.makedirs(OUT, exist_ok=True)
MODES = ["random", "frontier", "undo"]
LABEL = {"random": "random", "frontier": "frontier (no reversibility)", "undo": "undo-first (ours)"}
# dataviz reference palette, categorical slots 1-4, validated for the light surface
C = {"random": "#2a78d6", "frontier": "#eb6834", "undo": "#1baf7a", "yellow": "#eda100"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})

runs = {}  # mode -> list of result dicts (one per seed)
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
    d = json.load(open(f)); runs.setdefault(d["mode"], []).append(d)
games = [g["game"] for g in runs["undo"][0]["games"]]

def per_game(mode, key):
    """median over seeds of games[i][key]"""
    return [st.median(r["games"][i][key] for r in runs[mode]) for i in range(len(games))]

win = [g["win_levels"] for g in runs["undo"][0]["games"]]
levels = {m: per_game(m, "levels") for m in MODES}
score_total = {m: st.median(r["total"] for r in runs[m]) for m in MODES}
levels_total = {m: st.median(sum(g["levels"] for g in r["games"]) for r in runs[m]) for m in MODES}
games_any = {m: st.median(sum(g["levels"] > 0 for g in r["games"]) for r in runs[m]) for m in MODES}
def actions_to_first(m):
    xs = [g["level_actions"][0] for r in runs[m] for g in r["games"] if g["level_actions"]]
    return st.median(xs) if xs else float("nan")
first = {m: actions_to_first(m) for m in MODES}

# ---- Figure 1: levels completed per game, grouped bars (median over seeds) ----
fig, ax = plt.subplots(figsize=(10, 3.4))
x = range(len(games)); w = 0.27
for j, m in enumerate(MODES):
    ax.bar([i + (j - 1) * w for i in x], levels[m], width=w - 0.04, color=C[m], label=LABEL[m], linewidth=0)
ax.plot(list(x), win, linestyle="none", marker="_", color=MUTED, markersize=9, markeredgewidth=1.2, label="levels in game")
ax.set_xticks(list(x)); ax.set_xticklabels(games, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("levels completed (median of 3 seeds)"); ax.set_ylim(0, max(win) + 0.5)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.38), fontsize=8)
budget = runs["undo"][0]["max_actions"]
ax.set_title(f"Levels completed per public game, {budget:,} actions per game", loc="left", color=INK, fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_levels.png"), dpi=200); fig.savefig(os.path.join(OUT, "fig1_levels.pdf"))

# ---- Figure 2: what the undo-first agent learned: edge classes per game (stacked, share) ----
keys = ["reversible_edges", "irreversible_edges", "noop_edges", "fatal_edges"]
names = {"reversible_edges": "reversible (known way back)", "irreversible_edges": "no known way back",
         "noop_edges": "no-op", "fatal_edges": "game over"}
col = {"reversible_edges": C["undo"], "irreversible_edges": C["frontier"], "noop_edges": C["random"], "fatal_edges": C["yellow"]}
stats = [{k: st.median(r["games"][i]["stats"][k] for r in runs["undo"]) for k in keys} for i in range(len(games))]
tot = [max(1, sum(s.values())) for s in stats]
fig, ax = plt.subplots(figsize=(10, 3.0))
bottom = [0.0] * len(games)
for k in keys:
    vals = [100 * s[k] / t for s, t in zip(stats, tot)]
    ax.bar(list(x), vals, bottom=bottom, color=col[k], width=0.7, label=names[k], linewidth=0.8, edgecolor="#fcfcfb")
    bottom = [b + v for b, v in zip(bottom, vals)]
ax.set_xticks(list(x)); ax.set_xticklabels(games, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("share of tested transitions (%)"); ax.set_ylim(0, 100)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.38), fontsize=8)
ax.set_title("Transition classes learned by the undo-first agent (all levels reached, median of 3 seeds)", loc="left", color=INK, fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_edges.png"), dpi=200); fig.savefig(os.path.join(OUT, "fig2_edges.pdf"))

# ---- Figure 3: method schematic ----
fig, ax = plt.subplots(figsize=(10, 1.9)); ax.axis("off"); ax.grid(False)
def box(xc, text, color, w=1.9):
    ax.add_patch(FancyBboxPatch((xc - w / 2, 0.08), w, 0.84, boxstyle="round,pad=0.01,rounding_size=0.03", fc="#fcfcfb", ec=color, lw=1.4))
    ax.text(xc, 0.5, text, ha="center", va="center", fontsize=7.4, color=INK, linespacing=1.35)
steps = [("frame 64x64\nmask learned UI rows\nhash = state", C["random"]),
         ("actions at a state\nkeys, or one click per\ncolour component", C["random"]),
         ("transition graph per level\nno-op / fatal / successor;\nreversible once a way back\nis known", C["frontier"]),
         ("priors per action type\nresponsiveness, reversibility\nrate, winning-path types;\ninverse pairs, undo check", C["undo"]),
         ("policy\nprobe: win-path, undoable,\nresponsive first; route to the\ncheapest frontier incl.\nexpected cost of getting back", C["undo"])]
xs = [1.0, 3.0, 5.0, 7.0, 9.0]
for xc, (t, c) in zip(xs, steps): box(xc, t, c)
for a, b in zip(xs, xs[1:]):
    ax.add_patch(FancyArrowPatch((a + 0.96, 0.5), (b - 0.96, 0.5), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1))
ax.set_xlim(0, 10); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_method.png"), dpi=200); fig.savefig(os.path.join(OUT, "fig3_method.pdf"))

# ---- numbers.tex + results.md ----
edge_tot = {k: sum(s[k] for s in stats) for k in keys}; e_all = sum(edge_tot.values())
undo_games = sorted({g["game"] for r in runs["undo"] for g in r["games"] if g["undo_works"]})
inv_games = sorted({g["game"] for r in runs["undo"] for g in r["games"] if g["inverse_pairs"]})
n_seeds = len(runs["undo"])
with open(os.path.join(ROOT, "paper", "numbers.tex"), "w") as f:
    for m in MODES:
        f.write(f"\\newcommand{{\\{m}Score}}{{{score_total[m]:.2f}}}\n\\newcommand{{\\{m}Levels}}{{{int(levels_total[m])}}}\n"
                f"\\newcommand{{\\{m}Games}}{{{int(games_any[m])}}}\n\\newcommand{{\\{m}First}}{{{first[m]:.0f}}}\n")
    f.write(f"\\newcommand{{\\totalLevels}}{{{sum(win)}}}\n\\newcommand{{\\nSeeds}}{{{n_seeds}}}\n")
    for k in keys:
        f.write(f"\\newcommand{{\\share{k.split('_')[0]}}}{{{100 * edge_tot[k] / e_all:.0f}}}\n")
    f.write(f"\\newcommand{{\\undoVerified}}{{{len(undo_games)}}}\n\\newcommand{{\\invGames}}{{{len(inv_games)}}}\n")
    f.write(f"\\newcommand{{\\nStates}}{{{int(sum(st.median(r['games'][i]['stats']['states'] for r in runs['undo']) for i in range(len(games))))}}}\n")
with open(os.path.join(RES, "summary.md"), "w") as f:
    f.write("| mode | score (official formula, 25 games) | levels | games with >=1 level | median actions to first level |\n|---|---|---|---|---|\n")
    for m in MODES:
        f.write(f"| {LABEL[m]} | {score_total[m]:.2f} | {int(levels_total[m])}/{sum(win)} | {int(games_any[m])}/25 | {first[m]:.0f} |\n")
    f.write(f"\nedge classes (undo mode, all games, median of seeds): " + ", ".join(f"{names[k]} {100 * edge_tot[k] / e_all:.0f}%" for k in keys) + "\n")
    f.write(f"undo verified in: {undo_games}\ninverse pairs learned in: {inv_games}\n\n| game | levels | " + " | ".join(MODES) + " |\n|---|---|---|---|---|\n")
    for i, g in enumerate(games):
        f.write(f"| {g} | {win[i]} | " + " | ".join(f"{levels[m][i]:g}" for m in MODES) + " |\n")
print(open(os.path.join(RES, "summary.md")).read())
