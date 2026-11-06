"""Run an agent mode over the public ARC-AGI-3 games offline and score with the official formula.

usage: python run_local.py --mode undo --seed 0 --max-actions 3000 --games ls20,vc33 --out results/undo-s0.json
"""
import argparse, json, logging, os, sys, time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent"))
logging.disable(logging.INFO)

ENV_DIR = os.environ.get("ARC_ENV_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "games", "environment_files")


def level_score(baseline: int, actions: int) -> float:
    """arc_agi.scorecard.EnvironmentScoreCalculator: ((baseline/actions)^2*100, capped at 115)."""
    return min((baseline / actions) ** 2 * 100, 115.0) if actions > 0 else 0.0


def game_score(baselines: list[int], level_actions: list[int]) -> float:
    """Weighted by 1-indexed level number; unfinished levels score 0; capped by the finished-weight share."""
    tot = sum(range(1, len(baselines) + 1))
    done_w = sum(i + 1 for i in range(len(level_actions)))
    s = sum((i + 1) * level_score(b, a) for i, (b, a) in enumerate(zip(baselines, level_actions))) / tot
    return min(s, done_w / tot * 100)


def play(args):
    game_id, mode, seed, max_actions = args
    from arc_agi import Arcade, OperationMode
    from undo_agent import UndoAgent
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENV_DIR)
    info = next(e for e in arc.get_environments() if e.game_id.startswith(game_id))
    env = arc.make(info.game_id)
    fd = env.reset()
    agent = UndoAgent(mode=mode, seed=seed, max_actions=max_actions)
    t0 = time.time()
    n = 0
    while fd is not None and not agent.is_done([fd], fd):
        act = agent.choose_action([fd], fd)
        fd = env.step(act, data=agent.pending_data)
        n += 1
    levels = fd.levels_completed if fd is not None else agent.levels_done
    if fd is not None and levels > agent.levels_done:          # last action finished a level
        agent.level_actions.append(agent.actions_this_level)
    agent.close_level()
    base = info.baseline_actions or []
    return {
        "game": game_id, "mode": mode, "seed": seed, "actions": n, "levels": levels,
        "win_levels": fd.win_levels if fd is not None else len(base),
        "state": fd.state.name if fd is not None else "ERROR",
        "level_actions": agent.level_actions, "baseline_actions": base,
        "score": game_score(base, agent.level_actions), "seconds": round(time.time() - t0, 1),
        "stats": agent.stats, "undo_works": agent.undo_works, "inverse_pairs": len(agent.inverse) // 2,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="undo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-actions", type=int, default=10000)
    p.add_argument("--games", default="")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="")
    a = p.parse_args()
    games = a.games.split(",") if a.games else sorted(d for d in os.listdir(ENV_DIR) if not d.startswith("."))
    jobs = [(g, a.mode, a.seed, a.max_actions) for g in games]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(play, jobs))
    total = sum(r["score"] for r in rows) / len(rows)
    for r in rows:
        print(f"{r['game']:6s} levels {r['levels']}/{r['win_levels']}  score {r['score']:6.2f}  actions {r['actions']:5d}  "
              f"per-level {r['level_actions']}  {r['seconds']}s  inv={r['inverse_pairs']} undo={r['undo_works']} {r['stats']}")
    print(f"TOTAL {a.mode} seed {a.seed}: score {total:.2f}  levels {sum(r['levels'] for r in rows)}/{sum(r['win_levels'] for r in rows)}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"mode": a.mode, "seed": a.seed, "max_actions": a.max_actions, "total": total, "games": rows},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
