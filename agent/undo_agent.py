"""Reversibility-aware frontier exploration for ARC-AGI-3 (training-free, LLM-free, numpy only).

Modes (same code path, flags differ):
  random   : uniform random over the available actions (the Kaggle starter baseline)
  frontier : transition graph per level; untested actions first (ordered by a learned responsiveness
             prior and by action types seen on earlier winning paths); Dijkstra to the nearest frontier
  undo     : frontier + reversibility: a learned per-action-type reversibility rate orders probes
             (likely-undoable first), frontier choice adds the expected cost of getting back
             (RESET + replay) for a probe that cannot be undone, routes prefer edges with a known
             way back, and ACTION7 is verified as an undo early and then trusted.
"""
from __future__ import annotations

import hashlib
import heapq
import random
from collections import deque
from typing import Any

import numpy as np

try:
    from arcengine import GameAction, GameState
except ImportError:  # pragma: no cover - the Kaggle framework ships arcengine
    GameAction = GameState = None  # type: ignore

NOOP, OVER = "noop", "over"
UNDO_ID, CLICK_ID, RESET_ID = 7, 6, 0
EDGE_ROWS = 4            # UI strips (step counter, level indicator) live in the top/bottom rows
IRREVERSIBLE_PENALTY = 1.0


def frame_key(fd, mask: np.ndarray | None = None) -> tuple[bytes, np.ndarray] | None:
    """Hash of the last rendered frame with the learned UI cells zeroed; None when no frame was sent."""
    fr = getattr(fd, "frame", None)
    if not fr:
        return None
    arr = np.asarray(fr[-1], dtype=np.uint8)
    view = arr if mask is None else np.where(mask, 0, arr)
    return hashlib.blake2b(view.tobytes(), digest_size=8).digest(), arr


def click_targets(arr: np.ndarray, cap: int = 40) -> list[tuple[int, int, int]]:
    """One click target (x, y, colour) per 4-connected same-colour component, largest first.
    The component covering >40% of the grid is background. ponytail: pure-Python flood fill,
    a few ms per new state; vectorise if profiling says so."""
    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, tuple[int, int, int]]] = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0, x0]:
                continue
            c = int(arr[y0, x0])
            stack = [(y0, x0)]
            seen[y0, x0] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and arr[ny, nx] == c:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            n = len(cells)
            if n > 0.4 * h * w:
                continue
            cy = sum(p[0] for p in cells) / n
            cx = sum(p[1] for p in cells) / n
            y, x = min(cells, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
            comps.append((n, (int(x), int(y), c)))
    comps.sort(key=lambda t: -t[0])
    return [xyc for _, xyc in comps[:cap]]


def atype(a: Any) -> Any:
    """Action type used for the learned priors: the action id, or ('c', colour) for a click."""
    return ("c", a[3]) if isinstance(a, tuple) else a


class UndoAgent:
    """Duck-typed on the ARC-AGI-3 agent interface: choose_action(frames, latest_frame) -> GameAction."""

    MAX_ACTIONS = 10000

    def __init__(self, mode: str = "undo", seed: int = 0, max_actions: int | None = None):
        assert mode in ("random", "frontier", "undo")
        self.mode = mode
        self.rng = random.Random(seed)
        if max_actions:
            self.MAX_ACTIONS = max_actions
        # ---- game-level priors (survive level changes) ----
        self.tested: dict[Any, int] = {}     # action type -> transitions tested
        self.changed: dict[Any, int] = {}    # action type -> transitions that changed the state
        self.reversed_: dict[Any, int] = {}  # action type -> state-changing transitions with a known way back
        self.inverse: dict[int, int] = {}    # action id -> learned inverse id (keyboard games)
        self.undo_works: bool | None = None  # ACTION7 verified to return to the previous state
        self.win_types: set[Any] = set()     # action types on earlier levels' winning paths
        self.mask = np.zeros((64, 64), dtype=bool)
        self.prev_arr: np.ndarray | None = None
        # ---- bookkeeping ----
        self.levels_done = 0
        self.level_actions: list[int] = []   # actions spent on each completed level (RESETs included)
        self.actions_this_level = 0
        self.action_counter = 0
        self.pending_data: dict[str, Any] = {}
        self.stats = {"reversible_edges": 0, "irreversible_edges": 0, "noop_edges": 0, "fatal_edges": 0,
                      "states": 0, "game_overs": 0, "resets": 0, "plans": 0}
        self.nodes: dict[bytes, dict[Any, Any]] = {}
        self.rev_edges: set[tuple[bytes, Any]] = set()
        self._new_level()

    # ----- per-level state ------------------------------------------------------
    def close_level(self) -> None:
        """Fold the finished level's graph into the run statistics (final reversibility classes)."""
        self.stats["states"] += len(self.nodes)
        for s, node in self.nodes.items():
            for a, r in node.items():
                if r is None:
                    continue
                k = "noop_edges" if r == NOOP else "fatal_edges" if r == OVER else \
                    "reversible_edges" if (s, a) in self.rev_edges else "irreversible_edges"
                self.stats[k] += 1

    def _new_level(self) -> None:
        self.close_level()
        self.nodes = {}
        self.rev_edges = set()
        self.start: bytes | None = None
        self.prev: tuple[bytes, Any] | None = None          # (state, action) just taken
        self.prev_change: tuple[bytes, Any] | None = None   # last transition that changed the state
        self.plan: deque[tuple[bytes, Any]] = deque()       # (state to be in, action to take) ...

    def _action_keys(self, fd, arr: np.ndarray) -> list[Any]:
        keys: list[Any] = []
        for aid in fd.available_actions:
            if aid == RESET_ID:
                continue
            if aid == CLICK_ID:
                keys.extend(("c",) + t for t in click_targets(arr))
            else:
                keys.append(aid)
        return keys

    # ----- interface --------------------------------------------------------------
    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state == GameState.WIN or self.action_counter >= self.MAX_ACTIONS

    def choose_action(self, frames, latest):
        self.action_counter += 1
        self.actions_this_level += 1
        if latest.levels_completed > self.levels_done:          # the previous action finished a level
            self.level_actions.append(self.actions_this_level - 1)
            self.levels_done = latest.levels_completed
            self.actions_this_level = 1
            self._remember_winning_path()
            self._new_level()

        if latest.state == GameState.GAME_OVER:
            if self.prev:
                self.nodes[self.prev[0]][self.prev[1]] = OVER
            self.stats["game_overs"] += 1
            self.prev = self.prev_change = None
            self.plan.clear()
            return self._emit(RESET_ID)
        if latest.state == GameState.NOT_PLAYED:
            return self._emit(RESET_ID)

        fk = frame_key(latest, self.mask)
        if fk is None:
            return self._emit(RESET_ID)
        key, arr = fk
        if self._learn_mask(arr):                      # mask grew: old hashes are stale, restart the level graph
            key = frame_key(latest, self.mask)[0]
            self._new_level()
        self.prev_arr = arr
        if key not in self.nodes:
            self.nodes[key] = {a: None for a in self._action_keys(latest, arr)}
        if self.start is None:
            self.start = key

        self._record(key)
        akey = self._pick(key)
        self.prev = (key, akey) if akey != RESET_ID else None
        if akey == RESET_ID:
            self.prev_change = None
            self.plan.clear()
        return self._emit(akey)

    # ----- learning ---------------------------------------------------------------
    def _learn_mask(self, arr: np.ndarray) -> bool:
        """An edge row (top/bottom EDGE_ROWS) that changes by <= 2 cells in one transition is a UI
        counter bar, not game state: mask the whole row. ponytail: row-level, edge-only; learn
        arbitrary regions if a private game draws its counter elsewhere."""
        if self.prev_arr is None or self.prev_arr.shape != arr.shape:
            return False
        h = arr.shape[0]
        changed = np.argwhere(self.prev_arr != arr)
        grew = False
        for r in np.unique(changed[:, 0]):
            if (r < EDGE_ROWS or r >= h - EDGE_ROWS) and not self.mask[r].all() and (changed[:, 0] == r).sum() <= 2:
                self.mask[r] = True
                grew = True
        return grew

    def _record(self, key: bytes) -> None:
        """Classify the transition that led here and update the per-type priors."""
        if not self.prev:
            return
        s, a = self.prev
        t = atype(a)
        self.tested[t] = self.tested.get(t, 0) + 1
        if key == s:
            self.nodes[s][a] = NOOP
            if a == UNDO_ID and self.prev_change and self.undo_works is None:
                self.undo_works = False               # had something to undo, nothing happened
            return
        self.nodes[s][a] = key
        self.changed[t] = self.changed.get(t, 0) + 1
        # every known edge key -> s is now reversible (we just found a way back from s to key... no:
        # we found s -> key; so edges key -> s that were already known gain a way back)
        for b, r in self.nodes.get(key, {}).items():
            if r == s and (key, b) not in self.rev_edges:
                self.rev_edges.add((key, b))
                tb = atype(b)
                self.reversed_[tb] = self.reversed_.get(tb, 0) + 1
        if s in self.nodes.get(key, {}).values() and (s, a) not in self.rev_edges:
            self.rev_edges.add((s, a))
            self.reversed_[t] = self.reversed_.get(t, 0) + 1
        if self.prev_change and self.prev_change[0] == key:     # s_prev --a0--> s --a--> s_prev
            a0 = self.prev_change[1]
            if a == UNDO_ID:
                self.undo_works = True
            elif isinstance(a, int) and isinstance(a0, int) and UNDO_ID not in (a, a0):
                self.inverse[a0], self.inverse[a] = a, a0
        elif a == UNDO_ID and self.prev_change and self.undo_works is None:
            self.undo_works = False                   # undo went somewhere else
        self.prev_change = (s, a)

    def _remember_winning_path(self) -> None:
        """Action types on the shortest known path start -> winning transition get priority next level."""
        if not self.prev or self.start is None:
            return
        goal, last = self.prev
        self.win_types.add(atype(last))
        parent: dict[bytes, tuple[bytes, Any]] = {}
        q = deque([self.start])
        seen = {self.start}
        while q:
            s = q.popleft()
            if s == goal:
                break
            for a, r in self.nodes.get(s, {}).items():
                if r in (None, NOOP, OVER) or r in seen:
                    continue
                seen.add(r)
                parent[r] = (s, a)
                q.append(r)
        s = goal
        while s in parent:
            s, a = parent[s]
            self.win_types.add(atype(a))

    # ----- priors ------------------------------------------------------------------
    def p_change(self, a: Any) -> float:
        t = atype(a)
        return (self.changed.get(t, 0) + 1) / (self.tested.get(t, 0) + 2)

    def p_rev(self, a: Any) -> float:
        if a == UNDO_ID:
            return 0.0 if self.undo_works is False else 1.0
        if self.undo_works:
            return 1.0
        if isinstance(a, int) and a in self.inverse:
            return 1.0
        t = atype(a)
        return (self.reversed_.get(t, 0) + 1) / (self.changed.get(t, 0) + 2)

    def _order_key(self, a: Any) -> tuple:
        k: list[Any] = [0 if atype(a) in self.win_types else 1]
        if self.mode == "undo":
            k.append(-round(self.p_rev(a), 2))
        k.append(-round(self.p_change(a), 2))
        k.append(self.rng.random())
        return tuple(k)

    # ----- policy ------------------------------------------------------------------
    def _pick(self, key: bytes) -> Any:
        node = self.nodes[key]
        if self.mode == "random":
            aids = list({k if isinstance(k, int) else CLICK_ID for k in node})
            if not aids:
                return RESET_ID
            a = self.rng.choice(aids)
            return ("c", self.rng.randrange(64), self.rng.randrange(64), -1) if a == CLICK_ID else a
        untested = [a for a, r in node.items() if r is None]
        if untested:
            self.plan.clear()
            return min(untested, key=self._order_key)
        if self.plan and self.plan[0][0] == key:        # keep following the cached route
            return self.plan.popleft()[1]
        self.plan.clear()
        route = self._route_to_frontier(key)
        if route:
            self.stats["plans"] += 1
            self.plan = deque(route)
            return self.plan.popleft()[1]
        if key != self.start and self.start in self.nodes and any(r is None for r in self.nodes[self.start].values()):
            self.stats["resets"] += 1
            return RESET_ID
        live = [a for a, r in node.items() if r not in (NOOP, OVER, None)]   # graph exhausted: keep the game alive
        return self.rng.choice(live) if live else RESET_ID

    def _dijkstra(self, src: bytes) -> tuple[dict[bytes, float], dict[bytes, tuple[bytes, Any]]]:
        dist = {src: 0.0}
        parent: dict[bytes, tuple[bytes, Any]] = {}
        heap = [(0.0, 0, src)]
        tie = 0
        while heap:
            d, _, s = heapq.heappop(heap)
            if d > dist.get(s, float("inf")):
                continue
            for a, r in self.nodes.get(s, {}).items():
                if r in (None, NOOP, OVER):
                    continue
                w = 1.0 + (IRREVERSIBLE_PENALTY if self.mode == "undo" and (s, a) not in self.rev_edges else 0.0)
                nd = d + w
                if nd < dist.get(r, float("inf")):
                    dist[r] = nd
                    parent[r] = (s, a)
                    tie += 1
                    heapq.heappush(heap, (nd, tie, r))
        return dist, parent

    def _route_to_frontier(self, src: bytes) -> list[tuple[bytes, Any]] | None:
        """Cheapest frontier: travel cost, plus (undo mode) the expected cost of getting back after the
        probe we would make there: (1 - p_rev) x (RESET + replay from the level start)."""
        dist, parent = self._dijkstra(src)
        from_start = self._dijkstra(self.start)[0] if (self.mode == "undo" and self.start is not None) else {}
        best, best_cost = None, float("inf")
        for t, d in dist.items():
            if t == src:
                continue
            untested = [a for a, r in self.nodes.get(t, {}).items() if r is None]
            if not untested:
                continue
            cost = d
            if self.mode == "undo":
                a = min(untested, key=self._order_key)
                cost += (1.0 - self.p_rev(a)) * (1.0 + from_start.get(t, d))
            if cost < best_cost:
                best, best_cost = t, cost
        if best is None:
            return None
        route: list[tuple[bytes, Any]] = []
        s = best
        while s != src:
            ps, a = parent[s]
            route.append((ps, a))
            s = ps
        route.reverse()
        return route

    def _emit(self, akey: Any):
        if isinstance(akey, tuple):
            act = GameAction.ACTION6
            self.pending_data = {"x": akey[1], "y": akey[2]}
        else:
            act = GameAction.from_id(akey)
            self.pending_data = {}
        act.set_data(self.pending_data) if act.is_complex() else act.set_data({})
        return act
