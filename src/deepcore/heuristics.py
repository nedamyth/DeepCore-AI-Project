"""
heuristics.py
Student heuristic starter file for DeepCore:
  - h_naive_manhattan : provided baseline
  - h1_basic          : required student implementation
  - h2_advanced       : required student implementation
"""

from typing import Dict, List, Tuple
from collections import deque
import weakref
from .environment import DeepCoreEnv, DIRECTION_DELTAS
from .state import DeepCoreState, Pos

# Cache wall-aware shortest path distances per environment instance
# Keyed weakly by the env object: id(env) can be reused by a new env after GC,
# which would silently return another map's distances.
_WALL_DIST_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _get_wall_aware_distances(env: DeepCoreEnv) -> Dict[Tuple[Pos, Pos], int]:
    """Precomputes all-pairs shortest path steps ignoring dynamic obstacles (only walls # block)."""
    env_id = env
    if env_id in _WALL_DIST_CACHE:
        return _WALL_DIST_CACHE[env_id]

    walkable = [
        (r, c)
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in env.walls
    ]
    dist_map: Dict[Tuple[Pos, Pos], int] = {}

    for src in walkable:
        q = deque([(src, 0)])
        visited = {src}
        while q:
            curr, d = q.popleft()
            dist_map[(src, curr)] = d
            cr, cc = curr
            for dr, dc in DIRECTION_DELTAS.values():
                nxt = (cr + dr, cc + dc)
                if env.in_bounds(nxt) and nxt not in env.walls and nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, d + 1))

    _WALL_DIST_CACHE[env_id] = dist_map
    return dist_map


def _prim_mst_cost(nodes: List[Pos], dist_fn) -> float:
    """Computes Minimum Spanning Tree weight over `nodes` using `dist_fn(u, v)`."""
    if len(nodes) <= 1:
        return 0.0
    in_mst = {nodes[0]}
    total_weight = 0.0

    while len(in_mst) < len(nodes):
        best_edge = float("inf")
        best_node = None
        for u in in_mst:
            for v in nodes:
                if v not in in_mst:
                    w = dist_fn(u, v)
                    if w < best_edge:
                        best_edge = w
                        best_node = v
        if best_node is None:
            break
        in_mst.add(best_node)
        total_weight += best_edge

    return total_weight


def h_naive_manhattan(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """
    Baseline Naive Heuristic: Manhattan distance to nearest waiting miner (or Elevator if none).
    """
    if env.is_goal(state):
        return 0.0
    r, c = state.bot_pos
    if state.miners:
        return float(min(abs(r - mr) + abs(c - mc) for mr, mc in state.miners))
    er, ec = env.elevator_pos
    return float(abs(r - er) + abs(c - ec))




def _is_corner_deadlock(pos: Pos, env: DeepCoreEnv) -> bool:
    """Returns True if a boulder at `pos` is wedged against perpendicular rock walls (#)."""
    r, c = pos
    n_wall = (r - 1, c) in env.walls
    s_wall = (r + 1, c) in env.walls
    e_wall = (r, c + 1) in env.walls
    w_wall = (r, c - 1) in env.walls
    return (n_wall or s_wall) and (e_wall or w_wall)




# =============================================================================
# Shared helpers for H1 / H2
# =============================================================================
_INF = float("inf")

# Static sludge-aware entry-cost distances: entering a sludge cell costs at
# least 5 (MOVE) and any other cell at least 1, so these are lower bounds on the
# base (payload-free) part of every trajectory between two cells.
_WEIGHTED_DIST_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _wall_dist(env: DeepCoreEnv, a: Pos, b: Pos) -> float:
    return _get_wall_aware_distances(env).get((a, b), _INF)


def _get_weighted_distances(env: DeepCoreEnv) -> Dict[Pos, Dict[Pos, int]]:
    """Dijkstra from every walkable cell where entering sludge costs 5, else 1."""
    env_id = env
    if env_id in _WEIGHTED_DIST_CACHE:
        return _WEIGHTED_DIST_CACHE[env_id]

    import heapq

    table: Dict[Pos, Dict[Pos, int]] = {}
    walkable = [(r, c) for r in range(env.rows) for c in range(env.cols) if (r, c) not in env.walls]
    for src in walkable:
        dist: Dict[Pos, int] = {src: 0}
        pq = [(0, src)]
        while pq:
            d, cur = heapq.heappop(pq)
            if d > dist.get(cur, _INF):
                continue
            cr, cc = cur
            for dr, dc in DIRECTION_DELTAS.values():
                nxt = (cr + dr, cc + dc)
                if not env.in_bounds(nxt) or nxt in env.walls:
                    continue
                nd = d + (env.COST_SLUDGE if nxt in env.sludge else env.COST_CLEAR_FLOOR)
                if nd < dist.get(nxt, _INF):
                    dist[nxt] = nd
                    heapq.heappush(pq, (nd, nxt))
        table[src] = dist
    _WEIGHTED_DIST_CACHE[env_id] = table
    return table


def _tour_lower_bound(state: DeepCoreState, env: DeepCoreEnv, first_leg, sym) -> float:
    """
    Lower bound on the remaining trajectory "bot -> all miners (any order) -> E".

        T(s) = min_m first_leg(bot, m) + MST_sym(miners U {E})      (miners left)
        T(s) = first_leg(bot, E)                                    (no miners)

    The trajectory first reaches some miner m (>= first_leg(bot, m)), then
    visits the remaining points ending at E; that Hamiltonian path weighs at
    least the MST over miners U {E}.  Unlike an MST that also contains the bot,
    this bound changes by at most one edge cost when the bot moves one cell,
    which is what makes H1/H2 consistent (see the report for the proof).
    """
    bot = state.bot_pos
    if not state.miners:
        return first_leg(bot, env.elevator_pos)
    nearest = min(first_leg(bot, m) for m in state.miners)
    return nearest + _prim_mst_cost([*state.miners, env.elevator_pos], sym)


def _payload_lower_bound(state: DeepCoreState, env: DeepCoreEnv, moves_lb: float) -> float:
    """
    Lower bound on the payload surcharge still to be paid.

    * Every remaining move costs at least 2*k extra (k never decreases), and at
      least `moves_lb` moves remain.
    * After the last miner is picked up the pod carries all T miners, and the
      robot still has to walk from that miner to E: at least
      min_m dist(m, E) moves, each costing 2*(T - k) more than assumed above.
    """
    k = state.carried
    bound = 2 * k * moves_lb
    remaining = len(state.miners)
    if remaining:
        last_leg = min(_wall_dist(env, m, env.elevator_pos) for m in state.miners)
        bound += 2 * remaining * last_leg
    return bound


# =============================================================================
# H1: tour bound (wall-aware step distances) + payload surcharge
# =============================================================================
def h1_basic(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """
    Admissible and consistent MST-plus-payload lower bound.

    h1(s) = T(s) * 1 + payload(s), where T(s) is the tour bound
    `min_m d(bot, m) + MST(miners U {E})` on wall-aware BFS step distances
    (boulders, chasms and rubble are ignored, which only shortens distances).
    Every move costs at least 1, so T(s) bounds the base cost; the payload
    surcharge bound is added on top (`_payload_lower_bound`).
    """
    if env.is_goal(state):
        return 0.0
    dist = lambda u, v: _wall_dist(env, u, v)
    moves_lb = _tour_lower_bound(state, env, dist, dist)
    return moves_lb + _payload_lower_bound(state, env, moves_lb)


# =============================================================================
# H2: sludge-aware MST + payload + unavoidable-obstacle cost + deadlocks
# =============================================================================
def _reachable(env: DeepCoreEnv, start: Pos, blocked) -> set:
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRECTION_DELTAS.values():
            nxt = (r + dr, c + dc)
            if nxt in seen or not env.in_bounds(nxt) or nxt in env.walls or nxt in blocked:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return seen


_H2_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def h2_advanced(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """
    Dominates H1 and stays admissible.

    1. Base movement: the same tour bound as H1, but with sludge-aware weighted
       distances (entering ~ costs >= 5).  These are directed (entry cost of the
       destination), so the first leg uses dw(bot, m) and the MST uses the
       symmetric min(dw(u, v), dw(v, u)).  Every weighted distance is >= the
       step distance, so this term is >= H1's first term.
    2. Payload surcharge: identical to H1 (still uses step distances).
    3. Unavoidable obstacles (incremental cost only, never double-counting a move):
       - If some target (miner or E) cannot be reached without crossing current
         rubble, at least one DRILL (10, and drilling never moves the robot) is
         needed: +10.
       - If some target cannot be reached without crossing an open chasm, at
         least one PUSH into a chasm is needed. A push costs 6+2k while the move
         it performs is already counted with at most 5+2k: +1.
    4. Deadlock: if a chasm must be bridged but every remaining boulder is stuck
       in a wall corner, the state is unsolvable: return infinity.
    """
    cache = _H2_CACHE.setdefault(env, {})
    val = cache.get(state)
    if val is None:
        val = _h2_compute(state, env)
        cache[state] = val
    return val


def _h2_compute(state: DeepCoreState, env: DeepCoreEnv) -> float:
    if env.is_goal(state):
        return 0.0

    wd = _get_weighted_distances(env)
    directed = lambda u, v: wd.get(u, {}).get(v, _INF)
    symmetric = lambda u, v: min(directed(u, v), directed(v, u))
    base = _tour_lower_bound(state, env, directed, symmetric)
    steps = lambda u, v: _wall_dist(env, u, v)
    moves_lb = _tour_lower_bound(state, env, steps, steps)
    h = base + _payload_lower_bound(state, env, moves_lb)

    targets = [*state.miners, env.elevator_pos]
    bot = state.bot_pos

    if state.rubble:
        # Boulders can be pushed away and chasms bridged, so only rubble blocks here.
        reach = _reachable(env, bot, state.rubble)
        if any(t not in reach for t in targets):
            h += env.COST_DRILL

    if state.chasms:
        reach = _reachable(env, bot, state.chasms)
        if any(t not in reach for t in targets):
            if not any(not _is_corner_deadlock(b, env) for b in state.boulders):
                return _INF
            h += env.COST_PUSH_BASE - env.COST_SLUDGE

    return h
