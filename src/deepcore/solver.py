"""
solver.py
Student implementation file for DeepCore search algorithms.

- Breadth-First Search (BFS) is provided below as a complete reference example
  showing how to interact with `env.get_successors(state)` and `SearchMetrics`.
- Implement the remaining required algorithms. Do not change the public
  function signatures or the environment/state modules.
"""

from collections import deque
import heapq
import math
from typing import Callable, Dict, List, Optional, Set, Tuple

from .environment import DeepCoreEnv
from .state import DeepCoreState, SearchMetrics


def _reconstruct_path(
    parent_map: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]],
    goal_state: DeepCoreState,
) -> List[str]:
    """Helper function to trace actions from the initial state to `goal_state`."""
    actions: List[str] = []
    curr: Optional[DeepCoreState] = goal_state
    while curr is not None:
        prev, act = parent_map.get(curr, (None, None))
        if act is not None:
            actions.append(act)
        curr = prev
    actions.reverse()
    return actions


# =============================================================================
# 1. Breadth-First Search (BFS) — [PROVIDED AS REFERENCE EXAMPLE]
# =============================================================================
def breadth_first_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """
    Reference Graph-Search BFS implementation (FIFO queue).
    Minimizes the number of steps, but ignores terrain/payload energy costs!
    """
    metrics = SearchMetrics(algorithm_name="BFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    frontier = deque([start])
    frontier_set = {start}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))
        state = frontier.popleft()
        frontier_set.discard(state)

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if next_state not in explored and next_state not in frontier_set:
                parent[next_state] = (state, action.value)
                cost_so_far[next_state] = cost_so_far[state] + step_cost
                if env.is_goal(next_state):
                    metrics.solved = True
                    metrics.solution_actions = _reconstruct_path(parent, next_state)
                    metrics.total_cost = cost_so_far[next_state]
                    return metrics
                frontier.append(next_state)
                frontier_set.add(next_state)
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# Shared helpers
# =============================================================================
ParentMap = Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]]


def _path_cost(env: DeepCoreEnv, actions: List[str]) -> int:
    """Replays `actions` from the initial state and sums the exact step costs."""
    state = env.get_initial_state()
    total = 0
    for act in actions:
        state, step_cost = env.transition(state, act)
        total += step_cost
    return total


def _successor_cache(env: DeepCoreEnv):
    """
    Memoised `env.get_successors` for the iterative-deepening algorithms.
    IDS/IDA* regenerate the same states in every iteration; caching the
    successor lists trades memory for a large speed-up and does not change
    the order or result of the search.
    """
    cache: Dict[DeepCoreState, list] = {}

    def successors_of(state: DeepCoreState) -> list:
        succ = cache.get(state)
        if succ is None:
            succ = env.get_successors(state)
            cache[state] = succ
        return succ

    return successors_of


def _best_first_search(
    env: DeepCoreEnv,
    name: str,
    priority: Callable[[int, DeepCoreState], float],
    reopen: bool,
) -> SearchMetrics:
    """
    Generic best-first graph search used by UCS, A* and Weighted A*.

    The frontier is a binary heap of (priority, counter, g, state); the counter
    breaks ties in FIFO order so DeepCoreState objects are never compared.
    Stale heap entries (a cheaper g was found later) are skipped lazily.
    The goal test happens when a node is popped, which is what makes UCS/A*
    optimal.  With `reopen=True`, a closed state is re-opened when a strictly
    cheaper path to it is found (needed for A* with an inconsistent heuristic).
    """
    metrics = SearchMetrics(algorithm_name=name)
    start = env.get_initial_state()
    counter = 0
    frontier: List[Tuple[float, int, int, DeepCoreState]] = [(priority(0, start), counter, 0, start)]
    best_g: Dict[DeepCoreState, int] = {start: 0}
    parent: ParentMap = {start: (None, None)}
    closed: Set[DeepCoreState] = set()

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(closed))
        _, _, g, state = heapq.heappop(frontier)
        if g > best_g[state] or state in closed:
            continue  # stale entry

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = g
            metrics.status_message = "Solution found."
            return metrics

        closed.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, child, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            new_g = g + step_cost
            if new_g >= best_g.get(child, math.inf):
                continue
            if child in closed:
                if not reopen:
                    continue
                closed.discard(child)  # cheaper path found: re-open
            best_g[child] = new_g
            parent[child] = (state, action.value)
            p = priority(new_g, child)
            if p == math.inf:
                continue  # heuristic proved this state is a dead end
            counter += 1
            heapq.heappush(frontier, (p, counter, new_g, child))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 2. Depth-First Search (DFS) & Iterative Deepening Search (IDS)
# =============================================================================
def depth_first_search(env: DeepCoreEnv, max_expansions: Optional[int] = None) -> Optional[SearchMetrics]:
    """Depth-First Graph Search using a LIFO stack and an explored set."""
    metrics = SearchMetrics(algorithm_name="DFS")
    start = env.get_initial_state()
    stack: List[DeepCoreState] = [start]
    parent: ParentMap = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}
    explored: Set[DeepCoreState] = set()

    while stack:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(stack))
        metrics.max_memory_size = max(metrics.max_memory_size, len(stack) + len(explored))
        state = stack.pop()
        if state in explored:
            continue

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = cost_so_far[state]
            metrics.status_message = "Solution found."
            return metrics

        if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
            metrics.status_message = "Expansion limit reached."
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        # Push in reverse so the first legal action is explored first.
        for action, child, step_cost in reversed(env.get_successors(state)):
            metrics.nodes_generated += 1
            if child not in explored:
                parent[child] = (state, action.value)
                cost_so_far[child] = cost_so_far[state] + step_cost
                stack.append(child)

    metrics.status_message = "No solution found."
    return metrics


def iterative_deepening_search(
    env: DeepCoreEnv,
    max_depth: Optional[int] = None,
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """
    Iterative Deepening Search: depth-limited DFS restarted with limit 0, 1, 2, ...

    Each iteration keeps the states on the current path (cycle check) plus a
    table of the shallowest depth at which each state was reached during this
    iteration.  Re-reaching a state at the same or a greater depth cannot find
    anything new within the limit, so that branch is pruned.  The search stays
    complete and still returns a minimum-step solution, without the exponential
    re-exploration of transpositions.
    """
    metrics = SearchMetrics(algorithm_name="IDS")
    start = env.get_initial_state()
    successors_of = _successor_cache(env)
    if env.is_goal(start):
        metrics.solved = True
        metrics.status_message = "Solution found."
        return metrics

    limit = 1
    while max_depth is None or limit <= max_depth:
        best_depth: Dict[DeepCoreState, int] = {start: 0}
        path_actions: List[str] = []
        path_costs: List[int] = []
        on_path: Set[DeepCoreState] = {start}
        cutoff = False

        # Stack entries: (state, depth, iterator over its successors)
        stack = [(start, 0, iter(successors_of(start)))]
        metrics.nodes_expanded += 1
        metrics.explored_order.append(start.bot_pos)

        while stack:
            metrics.max_frontier_size = max(metrics.max_frontier_size, len(stack))
            metrics.max_memory_size = max(metrics.max_memory_size, len(stack) + len(best_depth))
            state, depth, successors = stack[-1]
            advanced = False
            for action, child, step_cost in successors:
                metrics.nodes_generated += 1
                child_depth = depth + 1
                if child in on_path or best_depth.get(child, math.inf) <= child_depth:
                    continue
                best_depth[child] = child_depth
                if env.is_goal(child):
                    metrics.solved = True
                    metrics.solution_actions = path_actions + [action.value]
                    metrics.total_cost = sum(path_costs) + step_cost
                    metrics.status_message = "Solution found."
                    return metrics
                if child_depth >= limit:
                    cutoff = True
                    continue
                if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
                    metrics.status_message = "Expansion limit reached."
                    return metrics
                metrics.nodes_expanded += 1
                metrics.explored_order.append(child.bot_pos)
                path_actions.append(action.value)
                path_costs.append(step_cost)
                on_path.add(child)
                stack.append((child, child_depth, iter(successors_of(child))))
                advanced = True
                break
            if not advanced:
                stack.pop()
                on_path.discard(state)
                if path_actions:
                    path_actions.pop()
                    path_costs.pop()

        if not cutoff:
            metrics.status_message = "No solution found."
            return metrics
        limit += 1

    metrics.status_message = "Depth limit reached."
    return metrics


# =============================================================================
# 3. Bidirectional Search
# =============================================================================
def bidirectional_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """
    Bidirectional breadth-first search.

    The forward search starts from the initial state and uses
    `env.get_successors`; the backward search starts from every state in
    `env.get_goal_states()` and uses `env.get_predecessors`.  The smaller
    frontier is expanded one full layer at a time; when a layer reaches a state
    already seen by the other side, the best meeting point of that layer gives a
    minimum-step solution.

    The goal state is not unique (extra boulders/bridges/rubble may differ), and
    on some maps none of the canonical goal states is reachable (e.g. level4 has
    more chasms than boulders).  The forward side therefore also accepts any
    state that passes `env.is_goal`, which keeps the search complete.
    """
    metrics = SearchMetrics(algorithm_name="Bidirectional")
    start = env.get_initial_state()
    goals = env.get_goal_states()

    # parent_f[s] = (previous state, action previous -> s)
    parent_f: ParentMap = {start: (None, None)}
    depth_f: Dict[DeepCoreState, int] = {start: 0}
    # parent_b[s] = (next state toward a goal, action s -> next)
    parent_b: ParentMap = {}
    depth_b: Dict[DeepCoreState, int] = {}
    for goal in goals:
        parent_b[goal] = (None, None)
        depth_b[goal] = 0

    def finish(meet: DeepCoreState) -> SearchMetrics:
        actions = _reconstruct_path(parent_f, meet)
        cur: Optional[DeepCoreState] = meet
        while cur is not None:
            nxt, act = parent_b[cur]
            if act is None:
                break
            actions.append(act)
            cur = nxt
        metrics.solved = True
        metrics.solution_actions = actions
        metrics.total_cost = _path_cost(env, actions)
        metrics.status_message = "Solution found."
        return metrics

    if start in depth_b:
        return finish(start)

    frontier_f: List[DeepCoreState] = [start]
    frontier_b: List[DeepCoreState] = list(goals)

    while frontier_f and frontier_b:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier_f) + len(frontier_b))
        metrics.max_memory_size = max(metrics.max_memory_size, len(depth_f) + len(depth_b))

        forward = len(frontier_f) <= len(frontier_b)
        layer = frontier_f if forward else frontier_b
        this_depth, other_depth = (depth_f, depth_b) if forward else (depth_b, depth_f)
        this_parent = parent_f if forward else parent_b
        next_layer: List[DeepCoreState] = []
        best_meet: Optional[DeepCoreState] = None
        best_total = math.inf

        for state in layer:
            metrics.nodes_expanded += 1
            if forward:
                metrics.explored_order.append(state.bot_pos)
                neighbours = env.get_successors(state)
            else:
                neighbours = env.get_predecessors(state)
            for action, other, _ in neighbours:
                metrics.nodes_generated += 1
                if other in this_depth:
                    continue
                this_depth[other] = this_depth[state] + 1
                this_parent[other] = (state, action.value)
                if other in other_depth:
                    total = this_depth[other] + other_depth[other]
                    if total < best_total:
                        best_total, best_meet = total, other
                elif forward and env.is_goal(other):
                    # Reached a goal not in the canonical backward roots.
                    parent_b[other] = (None, None)
                    depth_b[other] = 0
                    total = this_depth[other]
                    if total < best_total:
                        best_total, best_meet = total, other
                next_layer.append(other)

        if best_meet is not None:
            return finish(best_meet)

        if forward:
            frontier_f = next_layer
        else:
            frontier_b = next_layer

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 4. Uniform Cost Search (UCS)
# =============================================================================
def uniform_cost_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """Uniform Cost Search: priority queue ordered by path cost g(n)."""
    return _best_first_search(env, "UCS", lambda g, s: g, reopen=False)


# =============================================================================
# 5. Greedy Best-First Search (GBFS)
# =============================================================================
def greedy_best_first_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """Greedy Best-First Search: priority queue ordered strictly by h(n)."""
    metrics = SearchMetrics(algorithm_name="GBFS")
    start = env.get_initial_state()
    counter = 0
    frontier: List[Tuple[float, int, DeepCoreState]] = [(heuristic(start, env), counter, start)]
    parent: ParentMap = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}
    explored: Set[DeepCoreState] = set()

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))
        _, _, state = heapq.heappop(frontier)
        if state in explored:
            continue
        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = cost_so_far[state]
            metrics.status_message = "Solution found."
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, child, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if child in parent:
                continue  # already generated (in frontier or explored)
            parent[child] = (state, action.value)
            cost_so_far[child] = cost_so_far[state] + step_cost
            h = heuristic(child, env)
            if h == math.inf:
                continue
            counter += 1
            heapq.heappush(frontier, (h, counter, child))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 6. A* Search
# =============================================================================
def a_star_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """A* Graph Search: f(n) = g(n) + h(n), re-opening states reached more cheaply."""
    return _best_first_search(env, "A*", lambda g, s: g + heuristic(s, env), reopen=True)


# =============================================================================
# BONUS 1: Weighted A* Search
# =============================================================================
def weighted_a_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    weight: float = 1.5,
) -> Optional[SearchMetrics]:
    """Weighted A*: f(n) = g(n) + W * h(n). The cost is at most W times optimal."""
    return _best_first_search(
        env, f"WA*(W={weight:g})", lambda g, s: g + weight * heuristic(s, env), reopen=False
    )


# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*)
# =============================================================================
def ida_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """
    Iterative Deepening A*: depth-first search bounded by an f-cost threshold.

    Each iteration restarts from the root with threshold = the smallest f that
    exceeded the previous threshold.  Within an iteration, a table of the
    cheapest g seen per state prunes transpositions (a state re-reached with an
    equal or larger g cannot lead anywhere new under the same threshold), so
    the returned solution is still optimal for an admissible heuristic.
    """
    metrics = SearchMetrics(algorithm_name="IDA*")
    start = env.get_initial_state()
    successors_of = _successor_cache(env)
    h_cache: Dict[DeepCoreState, float] = {}

    def h(s: DeepCoreState) -> float:
        val = h_cache.get(s)
        if val is None:
            val = heuristic(s, env)
            h_cache[s] = val
        return val

    threshold = h(start)
    while threshold < math.inf:
        best_g: Dict[DeepCoreState, int] = {start: 0}
        next_threshold = math.inf
        path_actions: List[str] = []
        on_path: Set[DeepCoreState] = {start}
        stack: list = [(start, 0, None)]  # (state, g, successor iterator or None)

        while stack:
            metrics.max_frontier_size = max(metrics.max_frontier_size, len(stack))
            metrics.max_memory_size = max(metrics.max_memory_size, len(stack) + len(best_g))
            state, g, successors = stack[-1]

            if successors is None:
                f = g + h(state)
                if f > threshold:
                    next_threshold = min(next_threshold, f)
                    stack.pop()
                    on_path.discard(state)
                    if path_actions:
                        path_actions.pop()
                    continue
                if env.is_goal(state):
                    metrics.solved = True
                    metrics.solution_actions = list(path_actions)
                    metrics.total_cost = g
                    metrics.status_message = "Solution found."
                    return metrics
                if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
                    metrics.status_message = "Expansion limit reached."
                    return metrics
                metrics.nodes_expanded += 1
                metrics.explored_order.append(state.bot_pos)
                successors = iter(successors_of(state))
                stack[-1] = (state, g, successors)

            advanced = False
            for action, child, step_cost in successors:
                metrics.nodes_generated += 1
                child_g = g + step_cost
                if child in on_path or best_g.get(child, math.inf) <= child_g:
                    continue
                best_g[child] = child_g
                on_path.add(child)
                path_actions.append(action.value)
                stack.append((child, child_g, None))
                advanced = True
                break
            if not advanced:
                stack.pop()
                on_path.discard(state)
                if path_actions:
                    path_actions.pop()

        threshold = next_threshold

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*)
# =============================================================================
class _SMANode:
    __slots__ = ("state", "g", "f", "depth", "parent", "action", "children", "forgotten_f", "alive")

    def __init__(self, state, g, f, depth, parent, action):
        self.state: DeepCoreState = state
        self.g: int = g
        self.f: float = f
        self.depth: int = depth
        self.parent: Optional["_SMANode"] = parent
        self.action: Optional[str] = action
        self.children: List["_SMANode"] = []
        self.forgotten_f: float = math.inf  # best f among children dropped from memory
        self.alive = True


def sma_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    memory_limit: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """
    Simplified Memory-Bounded A* (SMA*).

    * Memory limit: at most `memory_limit` nodes are stored (default 50 000).
    * Expansion: the leaf with the lowest f (deepest on ties) is expanded and all
      its children are generated; f is kept monotone along a path:
      f(child) = max(f(parent), g(child) + h(child)).
    * Pruning: whenever memory is full, the leaf with the highest f (shallowest
      on ties) is deleted.  Its f is stored in the parent as `forgotten_f`
      (the "backed-up" value), so the parent knows how good the forgotten
      subtree was.  When a parent loses all its children it becomes a leaf
      again with f = forgotten_f and can be regenerated later.
    * Back-up: after an expansion, f(node) = min(f of children, forgotten_f),
      propagated up to the root while it changes.
    * Duplicate states: a child is skipped if the same state is already stored
      with a lower or equal g, or if it is an ancestor (cycle).
    """
    limit = memory_limit if memory_limit is not None else 50_000
    limit = max(limit, 2)
    metrics = SearchMetrics(algorithm_name="SMA*")
    start = env.get_initial_state()

    root = _SMANode(start, 0, heuristic(start, env), 0, None, None)
    stored: Dict[DeepCoreState, _SMANode] = {start: root}
    node_count = 1
    counter = 0
    # Leaves are tracked in two lazy heaps: best-first for expansion and
    # worst-first for deletion.  An entry is valid only if its f still matches.
    best_heap: List[Tuple[float, int, int, _SMANode]] = []
    worst_heap: List[Tuple[float, int, int, _SMANode]] = []

    def push_leaf(node: _SMANode) -> None:
        nonlocal counter
        if node.f == math.inf:
            return
        counter += 1
        heapq.heappush(best_heap, (node.f, -node.depth, counter, node))
        heapq.heappush(worst_heap, (-node.f, node.depth, counter, node))

    def valid_leaf(node: _SMANode, f: float) -> bool:
        return node.alive and not node.children and node.f == f

    def back_up(node: Optional[_SMANode]) -> None:
        while node is not None:
            new_f = min([c.f for c in node.children] + [node.forgotten_f])
            if new_f == node.f:
                return
            node.f = new_f
            if not node.children:
                push_leaf(node)
            node = node.parent

    def delete_worst_leaf(protect: _SMANode) -> bool:
        nonlocal node_count
        while worst_heap:
            neg_f, depth, cnt, node = heapq.heappop(worst_heap)
            if not valid_leaf(node, -neg_f):
                continue
            if node is protect or node.parent is None:
                # Cannot delete the node being expanded or the root; try others.
                rest = delete_worst_leaf(protect) if worst_heap else False
                heapq.heappush(worst_heap, (neg_f, depth, cnt, node))
                return rest
            parent = node.parent
            parent.children.remove(node)
            parent.forgotten_f = min(parent.forgotten_f, node.f)
            node.alive = False
            if stored.get(node.state) is node:
                del stored[node.state]
            node_count -= 1
            if not parent.children:
                parent.f = parent.forgotten_f
                push_leaf(parent)
            return True
        return False

    push_leaf(root)
    while best_heap:
        f, _, _, node = heapq.heappop(best_heap)
        if not valid_leaf(node, f):
            continue
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(best_heap) + 1)
        metrics.max_memory_size = max(metrics.max_memory_size, node_count)

        if env.is_goal(node.state):
            actions: List[str] = []
            cur: Optional[_SMANode] = node
            while cur is not None and cur.action is not None:
                actions.append(cur.action)
                cur = cur.parent
            actions.reverse()
            metrics.solved = True
            metrics.solution_actions = actions
            metrics.total_cost = node.g
            metrics.status_message = "Solution found."
            return metrics

        metrics.nodes_expanded += 1
        metrics.explored_order.append(node.state.bot_pos)
        node.forgotten_f = math.inf

        ancestors: Set[DeepCoreState] = set()
        cur = node
        while cur is not None:
            ancestors.add(cur.state)
            cur = cur.parent

        for action, child_state, step_cost in env.get_successors(node.state):
            metrics.nodes_generated += 1
            if child_state in ancestors:
                continue
            child_g = node.g + step_cost
            existing = stored.get(child_state)
            if existing is not None and existing.g <= child_g:
                continue
            h = heuristic(child_state, env)
            child_f = math.inf if h == math.inf else max(node.f, child_g + h)
            if child_f == math.inf:
                continue
            if node_count >= limit and not delete_worst_leaf(protect=node):
                node.forgotten_f = min(node.forgotten_f, child_f)
                continue
            child = _SMANode(child_state, child_g, child_f, node.depth + 1, node, action.value)
            node.children.append(child)
            stored[child_state] = child
            node_count += 1
            push_leaf(child)

        metrics.max_memory_size = max(metrics.max_memory_size, node_count)
        if node.children:
            back_up(node)
            continue
        if node.forgotten_f < math.inf:
            # Every child had to be forgotten immediately: stay a leaf with the
            # backed-up value so the subtree can be regenerated later.
            node.f = node.forgotten_f
            push_leaf(node)
            back_up(node.parent)
            continue
        # Dead end: no useful children at all. Remove it (and any ancestor that
        # becomes an empty dead end) from memory.
        while node.parent is not None and not node.children and node.forgotten_f == math.inf:
            parent = node.parent
            parent.children.remove(node)
            node.alive = False
            node_count -= 1
            node = parent
        if not node.children:
            node.f = node.forgotten_f
            push_leaf(node)
        back_up(node)

    metrics.status_message = "No solution found within the memory limit."
    return metrics
