"""Generic graph/tree search (BFS, DFS, UCS, A*).

Adapted from CS 4100 PA1 (itself based on the UC Berkeley Pacman AI projects,
http://ai.berkeley.edu). Nothing here depends on Pacman: a problem is any
object implementing `SearchProblem`.

Differences from the PA1 version:
  * no Pacman or `util` imports (stdlib only);
  * snake_case names;
  * nodes keep a parent pointer instead of copying the action list;
  * the frontier tracks which states it already holds with a dict, so checking
    "is this state already in the frontier?" is O(1) rather than a linear scan;
  * `max_expansions` stops runaway searches on huge or infinite state spaces;
  * the result is a `SearchResult` instead of an exception on failure.

States must be hashable (and comparable with ==) when `graph_search=True`.
"""

from __future__ import annotations

import heapq
import itertools
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, Optional

State = Hashable
Action = Any


class SearchProblem(ABC):
    """A deterministic, single-agent search problem."""

    @abstractmethod
    def start_state(self) -> State:
        """The initial state."""

    @abstractmethod
    def is_goal(self, state: State) -> bool:
        """True iff `state` is a goal state."""

    @abstractmethod
    def successors(self, state: State) -> Iterable[tuple[State, Action, float]]:
        """Return (next_state, action, step_cost) triples. Costs must be >= 0."""

    def cost_of_actions(self, actions: list[Action]) -> float:
        """Total cost of an action sequence (optional; used by tests and tools)."""
        state = self.start_state()
        total = 0.0
        for a in actions:
            for nxt, act, cost in self.successors(state):
                if act == a:
                    state, total = nxt, total + cost
                    break
            else:
                raise ValueError(f"illegal action {a!r} in state {state!r}")
        return total


Heuristic = Callable[[State, SearchProblem], float]


def null_heuristic(state: State, problem: Optional[SearchProblem] = None) -> float:
    return 0.0


@dataclass
class Node:
    state: State
    parent: Optional["Node"]
    action: Action
    cost: float  # path cost g(n)
    stale: bool = False  # replaced by a better node for the same state

    def actions(self) -> list[Action]:
        out: list[Action] = []
        node: Optional[Node] = self
        while node is not None and node.parent is not None:
            out.append(node.action)
            node = node.parent
        out.reverse()
        return out


@dataclass
class SearchResult:
    actions: Optional[list[Action]]  # None if no goal was found
    cost: float
    final_state: Optional[State]
    expanded: int  # number of nodes expanded

    @property
    def found(self) -> bool:
        return self.actions is not None


# ----------------------------------------------------------------------------
# Frontiers. Each one stores nodes and knows how to resolve two nodes that have
# the same state ("conflict"), as in PA1's `handleNodeConflict`.
# ----------------------------------------------------------------------------


class Frontier(ABC):
    def __init__(self) -> None:
        self._by_state: dict[State, Node] = {}

    def __bool__(self) -> bool:
        return bool(self._by_state)

    def __len__(self) -> int:
        return len(self._by_state)

    def offer(self, node: Node) -> None:
        """Add `node`, resolving a conflict if its state is already present."""
        old = self._by_state.get(node.state)
        if old is None:
            self._by_state[node.state] = node
            self._push(node)
        else:
            self._on_conflict(old, node)

    def pop(self) -> Node:
        """Remove and return the next node to expand."""
        while True:
            node = self._pop()
            if not node.stale:
                del self._by_state[node.state]
                return node

    def _replace(self, old: Node, new: Node) -> None:
        old.stale = True  # lazily deleted when popped
        self._by_state[new.state] = new
        self._push(new)

    @abstractmethod
    def _push(self, node: Node) -> None: ...

    @abstractmethod
    def _pop(self) -> Node: ...

    @abstractmethod
    def _on_conflict(self, old: Node, new: Node) -> None: ...


class StackFrontier(Frontier):
    """LIFO (DFS). On conflict the new node replaces the old one (PA1 DFS rule)."""

    def __init__(self) -> None:
        super().__init__()
        self._items: list[Node] = []

    def _push(self, node: Node) -> None:
        self._items.append(node)

    def _pop(self) -> Node:
        return self._items.pop()

    def _on_conflict(self, old: Node, new: Node) -> None:
        self._replace(old, new)


class QueueFrontier(Frontier):
    """FIFO (BFS). On conflict the old node is kept and the new one is dropped."""

    def __init__(self) -> None:
        super().__init__()
        self._items: deque[Node] = deque()

    def _push(self, node: Node) -> None:
        self._items.append(node)

    def _pop(self) -> Node:
        return self._items.popleft()

    def _on_conflict(self, old: Node, new: Node) -> None:
        pass


class PriorityFrontier(Frontier):
    """Min-priority queue (UCS, A*). On conflict the lower-priority node wins."""

    def __init__(self, priority: Callable[[Node], float]) -> None:
        super().__init__()
        self._priority = priority
        self._heap: list[tuple[float, int, Node]] = []
        self._counter = itertools.count()  # FIFO tie-break, avoids comparing Nodes

    def _push(self, node: Node) -> None:
        heapq.heappush(self._heap, (self._priority(node), next(self._counter), node))

    def _pop(self) -> Node:
        return heapq.heappop(self._heap)[2]

    def _on_conflict(self, old: Node, new: Node) -> None:
        if self._priority(new) < self._priority(old):
            self._replace(old, new)


# ----------------------------------------------------------------------------
# The generic algorithm (PA1's `genericSearch`)
# ----------------------------------------------------------------------------


def generic_search(
    problem: SearchProblem,
    frontier: Frontier,
    graph_search: bool = True,
    max_expansions: Optional[int] = None,
) -> SearchResult:
    """Search `problem` expanding nodes in the order chosen by `frontier`.

    graph_search: keep an explored set and never re-expand a state.
    max_expansions: give up (found=False) after this many expansions.

    Note: graph search over a frontier keyed by state keeps one node per state,
    exactly like the PA1 implementation.
    """
    frontier.offer(Node(problem.start_state(), None, None, 0.0))
    explored: set[State] = set()
    expanded = 0

    while frontier:
        node = frontier.pop()
        if graph_search:
            explored.add(node.state)
        if problem.is_goal(node.state):
            return SearchResult(node.actions(), node.cost, node.state, expanded)
        if max_expansions is not None and expanded >= max_expansions:
            break
        expanded += 1
        for next_state, action, step_cost in problem.successors(node.state):
            if graph_search and next_state in explored:
                continue
            frontier.offer(Node(next_state, node, action, node.cost + step_cost))

    return SearchResult(None, float("inf"), None, expanded)


def depth_first_search(problem, graph_search=True, max_expansions=None) -> SearchResult:
    return generic_search(problem, StackFrontier(), graph_search, max_expansions)


def breadth_first_search(problem, graph_search=True, max_expansions=None) -> SearchResult:
    return generic_search(problem, QueueFrontier(), graph_search, max_expansions)


def uniform_cost_search(problem, graph_search=True, max_expansions=None) -> SearchResult:
    return generic_search(
        problem, PriorityFrontier(lambda n: n.cost), graph_search, max_expansions
    )


def a_star_search(
    problem: SearchProblem,
    heuristic: Heuristic = null_heuristic,
    graph_search: bool = True,
    max_expansions: Optional[int] = None,
) -> SearchResult:
    """A*. Optimal if `heuristic` is admissible (tree search) or consistent (graph search)."""
    return generic_search(
        problem,
        PriorityFrontier(lambda n: n.cost + heuristic(n.state, problem)),
        graph_search,
        max_expansions,
    )


def weighted_a_star_search(
    problem: SearchProblem,
    heuristic: Heuristic,
    weight: float = 2.0,
    graph_search: bool = True,
    max_expansions: Optional[int] = None,
) -> SearchResult:
    """Priority g + weight*h. Faster but only bounded-suboptimal (cost <= weight * optimal
    when h is admissible). Useful when the heuristic is not admissible."""
    return generic_search(
        problem,
        PriorityFrontier(lambda n: n.cost + weight * heuristic(n.state, problem)),
        graph_search,
        max_expansions,
    )


# Abbreviations (as in PA1)
bfs = breadth_first_search
dfs = depth_first_search
ucs = uniform_cost_search
astar = a_star_search
