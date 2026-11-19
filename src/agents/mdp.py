"""Generic MDP interface and value iteration.

Adapted from CS 4100 PA3 (`mdp.py`, `valueIterationAgents.py`, and the
`ValueEstimationAgent` class from `learningAgents.py`), which are based on the
UC Berkeley Pacman AI projects (http://ai.berkeley.edu). No Pacman or `util`
dependencies remain.

Conventions (same as PA3):
  * a terminal state has no actions and value 0;
  * V(s)    = max_a Q(s, a)
  * Q(s, a) = sum_{s'} P(s' | s, a) * (R(s, a, s') + gamma * V(s'))

States must be hashable. Actions can be any object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Hashable, Optional

State = Hashable
Action = Any


class MarkovDecisionProcess(ABC):
    """A known-model MDP (transition probabilities and rewards are available)."""

    @abstractmethod
    def start_state(self) -> State: ...

    @abstractmethod
    def actions(self, state: State) -> list[Action]:
        """Legal actions in `state`. Must be empty for terminal states."""

    @abstractmethod
    def transitions(self, state: State, action: Action) -> list[tuple[State, float]]:
        """(next_state, probability) pairs; probabilities sum to 1.
        For a deterministic problem return a single pair with probability 1.0."""

    @abstractmethod
    def reward(self, state: State, action: Action, next_state: State) -> float: ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...

    def states(self) -> list[State]:
        """All states. By default this is every state reachable from the start state.
        Override it if you already have the list. It must be finite."""
        return reachable_states(self)


def reachable_states(mdp: MarkovDecisionProcess, limit: Optional[int] = None) -> list[State]:
    """Breadth-first enumeration of states reachable from the start state.

    This is how a large (but finite) MDP, for example a discretised physics
    simulator, can be given to value iteration without writing out the state
    list by hand. If `limit` is given and there are more states than that, a
    RuntimeError is raised, as a guard against accidentally enumerating a
    huge space.
    """
    start = mdp.start_state()
    seen = {start}
    order = [start]
    queue = deque([start])
    while queue:
        s = queue.popleft()
        if mdp.is_terminal(s):
            continue
        for a in mdp.actions(s):
            for s2, p in mdp.transitions(s, a):
                if p > 0 and s2 not in seen:
                    seen.add(s2)
                    order.append(s2)
                    queue.append(s2)
                    if limit is not None and len(seen) > limit:
                        raise RuntimeError(f"more than {limit} reachable states")
    return order


class ValueEstimationAgent(ABC):
    """Anything that can say how good a state or (state, action) pair is.

    Shared by value iteration (model-based) and Q-learning (model-free)."""

    @abstractmethod
    def get_q_value(self, state: State, action: Action) -> float: ...

    @abstractmethod
    def get_value(self, state: State) -> float:
        """max_a Q(s, a); 0 for terminal states."""

    @abstractmethod
    def get_policy(self, state: State) -> Optional[Action]:
        """argmax_a Q(s, a), or None if there are no actions. No exploration."""

    def get_action(self, state: State) -> Optional[Action]:
        """The action to actually take. Defaults to the greedy policy."""
        return self.get_policy(state)


class ValueIterationAgent(ValueEstimationAgent):
    """Runs value iteration on `mdp` at construction time, then acts greedily.

    iterations: maximum number of full sweeps.
    tolerance: stop early when the largest value change in a sweep is below this.
    After construction, `self.sweeps` is the number of sweeps actually run and
    `self.last_delta` is the largest value change in the final sweep.
    """

    def __init__(
        self,
        mdp: MarkovDecisionProcess,
        discount: float = 0.9,
        iterations: int = 100,
        tolerance: float = 0.0,
    ) -> None:
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.tolerance = tolerance
        self.values: defaultdict[State, float] = defaultdict(float)
        self.states = mdp.states()
        self.sweeps = 0
        self.last_delta = float("inf")
        self._run_value_iteration()

    def _q_from(self, state: State, action: Action, values) -> float:
        return sum(
            p * (self.mdp.reward(state, action, s2) + self.discount * values[s2])
            for s2, p in self.mdp.transitions(state, action)
        )

    def _run_value_iteration(self) -> None:
        for _ in range(self.iterations):
            new_values: defaultdict[State, float] = defaultdict(float)
            delta = 0.0
            for s in self.states:
                if self.mdp.is_terminal(s):
                    continue  # terminal value stays 0
                acts = self.mdp.actions(s)
                if not acts:
                    continue
                v = max(self._q_from(s, a, self.values) for a in acts)
                new_values[s] = v
                delta = max(delta, abs(v - self.values[s]))
            self.values = new_values
            self.sweeps += 1
            self.last_delta = delta
            if delta <= self.tolerance:
                break

    # --- ValueEstimationAgent -------------------------------------------------

    def get_q_value(self, state: State, action: Action) -> float:
        return self._q_from(state, action, self.values)

    def get_value(self, state: State) -> float:
        return self.values[state]

    def get_policy(self, state: State) -> Optional[Action]:
        if self.mdp.is_terminal(state):
            return None
        acts = self.mdp.actions(state)
        if not acts:
            return None
        return max(acts, key=lambda a: self.get_q_value(state, a))
