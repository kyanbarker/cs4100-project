"""Generic reinforcement learning: environment interface, Q-learning, approximate
Q-learning and a training loop.

Adapted from CS 4100 PA3 (`qlearningAgents.py` and `learningAgents.py`), which
are based on the UC Berkeley Pacman AI projects (http://ai.berkeley.edu). The
Pacman-specific parts (game scores, `observationFunction`, `registerInitialState`,
`final`) are replaced by a small gym-style `Environment` and an explicit
`train` loop.

Terminal handling: when `done` is True the update target is just `reward`. No
bootstrapping from the next state, so terminal states do not need to have an
empty action list (though they may).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Mapping, Optional

from .mdp import MarkovDecisionProcess, ValueEstimationAgent

State = Hashable
Action = Any


class Environment(ABC):
    """A simulator the agent can only interact with, not inspect.

    The environment owns the current state. `reset()` starts an episode and
    `step(action)` advances it.
    """

    @abstractmethod
    def reset(self) -> State:
        """Start a new episode and return the initial state."""

    @abstractmethod
    def actions(self, state: State) -> list[Action]:
        """Legal actions in `state`."""

    @abstractmethod
    def step(self, action: Action) -> tuple[State, float, bool]:
        """Apply `action`; return (next_state, reward, done)."""


class MDPEnvironment(Environment):
    """Wraps a known `MarkovDecisionProcess` as a sampling-only environment.

    Lets you run Q-learning on exactly the same problem that value iteration
    solves, which is a good correctness check for both.
    """

    def __init__(self, mdp: MarkovDecisionProcess, rng: Optional[random.Random] = None):
        self.mdp = mdp
        self.rng = rng or random.Random()
        self.state: State = mdp.start_state()

    def reset(self) -> State:
        self.state = self.mdp.start_state()
        return self.state

    def actions(self, state: State) -> list[Action]:
        return [] if self.mdp.is_terminal(state) else self.mdp.actions(state)

    def step(self, action: Action) -> tuple[State, float, bool]:
        outcomes = self.mdp.transitions(self.state, action)
        next_state = self.rng.choices(
            [s for s, _ in outcomes], weights=[p for _, p in outcomes]
        )[0]
        reward = self.mdp.reward(self.state, action, next_state)
        self.state = next_state
        return next_state, reward, self.mdp.is_terminal(next_state)


class QLearningAgent(ValueEstimationAgent):
    """Tabular Q-learning with epsilon-greedy exploration.

    actions_fn: state -> list of legal actions (e.g. `env.actions`).
    alpha: learning rate. epsilon: exploration probability. gamma: discount.
    rng: pass a seeded `random.Random` for reproducible runs.
    """

    def __init__(
        self,
        actions_fn: Callable[[State], list[Action]],
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.actions_fn = actions_fn
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = gamma
        self.rng = rng or random.Random()
        self.q: defaultdict[tuple[State, Action], float] = defaultdict(float)

    # Per PA3: always read Q-values through `get_q_value`, never `self.q`
    # directly, so that subclasses (e.g. approximate Q-learning) can override it.

    def get_q_value(self, state: State, action: Action) -> float:
        return self.q.get((state, action), 0.0)  # .get: don't grow the table on reads

    def get_value(self, state: State) -> float:
        actions = self.actions_fn(state)
        if not actions:
            return 0.0
        return max(self.get_q_value(state, a) for a in actions)

    def get_policy(self, state: State) -> Optional[Action]:
        actions = self.actions_fn(state)
        if not actions:
            return None
        best = max(self.get_q_value(state, a) for a in actions)
        # Break ties at random. Unseen actions have Q = 0, which can be optimal.
        return self.rng.choice([a for a in actions if self.get_q_value(state, a) == best])

    def get_action(self, state: State) -> Optional[Action]:
        """Epsilon-greedy: a uniformly random legal action with probability epsilon."""
        actions = self.actions_fn(state)
        if not actions:
            return None
        if self.rng.random() < self.epsilon:
            return self.rng.choice(actions)
        return self.get_policy(state)

    def _target(self, reward: float, next_state: State, done: bool) -> float:
        return reward if done else reward + self.discount * self.get_value(next_state)

    def update(
        self, state: State, action: Action, next_state: State, reward: float, done: bool = False
    ) -> None:
        """One Q-learning update from an observed transition."""
        diff = self._target(reward, next_state, done) - self.get_q_value(state, action)
        self.q[(state, action)] += self.alpha * diff


FeatureExtractor = Callable[[State, Action], Mapping[Any, float]]


class ApproximateQAgent(QLearningAgent):
    """Linear function approximation: Q(s, a) = sum_i w_i * f_i(s, a).

    Use this when the state space is too big or continuous for a table.
    `extractor(state, action)` returns a {feature_name: value} mapping.
    Keep feature values roughly in [-1, 1] (or use a small alpha) for stable learning.
    """

    def __init__(self, actions_fn, extractor: FeatureExtractor, **kwargs) -> None:
        super().__init__(actions_fn, **kwargs)
        self.extractor = extractor
        self.weights: defaultdict[Any, float] = defaultdict(float)

    def get_q_value(self, state: State, action: Action) -> float:
        feats = self.extractor(state, action)
        return sum(self.weights.get(k, 0.0) * v for k, v in feats.items())

    def update(self, state, action, next_state, reward, done=False) -> None:
        feats = self.extractor(state, action)
        diff = self._target(reward, next_state, done) - self.get_q_value(state, action)
        for k, v in feats.items():
            self.weights[k] += self.alpha * diff * v


@dataclass
class EpisodeStats:
    total_reward: float
    steps: int
    terminated: bool  # False if the episode was cut off by max_steps


@dataclass
class TrainingLog:
    episodes: list[EpisodeStats] = field(default_factory=list)

    def mean_reward(self, last: Optional[int] = None) -> float:
        eps = self.episodes[-last:] if last else self.episodes
        return sum(e.total_reward for e in eps) / len(eps) if eps else 0.0


def run_episode(
    env: Environment,
    agent: ValueEstimationAgent,
    max_steps: int = 10_000,
    learn: bool = False,
) -> EpisodeStats:
    """Run one episode. If `learn` is True the agent must have an `update` method
    and is updated after each step. Otherwise it only acts."""
    state = env.reset()
    total, steps, done = 0.0, 0, False
    while not done and steps < max_steps:
        action = agent.get_action(state)
        if action is None:  # no legal actions: treat as terminal
            done = True
            break
        next_state, reward, done = env.step(action)
        if learn:
            agent.update(state, action, next_state, reward, done)  # type: ignore[attr-defined]
        total += reward
        steps += 1
        state = next_state
    return EpisodeStats(total, steps, done)


def train(
    env: Environment,
    agent: QLearningAgent,
    episodes: int,
    max_steps: int = 10_000,
) -> TrainingLog:
    """Run `episodes` learning episodes and return the per-episode statistics."""
    log = TrainingLog()
    for _ in range(episodes):
        log.episodes.append(run_episode(env, agent, max_steps, learn=True))
    return log


def evaluate(
    env: Environment,
    agent: QLearningAgent,
    episodes: int,
    max_steps: int = 10_000,
) -> TrainingLog:
    """Run greedy episodes (epsilon = 0, no learning); the agent is left unchanged."""
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        log = TrainingLog()
        for _ in range(episodes):
            log.episodes.append(run_episode(env, agent, max_steps, learn=False))
        return log
    finally:
        agent.epsilon = saved_epsilon
