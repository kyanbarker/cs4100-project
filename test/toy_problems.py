"""Small problems with known answers, used to sanity-check `src/agents`.

All three interfaces (search, MDP, RL) are exercised on the same problems,
which also shows how a game would be adapted:

  GridMaze      - SearchProblem (shortest path with walls, optional step costs)
  TinyGridworld - MarkovDecisionProcess with stochastic moves
  ToyJetpack    - a drastically simplified jetpack game. One dynamics function
                  is wrapped three ways: SearchProblem, MDP, and (through
                  MDPEnvironment) Environment.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.mdp import MarkovDecisionProcess
from agents.search import SearchProblem

# ---------------------------------------------------------------------------
# Search: grid maze
# ---------------------------------------------------------------------------

MOVES = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}


class GridMaze(SearchProblem):
    """`layout` is a list of strings: '#' wall, 'S' start, 'G' goal, digits 2-9 are
    cells that cost that much to enter, anything else costs 1."""

    def __init__(self, layout):
        self.rows = layout
        self.h, self.w = len(layout), len(layout[0])
        for y, row in enumerate(layout):
            for x, c in enumerate(row):
                if c == "S":
                    self.start = (x, y)
                if c == "G":
                    self.goal = (x, y)

    def start_state(self):
        return self.start

    def is_goal(self, state):
        return state == self.goal

    def successors(self, state):
        x, y = state
        for name, (dx, dy) in MOVES.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h and self.rows[ny][nx] != "#":
                c = self.rows[ny][nx]
                yield (nx, ny), name, float(c) if c.isdigit() else 1.0

    def manhattan(self, state, problem=None):
        return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])


# ---------------------------------------------------------------------------
# MDP: 4x3 gridworld (Russell & Norvig style), stochastic moves
# ---------------------------------------------------------------------------


class TinyGridworld(MarkovDecisionProcess):
    """Moves succeed with prob 1-noise and slip to each side with prob noise/2.
    (3, 0) is +1 and (3, 1) is -1 (both terminal); (1, 1) is a wall. y grows downward."""

    WALL = (1, 1)
    EXITS = {(3, 0): 1.0, (3, 1): -1.0}

    def __init__(self, noise=0.2, living_reward=0.0):
        self.noise, self.living = noise, living_reward

    def start_state(self):
        return (0, 2)

    def states(self):
        return [(x, y) for x in range(4) for y in range(3) if (x, y) != self.WALL]

    def is_terminal(self, state):
        return state in self.EXITS

    def actions(self, state):
        return [] if self.is_terminal(state) else ["N", "S", "W", "E"]

    def _move(self, state, name):
        dx, dy = MOVES[name]
        nxt = (state[0] + dx, state[1] + dy)
        if not (0 <= nxt[0] < 4 and 0 <= nxt[1] < 3) or nxt == self.WALL:
            return state
        return nxt

    def transitions(self, state, action):
        sides = {"N": "WE", "S": "WE", "W": "NS", "E": "NS"}[action]
        out = {}
        for name, p in [(action, 1 - self.noise)] + [(s, self.noise / 2) for s in sides]:
            nxt = self._move(state, name)
            out[nxt] = out.get(nxt, 0.0) + p
        return list(out.items())

    def reward(self, state, action, next_state):
        return self.EXITS.get(next_state, self.living) if next_state != state else self.living


# ---------------------------------------------------------------------------
# Toy jetpack: discrete, vertical-only version of the real game
# ---------------------------------------------------------------------------

DEAD = ("dead",)  # absorbing state


class ToyJetpackDynamics:
    """The "laws of physics" for the toy game, in one place.

    The world is `length` columns wide and advances one column per step.
    A state is (column, y, vy). y in [0, height); vy in [-max_v, max_v]; y grows
    downward (like pygame). Action 'up' = thrust (vy -= 1), 'none' = fall (vy += 1).
    `obstacles` maps column -> (gap_lo, gap_hi); on that column the player must
    have gap_lo <= y <= gap_hi after moving, or it dies. Hitting the ceiling or floor
    zeroes vy (it is not fatal), as in the real game.
    """

    def __init__(self, height=8, length=12, max_v=2, obstacles=None):
        self.height, self.length, self.max_v = height, length, max_v
        self.obstacles = obstacles if obstacles is not None else {3: (5, 7), 6: (1, 3), 9: (3, 5)}
        self.actions = ["up", "none"]

    def start(self):
        return (0, self.height // 2, 0)

    def is_goal(self, state):
        return state != DEAD and state[0] >= self.length

    def step(self, state, action):
        col, y, vy = state
        vy = max(-self.max_v, min(self.max_v, vy + (-1 if action == "up" else 1)))
        y += vy
        if y < 0 or y > self.height - 1:
            y = max(0, min(self.height - 1, y))
            vy = 0
        col += 1
        gap = self.obstacles.get(col)
        if gap is not None and not (gap[0] <= y <= gap[1]):
            return DEAD
        return (col, y, vy)


class ToyJetpackSearch(SearchProblem):
    """Search view: unit step cost, goal = reach the last column, death = no successor."""

    def __init__(self, dyn):
        self.dyn = dyn

    def start_state(self):
        return self.dyn.start()

    def is_goal(self, state):
        return self.dyn.is_goal(state)

    def successors(self, state):
        for a in self.dyn.actions:
            nxt = self.dyn.step(state, a)
            if nxt != DEAD:  # dying is never useful for a path search, so prune it
                yield nxt, a, 1.0

    def heuristic(self, state, problem=None):
        return self.dyn.length - state[0]  # exact remaining steps: admissible and consistent


class ToyJetpackMDP(MarkovDecisionProcess):
    """MDP view: +1 per step survived, -100 for dying, goal and death are terminal."""

    def __init__(self, dyn, death_penalty=-100.0):
        self.dyn, self.death_penalty = dyn, death_penalty

    def start_state(self):
        return self.dyn.start()

    def is_terminal(self, state):
        return state == DEAD or self.dyn.is_goal(state)

    def actions(self, state):
        return [] if self.is_terminal(state) else list(self.dyn.actions)

    def transitions(self, state, action):
        return [(self.dyn.step(state, action), 1.0)]  # deterministic

    def reward(self, state, action, next_state):
        return self.death_penalty if next_state == DEAD else 1.0
