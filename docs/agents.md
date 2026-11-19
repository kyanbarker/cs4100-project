# `src/agents/`: generic search, MDP and RL code

Game-independent implementations of the algorithms from PA1 (search) and PA3 (MDPs and Q-learning). They use only the Python standard library. Developed and tested on Python 3.13; the code should work on 3.9+.

| Module      | Contents                                                                                                   | Source                                                       |
| ----------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `search.py` | `SearchProblem`, `generic_search`, BFS, DFS, UCS, A\*, weighted A\*                                        | PA1 `search.py`                                              |
| `mdp.py`    | `MarkovDecisionProcess`, `reachable_states`, `ValueEstimationAgent`, `ValueIterationAgent`                 | PA3 `mdp.py`, `valueIterationAgents.py`, `learningAgents.py` |
| `rl.py`     | `Environment`, `MDPEnvironment`, `QLearningAgent`, `ApproximateQAgent`, `train`, `evaluate`, `run_episode` | PA3 `qlearningAgents.py`, `learningAgents.py`                |

PA2 (minimax / expectimax) is not ported. Jetpack Joyride is a single-agent game. If a ghost-like adversary is ever needed, it can be added as a fourth module.

**Attribution and licence.** The PA code is derived from the UC Berkeley Pacman AI projects (http://ai.berkeley.edu). The Berkeley licence asks that people retain its notice, give attribution, and **not publish solutions**. The module docstrings carry the attribution. The ported code contains our own PA solutions and the repository's `origin` is on GitHub, so check that the repo is private (or the course allows it) before pushing.

## What changed from the PAs, and why

The PA code is tied to Pacman through `game.Agent`, `util.Counter` / `util.PriorityQueue`, `state.getLegalActions()`, `state.getScore()`, and the `observationFunction` / `final` callbacks. Search was already fairly generic. The MDP and RL code needed more work.

- **No Pacman dependency.** Nothing imports `game`, `util` or `featureExtractors`. `Counter` became `collections.defaultdict(float)`, and the priority queue is `heapq`.
- **Agents no longer depend on a game object.** Q-learning gets its legal actions from a function passed to the constructor (`actions_fn`), not from `state.getLegalActions()`. Rewards come from the environment, not from score differences.
- **Gym-style `Environment`** (`reset`, `actions`, `step`) replaces Pacman's game-loop callbacks. A `train` function now runs the training loop that the Pacman framework used to run.
- **Explicit `done` flag.** A terminal transition updates with target `reward`. The PA code relied on terminal states having no legal actions. Both work, but `done` is easier for simulators that only report "dead".
- **Efficient search.** PA1 scanned the whole frontier for each successor (O(n) each) and copied the action list into every node. The port keeps a `state -> node` dict in each frontier and stores parent pointers, so each step is O(1). The PA1 conflict rules are kept (DFS replaces the old node, BFS keeps it, UCS and A\* keep the cheaper one).
- **Search results** are returned as a `SearchResult(actions, cost, final_state, expanded)`. A failed search returns `found == False` and does not raise. `max_expansions` bounds the work.
- **Reproducibility.** Agents and environments accept a `random.Random`, so a run can be repeated from a seed.
- **Python naming.** snake_case instead of the PA's camelCase.
- **A bug that was fixed:** reading `agent.q[key]` on a `defaultdict` inserts the key, so evaluating a policy silently grew the Q-table. The port reads with `.get`. A test covers it.

Not ported (not needed yet): `PrioritizedSweepingValueIterationAgent` (a stub in the PA), the Pacman feature extractors, `PacmanQAgent`, and the Crawler and Gridworld displays. Policy iteration is not in PA3 either. The proposal mentions it, so it would have to be written.

## How to use each module

### Search

Subclass `SearchProblem` and implement three methods:

```python
class MyProblem(SearchProblem):
    def start_state(self): ...                   # a hashable state
    def is_goal(self, state): ...                # bool
    def successors(self, state):                 # iterable of (next_state, action, step_cost)
        ...

result = a_star_search(MyProblem(), heuristic)   # heuristic(state, problem) -> float
if result.found:
    print(result.actions, result.cost, result.expanded)
```

- Pick the algorithm: `bfs`, `dfs`, `ucs`, `astar`, `weighted_a_star_search`. All take `graph_search=True` (an explored set, the default) and `max_expansions=None`.
- **States must be hashable and compare with `==`.** A continuous state needs rounding or binning before it is useful in graph search.
- Step costs must be non-negative. A\* is optimal for an admissible heuristic (tree search) or a consistent one (graph search).
- Custom strategies: build a `Frontier` subclass (`_push`, `_pop`, `_on_conflict`) and call `generic_search(problem, frontier)`.

### MDP / value iteration

```python
class MyMDP(MarkovDecisionProcess):
    def start_state(self): ...
    def actions(self, state): ...                  # [] for terminal states
    def transitions(self, state, action): ...      # [(next_state, probability), ...]
    def reward(self, state, action, next_state): ...
    def is_terminal(self, state): ...
    # optional: def states(self): ...  (default: all states reachable from start_state)

vi = ValueIterationAgent(MyMDP(), discount=0.95, iterations=500, tolerance=1e-8)
vi.get_value(s); vi.get_q_value(s, a); vi.get_policy(s)
```

- The state set must be finite. If `states()` is not overridden it is found by breadth-first search from the start state (`reachable_states`). Pass a `limit` if you are unsure how big it is.
- Deterministic problems return `[(next_state, 1.0)]`.
- Terminal states have value 0 and no actions.
- `tolerance` stops sweeping once the largest value change is below it. `vi.sweeps` and `vi.last_delta` report how long it ran.
- Memory and time are O(|S| · |A| · branching) per sweep. The code is plain Python dictionaries, so expect a practical limit of about 10⁵–10⁶ states. Beyond that use numpy arrays or a coarser discretisation.

### Reinforcement learning

```python
class MyEnv(Environment):
    def reset(self): ...                   # returns the start state
    def actions(self, state): ...          # legal actions
    def step(self, action): ...            # returns (next_state, reward, done)

env = MyEnv()
agent = QLearningAgent(env.actions, alpha=0.5, epsilon=0.1, gamma=0.9, rng=random.Random(0))
log = train(env, agent, episodes=5000, max_steps=1000)
result = evaluate(env, agent, episodes=100)      # greedy, no learning
```

- `evaluate` temporarily sets epsilon to 0 and does not update, then restores epsilon.
- `ApproximateQAgent(env.actions, extractor)` replaces the table with linear weights over `extractor(state, action) -> {feature: value}`. Use it when the state space is large or continuous.
- `MDPEnvironment(mdp)` turns any known `MarkovDecisionProcess` into an `Environment`. This runs Q-learning on exactly the problem value iteration solves, so the two results can be compared.
- Episodes stop at `done` or after `max_steps` (reported as `terminated=False`).
- There is no epsilon or alpha decay. Change `agent.epsilon` and `agent.alpha` between `train` calls if you want a schedule.

## Tests and example problems

`test/toy_problems.py` has small problems with known answers: `GridMaze` (search), `TinyGridworld` (stochastic MDP) and `ToyJetpack` (a simplified vertical jetpack game, defined once and used as a search problem, an MDP, and, via `MDPEnvironment`, an RL environment).

`ToyJetpackDynamics` is the template for adapting the real game: one function holds the physics, and each module gets a thin wrapper around it.

Run the tests from the repo root:

```text
python -m unittest discover -s test -v
```

They check, among other things: BFS, UCS and A\* agree on unit-cost mazes, UCS handles weighted cells, failure is reported on unreachable goals, value iteration matches hand-computed values and satisfies the Bellman equation, Q-learning matches value iteration on the gridworld, and all three solve the toy jetpack level.

Q-learning is stochastic. The tests use fixed seeds, and the settings were also checked on 10 seeds each.

## Limits to know about

- **Continuous states** must be discretised, or approximated with features, by the problem definition. The modules do not do this for you.
- **Large state spaces.** Graph search stores every explored state. With a 2-action game and long horizons, use a good heuristic, a frame-skip, `max_expansions`, or receding-horizon replanning.
- **Partial observability and multiple agents** are not supported (not needed for the proposal).
- **Performance** has not been benchmarked on anything larger than the toy problems.
