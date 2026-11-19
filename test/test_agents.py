"""Sanity checks for src/agents. Run from the repo root:

    python -m unittest discover -s test -v
"""

import random
import unittest

import toy_problems as tp  # also puts src/ on sys.path
from agents import search
from agents.mdp import MarkovDecisionProcess, ValueIterationAgent, reachable_states
from agents.rl import ApproximateQAgent, MDPEnvironment, QLearningAgent, evaluate, train

MAZE = [
    "#########",
    "#S..#...#",
    "#.#.#.#.#",
    "#.#...#G#",
    "#########",
]


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.maze = tp.GridMaze(MAZE)

    def test_bfs_ucs_astar_agree_on_unit_costs(self):
        costs = {
            "bfs": search.bfs(self.maze).cost,
            "ucs": search.ucs(self.maze).cost,
            "astar": search.astar(self.maze, self.maze.manhattan).cost,
        }
        self.assertEqual(set(costs.values()), {12.0}, costs)

    def test_solution_is_a_legal_path_with_matching_cost(self):
        for result in (search.bfs(self.maze), search.dfs(self.maze), search.ucs(self.maze)):
            self.assertTrue(result.found)
            self.assertEqual(self.maze.cost_of_actions(result.actions), result.cost)

    def test_astar_expands_no_more_than_ucs(self):
        self.assertLessEqual(
            search.astar(self.maze, self.maze.manhattan).expanded, search.ucs(self.maze).expanded
        )

    def test_ucs_avoids_expensive_cells(self):
        layout = [
            "#######",
            "#S999G#",  # direct route costs 28
            "#.###.#",
            "#.....#",  # detour is 8 moves, costs 8
            "#######",
        ]
        maze = tp.GridMaze(layout)
        self.assertEqual(search.ucs(maze).cost, 8.0)
        self.assertEqual(search.astar(maze, maze.manhattan).cost, 8.0)
        self.assertEqual(search.bfs(maze).cost, 28.0)  # BFS minimises moves (4), not cost; documents the difference

    def test_unreachable_goal(self):
        maze = tp.GridMaze(["#####", "#S#G#", "#####"])
        for fn in (search.bfs, search.dfs, search.ucs):
            self.assertFalse(fn(maze).found)

    def test_max_expansions(self):
        self.assertFalse(search.bfs(self.maze, max_expansions=2).found)

    def test_tree_search_finds_solution_too(self):
        self.assertEqual(search.bfs(self.maze, graph_search=False).cost, 12.0)


class ValueIterationTests(unittest.TestCase):
    class Det(MarkovDecisionProcess):
        """s0 --go--> s1 --go--> end (reward 1 on the second step).
        s0 --gamble--> win (reward 10) or lose (reward 0), 50/50. All three outcomes are terminal."""

        def start_state(self):
            return "s0"

        def states(self):
            return ["s0", "s1", "end", "win", "lose"]

        def is_terminal(self, s):
            return s in ("end", "win", "lose")

        def actions(self, s):
            return {"s0": ["go", "gamble"], "s1": ["go"]}.get(s, [])

        def transitions(self, s, a):
            if s == "s0" and a == "go":
                return [("s1", 1.0)]
            if s == "s0" and a == "gamble":
                return [("win", 0.5), ("lose", 0.5)]
            return [("end", 1.0)]

        def reward(self, s, a, s2):
            return {"win": 10.0, "lose": 0.0}.get(s2, 1.0 if s == "s1" else 0.0)

    def test_hand_computed_values(self):
        vi = ValueIterationAgent(self.Det(), discount=0.9, iterations=50)
        self.assertAlmostEqual(vi.get_value("s1"), 1.0)
        self.assertAlmostEqual(vi.get_q_value("s0", "go"), 0.9)  # 0 + 0.9 * V(s1)
        self.assertAlmostEqual(vi.get_q_value("s0", "gamble"), 5.0)  # 0.5 * 10
        self.assertEqual(vi.get_policy("s0"), "gamble")
        self.assertIsNone(vi.get_policy("end"))

    def test_gridworld_policy_and_convergence(self):
        mdp = tp.TinyGridworld(noise=0.2)
        vi = ValueIterationAgent(mdp, discount=0.9, iterations=1000, tolerance=1e-10)
        self.assertLess(vi.sweeps, 1000)  # converged early
        self.assertEqual(vi.get_policy((2, 0)), "E")  # next to +1 exit
        self.assertEqual(vi.get_policy((0, 0)), "E")
        self.assertEqual(vi.get_value((3, 0)), 0.0)  # terminal values are 0
        # Bellman optimality holds at convergence
        for s in mdp.states():
            if not mdp.is_terminal(s):
                best = max(vi.get_q_value(s, a) for a in mdp.actions(s))
                self.assertAlmostEqual(vi.get_value(s), best, places=8)

    def test_reachable_states_matches_declared_states(self):
        mdp = tp.TinyGridworld()
        self.assertEqual(set(reachable_states(mdp)), set(mdp.states()))
        with self.assertRaises(RuntimeError):
            reachable_states(mdp, limit=3)


class QLearningTests(unittest.TestCase):
    def test_q_learning_matches_value_iteration_on_gridworld(self):
        mdp = tp.TinyGridworld(noise=0.2)
        vi = ValueIterationAgent(mdp, discount=0.9, iterations=500, tolerance=1e-10)
        rng = random.Random(0)
        env = MDPEnvironment(mdp, rng)
        agent = QLearningAgent(env.actions, alpha=0.1, epsilon=0.3, gamma=0.9, rng=rng)
        train(env, agent, episodes=20000, max_steps=100)
        # Rarely visited (state, action) pairs are still under-trained after 20k episodes,
        # so compare only where the optimal action wins by a clear margin, and values on
        # states along the main route.
        for s in mdp.states():
            if mdp.is_terminal(s):
                continue
            qs = sorted(vi.get_q_value(s, a) for a in mdp.actions(s))
            if qs[-1] - qs[-2] > 0.15:
                self.assertEqual(agent.get_policy(s), vi.get_policy(s), s)
        for s in [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0)]:
            self.assertAlmostEqual(agent.get_value(s), vi.get_value(s), delta=0.1, msg=str(s))

    def test_evaluate_restores_epsilon_and_does_not_learn(self):
        mdp = tp.TinyGridworld()
        env = MDPEnvironment(mdp, random.Random(1))
        agent = QLearningAgent(env.actions, epsilon=0.25, rng=random.Random(1))
        evaluate(env, agent, episodes=5, max_steps=50)
        self.assertEqual(agent.epsilon, 0.25)
        self.assertEqual(len(agent.q), 0)

    def test_approximate_q_learns_a_linear_problem(self):
        # one state, two actions; reward = +1 for "a", -1 for "b", episode ends immediately
        class Bandit(tp.MarkovDecisionProcess):
            def start_state(self): return "s"
            def states(self): return ["s", "end"]
            def is_terminal(self, s): return s == "end"
            def actions(self, s): return [] if s == "end" else ["a", "b"]
            def transitions(self, s, a): return [("end", 1.0)]
            def reward(self, s, a, s2): return 1.0 if a == "a" else -1.0

        env = MDPEnvironment(Bandit(), random.Random(0))
        agent = ApproximateQAgent(
            env.actions, lambda s, a: {("is", a): 1.0}, alpha=0.2, epsilon=0.5, rng=random.Random(0)
        )
        train(env, agent, episodes=200)
        self.assertEqual(agent.get_policy("s"), "a")


class ToyJetpackTests(unittest.TestCase):
    """The same toy game solved with search, value iteration and Q-learning."""

    def setUp(self):
        self.dyn = tp.ToyJetpackDynamics()

    def _replay(self, actions):
        state = self.dyn.start()
        for a in actions:
            state = self.dyn.step(state, a)
            self.assertNotEqual(state, tp.DEAD)
        return state

    def test_search_solves_it(self):
        prob = tp.ToyJetpackSearch(self.dyn)
        results = [
            search.bfs(prob),
            search.ucs(prob),
            search.astar(prob, prob.heuristic),
        ]
        for r in results:
            self.assertTrue(r.found)
            self.assertEqual(r.cost, 12.0)
            self.assertTrue(self.dyn.is_goal(self._replay(r.actions)))
        self.assertLessEqual(results[2].expanded, results[1].expanded)

    def test_search_reports_failure_on_impossible_level(self):
        # a gap at the very top and bottom in consecutive columns cannot both be passed
        dyn = tp.ToyJetpackDynamics(obstacles={2: (0, 0), 3: (7, 7)})
        self.assertFalse(search.bfs(tp.ToyJetpackSearch(dyn)).found)

    def test_value_iteration_policy_reaches_goal(self):
        mdp = tp.ToyJetpackMDP(self.dyn)
        vi = ValueIterationAgent(mdp, discount=0.95, iterations=100, tolerance=1e-9)
        state, steps = mdp.start_state(), 0
        while not mdp.is_terminal(state) and steps < 50:
            state = self.dyn.step(state, vi.get_policy(state))
            steps += 1
        self.assertTrue(self.dyn.is_goal(state))
        self.assertEqual(steps, 12)

    def test_q_learning_reaches_goal(self):
        mdp = tp.ToyJetpackMDP(self.dyn)
        rng = random.Random(0)
        env = MDPEnvironment(mdp, rng)
        agent = QLearningAgent(env.actions, alpha=0.5, epsilon=0.2, gamma=0.95, rng=rng)
        train(env, agent, episodes=3000, max_steps=50)
        log = evaluate(env, agent, episodes=20, max_steps=50)
        self.assertTrue(all(e.terminated and e.total_reward == 12.0 for e in log.episodes), log.episodes[:3])


if __name__ == "__main__":
    unittest.main()
