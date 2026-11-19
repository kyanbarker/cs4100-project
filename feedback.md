# Feedback on the project proposal: *Barry's Big Blast*

Reviewed: `docs/project-proposal_barrys-big-blast.md`, against the course handout (`docs/project.pdf`) and the forked game in `src/game/` (see `docs/jetpack-goodride.md` for what the game actually does).

## Summary

The topic is a good fit. The game is easy to formulate, can be solved with all three course modules (search, MDP, RL), and has a natural ladder of difficulty. The proposal already follows the handout's main advice: start with the simplest subproblem and add complexity. The ideal-outcome and milestone sections are also concrete enough to test.

The main weakness is that the formulation is not yet precise enough to implement. Four things need fixing before the algorithms can run:

1. The state space is continuous, and none of the three chosen algorithms works on one directly.
2. The time step is unspecified.
3. The game is random, but the proposal calls it deterministic.
4. The search goal and the MDP rewards are not defined.

These matter more than anything else below. They decide whether the first implementation works.

## Priority issues

### 1. The state space is continuous and there is no plan to make it finite (high)

The proposal's state is floats for position, velocity and acceleration, plus obstacle colliders. That is a continuous space. Each planned algorithm needs more than that:

- **Value iteration** needs an enumerable state set. The PA3 code loops over `mdp.getStates()`.
- **Tabular Q-learning** needs a finite, reasonably small table. Otherwise no state is ever visited twice and nothing is learned.
- **Graph search** needs states that can be compared and hashed. Two float states are almost never equal, so the explored set never prunes anything.

The proposal says Q-learning will "teach itself", but it never says how states are generalised. Pick one and write it down:

- **Discretise.** Bin `y` and `vy` (and distance to the next obstacle) on a fixed grid, using a fixed `dt`.
- **Approximate Q-learning with features.** Reuse PA3's `ApproximateQAgent` with features such as `dy_to_gap_center`, `vy`, `distance_to_next_obstacle`, and `distance_to_ceiling` and `distance_to_floor`.
- **Both.** Use the discretised version as the sanity-checkable baseline and the feature version as the extension.

A rough size estimate is worth doing early. For example, 40 bins of `y` × 30 bins of `vy` × 20 bins of distance × 40 bins of next-gap `y` is about 960k states × 2 actions. That is feasible for value iteration in numpy but slow in pure Python dictionaries.

### 2. The time step is missing from the formulation (high)

The game runs a variable timestep: `dt = clock.tick(300)/1000`. It is capped at 300 FPS and varies with the machine. The proposal says "at every frame, the player chooses an action", but never says how long a frame is. For a Markov model, `dt` has to be a constant, for example 1/60 s. It is part of the transition function.

You should also consider frame-skip, meaning one decision every k frames. At 60 Hz, "avoid 10 obstacles" is about 800 decisions (obstacles are spaced 680 px apart and scroll at about 528 px/s at the start, so roughly 1.3 s each). With 2 actions that is a huge tree for A*. Choosing a decision interval of about 0.1 s changes the problem a lot: the horizon is about 130 decisions, not 800. Flag this as a design parameter.

### 3. "Deterministic transitions" is only true if the layout is fixed (high)

The proposal says that "if we know the laws of physics … transition probabilities are deterministic". The physics is deterministic, but the world is not: in the fork, each zapper's height is `random.randint(180, 554)` and rockets are spawned by `random.randint(1, 6) == 1`.

This affects each module differently:

- **Search** (and open-loop planning in general) needs the future obstacle positions. Either use seeded or fixed levels for the first stage (reasonable, and it also helps testing), or plan online with a receding horizon over the obstacles currently visible. The handout explicitly describes replanning as an accepted approach for the milestone.
- **MDP** has to model the next obstacle's height as a random variable. A uniform distribution over 375 possible values gives a real transition model. Alternatively, treat the spawn as part of the environment and only plan over a lookahead window.
- **RL** is unaffected, which is a point in its favour.

State this assumption explicitly and decide which version is "stage 1".

### 4. Goal test, rewards and the heuristic are underspecified (high)

- **Search needs a goal.** "Survive forever" has no goal state, so A* and UCS have nothing to terminate on. Define a finite goal, such as "reach distance D" or "pass N obstacles". The milestone's "avoid 10 obstacles" is a natural choice.
- **Admissible heuristic.** For "reach distance D at minimum time", `h = remaining_distance / v_max` is admissible. A heuristic built from "velocity, acceleration and distance to obstacles" is much harder to make admissible, because obstacle avoidance acts as a constraint (states pruned), not as a cost. If it is only an inadmissible guide, say so and call the search "greedy" or "weighted A*".
- **MDP / RL reward.** The proposal says "we will define rewards" but gives none. A death-only penalty gives a sparse signal and slow Q-learning. A common choice is `+1` per step alive (or per unit of distance) and a large negative reward on death, with `gamma < 1`. Say which one you plan to use, and whether it is shaped.
- **Terminal handling** ("empty action space, reward 0") is fine and matches the PA3 convention.

### 5. The "adapt the forked repo" step is bigger than the plan suggests (high)

Reading `src/game/scripts/game.py`:

- Game logic and drawing are interleaved. `move_things()` blits sprites, and `update_game()` ends in `pygame.display.update()`.
- The timestep comes from the real clock (`main.dt`).
- Death is signalled through `pygame.event.post`.
- State is spread over about 30 attributes on one `Game` object.
- The file's own header says it needs to be made more object-oriented.

There is no headless mode. Training Q-learning on a real-time pygame loop at ≤300 FPS, with rendering, would be far too slow. The handout says dev problems should run "in seconds … certainly under a minute".

I recommend the following instead of "adapt the repo":

- Write a small headless simulator (pure Python or numpy, no pygame) that reimplements the physics documented in `docs/jetpack-goodride.md`.
- Check it against traces recorded from the real game.
- Keep pygame only for visualising a learned policy.

The week-by-week plan has one week for this. I would budget at least that, and probably more.

### 6. The state description does not match the game (medium)

- **Acceleration is not state.** In the game the vertical acceleration is fully determined by the action and the horizontal speed: `±1.7 × 1.8 × v_x`. Storing it in the state is redundant. If your actions set acceleration directly, define that explicitly. In the game, the action is just "thrust on or off".
- **The actions are not "increase / decrease" acceleration.** The game has two actions, thrust on and thrust off (hold or release `W`). Gravity and thrust are the same magnitude and opposite direction, so the player is effectively bang-bang controlled. The proposal wording is ambiguous.
- **Horizontal speed is a state variable.** `v_x` is not constant. It ramps from 0 to 480 px/s over the 4 s intro, then grows by 2.5 px/s² up to 690 px/s. Because vertical acceleration scales with `v_x`, the dynamics change over a run. This breaks "obstacle horizontal velocity strictly < 0 at all times" only during the intro, but it does mean a time-invariant model needs `v_x` (or elapsed time) in the state.
- **Option 1 vs Option 2.** The game is exactly Option 1: the player's `x` is fixed at 256 after the intro and the world scrolls. Say so and drop Option 2.
- **Colliders.** The proposal's formal definition (bounded convex open sets) is correct but heavier than needed. The game uses axis-aligned `pygame.Rect`s. The obstacle hitbox is inset 25 px horizontally and 18 px vertically from its sprite. The player hitbox is the full 64×68 sprite. Rects that only touch along an edge do not collide (this matches your "open set" definition). Use "axis-aligned rectangle" and note the inset.

### 7. Scope: "every feature of the original game" (medium)

The handout's most common mistake is setting the scope too large. "Formulate a problem that captures the behaviour of every feature in the original game" is more than the forked game contains:

- It has zappers (static, non-rotated) and homing rockets.
- Coins are a TODO.
- There are no vehicles, power-ups or lasers.

Reaching those means implementing them in the game first. Reword the ideal outcome as a staged ladder with a stopping rule, for example:

1. Fixed-layout zappers, fixed speed.
2. Random zappers.
3. Increasing speed.
4. Rockets.
5. (Stretch) coins.

Each stage should have a success criterion. Rockets are the first stage with state-dependent obstacle motion. They home vertically toward the player while off-screen, so they should be a stage of their own.

### 8. Milestone and evaluation protocol (medium)

- "Avoid 10 obstacles at least 90% of the time" is a good measurable target. Add how it will be measured: the number of episodes (for example 100 or more), seeds, and which stage's layout.
- Check feasibility. The zapper spacing and the vertical accelerations (about 1470 px/s² at the start) probably permit 10 in a row for every layout, but verify. Compute the worst consecutive-gap change versus the time available. That tells you whether 90% is a ceiling set by the game or a ceiling set by the agent.
- Add baselines so results can be interpreted: a random policy, a "hold at the gap center" PD controller, and a human score.
- The milestone date is 12/4 and bundles formulation, implementation and "a decent agent". The week plan reflects that, but it is tight. Consider making "a first agent runs end-to-end on stage 1" the milestone, and "decent agent" the stretch.

### 9. Division of labour (medium)

- Implementation is assigned to one person (Jesse), Documentation to one (Kyan) and Testing to one (Ran). If any one of them is unavailable, a whole phase stops.
- The rationale for three people is "try multiple approaches". The division of labour does not reflect that. A more natural split is one person per approach (search, MDP, RL) with shared ownership of the simulator and the evaluation harness. That also helps each person show what they implemented, which is one of the grading dimensions.

## Smaller points

- Typos: "Bary" (twice), "than" for "then" in the team rationale, and the stray backslash in the URL escapes.
- Name: the proposal says *Barry's Big Blast* while the repo and game say *Jetpack Goodride / Joyride*. Pick one.
- "Libraries / topics to learn" lists only the game repo. Add pygame (needed to read the game) and, if used, function-approximation Q-learning or numpy for value iteration. The handout asks you to say what you had to learn and roughly how long it took, so start recording that.
- The proposal says "Value/policy iteration" but PA3 only has value iteration. Policy iteration needs to be written if you want it.
- Discounting and the horizon are not mentioned for the MDP. Say whether you will use a finite horizon or `gamma < 1`.
- Mention the randomness in the heading of the MDP section, not only in the transition model.
- The handout requires the final report to explain "what the existing work did and what you did differently". Start keeping a short log of changes to the forked game from day one, to make that section easy.

## Status against the week plan (as of 11/18)

- 11/08–11/14 (environment setup, formulation, general PA-based classes): the repo, fork, and proposal exist. The general code is only now being written (see `src/agents/`).
- 11/15–11/21 (adapt the game): not started. See issue 5 above before starting.

The plan is still achievable, but the headless simulator should be the next priority, because every agent depends on it.
