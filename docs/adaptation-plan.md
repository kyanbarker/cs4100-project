# Plan: adapting Jetpack Goodride to the generic search / MDP / RL code

This is a plan only. No game code has been changed. It builds on `docs/jetpack-goodride.md` (how the game works), `docs/agents.md` (the generic modules), and `feedback.md` (open problems in the formulation).

## 1. Approach: a headless simulator beside the game, not edits to the game loop

`game.py` mixes logic and drawing, takes `dt` from the real clock, signals death through pygame events, and keeps state in about 30 attributes (`jetpack-goodride.md`, section 4). Wrapping that loop in `Environment.step` would mean refactoring it first, and it would still run at rendering speed.

Instead, write a small **pure-Python simulator** that reimplements the physics from `jetpack-goodride.md`, section 8. Then:

1. The generic modules are driven through thin adapters, exactly as `ToyJetpackDynamics` does in `test/toy_problems.py`.
2. The simulator is **validated against the real game** (section 5), so we can say it is faithful.
3. pygame is kept only to **replay** a learned policy or plan visually (section 7).

Proposed layout (new, under `src/`):

```text
src/jetpack/
  constants.py     # numbers copied from game/scripts/settings.py + sprite sizes
  sim.py           # State, Action, step(state, action, dt) -> State   (the transition function)
  layout.py        # obstacle layouts: seeded random, fixed, or hand-written
  discretize.py    # snapping continuous states to hashable keys
  problems.py      # JetpackSearchProblem, JetpackMDP, JetpackEnv (adapters)
  features.py      # feature extractor for ApproximateQAgent
  replay.py        # drive the real pygame game with a policy (needs pygame, optional)
  validate.py      # compare sim against recorded game traces
```

`sim.py` has no dependency on `agents` and `problems.py` is the only module that imports both. That keeps the generic code generic.

## 2. How game concepts map onto the generic interfaces

| Game concept | Search (`SearchProblem`) | MDP (`MarkovDecisionProcess`) | RL (`Environment`) |
|---|---|---|---|
| State | snapped key (section 3) | snapped key | snapped key (or raw state plus features) |
| Actions | `["up", "none"]` (thrust / no thrust) | same | same |
| Transition | `successors`: apply `sim.step` once per decision interval, **skip dead states** | `transitions`: `[(sim.step(...), 1.0)]`, or a list over the possible next-zapper heights (stage 2) | `step`: `sim.step`, returns `(s', r, done)` |
| Death | state with no successor | terminal state, reward `-D` | `done = True` |
| Goal / horizon | `is_goal`: passed `N` zappers or reached distance `X` | terminal when goal reached | `done` at goal or `max_steps` |
| Step cost / reward | cost 1 per decision (time), or distance | `+1` per step alive (or per px), `-D` on death | same as MDP |
| Heuristic | remaining distance ÷ max speed (admissible); see section 6 | n/a | n/a |
| Randomness | fixed or seeded layout, or replan online | chance node over next zapper `y` | seeded RNG in `reset()` |

## 3. State definition and discretisation

**Simulator state** (what `sim.step` takes and returns, a frozen dataclass or tuple):

```text
y, vy            # float; player vertical position and velocity
vx               # float; world speed (constant in stage 1)
scroll           # float; total distance scrolled (replaces foreground_pos_x)
zappers          # tuple of (x, center_y) for zappers currently in the world
rockets          # tuple of (x, y)    (stage 4 only)
dead             # bool
```

Not in the state: acceleration (derived from the action and `vx`), sprites, score, and the intro animation (skip it; start at steady state with `y` mid-screen and `vy = 0`).

**Decision interval and `dt`.** The real game has no fixed `dt`. Pick one and keep it everywhere:

- Simulator tick: `dt = 1/60 s`.
- Decision interval: `k` ticks per action, where `k` is a parameter (start with `k = 6`, so 0.1 s, and try 3 and 12). The action is held for `k` ticks, and a collision on any tick counts.
- Horizon check: 10 zappers at 0.9–1.3 s each is about 13 s, so 130 decisions at `k = 6`. That is far shorter than the 800 ticks of a per-tick decision.

**Snapped key for tabular methods and graph search.** The adapters work on the key and call `sim.step` on the un-snapped version of it, so the discretised model is "snap, step, snap". Candidate key for stage 1 (obstacles only, fixed `vx`):

```text
(y_bin, vy_bin, dx_bin, gap_bin)
  y_bin   : round((y - 80) / 20)               -> about 26 values  (80..581)
  vy_bin  : round(vy / 60), clipped            -> about 40 values  (|vy| <= ~1200 px/s)
  dx_bin  : round((zapper_x - player_right) / 40), clipped   -> about 25 values (0..1000 px)
  gap_bin : round((zapper_center_y - 180) / 25)              -> about 15 values  (180..554)
```

- That is at most about 26 × 40 × 25 × 15 ≈ 390k keys. Far fewer are reachable, so **measure `len(reachable_states(mdp))` first** before running value iteration. With plain-Python dicts, aim for under about 10⁵ keys, and coarsen the bins if it is bigger.
- Only the **next zapper** is in the key. The following one is 680 px behind it, so it matters later, but leaving it out keeps the table small. Add `dx2`/`gap2` as a stage 2 extension if the agent clips zappers when planning ahead.
- Snapping makes the model approximate: a bin hides up to half a bin of `y` or `vy`. Always **check the resulting policy by rolling it out in the exact simulator** and report that success rate (not the success rate inside the discretised model).
- `vx` is added to the key as `vx_bin` (for example 5 bins) in stage 3.

**Features for `ApproximateQAgent`** (continuous alternative, no binning):

```text
bias, dy_to_gap_center / 400, vy / 1000, dist_to_ceiling / 500, dist_to_floor / 500,
(dy_to_gap_center / 400) * (dx / 1000), thrust_is_on * (vy / 1000)  ...
```

Features are returned per (state, action) pair (for example, multiply by `+1`/`-1` for the action direction). Keep values roughly in [-1, 1]. Expect to iterate on these.

## 4. Stages (the "ladder")

Each stage must work for all three modules before moving on. Stages 1–2 are the realistic goal for the 12/4 milestone, per the feedback.

| Stage | World | Search | MDP / VI | Q-learning | Success criterion |
|---|---|---|---|---|---|
| 0 | `ToyJetpack` (already in `test/`) | ✅ | ✅ | ✅ | exists, tests pass |
| 1 | Zappers only, **fixed layout** (seeded, same 10+ zappers every run), constant `vx = 480`, no intro | A\* to "pass N zappers" | VI on snapped key | tabular Q on snapped key | passes 10 zappers in 100/100 runs |
| 2 | Zappers, **random layout** (the real `randint(180, 554)`) | receding-horizon A\* over visible zappers | VI with chance node over next zapper's `gap_bin` | Q-learning over many seeds (no model needed) | passes 10 zappers in at least 90% of 100+ random layouts (the proposal's milestone) |
| 3 | `vx` ramps 480 → 690 | same, with `vx` in state | add `vx_bin` | same | same criterion at speed |
| 4 | Homing rockets | rockets in the search state (the heuristic gets weaker) | rocket `(dx, dy)` bins in the key (large) | features or bins | survival time vs. baseline |
| 5 | (stretch) coins, anything else | n/a | n/a | n/a | only if time remains |

**Baselines** (add in stage 1, report in every stage): a random policy, a hand-written "steer to the gap centre" controller, and a human run if one is available.

## 5. Making sure the simulator matches the game

Without this step the results describe our model, not the game.

1. **Record traces from the real game.** Add an optional logger (gated by an env var, so normal play is unchanged) that writes one row per `update_game` call: `dt, is_moving_up, player_pos_y, player_vel_y, player_vel_x, foreground_pos_x, died`. Scripted input is easiest: have a tiny driver post `KEYDOWN`/`KEYUP` events on a schedule. `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy` let it run without a window.
2. **Replay each trace through `sim.step`** using the logged `dt` (the real game's `dt` varies, so use the logged values, not 1/60). Compare `y`, `vy` and `foreground_pos_x` frame by frame.
3. **Acceptance:** differences below a small tolerance (for example 1 px in `y` over a 10 s run; `dt` rounding and `Rect` truncation are the expected sources). Anything larger means a mistake in the transition function. The quirks listed in `jetpack-goodride.md` (floor clamp using sprite width, one-frame overshoot, checks on the old `y`) must be reproduced, or the traces will drift.
4. **Collision check:** confirm separately that the simulator's death frame matches `died` in the trace, at least for a few runs that crash into a zapper from above, from below, and head-on.
5. **Resolution of `dt`:** run the simulator at 1/60 s and at the logged `dt` and measure how far apart they end up after 10 s. This tells us whether `dt = 1/60` is an acceptable modelling choice (the estimate from `jetpack-goodride.md`, section 8, is about 0.4 px per step).

## 6. Algorithm notes for this problem

- **Search.** "Survive forever" has no goal, so use "pass N zappers" and plan on the visible zappers only. For online play, replan every `m` decisions (receding horizon) and execute the first few actions.
  - **Heuristic.** `h = (X_goal - x) / v_max` is admissible when cost is time. A heuristic based on distance to the gap is not obviously admissible. If used, label the search "weighted A\*" (`weighted_a_star_search` exists) and do not claim optimality.
  - **Pruning.** Dead states are not generated, so collisions act as constraints, not costs.
  - **Budget.** Use `max_expansions`. The tree is 2^130 deep without pruning, so success depends on the `(y_bin, vy_bin)` key merging equivalent states.
- **Value iteration.** Use `reachable_states` to enumerate the key set. Choose `gamma` so the horizon matters (for example 0.99 with a 0.1 s decision interval is about a 10 s horizon). Start with `+1` per step alive and `-100` on death. Death is absorbing and terminal. The stage 2 chance node is a list of 15 `(next_state, 1/15)` pairs, one per next-zapper `gap_bin`.
- **Q-learning.** Training needs roughly 10⁴–10⁶ episodes. With the pure-Python simulator that is feasible only if each episode is short, so start with `max_steps` set to about 300 decisions. Use a seeded RNG, a decaying epsilon (change `agent.epsilon` between `train` calls), and record the learning curve. If the table is too sparse, switch to `ApproximateQAgent`. Compare against the value-iteration policy on stage 1 (they should agree), which is the main correctness check.
- **Evaluation harness** (shared by all three): fixed list of seeds, `N >= 100` episodes, report the fraction passing 10 zappers with a binomial confidence interval, plus mean survival distance.

## 7. Replay in the real game

`replay.py` would run the pygame game with a policy choosing `is_moving_up` each frame, so we can demonstrate and spot-check what the agent does. This needs only a small hook in `Game.check_events` or `move_things` (set `self.is_moving_up` from a callable). It also needs the game's zapper heights to be the layout the policy was trained on (seed `random` before `Game()`, or inject a layout). Keep the hook behind a flag so the game still plays normally for humans. Remember the real game's `dt` is variable, which is another reason to run the agent's decisions on game time, not frames.

## 8. Work order (calendar from the proposal; today is 11/18)

| When | Task | Output |
|---|---|---|
| 11/18–11/21 | `constants.py`, `sim.py` (stage 1: zappers, fixed `vx`), seeded `layout.py`, unit tests (clamping, thrust sign, scroll speed, collision inset) | simulator passing hand-computed checks |
| 11/18–11/21 | Trace logger and `validate.py` | simulator matches the game within tolerance, or a list of the differences |
| 11/22–11/28 | `discretize.py`, `problems.py`; stage 1 with all three modules; baselines; evaluation harness | stage 1 results table |
| 11/22–11/28 | Count `reachable_states`; tune bins and `k`; compare VI vs Q-learning | a documented choice of discretisation |
| 11/29–12/4 | Stage 2 (random layouts) and the milestone report; stage 3 if time allows | milestone numbers: success rate over 10 zappers |
| 12/5–12/11 | Rockets (stage 4) or deepen analysis; draft report | draft report |
| 12/12–12/18 | Polish, final report | final report |

Stage 4 and later are stretch. Cut them first if the schedule slips.

A possible way to split the work, matching the feedback: one person each for the simulator plus validation, search plus MDP, and RL plus the evaluation harness. The proposal's current split (one implementer, one tester, one documenter) puts the simulator on a single person's critical path.

## 9. Decisions needed from the team

1. **Decision interval `k`** (and `dt = 1/60`): 6 ticks (0.1 s) is only a starting guess.
2. **Reward and discount:** `+1` per step with `-100` on death, or distance-based; `gamma`.
3. **Stage 1 goal:** "pass 10 zappers" matches the milestone. Confirm.
4. **Fixed versus random layouts for the milestone:** stage 1 only, or stage 2 too?
5. **Intro and speed ramp:** skip the intro in all stages (recommended), and when to turn on the ramp (stage 3)?
6. **Repo visibility:** the PA code is Berkeley-licensed, so confirm that the GitHub repo is private (or acceptable under the course policy) before pushing `src/agents/`.

## 10. Risks

- **Simulator drift** from the real game if a quirk is missed. Mitigation: section 5 before any results.
- **State-space blow-up** with stages 2–4. Mitigation: measure the reachable count early and use features instead of bins when needed.
- **Q-learning sample efficiency** in pure Python. Mitigation: shorter episodes, seeded runs, vectorise the simulator only if profiling shows it is the bottleneck.
- **Discretisation bias.** A policy that is optimal on snapped states can fail in the exact simulator. Mitigation: always evaluate in the exact simulator.
- **Schedule.** The headless simulator was not in the original week-by-week plan, and the first week's "general classes" step has just been completed. Stages 1–2 are the realistic target for 12/4.
