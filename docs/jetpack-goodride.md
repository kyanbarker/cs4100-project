# Jetpack Goodride: how the game models states, actions and physics

Source: `src/game/` (fork of Gustavo-Pauli/Jetpack-Goodride). Almost all gameplay logic is in `src/game/scripts/game.py`. Constants are in `src/game/scripts/settings.py`, and the main loop is in `src/game/main.py`.

Everything below is from reading the code. Nothing here has been checked against a running game, so numbers marked "derived" are computed by hand from the code.

Coordinates are pygame's: **x grows to the right, y grows downward**. "Up" is negative y.

## 1. Constants that matter (`settings.py`)

| Constant | Value | Meaning |
|---|---|---|
| `WIDTH`, `HEIGHT` | 1280, 720 | window size (px) |
| `FPS` | 300 | frame-rate cap passed to `clock.tick` |
| `DEFAULT_X_VELOCITY` | 480 | initial world speed (px/s) |
| `MAX_X_VELOCITY` | 690 | speed cap (px/s) |
| `GRAVITY` | 1.7 | dimensionless gravity factor |
| `MAX_HEIGHT` | 80 | ceiling (smallest allowed `player_pos_y`) |
| `MIN_HEIGHT` | 645 | floor line (see the clamp in section 6) |
| `FIRST_OBSTACLE_OFFSET` | 1920 | x where a new obstacle appears |
| `OBSTACLE_OFFSET` | 680 | spacing between obstacles (px of scroll) |

`MIN_OBSTACLE_OFFSET` and `MAX_OBSTACLE_OFFSET` are defined but never used.

## 2. How are actions and states represented?

### Actions

There is no action object. The only player input that affects gameplay is whether `W` is held. This is stored as one boolean on the game object: **`self.is_moving_up`**.

- `KEYDOWN W` sets `is_moving_up = True`; `KEYUP W` sets it to `False` (`Game.check_events`).
- So the action space is `{thrust, no thrust}`, which is a binary action held across frames. It is not an "increase / decrease acceleration" command.
- `ESC` toggles `paused`. That is not a gameplay action.
- Input is ignored once `dead` is `True`.

### State

There is no state object either. The state is spread over attributes on `Game` (created in `Game.__init__`):

| Group | Attributes | Type |
|---|---|---|
| Player | `player_pos_x`, `player_pos_y`, `player_vel_x`, `player_vel_y` | Python `float` |
| Player | `is_moving_up`, `dead`, `paused` | `bool` |
| Player | `player_rect` | `pygame.Rect` (integer; x, y set from `player_pos_*` every frame) |
| World scroll | `foreground_pos_x` (starts 0, decreasing), `bg_pos_x` | `float` |
| Obstacles | `obstacles_list`: list of `[pygame.Rect, relative_pos]` pairs; `obstacle_num` (count created) | list / `int` |
| Rockets | `rocket_spawner.rocket_list`: list of `Rocket` (`position` tuple `(x, y)`, `warning_active`) | objects |
| Start-up animation | `lerp_start_velocity`, `timer1`, `lerp_x_vel`, `lerp_factor`, `player_vel_x_start` | `bool` / `float` |
| Bookkeeping | `score`, `high_score`, `coins_collected`, `died_by` (`'eletricity'` / `'rocket'`) | `float` / `int` / `str` |

Notes:

- Positions, velocities and `dt` are Python floats (64-bit doubles). The only integers are the `pygame.Rect` values. Assigning a float to `Rect.x` or `Rect.y` truncates it, so **collisions use truncated integer positions** while physics uses floats.
- There is no vector type. The player is `(pos_x, pos_y)` and `(vel_x, vel_y)` as separate scalars. After the intro, `player_pos_x` is a constant 256, so the player's motion is 1-D (vertical) and the world scrolls left.
- This is the proposal's **Option 1** (player does not move horizontally; obstacles move left).
- **Acceleration is not stored.** It is computed inside `move_things` from `is_moving_up` and `player_vel_x` and applied directly to `player_vel_y`.
- `Zapper` is a single sprite, `Zapper1.png`, 46×109 px, scaled 2× to 92×218. The obstacle's `Rect` is that sprite size.

## 3. How are actions performed?

`is_moving_up` is read once per frame in `Game.move_things`:

```python
if not self.is_moving_up:
    self.player_vel_y += self.gravity * self.main.dt * self.player_vel_x * 1.8
else:
    self.player_vel_y -= self.gravity * self.main.dt * self.player_vel_x * 1.8
```

- With the key released, `vel_y` increases (the player falls). With it held, `vel_y` decreases (the player rises).
- Both branches have the same magnitude. Thrust is not "thrust plus gravity". The net vertical acceleration is `+a` or `-a`.
- The acceleration magnitude is `a = 1.7 × 1.8 × player_vel_x = 3.06 · v_x` px/s². At `v_x = 480` that is 1468.8 px/s², and at `v_x = 690` it is 2111.4 px/s². Because it depends on `v_x`, the vertical dynamics change as the game speeds up.
- The jetpack sprites, particles and sound are cosmetic (`ParticleGenerator.update(..., self.is_moving_up)`) and do not feed back into the physics.

## 4. How are states updated, and how often?

**The update function is `Game.update_game(main)`** (`game.py:141`). The main loop (`main.py:40`) calls it once per iteration while `main.playing` is true:

```python
while self.playing:
    self.update_dt()                 # dt = clock.tick(FPS) / 1000
    self.game.update_game(self)
```

- `clock.tick(300)` caps the loop at **300 calls per second at most**. The real rate is lower if a frame takes longer than 3.3 ms (drawing a 1280×720 window with scaled sprites and particles may well do this). It is machine dependent and not measured here.
- **The timestep is variable**: `dt` is the real elapsed time in seconds. It is not fixed and not recorded. Two runs with the same inputs are not frame-for-frame identical.
- Pausing sets `main.dt = 0` for that frame.

Order of work inside one `update_game` call:

1. `check_events` drains `pygame.event.get()`. It handles quit, key presses, the `DIED` event (marks `dead`, starts the speed-to-zero lerp, awards coins, updates high score), the music event, and the 1-second `TRY_SPAWN_ROCKET` timer event.
2. `check_obstacles` spawns an obstacle when `-foreground_pos_x > OBSTACLE_OFFSET × obstacle_num`, and drops the oldest when more than 12 exist.
3. `move_things` runs, in order: the death or start lerps; background scroll; `foreground_pos_x` update; obstacle positions; particles; **vertical velocity update; vertical clamp and position update**; blit player; score.
4. `check_rockets`, then `rocket_spawner.update` (move, delete, collide, draw rockets).
5. Draw HUD or death screen, `check_collisions` (zappers), `update_x_velocity` (speed ramp), debug overlays, pause overlay, `pygame.display.update()`.

Things that matter for anyone reusing this code:

- **Logic and rendering are interleaved.** `move_things` and `move_obstacles` blit sprites as they update state, and `update_game` ends with `display.update()`. There is no headless step function.
- **State changes happen in event handlers.** Collisions only post a `DIED` event with `pygame.event.post`, and `dead` is set when that event is read on the **next** frame.
- State is mutated in place across many methods. There is no function from `(state, action)` to the next state.

## 5. Positions, velocities, acceleration: representation and update laws

Notation: `y` = `player_pos_y`, `v` = `player_vel_y`, `vx` = `player_vel_x`, `F` = `foreground_pos_x`, `g = 1.7`, `dt = main.dt`.

| Quantity | Representation | Update |
|---|---|---|
| Player `x` | float | Fixed at 256 after the intro (see below). |
| Player `y` | float, px, down is positive | Semi-implicit Euler: `y += v·dt` (using the already-updated `v`), with clamping (section 6) |
| Player `v` (vertical) | float, px/s | `v += s·g·1.8·vx·dt`, where `s = +1` (no thrust) or `-1` (thrust) |
| Vertical acceleration | not stored | `a = s · 3.06 · vx` |
| World speed `vx` | float, px/s | `vx += 2.5·dt` per frame while `vx < 690`, not dead and the intro is over. Applied at the **end** of the frame |
| Background | float | `bg_pos_x -= vx·dt`, wrapping every 1536 px |
| Foreground offset `F` | float | `F -= vx·dt·1.1` (1.1 gives parallax) |
| Obstacle `x` | `Rect.x` | `obstacle.x = F + relative_pos` (recomputed every frame) |
| Rocket `x`, `y` | tuple of floats | see section 7 |
| Score | float | `score += dt·vx·0.05` while alive |

**Intro (first ~4 s).** `lerp_start_velocity` starts `True`. `timer1` grows by `dt/4`. While it is below 1, `vx = 480 - lerp(0, 480, t)` and `x = 256 - lerp(100, 256, t)`. It ends when `t ≥ 1`, then `vx = 480` and `x = 256`.

- The helper `Game.lerp(A, B, C)` is written `(1-C)·A + (1-C)·B`. The usual formula is `(1-C)·A + C·B`. This is a bug, but all three call sites happen to give sensible results. The start-up ramp is `vx = 480·t` (from 0 to 480) and `x = -100 + 356·t` (from -100 to 256). The death slow-down is `vx = (1-f)·vx_start` (the `B = 0` case).
- **During the intro `vx ≈ 0`, so the vertical acceleration is also ≈ 0.** The player starts at `y = 645`, which is below the floor clamp, so the first frame snaps it to the floor.

**Death.** `dead = True`, `is_moving_up = False` (input ignored), and `vx` lerps linearly to 0 at rate `0.75/s`. The player keeps falling under the normal rule until the floor. The player sprite changes to a larger dead sprite, 82×74, so the floor clamp changes too.

## 6. Bounds and collisions

### Vertical clamp (`move_things`)

The checks use the position from the **previous** frame, then integrate:

```python
if y < MAX_HEIGHT:                         # 80, "touching the ceiling"
    y = MAX_HEIGHT; v = 0
elif y > MIN_HEIGHT - player_surface.get_size()[0]:   # 645 - 64 = 581 (flying)
    y = MIN_HEIGHT - player_surface.get_size()[0]; v = 0
else:
    y += v * dt
```

- The upper bound on `y` is 581 while flying (563 for the 82-px-wide dead sprite). The code uses `get_size()[0]` (the **width**) where the sprite **height** (68 or 74) looks intended. This is probably a bug, though it only shifts the floor by a few pixels.
- Because bounds are checked before integrating, the player can overshoot a bound by one frame's motion before being snapped back.
- Hitting a bound zeroes `v`. It is not a death.

### Collision tests (every frame, after movement)

- **Player collider:** `player_rect`, the full sprite rectangle at `(int(x), int(y))`, size 64×68 flying or 82×74 dead. (The `PlayerFly.png` file is 53×57 and is scaled to 64×68 at load.)
- **Zapper collider:** the obstacle's rect inset: `Rect(ox+25, oy+18, 92-50, 218-36)`, which is **42 × 182 px**.
- **Rocket collider:** `Rect(position, sprite size)`; the rocket sprite is 184×112 scaled by 0.75 to **138 × 84**. There is no inset.
- All colliders are **axis-aligned integer rectangles** tested with `Rect.colliderect`. They are convex, so they fit the proposal's definition. Rects that only touch along an edge do not collide.
- Collision is **discrete**, once per frame, so it can miss thin overlaps at very low frame rates. At normal speed (≤ 759 px/s world speed) a frame at 60 FPS moves about 13 px, which is smaller than the 42-px zapper hitbox width.
- A hit posts the `DIED` event and sets `died_by`. There are no other terminal conditions (no ceiling or floor death).

## 7. Obstacle generation and rockets

**Zappers.**

- `check_obstacles` creates one when `-F > 680 × obstacle_num`, so one is created every 680 px of foreground scroll, starting immediately.
- A new zapper has its **center y** uniform in `randint(180, 554)` (inclusive), and stored `relative_pos = 1920 - F`, so it first appears at `x = 1920` (640 px to the right of the visible screen).
- Zappers then move left at `1.1·vx` px/s. At `vx = 480` that is 528 px/s, so one zapper arrives every `680 / 528 ≈ 1.29 s`. At `vx = 690` it is every 0.90 s.
- Keeps at most 12 in the list (oldest popped by count, not by position).
- The zapper's center y is the **only random value** in the level, apart from rockets.

**Rockets** (`Rocket`, `RocketSpawner`).

- Spawn condition, by my reading of `check_rockets`: only when `obstacle_num > 5` and the scroll has just crossed the next obstacle threshold (this is true for about one frame per obstacle), with a 1-in-6 chance. The `TRY_SPAWN_ROCKET` timer event also calls it every second, but it rarely lines up with a crossing frame. So roughly **1 rocket per 6 obstacles** after the first 5. This is derived, not measured.
- A rocket starts at `x = player_x + 3840`, with `y` uniform in `randrange(80, 645)`.
- Motion: x moves left at `vx · dt · 2.4` px/s (so 2.4 × world speed). While `x > 1280` (off-screen) it also homes vertically toward the player's `y` at **100 px/s**, up if the player is above, down if below. Once on-screen it moves in a straight line. A warning sprite is drawn at `x = 1200` while the rocket is off-screen.
- Edge cases: if the player's `y` equals the rocket's `y` exactly, neither branch runs and the rocket does not move that frame. Deleting while iterating with `pop(i)` can skip a rocket.

**Not implemented in this fork:** coins (the code only computes `coins_collected = int(score/11)` on death), power-ups, vehicles, and lasers. See `src/game/todo.md`.

## 8. Transition function: next state for a (state, action) pair

The code has no such function, so here it is, written out from sections 5–7. Assume **a fixed time step `dt`** (a constant such as 1/60 s, or a frame-skip multiple). A deterministic model needs this, because the game itself uses the real clock.

**State** `s = (y, v, vx, F, zappers, rockets, dead)`
**Action** `u ∈ {thrust, none}`, with `sgn = -1` for thrust and `+1` for none.

One step, in the order the code does it:

```text
# 0. gravity/thrust uses the CURRENT vx
v1  = v + sgn · 1.7 · 1.8 · vx · dt           # = v + sgn · 3.06 · vx · dt

# 1. scrolling uses the CURRENT vx
F'  = F − 1.1 · vx · dt
zapper.x = F' + relative_pos                   # for every zapper

# 2. vertical clamp, tested on the OLD y, then integrate with the NEW v
if   y < 80:                  y' = 80;        v' = 0
elif y > 645 − sprite_width:  y' = 645 − w;   v' = 0     # w = 64 (alive)
else:                         y' = y + v1·dt; v' = v1

# 3. collisions on (int(256), int(y')) with 64×68, vs shrunk zapper rects
#    and un-shrunk rocket rects
dead' = dead or any collision

# 4. speed ramp applied last (only when alive, intro over)
vx' = min(vx + 2.5·dt, ~690)         # overshoots the cap by < 2.5·dt
```

New zappers appear at `x = 1920` whenever `−F' > 680·n` (this is the only stochastic step, apart from rockets). If `vx` is treated as constant, the vertical motion is the same piecewise-constant-acceleration motion in `±a`: `v(t+Δ) = v(t) ± aΔ`, `y(t+Δ) = y(t) + v(t)Δ ± ½aΔ²`. The code's semi-implicit Euler differs from this by a term of order `a·dt²`, so with `dt = 1/60` the error is about 0.4 px per step at `vx = 480`.

**Terminal states:** `dead = True`. The game keeps updating after death (falling, slowing down), but a model can treat death as absorbing.

**Reduced model.** Ignoring rockets and the intro, the transition above depends on only `(y, v, vx, x-positions of upcoming zappers and their y)`, and the action. That is enough to write a headless simulator, which is the approach the feedback recommends.

## 9. Pitfalls to remember when adapting this for search, MDP or RL

- The model needs a **fixed `dt`**, which the game does not have.
- `vx` (or elapsed time) belongs in the state, because acceleration and obstacle speed both scale with it.
- The state is continuous. Discretise it (or use features) before using tabular methods.
- Zapper heights are random, so "deterministic transitions" holds only for a known layout.
- The game cannot run a step without drawing, and it depends on pygame events. A headless re-implementation, verified against recorded traces, is safer than editing the game loop.
