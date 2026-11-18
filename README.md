# CS 4100 Fall 2026 Final Project Repository

- todo: convert project.pdf to project.md
- todo: create a contributing.md

## Repository Outline

- `feedback.md`: review of the project proposal
- `docs/`
  - `docs/project-proposal_barrys-big-blast.md`:
    - Formulates the problem (states, actions)
    - Defines division of labor (tenative)
  - `docs/jetpack-goodride.md`: how the game represents and updates states, actions and physics, including the transition function
  - `docs/agents.md`: documentation for `src/agents/`
  - `docs/adaptation-plan.md`: plan for connecting the game to the generic code
- `src/`
  - `game/`
    - A forked GitHub repository containing a python clone of jetpack joyride.
  - `agents/`
    - General code for solving search, mdp and rl problems (not specific to jetpack joyride):
      - `search.py`
      - `mdp.py`
      - `rl.py`
- `test/`
  - `test/test_agents.py`, `test/toy_problems.py`: sanity checks on small problems with known answers
  - run from the repo root: `python -m unittest discover -s test -v`
