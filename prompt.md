* Make Claude review the project proposal and give feedback. Save feedback to a `feedback.md` file

* Make Claude generate documentation about Jetpack Goodride. Save documentation in `docs/`

  * How are actions and states represented?

  * How are actions performed?

  * How are states updated?

  * Is there an update function? How many times per second is it called?

  * How are position, velocity and acceleration represented? (eg floats, vectors)

  * How are position, velocity and acceleration updated? (ie the laws of physics)

    * Using the laws of physics, what is the next state for a state, action pair? (ie transition function)

* Copy PA1-3 code to the repo.

  * Want generic code so we can try search, MDP and RL on multiple subproblems with varying levels of complexity

  * If code is not generic enough to be adapted for jetpack joyride, refactor to be more abstract.

  * Write documentation for this code in `docs/`

  * Want to have 1 module for search and 1 or 2 modules for mdp and rl (q learning)

* Once the above are finished, plan how to adapt Jetpack Goodride to generic search, mdp and rl code