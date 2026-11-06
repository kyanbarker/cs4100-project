**CS 4100 Project Proposal**

**Project Name: Barry’s Big Blast**

**Group Members: Jesse Garcia, Kyan Barker, Ran Guo**

- **Who is on the team? If you are working in a team of two, is there a clear division of labor? If the team is smaller/larger than recommended, provide a rationale.**
  - Group Members: Jesse Garcia, Kyan Barker, and Ran Guo

  - Division of labor:
    - Research: Jesse
    - Ideation: Kyan/Ran
    - Planning: Kyan/Ran
    - Implementation: Jesse
    - Testing: Ran
    - Documentation: Kyan
  - Why the team is larger than recommended (3 instead of 2):
    - We all love Jetpack Joyride and want to apply different concepts from class to the game. Having 3 people instead of 2 will allow us to try multiple approaches and hopefully allow us to solve a more complex subproblem then we would be able to do if we only had 2 people.

- **Describe the problem you are trying to address, and provide a formulation of it. If it involves an algorithm (i.e., any implementation-based project), describe the input to the algorithm and the desired output.**
  - Context:
    - Jetpack Joyride is a mobile game in which the player, Barry, aims to avoid obstacles by flying up or down with his jetpack as he moves right continuously. Human players hold a finger on the screen to make Barry fly up with his jetpack while the finger is held. Human players release their finger from the screen to make Barry stop flying up; Barry falls due to gravity while no finger is held on the screen. Upon colliding with an obstacle in the environment, Barry dies and the round is over.
    - Example gameplay video: [https\://youtu.be/Jzxi8nid9BQ?t=37](https://youtu.be/Jzxi8nid9BQ?t=37)
  - We need to formulate a search, MDP, or RL problem. In this proposal we start by formulating the simplest possible subproblem of the game.
  - Actions:
    - At every frame, the player (human or AI) chooses one of two actions: increase vertical velocity/acceleration, or decrease vertical velocity/acceleration.
  - States:
    - Definitions:
      - A bounded convex domain D ⊂ ℝ² is a nonempty, open set of points that is finite in extent (contained in some disk of finite radius) and convex (the line segment between any two points of D lies entirely within D). Intuitively, it is the interior of a shape such as a triangle, rectangle, or disk, with no indentations and no holes.
    - Given we are working in the context of a video game, we will define a collider to be a bounded convex domain. If C1 and C2 are colliders, then we say C1 and C2 collide if and only if the intersection of C1 and C2 is non empty (ie the sets C1 and C2 are not disjoint)
    - The components of any state are Barry and obstacles:
    - Barry has position, velocity, acceleration and a collider centered at the position
    - Option 1: Barry does not move horizontally, obstacles move left
      - Barry has floats (representing vertical position, velocity and acceleration), and a collider centered at the position
      - Each obstacle has a collider and a horizontal velocity that is strictly less than 0 at all times (since obstacles move left). The 2D points in the collider have positive horizontal components if the point is to the right of Barry and negative horizontal components if the point is to the left of Bary
    - Option 2: Barry moves right, obstacles do not move
      - Barry has 2D vectors (representing position, velocity and acceleration), and a collider centered at the position. Barry’s horizontal position is strictly increasing (i.e. Barry’s horizontal velocity is strictly greater than 0 at all times)
      - Each obstacle has a collider
  - We found a [GitHub repository of a remake of Jetpack Joyride](https://github.com/Gustavo-Pauli/Jetpack-Goodride/tree/master/) that we will likely use. We will investigate how it models states and actions. We will also need to investigate how it determines terminal states. If we use search, we will likely use a function from states to bool that indicates which states are terminal. If we use MDP or RL we will likely define the action space to be empty for terminal states (and include a boolean component in states).
  - If we use search:
    - We will use a function from states to bool that indicates which states are terminal.
    - We will develop a heuristic that considers its current velocity and acceleration as well as the distance between Bary and any obstacles
  - If we use MDP or RL:
    - For terminal states, we will define the action space to be empty (or contain only the identity action) and the reward to be 0
  - If we use MDP we will define rewards, but not transition probabilities (if we know the laws of physics of the environment, which we likely will, then transition probabilities are deterministic).
  - If we use RL we will use Q learning to have the agent teach itself how to behave.

- **What is the ideal outcome of the project? What do you expect to show?**
  - Formulate a problem that captures the behavior of every feature in the original game
  - Implement the formulated problem
  - Create an agent capable of finding an optimal solution or policy
  
  The formulation will be more difficult if we choose to include power ups and vehicles, so our first step will be to formulate the simplest subproblem, then gradually work our way up in terms of complexity.

- **What algorithms do you expect to use?**

  If we use search: A\* with a consistent or admissible heuristic

  If we use MDP: Value/policy iteration

  If we use RL: Q learning

- **What topics / libraries / platforms, if any, will you have to learn in order to undertake your project? Provide references where applicable.**
  - [https\://github.com/Gustavo-Pauli/Jetpack-Goodride/tree/master/scripts](https://github.com/Gustavo-Pauli/Jetpack-Goodride/tree/master/scripts)

- **If your project has a machine learning component, it will most likely involve a dataset. Where will this come from? Is it an existing dataset, or are you making a new one? If the latter, do you have the resources to do so?**

  We will not require a dataset. The transition probabilities of an MDP for this problem would be deterministic, and Q learning agent would teach itself.

- **Define a halfway milestone for your project. This will be due on 12/4 – note that this is only a week before your draft report is due. Milestones can include learning about certain topics / algorithms, acquiring and processing datasets, implementing an algorithm, analyzing results, etc. Again, we will not penalize you if you do not achieve your specific milestone. At the due date, we will request a short progress report (see next page) that addresses progress toward the milestones, and if they were not achieved, what turned out to be more challenging than expected. You may also find that your initial milestones were inappropriate, and you can choose different milestones, work toward those, and report on them. That is completely fine too – the point is to decompose your project into smaller pieces and ensure that you are making consistent progress over the next few weeks.**

  We will have completed the formulation, research, ideation, planning, implementation, and creation of a decent agent.

  Definition: A decent agent will successfully avoid 10 obstacles before dying at least 90% of the time.

- **Provide a week-by-week plan for your project.**
  - 11/08 \- 11/14
    - Environment setup: fork repository, setup IDE if necessary, setup git collaboration between members

    - Formulate problem

    - Using code from PAs, create general classes and functions for solving search, MDP, and RL problems.

  - 11/15 \- 11/21
    - Adapt forked Jetpack Joyride repository to the general codebase

  - 11/22 \- 11/28
    - Address technical issues or formulate a more complex subproblem (e.g. one with moving obstacles, power ups, vehicles)

  - 11/29 \- 12/05
    - Project Milestone

  - 12/06 \- 12/12
    - Draft Project Report

  - 12/13 \- 12/19
    - Final Project Report
