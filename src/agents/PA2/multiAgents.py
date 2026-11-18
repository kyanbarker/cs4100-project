# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from http.cookiejar import cut_port_re
from math import nextafter

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')

        for ghostState, scaredTime, in zip(newGhostStates, newScaredTimes):
            if scaredTime == 0 and manhattanDistance(newPos, ghostState.getPosition()) <= 1:
                return float('-inf')

        nearestFood = min(manhattanDistance(newPos, food) for food in newFood.asList())
        value = successorGameState.getScore() + 1 / nearestFood

        if action == Directions.STOP:
            value -= 1

        return value

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def isCutoff(self, state: GameState, depth: int):
        return state.isWin() or state.isLose() or depth == self.depth

    def nextTurn(self, state: GameState, agentIndex: int, depth: int):
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth + 1 if nextAgent == 0 else depth
        return nextAgent, nextDepth

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        _, action = self.value(gameState, 0, 0)
        return action

    "returns the minmax value for whichever agent moves at this state"
    def value(self, state: GameState, agentIndex: int, depth: int):
        if self.isCutoff(state, depth):
            return self.evaluationFunction(state), None
        if agentIndex == 0:
            return self.maxValue(state, depth)
        return self.minValue(state, agentIndex, depth)

    "pacman turn: take child w hihest value"
    def maxValue(self, state: GameState, depth: int):
        actions = state.getLegalActions(0)
        if not actions:
            return self.evaluationFunction(state), None
        nextAgent, nextDepth = self.nextTurn(state, 0, depth)
        bestValue = float('-inf')
        bestAction = None
        for action in actions:
            successor = state.generateSuccessor(0, action)
            childValue, _ = self.value(successor, nextAgent, nextDepth)
            if childValue > bestValue:
                bestValue = childValue
                bestAction = action
        return bestValue, bestAction

    "ghost turn: assume it picks child that is wrost for pacman"
    def minValue(self, state: GameState, agentIndex: int, depth: int):
        actions = state.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(state), None
        nextAgent, nextDepth = self.nextTurn(state, agentIndex, depth)
        bestValue = float('inf')
        bestAction = None
        for action in actions:
            successor = state.generateSuccessor(agentIndex, action)
            childValue, _ = self.value(successor, nextAgent, nextDepth)
            if childValue < bestValue:
                bestValue, bestAction = childValue, action
        return bestValue, bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        _, action = self.value(gameState, 0, 0)
        return action

    def value(self, state: GameState, agentIndex: int, depth: int):
        if self.isCutoff(state, depth):
            return self.evaluationFunction(state), None
        if agentIndex == 0:
            return self.maxValue(state, depth)
        return self.expValue(state, agentIndex, depth)

    def maxValue(self, state: GameState, depth: int):
        actions = state.getLegalActions(0)
        if not actions:
            return self.evaluationFunction(state), None
        nextAgent, nextDepth = self.nextTurn(state, 0, depth)
        bestValue = float("-inf")
        bestAction = None
        for action in actions:
            successor = state.generateSuccessor(0, action)
            childValue, _ = self.value(successor, nextAgent, nextDepth)
            if childValue > bestValue:
                bestValue = childValue
                bestAction = action
        return bestValue, bestAction

    def expValue(self, state: GameState, agentIndex: int, depth: int):
        actions = state.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(state), None
        nextAgent, nextDepth = self.nextTurn(state, agentIndex, depth)
        prob = 1.0/ len(actions)
        ev = 0.0
        for action in actions:
            successor = state.generateSuccessor(agentIndex, action)
            childValue, _ = self.value(successor, nextAgent, nextDepth)
            ev += prob * childValue
        return ev, None




def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    start from game score which rewards eating food (+10). eating ghost (+200) and winning (+500)
    and charges 1 per time step

    additionally:
    win is worth its score
    lose gets extra -1000 bc dying means you also lose the remaining food and win bonus. cant use
    -inf bc it would make every move wth a chance of death look equally bad

    food is +10 / dis to nearest pellet bc this moves pacman toward food beyond 2 ply and since it
    cant ever exceed 10 pacman will choose to hover next to a dog

    capsule is -40 per capsule left on the board so the nearby ones are worth getting

    scared ghost pacman can reach before its timer runs out adds +200 / dis + 1 so he will hunt it
    down, and is always less than eating so pacman will eat is possible
    non-scared ghosts within one square costs -300 so pacman will keep his distance
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    if currentGameState.isWin():
        return score
    if currentGameState.isLose():
        return - 1000

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()

    value = score

    nearestFood = min(manhattanDistance(pos, food) for food in foodList)
    value += 10.0 / nearestFood

    value -= 40 * len(capsules)

    for ghost in currentGameState.getGhostStates():
        distance = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > distance:
            value += 200.0 / (distance + 1)
        elif ghost.scaredTimer == 0 and distance <= 1:
            value -= 300

    return value

# Abbreviation
better = betterEvaluationFunction
