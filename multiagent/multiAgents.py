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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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

        ghostPositions = successorGameState.getGhostPositions()
        currentFood = currentGameState.getFood()

        closestFood = 9999999

        score = 0.0

        foodPositions = newFood.asList()
        currentFoodPositions = currentFood.asList()

        for ghost in ghostPositions:
            distance = manhattanDistance(ghost, newPos)
            if ghost == newPos or distance == 1:
                return -9999999

        for food in currentFoodPositions:
            distance = manhattanDistance(food, newPos)
            if distance < closestFood:
                closestFood = distance

        score = 1.0 / (1 + closestFood)
        return score

def scoreEvaluationFunction(currentGameState):
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

        #new stuff
        self.bestMove = 'Stop'

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
            Returns the total number of agents in th
            e game
        """
        "*** YOUR CODE HERE ***"

        def minimaxRecursion(gameState, index, currentDepth):
            if gameState.isLose() or gameState.isWin() or currentDepth >= self.depth:
                return self.evaluationFunction(gameState)
            if index == 0:
                value = -float('inf')
                bestMove = 'Stop'
                for action in gameState.getLegalActions(index):
                    if action != 'Stop':
                        playerGameState = gameState.generateSuccessor(index, action)
                        newValue = minimaxRecursion(playerGameState, index + 1, currentDepth)
                        if value < newValue:
                            value = newValue
                            bestMove = action
                self.bestMove = bestMove
                return value

            elif index != 0:
                value = float('inf')
                for action in gameState.getLegalActions(index):
                    if action != 'Stop':
                        ghostGameState = gameState.generateSuccessor(index, action)
                        resetOrNot = index
                        nextDepth = currentDepth
                        if index >= gameState.getNumAgents() - 1:
                            resetOrNot = 0
                            nextDepth += 1
                        else:
                            resetOrNot = index + 1
                        newValue = minimaxRecursion(ghostGameState, resetOrNot, nextDepth)
                        if value > newValue:
                            value = newValue
                return value

        minimaxRecursion(gameState, self.index, 0)

        return self.bestMove

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        def ABRecursion(gameState, index, alpha, beta, currentDepth):
            if gameState.isLose() or gameState.isWin() or currentDepth >= self.depth:
                return self.evaluationFunction(gameState)
            if index == 0:
                value = float('-inf')
                bestMove = 'Stop'
                for action in gameState.getLegalActions(index):
                    if action != 'Stop':
                        playerGameState = gameState.generateSuccessor(index, action)
                        newValue = ABRecursion(playerGameState, index + 1, alpha, beta, currentDepth)
                        if value < newValue:
                            value = newValue
                            bestMove = action
                        
                        if value > beta:
                          return value
                        alpha = max(alpha, value)
                self.bestMove = bestMove
                return value

            elif index != 0:
                value = float('inf')
                for action in gameState.getLegalActions(index):
                    if action != 'Stop':
                        ghostGameState = gameState.generateSuccessor(index, action)
                        resetOrNot = index
                        nextDepth = currentDepth
                        if index >= gameState.getNumAgents() - 1:
                            resetOrNot = 0
                            nextDepth += 1

                        else:
                            resetOrNot = index + 1
                        newValue = ABRecursion(ghostGameState, resetOrNot, alpha, beta, nextDepth)
                        if value > newValue:
                            value = newValue
                        
                        if value <= alpha:
                            return value
                        beta = min(beta, value)
                return value

        ABRecursion(gameState, 0, float("-inf"), float("inf"), 0)

        return self.bestMove


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimaxRecursion(gameState, index, currentDepth):
            if gameState.isLose() or gameState.isWin() or currentDepth >= self.depth:
                return self.evaluationFunction(gameState)
            if index == 0:
                value = -float('inf')
                bestMove = 'Stop'
                for action in gameState.getLegalActions(index):
                    if action != 'Stop':
                        playerGameState = gameState.generateSuccessor(index, action)
                        newValue = expectimaxRecursion(playerGameState, index + 1, currentDepth)
                        if value < newValue:
                            value = newValue
                            bestMove = action
                self.bestMove = bestMove
                return value

            elif index != 0:
                value = 0.0
                legalActions = gameState.getLegalActions(index)
                for action in legalActions:
                    if action != 'Stop':
                        ghostGameState = gameState.generateSuccessor(index, action)
                        resetOrNot = index
                        nextDepth = currentDepth
                        if index >= gameState.getNumAgents() - 1:
                            resetOrNot = 0
                            nextDepth += 1
                        else:
                            resetOrNot = index + 1
                        newValue = expectimaxRecursion(ghostGameState, resetOrNot, nextDepth)
                        value = value + ((1.0 / len(legalActions)) * newValue)
                return value

        expectimaxRecursion(gameState, self.index, 0)

        return self.bestMove

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

