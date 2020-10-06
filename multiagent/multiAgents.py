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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        closestFood = float("inf") # we want close food
        closestGhost = float("inf") # we want far ghosts - so check the closest and create some val off of that
        for food in newFood:
            foodDist = manhattanDistance(newPos, food)
            if foodDist < closestFood:
                closestFood = foodDist

        for ghost in newGhostStates:
            ghostDist = manhattanDistance(newPos, ghost.getPosition())
            if ghostDist < closestGhost:
                closestGhost = ghostDist

        retVal = (.21 * (closestGhost + closestFood)) + (13 / (closestFood + 1)) + (11 * successorGameState.getScore()) #+ sum(newScaredTimes)
        return retVal

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
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        agentIndex = 0
        bestScore = -float("inf")
        bestMove = Directions.STOP
        # Iterate through each action and determine which will yield the max score, recursively.
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            score = self.minAgent(nextState, self.depth, ((agentIndex + 1) % gameState.getNumAgents()))
            if score > bestScore:
                bestMove = action
                bestScore = score
        return bestMove

    # Returns the min achieveable score for a given agent in the given 
    def minAgent(self, gameState, depth, agentIndex):
        # Handle leaf return values
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        lowestScore = float("inf")
        # If the next agent is pacman, call maxAgent and proceed to next depth.
        if agentIndex == (gameState.getNumAgents() - 1):
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                lowestScore = min(lowestScore, self.maxAgent(nextState, depth - 1))
        # Else, call minAgent on the next ghost and remain on same depth.
        else:
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                lowestScore = min(lowestScore, self.minAgent(nextState, depth, ((agentIndex + 1) % gameState.getNumAgents())))
        return lowestScore

    # Returns the max in minimax -> always for pacman.
    def maxAgent(self ,gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        highestScore = -float("inf")
        for action in gameState.getLegalActions(0): # Function call always for pacman
            nextState = gameState.generateSuccessor(0, action)
            highestScore = max(highestScore, self.minAgent(nextState, depth, (1 % gameState.getNumAgents())))
        return highestScore
        



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agentIndex = 0
        bestScore = -float("inf")
        bestMove = Directions.STOP
        alpha, beta = -float("inf"), float("inf")
        # Iterate through each action and determine which will yield the max score, recursively.
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            score = self.minAgent(nextState, self.depth, ((agentIndex + 1) % gameState.getNumAgents()), alpha, beta)
            if score > bestScore:
                bestMove = action
                bestScore = score
            if score > beta:
                return bestMove
            alpha = max(alpha, score)
        return bestMove

    # Returns the min achieveable score for a given agent in the given 
    def minAgent(self, gameState, depth, agentIndex, alpha, beta):
        # Handle leaf return values
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        lowestScore = float("inf")
        # If the next agent is pacman, call maxAgent and proceed to next depth.
        if agentIndex == (gameState.getNumAgents() - 1):
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                lowestScore = min(lowestScore, self.maxAgent(nextState, depth - 1, alpha, beta))
                if lowestScore < alpha:
                    return lowestScore
                beta = min(beta, lowestScore)
        # Else, call minAgent on the next ghost and remain on same depth.
        else:
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                lowestScore = min(lowestScore, self.minAgent(nextState, depth, ((agentIndex + 1) % gameState.getNumAgents()), alpha, beta))
                if lowestScore < alpha:
                    return lowestScore
                beta = min(beta, lowestScore)
        return lowestScore

    # Returns the max in minimax -> always for pacman.
    def maxAgent(self, gameState, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        highestScore = -float("inf")
        for action in gameState.getLegalActions(0): # Function call always for pacman
            nextState = gameState.generateSuccessor(0, action)
            highestScore = max(highestScore, self.minAgent(nextState, depth, (1 % gameState.getNumAgents()), alpha, beta))
            if highestScore > beta:
                return highestScore
            alpha = max(alpha, highestScore)
        return highestScore

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
        agentIndex = 0
        bestScore = -float("inf")
        bestMove = Directions.STOP
        # Iterate through each action and determine which will yield the max score, recursively.
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            score = self.expAgent(nextState, self.depth, ((agentIndex + 1) % gameState.getNumAgents()))
            if score > bestScore:
                bestMove = action
                bestScore = score
        return bestMove

    # Returns the max in minimax -> always for pacman.
    def maxAgent(self ,gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        highestScore = -float("inf")
        for action in gameState.getLegalActions(0): # Function call always for pacman
            nextState = gameState.generateSuccessor(0, action)
            highestScore = max(highestScore, self.expAgent(nextState, depth, (1 % gameState.getNumAgents())))
        return highestScore

    # Returns the min achieveable score for a given agent in the given 
    def expAgent(self, gameState, depth, agentIndex):
        # Handle leaf return values
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        score = 0
        numActions = len(gameState.getLegalActions(agentIndex))
        # If the next agent is pacman, call maxAgent and proceed to next depth.
        if agentIndex == (gameState.getNumAgents() - 1):
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                score += self.maxAgent(nextState, depth - 1)
        # Else, call minAgent on the next ghost and remain on same depth.
        else:
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                score += self.expAgent(nextState, depth, ((agentIndex + 1) % gameState.getNumAgents()))
        return score / numActions



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    closestFood = float("inf") # we want close food
    nextGhost = float("inf") # we want far ghosts - so check the closest and create some val off of that
    closestCapsule = float("inf")
    for food in foodPos:
        foodDist = manhattanDistance(pos, food)
        if foodDist < closestFood:
            closestFood = foodDist

    for capsule in capsules:
        capDist = manhattanDistance(pos, capsule)
        if capDist < closestCapsule:
            closestCapsule = capDist

    for ghost in ghostStates:
        ghostDist = manhattanDistance(pos, ghost.getPosition())            
        if ghostDist < nextGhost:
            if ghost.scaredTimer >= ghostDist:
                nextGhost = 1/(ghostDist + 1)
            else:
                nextGhost = ghostDist

    retVal = (.21 * (nextGhost + closestFood)) + (11 * currentGameState.getScore()) + (2.5 * sum(scaredTimes)) + (16 / (len(capsules) + closestFood)) + (12 / (closestCapsule + 1)) + (12 / (closestFood + 1)) 
    return retVal

# Abbreviation
better = betterEvaluationFunction
