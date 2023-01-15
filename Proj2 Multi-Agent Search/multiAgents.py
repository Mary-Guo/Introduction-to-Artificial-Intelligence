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
        if action is Directions.STOP:
            return -100
        nearest_ghost_dist = 1e9
        for g in newGhostStates:
            if g.scaredTimer is 0:
                nearest_ghost_dist = min(nearest_ghost_dist, int(
                    manhattanDistance(newPos, g.getPosition())))
        if nearest_ghost_dist is 0:
            return -100
        nearest_food_dist = 1e9
        if newFood:
            for f in newFood:
                nearest_food_dist = min(
                    nearest_food_dist, int(manhattanDistance(newPos, f)))
        #print(action, nearest_food_dist)
        return successorGameState.getScore() - 10/(nearest_ghost_dist) + 6/(nearest_food_dist+1)

    

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

        actionsforpacman = gameState.getLegalActions()
        successors = []
        for a in actionsforpacman:
            successors.append(gameState.generateSuccessor(0, a))
        ghostscores = []
        for s in successors:
            ghostscores.append(self.minmaxhelper(self.depth, 1, s))
        return actionsforpacman[ghostscores.index(max(ghostscores))]        



    def minmaxhelper(self, depth, agentIndex, state):

        if state.isWin() or state.isLose() or depth is 0:  #base case
                return self.evaluationFunction(state)

        actions = state.getLegalActions(agentIndex)
        successors = []
        for i in actions:
            successors.append(state.generateSuccessor(agentIndex, i))

        if agentIndex is 0: #pacman turn 
            next_agentIndex = 1
            results = []
            for j in successors:
                results.append(self.minmaxhelper(depth, next_agentIndex, j))
            return max(results)


        else: #ghost turn 
            #print(state.getNumAgents())
            next_agentIndex = (1 + agentIndex) % state.getNumAgents()

            if next_agentIndex is 0:
                depth -= 1
            results = []
            for j in successors:
                results.append(self.minmaxhelper(depth, next_agentIndex, j))
            return min(results)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        actionsforpacman = gameState.getLegalActions()
        successors = []
        for a in actionsforpacman:
            v = max(v, self.min_value(gameState.generateSuccessor(0, a), alpha, beta, 1, self.depth))
            if v > alpha:
                alpha = v
                action = a
            if v > beta:
                return v

        return action
    

    def max_value(self, state, alpha, beta, agentIndex, depth):
        v = float("-inf")
        if state.isWin() or state.isLose() or depth is 0:  #base case
                return self.evaluationFunction(state)
        
        actions_for_pacman = state.getLegalActions(agentIndex)
        successors = []
        for a in actions_for_pacman:
            v = max(v, self.min_value(state.generateSuccessor(agentIndex, a), alpha, beta, 1, depth))
            if v > beta:
                return v
            alpha = max(alpha,v)
        return v

    def min_value(self, state, alpha, beta, agentIndex, depth):
        if agentIndex is 0:
            return self.max_value(state, alpha, beta, 0, depth -1)
        v = float("inf")

        if state.isWin() or state.isLose() or depth is 0:  #base case
                return self.evaluationFunction(state)

        actions_for_ghosts = state.getLegalActions(agentIndex)
        successors = []
        for a in actions_for_ghosts:
            v = min(v, self.min_value(state.generateSuccessor(agentIndex, a), alpha, beta, (1 + agentIndex) % state.getNumAgents(), depth))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v
        util.raiseNotDefined()

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
        actionsforpacman = gameState.getLegalActions()
        successors = []
        for a in actionsforpacman:
            successors.append(gameState.generateSuccessor(0, a))
        scores = []
        for s in successors:
            scores.append(self.expectimaxhelper(1, self.depth, s))

        return actionsforpacman[scores.index(max(scores))]

    def expectimaxhelper(self, agentIndex, depth, state):
        if state.isWin() or state.isLose() or depth is 0:  #base case
            return self.evaluationFunction(state)

        actions = state.getLegalActions(agentIndex)
        successors = []
        for a in actions:
            successors.append(state.generateSuccessor(agentIndex, a))
    


        if agentIndex == 0: 
            next_agentIndex = 1
            scores = []
            for s in successors:
                scores.append(self.expectimaxhelper(next_agentIndex, depth, s))
            return max(scores)

        else: 
            next_agentIndex = (1 + agentIndex) % state.getNumAgents()
            if next_agentIndex == 0:
                depth -= 1
            scores = []
            for s in successors:
                scores.append(self.expectimaxhelper(next_agentIndex, depth, s))
            return sum(scores)/len(scores)

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <basically it's the same as the reflex agent. 
    The only difference is we evaluate the states instead of successors. 
    Also, we have a better score if the ghost got scared of. >
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()

    nearest_ghost_dist = 1e9
    for g in newGhostStates:
        if g.scaredTimer == 0:
            nearest_ghost_dist = min(nearest_ghost_dist, int(
                manhattanDistance(newPos, g.getPosition())))
        else:
            nearest_ghost_dist = -10
    if nearest_ghost_dist == 0:
        return -100
    nearest_food_dist = 1e9
    if newFood:
        for f in newFood:
            nearest_food_dist = min(
                nearest_food_dist, int(manhattanDistance(newPos, f)))
    return currentGameState.getScore() - 12/(nearest_ghost_dist+1) + 5/(nearest_food_dist+1)

# Abbreviation
better = betterEvaluationFunction
