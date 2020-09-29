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
import math

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
        # print(scores, bestScore)
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
        score = successorGameState.getScore()

        foodScore = 0
        if successorGameState.getNumFood() == currentGameState.getNumFood():
            foodList = [(x,y) for x in range(0, newFood.width) for y in range(0, newFood.height) if newFood[x][y] == True]
            nearestFood = min([util.manhattanDistance(newPos, (x,y)) for x,y in foodList])
            foodScore += (1/(nearestFood)) 
        else:
            foodScore += 1
            
        ghostScore = 0
        ghostPositions = successorGameState.getGhostPositions()
        if ghostPositions:
            nearestGhost = min([util.manhattanDistance(newPos, gp) for gp in ghostPositions])
            if nearestGhost == 0:
                return -(math.inf)
            ghostScore = nearestGhost

        scaredScore = sum(newScaredTimes)    

        score += foodScore + ghostScore + scaredScore
        
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

        # Stop the game when needed 
        self.stopGame = Directions.STOP

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

        # print(gameState.generateSuccessor(1, gameState.getLegalActions()[0]))
        return self.miniMaxValue(gameState, 1, 0)[1]

    def miniMaxValue(self, gameState, depth, agentIndex):
        # if the state is a terminal state: return the state's utility
        if depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        
        legalActions = gameState.getLegalActions(agentIndex)
        nextAgent = agentIndex + 1
        nextDepth = depth
        if nextAgent >= gameState.getNumAgents():
            nextAgent = 0
            nextDepth = depth + 1
        values = [self.miniMaxValue(gameState.generateSuccessor(agentIndex, a), nextDepth, nextAgent)[0] for a in legalActions]

        # if the next agent is MAX: return max-value(state)
        if agentIndex == 0:
            bestValue = max(values)

        # if the next agent is MIN: return min-value(state)
        else:
            bestValue = min(values)
        
        bestMoveIndex = values.index(bestValue)
        return bestValue, legalActions[bestMoveIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # intializing alpha & beta as inifinity 
        value, path = self.alpha_pacM_value(gameState, float ('-inf'),float ('inf'), 0, self.depth)
        # if value & path == 0 then return the path 
        return path 
    
    # maximum value of the PacMan 
    def alpha_pacM_value (self, pos_state, alpha, beta, numAgt, depth):
        # whether its win or lose, return the value of the game
        if pos_state.isWin() or pos_state.isLose():
            return self.evaluationFunction(pos_state), 'None'
        
        maxEvaluation = float ('-inf')
        # returns a list of agent's path from one location to the next 
        path = pos_state.getLegalActions(numAgt)
        # score of the best path
        best_path_score = path[0]

        # for loop that finds the max value of the path 
        # and pruning through the path for the max value
        for pathWay in path:
            # variable for the previous maxEvaluation 
            prevMaxEval = maxEvaluation
            # obtain the successor game state after the agent takes a path 
            successorGameState = pos_state.generateSuccessor(numAgt, pathWay)
            # if the depth is 0 whether agent lost or won, then take the max
            # of the maxEvaluation and self.evaluationFunction 
            if(depth == 0 or successorGameState.isWin() or successorGameState.isLose()):
                #takes the max of the maxEvaluation and self.evaluationFunction 
                maxEvaluation = max(maxEvaluation,self.evaluationFunction(successorGameState))
            else: 
                #take the max of the maxEvaluation and minEvaluation of the ghost (beta)
                maxEvaluation = max(maxEvaluation, self.beta_ghost_value(successorGameState,alpha,beta,numAgt+1,depth))
            
            # checks for pruning through the tree
            # if maxEvaluation is greater than beta, the return the maxEvaluation and path 
            if maxEvaluation > beta: 
                return maxEvaluation, pathWay
            #set alpha to be the max of the alpha and maxEvaluation 
            alpha = max(alpha,maxEvaluation)
            # if the maxEvaluation does not equal the past maxEval 
            # then store the path to the bestpathScore 
            if maxEvaluation != prevMaxEval:
                best_path_score = pathWay

        # loop does not have maxValue/pruning then return the current path and maxEvaluation 
        return (maxEvaluation, best_path_score)

    # minimum value of the ghost 
    def beta_ghost_value(self, pos_state, alpha, beta, numAgt, depth):
        # whether its win or lose, return the value of the game
        if pos_state.isWin() or pos_state.isLose():
            return self.evaluationFunction(pos_state), 'None'
        
        ghMinEval = float ('inf')
        # returns a list of agent's path from one location to the next 
        gh_path = pos_state.getLegalActions(numAgt)
        # decreasing depth boolean 
        decreasingDepth = False 
        # for loop that finds the min value of the path 
        for path in gh_path:
            #obtain the successor game state after the agent takes a path 
            successorGameState = pos_state.generateSuccessor(numAgt, path)
            # if the depth is 0 whether agent lost or won, then take the min
            # of the ghMinEval and self.evaluationFunction 
            if(depth == 0 or successorGameState.isWin() or successorGameState.isLose()):
                #takes the min of the ghMinEval and self.evaluationFunction 
                ghMinEval = min(ghMinEval,self.evaluationFunction(successorGameState))
            elif numAgt == (pos_state.getNumAgents() - 1):
                # if decreasing Depth is false than 
                # it avoids deacreaing the depth of the same level more than once 
                if decreasingDepth == False: 
                    depth -= 1
                    decreasingDepth = True 
                # if the last level of the tree is reached 
                if depth == 0:
                    # then return the min of the ghMinEval and self.evaluationFunction 
                    ghMinEval = min(ghMinEval,self.evaluationFunction(successorGameState))
                else: 
                    # if not the last level then print out the min of the ghMinEvla and the alpha_pacM_value at 0
                    ghMinEval = min(ghMinEval, self.alpha_pacM_value(successorGameState,alpha, beta, 0, depth)[0])
            else:
                # take the min of the ghMinEval and the beta_ghost_value
                ghMinEval = min(ghMinEval, self.beta_ghost_value(successorGameState,alpha,beta,numAgt+1, depth))
            
            # checks for pruning through the tree
            if ghMinEval < alpha:
                #return the ghMinEval 
                return ghMinEval
            #set beta to the min of the ghMinEval and beta
            beta = min(beta,ghMinEval)
        return ghMinEval
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
        # getting the number of agents 
        numOfAgt = gameState.getNumAgents()

        # size of the tree (depth of the tree)
        depthOfTree = self.depth*numOfAgt

        # enter the function that will go through the tree, 
        # get the average of the bottom nodes, then 
        # get the min value of the calucated average 
        # finally get the max of the min values 
        self.getAverageMaxMin(gameState, depthOfTree, numOfAgt)
        
        # stop the game once the final max is found 
        return self.stopGame

        # this method would get the average of the nodes from the subtrees
        # take the min value of the average 
        # take the max value of the min values 
        def getAverageMaxMin(self, gameState, depth, numAgt):
            # list for the maxValues of the nodes
            maxValues = []

            # list for the average values of the node 
            avgValues = []

            # whether its win or lose, return the value of the game
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # if the depth of the tree is greater than 0 
            # if depthOfTree > 0:
                #

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin():
        return math.inf
    if currentGameState.isLost():
        return -math.inf

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    score = currentGameState.getScore()
    pellets = currentGameState.getCapsules()

    foodWeight, pelletWeight, ghostWeight = 1, 1, 1

    # manhattan distance to closest food
    foodList = [(x,y) for x in range(0, food.width) for y in range(0, food.height) if food[x][y] == True]
    nearestFood = min([util.manhattanDistance(pos, (x,y)) for x,y in foodList])
    foodScore = (1/(nearestFood)) 
    foodScore += (1/food)

    # pellets score
    pelletsScore = (1/pellets) * pelletsWeight

    activeGhosts = [ghost for ghost in ghostStates if not ghost.scaredTimer]
    scaredGhosts = [ghost for ghost in ghostStates if ghost.scaredTimer]

    nearestActGh = min([util.manhattanDistance(pos, ghost.getPosition()) for ghost in activeGhosts])
    actGhScore = nearestActGh * ghostWeight

    nearestScGh = min([util.manhattanDistance(pos, ghost.getPosition()) for ghost in scaredGhosts])
    actGhScore = (1/nearestScGh)

    return foodScore + pelletsScore + nearestActGh + nearestScGh

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
