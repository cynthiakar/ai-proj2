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


# Project 2 by Arvin Lin and Cynthia Kar
# CIS 467

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

        # set score to current score
        score = successorGameState.getScore()
 
        foodScore = 0
        # if food is not eaten in this action, get distance to nearest food
        if successorGameState.getNumFood() == currentGameState.getNumFood():
            foodList = [(x,y) for x in range(0, newFood.width) for y in range(0, newFood.height) if newFood[x][y] == True]
            nearestFood = min([util.manhattanDistance(newPos, (x,y)) for x,y in foodList])
            # use reciprocal so closer distance means higher score
            foodScore = (1/(nearestFood)) 
        # if food is eaten, incentivize action with full point
        else:
            foodScore += 1
            
        # get distance to nearest ghost
        ghostScore = 0
        ghostPositions = successorGameState.getGhostPositions()
        if ghostPositions:
            nearestGhost = min([util.manhattanDistance(newPos, gp) for gp in ghostPositions])
            if nearestGhost == 0:
                return -(math.inf)
            ghostScore = 1/nearestGhost

        # higher number of scared ghosts is better
        scaredScore = sum(newScaredTimes)    

        score += foodScore + (foodScore/ghostScore) + scaredScore 
        
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
        return self.miniMaxValue(gameState, 1, 0)[1]

    def miniMaxValue(self, gameState, depth, agentIndex):
        # if the state is a terminal state: return the state's utility
        if depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        
        legalActions = gameState.getLegalActions(agentIndex)
        # set nextAgent to search
        nextAgent = agentIndex + 1
        nextDepth = depth
        # if we already went through all agents,
        # increment depth and restart at pacman agent
        if nextAgent >= gameState.getNumAgents():
            nextAgent = 0
            nextDepth = depth + 1
        # get list of values
        values = [self.miniMaxValue(gameState.generateSuccessor(agentIndex, a), nextDepth, nextAgent)[0] for a in legalActions]

        # if the next agent is MAX: return max-value(state)
        if agentIndex == 0:
            bestValue = max(values)

        # if the next agent is MIN: return min-value(state)
        else:
            bestValue = min(values)
        
        # return either min and max value and action associated with best value
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
        path_values, path = self.alpha_pacM_value(gameState, float ('-inf'),float ('inf'), 0, self.depth)
        # if value & path == 0 then return the path 
        return path 
    
    # maximum value of the PacMan 
    def alpha_pacM_value (self, pos_state, alpha, beta, numAgt, depthTree):

        # whether its win or lose, return the value of the game
        if pos_state.isWin() or pos_state.isLose():
            return self.evaluationFunction(pos_state)
        
        maxEvaluation = -math.inf
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

            # if the depth is 0, then set maxEvaluation to the max of maxEvaluation and self.evaluationFunction 

            if(depthTree == 0):
                #if self.evaluationFunction is greater than maxEvaluation than set maxEvaluation to self.evaluationFunction
                if maxEvaluation < (self.evaluationFunction(successorGameState)):
                    maxEvaluation = self.evaluationFunction(successorGameState)              
                
            else: 
                #if self.ghost_value is greater than set the self.ghost_value to maxEvaluation 
                if maxEvaluation < (self.ghost_value(successorGameState,alpha,beta,numAgt+1,depthTree)):
                    maxEvaluation = self.ghost_value(successorGameState,alpha,beta,numAgt+1,depthTree)
            
            # checks for pruning through the tree

            # if the maxEvaluation does not equal the past maxEval 
            # then store the path to the bestpathScore 
            if maxEvaluation != prevMaxEval:
                best_path_score = pathWay

            # if maxEvaluation is greater than beta, the return the maxEvaluation and path 
            if maxEvaluation > beta: 
                return maxEvaluation, pathWay

            #set alpha to be the max of the alpha and maxEvaluation 
            alpha = max(alpha,maxEvaluation)
        # loop does not have maxValue/pruning then return the current path and maxEvaluation 
        return (maxEvaluation, best_path_score)

    # minimum value of the ghost 
    def ghost_value(self, pos_state, alpha, beta, numAgt, depthTree):

        # whether its win or lose, return the value of the game
        if pos_state.isWin() or pos_state.isLose():
            return self.evaluationFunction(pos_state)
        
        ghMinEval = math.inf
        # returns a list of agent's path from one location to the next 
        gh_path = pos_state.getLegalActions(numAgt)
        # set decreasing depth to current depth 
        decreasingDepth = depthTree 
        # for loop that finds the min value of the path 
        for path in gh_path:
            # obtain the successor game state after the agent takes a path 
            successorGameState = pos_state.generateSuccessor(numAgt, path)
            win = successorGameState.isWin()
            lose = successorGameState.isLose()

            # if the depth is 0 whether agent lost or won, then take the min
            # of the ghMinEval and self.evaluationFunction 
            if(depthTree == 0 or win or lose):
                #if the ghMinEval is grater than set ghMinEval to self.evaluationFunction
                if ghMinEval > (self.evaluationFunction(successorGameState)):
                    ghMinEval = self.evaluationFunction(successorGameState)

            elif numAgt == (pos_state.getNumAgents() - 1):
                # checks if the depth is still in the same section
                # if decreasing Depth is at the same depth 
                # it will decrease the depth to move onto the next section of the tree 
                if decreasingDepth == depthTree: 
                    depthTree = depthTree - 1

                # if the last level of the tree is reached 
                if depthTree == 0:
                    # if the ghMinEval is grater than set ghMinEval to self.evaluationFunction
                    if ghMinEval > (self.evaluationFunction(successorGameState)):
                        ghMinEval = self.evaluationFunction(successorGameState)
                    
                else: 
                    # if not the last level then 
                    # if the ghMinEval is grater than set ghMinEval to self.alpha_pacM_value
                    if ghMinEval > (self.alpha_pacM_value(successorGameState,alpha, beta, 0, depthTree)[0]):
                        ghMinEval = self.alpha_pacM_value(successorGameState,alpha, beta, 0, depthTree)[0]                    

            else:
                # if the ghMinEval is grater than set ghMinEval to self.ghost_value
                if ghMinEval > (self.ghost_value(successorGameState,alpha,beta,numAgt+1, depthTree)):
                    ghMinEval = self.ghost_value(successorGameState,alpha,beta,numAgt+1, depthTree) 

            
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

        return self.expectimaxValue(gameState, self.depth, 0)[1]

    def expectimaxValue(self, gameState, depth, agentIndex):
        # if the state is a terminal state: return the state's utility
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        legalActions = gameState.getLegalActions(agentIndex)
        # set nextAgent to search
        nextAgent = agentIndex + 1
        nextDepth = depth
        # if we already went through all agents,
        # decrement depth
        if nextAgent >= gameState.getNumAgents():
            nextAgent = 0
            nextDepth = depth - 1

        bestAction = None
        # if the next agent is MAX: initialize bestValue to negative infinity
        if agentIndex == 0:
            bestValue = -math.inf
        # if the next agent is EXP: initialize bestValue to 0
        else:
            bestValue = 0

        for a in legalActions:
            # get search results for action a
            result = self.expectimaxValue(gameState.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
            # if the next agent is MAX: return max-value
            if agentIndex == 0:
                if result[0] > bestValue:
                    bestValue = result[0]
                    bestAction = a
            # if the next agent is EXP: return exp-value
            else:
                # get average of values
                bestValue += result[0]/len(legalActions)
                bestAction = a

        return bestValue, bestAction 

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    First, we check if the state is a terminal state. Winning state returns positive
    infinity. Losing state returns negative infinity

    Score is comprised of 4 attributes.

    First is food score, which we calculate by finding the distance to the nearest food.
    We use the reciprocal of this distance so that closer food means higher score.
    We also add the reciprocal of the number of foods left. Less food left means higher score.
    Then we used weights to incentivize eating foods over being close to food.

    Second is pellets score. Again, we used the reciprocal of the number of pellets left so
    that less pellets means higher score. Eating a pellet has a higher weight than eating
    food because pellets give you scared ghosts.

    Third is active ghost score. This is calculated by the distance the nearest ghost
    that is not scared. Then it is normalized by multiplying it by .03. Since all the other
    scores are reciprocals, the numbers are overall very small. When we used only the
    distance value, it skewed our results because the distance is not a fraction.
    We tried different values but .03 gave the best results. 

    Lastly, we calculated the scared ghost score. Similar to food score, we used
    the reciprocal of the distance to the nearest scared ghost. We wanted to incentivize eating
    the ghost because it gave bonus points. 

    Both ghost scores have the same weight, which is lower than eating food because we
    want the ghost to prioritize eating food to reach its goal

    Then we added all the scores together to get the overall score of the state. 
    """
    if currentGameState.isWin():
        return math.inf
    if currentGameState.isLose():
        return -math.inf

    # useful information from game state
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    pellets = currentGameState.getCapsules()

    # set weights of different scores
    # see description for explanation
    foodWeight1, foodWeight2, pelletWeight, ghostWeight = 2, 4, 20, 3

    # initialize scores
    foodScore, pelletsScore, actGhScore, scGhScore = 0, 0, 0, 0
    score = currentGameState.getScore()
    
    if food:
        # manhattan distance to closest food
        foodList = [(x,y) for x in range(0, food.width) for y in range(0, food.height) if food[x][y] == True]
        nearestFood = min([util.manhattanDistance(pos, (x,y)) for x,y in foodList])
        foodScore = foodWeight1*(1/(nearestFood)) 
        foodScore += foodWeight2*(1/len(foodList))

    # pellets score
    if pellets:
        pelletsScore = (1/len(pellets)) * pelletWeight

    activeGhosts = [ghost for ghost in ghostStates if not ghost.scaredTimer]
    scaredGhosts = [ghost for ghost in ghostStates if ghost.scaredTimer]

    # active ghosts score
    if activeGhosts:
        nearestActGh = min([util.manhattanDistance(pos, ghost.getPosition()) for ghost in activeGhosts])
        actGhScore = nearestActGh * .03 * ghostWeight

    # scared ghosts score
    if scaredGhosts:
        nearestScGh = min([util.manhattanDistance(pos, ghost.getPosition()) for ghost in scaredGhosts])
        scGhScore = (1/nearestScGh) * ghostWeight

    score += foodScore + pelletsScore + actGhScore + scGhScore
    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
