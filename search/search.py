# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # Fringe is a stack containing tuples of (node, pathTo)
    closed = set()
    startPath = list()
    fringe = util.Stack()
    fringe.push((problem.getStartState(), startPath))

    # Main loop
    while not fringe.isEmpty():
        element = fringe.pop()
        pathTo = element[1]
        state = element[0]

        # Test for goal
        if problem.isGoalState(state):
            # Return path
            return pathTo

        # If state not in closed, add to closed and expand node.
        if state not in closed:
            closed.add(state)
            for child in problem.getSuccessors(state):
                thisPath = pathTo.copy()
                thisPath.append(child[1])
                fringe.push((child[0], thisPath))
    return pathTo


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Fringe is a queue containing tuples of (node, pathTo)
    closed = set()
    startPath = list()
    overallPath = list()
    fringe = util.Queue()
    fringe.push((problem.getStartState(), startPath))

    # Main loop
    while not fringe.isEmpty():
        element = fringe.pop()
        state = element[0]
        pathTo = element[1]

        # Test for goal
        isGoal = problem.isGoalState(state)
        if isGoal == True:
            # Return path
            return pathTo

        # If state not in closed, add to closed and expand node.
        if state not in closed:
            closed.add(state)
            for child in problem.getSuccessors(state):
                thisPath = pathTo.copy()
                thisPath.append(child[1])
                fringe.push((child[0], thisPath))
    return startPath

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Fringe is a Priority Queue contain tuples of (node, pathTo, backwardCost)
    closed = set()
    startPath = list()
    startCost = 0
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), startPath, startCost), 0)

    # Main loop
    while not fringe.isEmpty():
        element = fringe.pop()
        state = element[0]
        pathTo = element[1]
        backwardCost = element[2]
        

        # Test for goal
        if problem.isGoalState(state):
            # Return path
            return pathTo

        # If state not in closed, add to closed and expand node.
        if state not in closed:
            closed.add(state)
            for child in problem.getSuccessors(state):
                thisPath = pathTo.copy()
                thisPath.append(child[1])
                thisCost = backwardCost + child[2]
                fringe.push((child[0], thisPath, thisCost), thisCost)
    return startPath

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Fringe is a Priority Queue contain tuples of (node, pathTo, backwardCost)
    closed = set()
    startPath = list()
    startCost = 0
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), startPath, startCost), 0)

    # Main loop
    while not fringe.isEmpty():
        element = fringe.pop()
        state = element[0]
        pathTo = element[1]
        backwardCost = element[2]
        

        # Test for goal
        if problem.isGoalState(state):
            # Return path
            return pathTo

        # If state not in closed, add to closed and expand node.
        if state not in closed:
            closed.add(state)
            for child in problem.getSuccessors(state):
                if child[0] not in closed:
                    thisPath = pathTo.copy()
                    thisPath.append(child[1])
                    # Priority cost
                    estCost = backwardCost + child[2] + heuristic(child[0], problem)
                    # Actual cost
                    actualCost = backwardCost + child[2]
                    fringe.push((child[0], thisPath, actualCost), estCost)
    return startPath


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
