# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # Start the generic value update function -> start at 1 bc 0 already handled in init.
        iteration = 0
        while iteration < self.iterations:

            # Assign all new values to temporary Counter.
            tempValues = self.values.copy()

            # Run update function on every state in the mdp.
            for state in self.mdp.getStates():
                bestVal = -999999
                # Run the Q val update func for each action and choose the 
                # value associated with the optimal action.
                if self.mdp.getPossibleActions(state):
                    for action in self.mdp.getPossibleActions(state):
                        currVal = self.computeQValueFromValues(state, action)
                        if bestVal < currVal:
                            bestVal = currVal
                else:
                    bestVal = 0

                # Assign best possible value to the value dictionary.    
                tempValues[state] = bestVal

            
            # Replace values Counter with update values and increment iteration counter.
            self.values = tempValues
            iteration += 1
    # End runValueIteration()


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        summedReward = 0
        for nextState in self.mdp.getTransitionStatesAndProbs(state, action):
            sPrime, prob = nextState[0], nextState[1]
            # Iteratively sum the value for the possible next states
            if not self.mdp.isTerminal(sPrime):
                summedReward += prob*(self.mdp.getReward(state, action, sPrime)+self.discount*(self.values[sPrime]))
            else:
                summedReward = prob*self.mdp.getReward(state, action, sPrime)
        return summedReward
                        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        bestVal = -999999

        # Iterate through all possible actions and choose the action with the highest associated value.
        for action in self.mdp.getPossibleActions(state):
            currVal = self.computeQValueFromValues(state, action)
            if bestVal < currVal:
                bestVal = currVal
                bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # States all set to 0, start iterations at 0.
        iteration = 0
        
        while iteration < self.iterations:

            # Run update function on each state one at a time - determined by indexing with curr iteration.
            state = self.mdp.getStates()[iteration % len(self.mdp.getStates())]
            bestVal = -999999
            # Run the Q val update func for each action and choose the 
            # value associated with the optimal action.
            if self.mdp.getPossibleActions(state):
                for action in self.mdp.getPossibleActions(state):
                    currVal = self.computeQValueFromValues(state, action)
                    if bestVal < currVal:
                        bestVal = currVal
            else:
                bestVal = 0

            # Assign best possible value to the value dictionary.    
            self.values[state] = bestVal

            # Increment iteration counter.
            iteration += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states and store in a dictionary of sets.
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState in self.mdp.getTransitionStatesAndProbs(state, action):
                    sPrime = nextState[0]
                    predecessors[sPrime].add(state)

        # Initialize empty priority queue.
        pQueue = util.PriorityQueue()

        # Push states to pQueue.
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                diff = self.values[state]
                highestQ = -999999
                for action in self.mdp.getPossibleActions(state):
                    qVal = self.computeQValueFromValues(state, action)
                    if qVal > highestQ:
                        highestQ =  qVal
                diff = abs(diff - highestQ)
                pQueue.push(state, -diff)
        
        # Iterate through pQueue 
        iteration = 0
        while iteration < self.iterations:
            if pQueue.isEmpty():
                break
            s = pQueue.pop()
            # Update value of s if not terminal state.
            if not self.mdp.isTerminal(s):
                bestVal = -999999
                # Run the Q val update func for each action and choose the 
                # value associated with the optimal action.
                if self.mdp.getPossibleActions(s):
                    for action in self.mdp.getPossibleActions(s):
                        currVal = self.computeQValueFromValues(s, action)
                        if bestVal < currVal:
                            bestVal = currVal
                else:
                    bestVal = 0

                # Assign best possible value to the value dictionary.    
                self.values[s] = bestVal

            # Find abs val of diff between curr value of each predecessor and 
            # highest q val across all possible actions from p.
            for p in predecessors[s]:
                diff = self.values[p]
                highestQ = -999999
                for action in self.mdp.getPossibleActions(p):
                    qVal = self.computeQValueFromValues(p, action)
                    if qVal > highestQ:
                        highestQ =  qVal
                diff = abs(diff - highestQ)
                if diff > self.theta:
                    pQueue.update(p, -diff)
            
            # Increment iteration counter.
            iteration += 1