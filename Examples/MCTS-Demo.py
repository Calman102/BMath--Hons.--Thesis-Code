import numpy as np
from copy import deepcopy
from mcts import mcts
from scipy.stats import mode


def P_func(s, a, s_dash):
    """ Function representation of the transition probabilities. """
    if s == 3 and s_dash == 3:
        return 1
    elif s_dash == s + 1:
        return (a + 1) / 10
    elif (s == 0 and s_dash == 0) or (s_dash == s - 1 and s != 3):
        return 1 - (a + 1) / 10
    else:
        return 0
        

def R_func(s, a, s_dash):
    """ Function representation of the expected reward. """
    if s == 3:
        return 0
    elif s_dash == s + 1:
        return 50 * (s + 1)**2 / (a + 1)**2 - 10
    else:
        return -10


S = [0, 1, 2, 3]
A = [0, 1, 2, 3, 4]
P = np.array([[[P_func(s, a, s_dash) for s_dash in S] for a in A] for s in S])
R = np.array([[[R_func(s, a, s_dash) for s_dash in S] for a in A] for s in S])


class MCTS_State():
    def __init__(self, S, A, P, R, s0=0):
        self.S = S
        self.A = A
        self.P = P
        self.R = R
        self.state = s0
        self.path = []
        self.reward = 0

    def getCurrentPlayer(self):
        return 1  # only a one player game

    def getPossibleActions(self):
        return self.A

    def takeAction(self, action):
        s = self.state
        s_dash = np.random.choice(S, p=self.P[s][action])
        newState = deepcopy(self)
        newState.state = s_dash
        newState.reward = self.reward + self.R[s][action][s_dash]
        newState.path.append(s_dash)
        return newState

    def isTerminal(self):
        return self.state == 3

    def getReward(self):
        return self.reward


C = 0.6

initial_state = MCTS_State(S, A, P, R, s0=0)
tree = mcts(timeLimit=1000, explorationConstant=C)
action = tree.search(initialState=initial_state)

print(action)
