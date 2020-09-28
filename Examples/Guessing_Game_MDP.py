import numpy as np


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
        

def R_func(s, a):
    """ Function representation of the expected reward. """
    if s == 3:
        return 0
    else:
        return 5 * (s + 1)**2 / (a + 1) - 10


# State Space
S = [0, 1, 2, 3]

# Action Space
A = [0, 1, 2, 3, 4]

# Transition Probabilities
P = np.array([[[P_func(s, a, s_dash) for s_dash in S] for a in A] for s in S])

# Expected Reward
R = np.array([[R_func(s, a) for a in A] for s in S])
