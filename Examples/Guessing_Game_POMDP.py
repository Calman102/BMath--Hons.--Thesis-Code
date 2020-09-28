from Guessing_Game_MDP import S, A, P, R

import numpy as np


def Obs_func(s_dash, a, o):
    """ Function representation of the observation probabilities. """
    if o == 3 and s_dash == 3:
        return 1
    elif o == s_dash and s_dash <= 2:
        return 1 / 2
    elif o != 3 and s_dash <= 2:
        return 1 / 4
    else:
        return 0


# Observation Space
O = [0, 1, 2, 3]

# Observation Probabilities
Obs = np.array([[[Obs_func(s_dash, a, o) for o in O] for a in A] for s_dash in S])   
