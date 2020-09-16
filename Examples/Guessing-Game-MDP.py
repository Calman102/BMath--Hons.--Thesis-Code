from ValueIteration import VI
from PolicyIteration import PI
import numpy as np
import time


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

# Tolerance
ε = 10**(-3)


print("\nVALUE ITERATION")    
print("-" * 50)
for γ in [0.25, 0.6, 0.95, 0.99]:
    start = time.time()
    V, π = VI(S, A, P, R, ε, γ)
    V = [round(v, 6) for v in V]
    π = [int(a + 1) for a in π]
    stop = time.time()
    print("Discount Factor =", γ)
    print("  Time:", stop - start)  
    print(" Value:", V, "\nPolicy:", π)
    print('-' * 50)  

print("\nPOLICY ITERATION")
print("-" * 50)
for γ in [0.25, 0.6, 0.95, 0.99]:
    start = time.time()
    V, π = PI(S, A, P, R, ε, γ)
    V = [round(v, 6) for v in V]
    π = [int(a + 1) for a in π]
    stop = time.time()
    print("Discount Factor =", γ)
    print("  Time:", stop - start)  
    print(" Value:", V, "\nPolicy:", π)
    print('-' * 50)
