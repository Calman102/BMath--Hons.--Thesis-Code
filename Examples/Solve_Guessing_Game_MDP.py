from ValueIteration import VI
from PolicyIteration import PI
from Guessing_Game_MDP import S, A, P, R

import time

ε = 10**(-3)  # Tolerance

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
