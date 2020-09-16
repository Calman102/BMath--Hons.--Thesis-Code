import numpy as np


def VI(S, A, P, R, ε, γ):
    """ Value Iteration Algorithm. """
    # Initialization
    π = np.zeros(len(S))
    V = np.max(R, 1)
    
    # Value Improvement
    while True:
        V_dash = V.copy()
        for s in S:
            V[s] = np.max(R[s] + γ*(P @ V)[s])
        if np.max(np.abs(V - V_dash)) < ε:
            break
            
    # Policy Evaluation
    for s in S:
        π[s] = np.argmax(R[s] + γ*(P @ V)[s])

    return (V, π)
