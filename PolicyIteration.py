import numpy as np


def PI(S, A, P, R, ε, γ):
    """ Policy Iteration Algorithm. """
    # Initialization
    π = np.random.choice(A, size=len(S))
    V = np.zeros(len(S))

    while True:
        # Policy Evaluation
        while True:
            V_dash = V.copy()
            for s in S:
                V[s] = R[s][π[s]] + γ*(P @ V)[s][π[s]]
            if np.max(np.abs(V - V_dash)) < ε:
                break
        # Policy Improvement
        π_dash = π.copy()
        for s in S:
            π[s] = np.argmax(R[s] + γ*(P @ V)[s])
        if np.array_equal(π_dash, π):
            break
    
    return (V, π)
    