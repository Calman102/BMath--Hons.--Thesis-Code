from Guessing_Game_POMDP import S, A, P, R, O, Obs

import matplotlib.pyplot as plt
import time
import random
import numpy as np 


def τ(b, a, o, plot=False):
    """
    Gets the new belief state b' using an exact belief update.
    """
    b_dash = np.zeros(len(S))
    for s_dash in S:
        norm_const = (Obs.T @ ((P.T @ b).T)[a])[o,a]
        b_dash[s_dash] = Obs[s_dash,a,o] * (P.T @ b)[s_dash,a] / norm_const
        
    if plot:
        plt.plot(S, b_dash)
        plt.show()    
        
    return b_dash


def collect_belief_points(b_0, n):
    """
    Collects a set of reachable belief points.
    """
    B = [b_0]
    for _ in range(2*len(A)):
        b_dash = b_0
        for _ in range(n-1):
            b = b_dash.copy()
            s = np.random.choice(S, p=b)
            a = 4 if s < 2 else 0  # MDP Optimal Policy
            s_dash = np.random.choice(S, p=P[s,a])
            o = np.random.choice(O, p=Obs[s_dash,a])
            b_dash =  τ(b,a,o)
            if True not in [np.array_equal(b, b_dash) for b in B]:
                B.append(b_dash)
    return B
    
    
def backup(b, Γ, γ):
    """
    Backup function for generating new α-vectors.
    """
    βi_ao = np.zeros([len(Γ), len(A), len(O), len(S)])
    β_ao = np.zeros([len(A), len(O), len(S)])
    for a in A:
        for o in O:
            for i in range(len(Γ)):
                for s in S:
                    βi_ao[i,a,o,s] = np.sum(P[s,a,:] * Obs[:,a,o] * Γ[i][0])
            n = np.argmax(np.array([b.dot(βi_ao[i,a,o]) for i in range(len(Γ))]))
            β_ao[a,o] = βi_ao[n,a,o]
    β_a = (R + γ * np.sum(β_ao, 1).T).T
    β = β_a[np.argmax(β_a.dot(b))]
    a = A[np.argmax(β_a.dot(b))]
    return (β, a)
    
    
def PBVI(ε, γ, b_0="uniform"):
    """
    Point-Based Value Iteration using the Perseus backup stage.
    Takes a tolerance ε, discount factor γ and initial belief b_0.
    """
    if b_0 == "uniform":
        b_0 = np.ones(len(S)) / len(S)
    B = collect_belief_points(b_0, 10)
    
    π_unchanged = 0
    V = {tuple(b): 0 for b in B}
    V_π = {}
    π = {}
    π_dash = {}
    
    V_dash = {tuple(b): -np.inf for b in B}            # Old V
    Γ_dash = [(np.min(R)/(1-γ) * np.ones(len(S)), 0)]  # Old Γ
    
    # Value Improvement
    start = time.time()
    n = 1
    while True:
        print(f"ITERATION {n}")
        Γ = []  # new Γ
        B_tilde = B.copy()
        # _____________________Perseus backup stage_____________________
        while len(B_tilde) > 0:
            b_tmp = random.choice(B_tilde)
            α_tmp = backup(b_tmp,Γ_dash,γ)
            if b_tmp.dot(α_tmp[0]) >= V_dash[tuple(b_tmp)]:
                B_tilde = [b for b in B_tilde 
                           if b_tmp.dot(α_tmp[0]) < V_dash[tuple(b)]]
            else:
                B_tilde = [b for b in B_tilde 
                           if not np.array_equal(b, b_tmp)]
                α_list = np.array([b_tmp.dot(α[0]) for α in Γ_dash])
                α_tmp = Γ_dash[np.argmax(α_list)]
            Γ.append(α_tmp)
        # ______________________________________________________________
        for b in B:
            V_π[tuple(b)] = max((b.dot(α[0]), α[1]) for α in Γ)
            V[tuple(b)] = V_π[tuple(b)][0]
            π[tuple(b)] = V_π[tuple(b)][1]
        π_unchanged = π_unchanged + 1 if π == π_dash else 0
        Δ = max(abs(V[tuple(b)] - V_dash[tuple(b)]) for b in B)
        if Δ < 1e-5 or π_unchanged == 10 or n == 5000:
            stop = time.time()
            print(f"Δ = {Δ}\n")
            print(f"PBVI terminated after {stop-start} seconds.")
            break
        Γ_dash = Γ.copy()
        V_dash = V.copy()
        π_dash = π.copy()
        print(f"Δ = {Δ}\n")
        n += 1
    return V, π

    
if __name__ == "__main__":
    ε = 10**(-3)
    γ = 0.95
    b_0 = np.array([0.5, 0.5, 0, 0])
    V, π = PBVI(ε, γ, b_0)
    