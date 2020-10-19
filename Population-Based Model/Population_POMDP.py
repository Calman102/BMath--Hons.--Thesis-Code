from scipy.stats import norm
from ThresholdPolicies import POMDPThresholdPolicy
from SCO import SCO
from ProblemSpec import S, A, O, Reward, TransitionProbability, ObservationProbability
import matplotlib.pyplot as plt
import numpy as np


# Initial Belief State
b = np.array([norm(50,10).pdf(s) for s in S])  
b = 1/np.sum(b) * b

R = Reward().as_matrix(exact=True)
P = TransitionProbability().as_matrix()
Obs = ObservationProbability().as_matrix()


def τ(b, a, o, plot=False):
    """
    Gets the new belief state b' using an exact belief update.
    """
    b_dash = np.zeros(len(S))
    for s_dash in S:
        norm_const = (Obs.T @ ((P.T @ b).T)[a])[o,a]
        b_dash[s_dash] = Obs[s_dash,a,o] * (P.T @ b)[s_dash,a] / norm_const
        
    if plot:
        plt.plot(S, b, "C1")
        plt.plot(S, b, "C0.")
        plt.xlabel("$s$")
        plt.ylabel("P($S = s$)")
        plt.show()
        
    return b_dash
    

def τ_hat(b, a, o, K, plot=False):
    """
    Gets the new belief state b'.
    """
    S_dash_tilde = np.zeros(K)
    b_dash = np.zeros(len(S))
    S_tilde = np.random.choice(S, size=K, p=b)
    i = 0
    while i < K:
        s_tilde  = np.random.choice(S_tilde)
        s_dash_tilde, o_tilde = state_transition(s_tilde,a,b)
        if o_tilde == o:
            S_dash_tilde[i] = s_dash_tilde
            i += 1
    for s_dash in S:
        b_dash[s_dash] = 1/K * np.sum(S_dash_tilde == s_dash)
    
    if plot:
        plt.plot(S, b, "C1")
        plt.plot(S, b, "C0.")
        plt.xlabel("$s$")
        plt.ylabel("P($S = s$)")
    
    return b_dash
    
    
def state_transition(s, a, b):
    s_dash = np.random.choice(S, p=P[s,a])
    o = np.random.choice(O, p=Obs[s_dash,a])
    
    return s_dash, o
    

def rollout(b_0, π, γ, depth=10**2, start=None, loop=1, console=False):
    """ 
    Simulates a rollout.
    """
    V = np.array([0 for _ in range(loop)])
    N = int(depth)
    
    for i in range(loop):
        if console:
            print("ROLLOUT START")
        b = b_0.copy()
        s = np.random.choice(S, p=b) if start == None else start
        for n in range(N):
            a = π.get_action(b)
            s_dash = np.random.choice(S, p=P[s][a])
            r = γ**n * R[s][a][s_dash]
            V[i] = V[i] + r
            if s_dash == 0 and console:
                print("Extinction!")
            if console:
                print("-" * 50)
                print(f"Population:  {s_dash}")
                print(f"mean(S):     {π.get_mean(b)}")
                print(f"sd(S):       {π.get_sd(b)}")
            o = s - s_dash
            # o = np.random.choice(O, p=Obs[s_dash][a])
            # b = τ_hat(b, a, o, 1000, plot=True)
            b = τ(b, a, o, plot=console)
            s = s_dash
        if console:
            print(f"\nROLLOUT completed with total V = {V / loop}")
            print("-" * 50)
    
    return np.mean(V)

    
if __name__ == "__main__":
    print("Initial Belief:", end="")
    plt.plot(S, b, "C1")
    plt.plot(S, b, "C0.")
    plt.xlabel("$s$")
    plt.ylabel("P($S = s$)")
    plt.show()
    print("\n")
    
    np.random.seed(123)
    
    f = lambda X: rollout(b, POMDPThresholdPolicy(A, [X[0]], X[1]), 0.95, loop=10)
    Ψ, V = SCO(f, 10, 0.6, 0.5, [np.array([0, -3]), np.array([100, 3])], MaxTry=3, T=10)
    print(Ψ)
    print(V)
    print(f"Mean Value:     {np.mean(list(V.values()))}")
    print(f"Standard Error: {np.std(list(V.values())) / np.sqrt(len(V))}")
