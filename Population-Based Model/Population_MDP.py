from scipy.stats import mode
from ThresholdPolicies import MDPThresholdPolicy
from ProblemSpec import *
import numpy as np
import matplotlib.pyplot as plt
import timeit


A.append(2)  # Big Catch

R = Reward().as_matrix() 
P = TransitionProbability().as_matrix()


def PI(S, A, ε, γ):
    """
    A policy iteration algorithm.
    """
    π = A[1]*np.ones(len(S))
    V = np.zeros(len(S))

    while True:
        while True:
            delta = 0
            V_dash = V.copy()
            for s in S:
                a = int(π[s])
                V[s] = R[s][a] + γ*(P @ V)[s][a]
                delta = max(delta, abs(V_dash[s] - V[s]))
            if delta < ε:
                break 
        π_dash = π.copy()
        for s in S:
            π[s] = np.argmax(R[s] + γ*(P @ V)[s])
        if np.array_equal(π_dash, π):
            break
        
    return (V, π)
     

def EvaluatePolicy(π, ε, γ):
    """ Evaluates a policy. """
    V = np.zeros(len(S))
    
    while True:
        Δ = 0
        V_dash = V.copy()
        for s in S:
            a = int(π[s])
            V[s] = R[s][a] + γ*(P @ V)[s][a]
            Δ = max(Δ, abs(V_dash[s] - V[s]))
        if Δ < ε:
            break
            
    return V
    
    
def PolicyImprovement(ε, γ):
    V_max = np.array([-np.inf for _ in range(len(S))])
    π_max = np.array([0 for _ in range(len(S))])
    ψ_max = [0]
    
    πs = [MDPThresholdPolicy(S, A, [i] + [max(S) for _ in range(len(A) - 2)]) for i in [0, 1]]

    loop_start = timeit.default_timer()    
    for n in range(0, len(A)-1):
        print(f"\nBEGINNING PASS {n} OUT OF {len(A)-1}")
        print("-" * 50)
        
        k_lb = min(S) if n == 1 else ψ_max[n-1]
        k_ub = max(S)    
        
        i = 1
        while True:
            k1 = np.random.randint(k_lb, k_ub)
            k2 = k1 + 1
            πs[0].update(k1, index=n)
            πs[1].update(k2, index=n)
            
            print(f'ITERATION {i}')
            print(f'  Current Bounds on η_{n+1}:   {[k_lb, k_ub]}')
            print(f'  Current Best Threshold:  {ψ_max}')
            print(f'  Comparing Thresholds:    {πs[0].thresholds} & {πs[1].thresholds}')
            start = timeit.default_timer()
            
            count = 0
            Vs = [EvaluatePolicy(π.asarray(),ε,γ) for π in πs]
            SumVs = [np.sum(V) for V in Vs]
        
            j = int(SumVs[1] > SumVs[0])
            if j == 0:  # k1 was better than k2
                k_ub = k1 - 1
                print(f'    {ψ_max[:-1] + [k1]} is better...')
            else:       # k2 was better than k1
                k_lb = k2
                print(f'    {ψ_max[:-1] + [k2]} is better...')
                
            if SumVs[j] > np.sum(V_max):
                V_max = Vs[j].copy()
                π_max = πs[j].asarray()
                ψ_max = πs[j].thresholds.copy()
                
            stop = timeit.default_timer()
            print(f'  Time taken: {round(stop-start,4)} seconds')
            print("-" * 50) 
                
            if k_lb >= k_ub:
                πs[0].update(ψ_max[n], index=n)
                πs[1].update(ψ_max[n], index=n)
                break
            
            i += 1
    
    loop_end = timeit.default_timer()
    print(f'\nTotal duration was {loop_end-loop_start} seconds\n')
    
    return V_max, π_max
    

def main():
    V, π = PolicyImprovement(10**(-5), 0.95)
    print(f"Summed value = {np.sum(V)}")
    print(π, "\n")

    print("Policy Iteration Result:")
    start = timeit.default_timer()
    V2, π2 = PI(S, A, 10**(-5), 0.95)
    print(f"Summed Value = {np.sum(V2)}\n{π2}") 
    stop = timeit.default_timer()
    print('Time:', stop - start)
    plt.scatter(S, π, s=2)
    plt.xlabel("State (Population)")
    plt.yticks(A)
    plt.ylabel("Action")
    plt.title("Actions = {Wait, Catch}" if len(A) == 2 
              else "Actions = {Wait, Small Catch, Big Catch}")
    plt.show()
    

if __name__ == "__main__":
    main()
