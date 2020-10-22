import numpy as np
import timeit


def SCO(S, N, ς, w, B, MaxTry=5, T=10):
    """ Application of SCO for solving a threshold policy with a single threshold. """
    time_start = timeit.default_timer()
    
    Y_SET = {0: [np.random.uniform(B[0], B[1]) for _ in range(N)]}
    X_SET = {}
    X_best = {}
    V = {}
    t = 0
    N_elite = int(np.ceil(N * ς))
    
    R = np.zeros(N_elite) 
    σ = {}              
    I_SET = np.arange(N_elite)
    
    B_i = np.concatenate((np.ones(N - N_elite), np.zeros(2*N_elite - N)))
    
    best = (None, -np.inf)
    while True:
        print(f"ITERATION {t}...")
        S_X = np.array([S(X) for X in Y_SET[t]])
        idx = np.argsort(S_X)[::-1][:N_elite]
        S_X = S_X[idx]
        X = [Y_SET[t][i] for i in idx]
        X_SET[t+1] = X
        V[t+1] = S_X[0]
        X_best[t+1] = X[0].copy()
        
        np.random.shuffle(B_i)      
        
        for i in range(N_elite):
            R[i] = int(np.floor(N / N_elite) + B_i[i])  # random splitting factor
            Y = X[i].copy()
            Y_dash = Y.copy()
            
            for j in range(int(R[i])):
                I = np.random.choice(I_SET[I_SET != i])
                σ[i] = w * np.abs(X[i] - X[I])
                μ = np.random.permutation(2)
                
                for Try in range(MaxTry):  # optimising the threshold
                    Z = np.random.normal()
                    Y_dash[μ[0]] = max(0, min(100, Y[μ[0]] + σ[i][μ[0]] * Z))
                    if S(Y_dash) > S(Y):
                        Y = Y_dash.copy()
                        break
                    
                for Try in range(MaxTry):  # optimising the quantile
                    Z = np.random.normal()
                    Y_dash[μ[1]] = Y[μ[1]] + σ[i][μ[1]] * Z
                    if S(Y_dash) > S(Y):
                        Y = Y_dash.copy()
                        break
                        
                if Y_SET.get(t+1) == None:
                    Y_SET[t+1] = []
                Y_SET[t+1].append(Y.copy())
                
        t = t + 1
        if V[t] > best[1]:
            best = (X_best[t], V[t])
            print(f"Best value:      {best[1]}")
            print(f"Best threshold:  {best[0][0]}")
            print(f"Best quantile:   {best[0][1]}")
        
        if t == T:
            break
        
    time_stop = timeit.default_timer()
    print(f"\nTerminated after {time_stop - time_start} seconds.")
        
    return (X_best, V)        
