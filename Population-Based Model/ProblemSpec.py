import numpy as np
from scipy.stats import norm


# Action Space 
A = [0, 1]

#  State Space
S = list(range(101))

# Observation Space
O = list(range(int(np.max(S)/5)+1))

# Initial Belief State (for use in Obs)
b_0 = 1 / len(S) * np.ones(len(S))
#b_0 = np.array([norm(50,10).pdf(s) for s in S])      
#b_0 = b_0 / np.sum(b_0) 


class Reward:
    def __init__(self):
        pass 
    
    def R(self, s, a, s_new=None):
        """
        Reward from taking action a in state s.
        If s_new is not provided, this is the expected reward.
        """
        if s == 0:    # terminating state
            return -10000
        
        elif a == 0:  # do nothing
            return 0
        
        elif s_new:   # exact reward if new state is known
            return 50*(s - s_new)
        
        elif a == 1:  # expected reward from small catch
            return 50*np.ceil(s/5) / 2
        
        elif a == 2:  # expected reward from big catch
            return 50*(np.ceil(4*s/5) + np.ceil(s/5)) / 2
        
    def as_matrix(self, exact=False):
        if not exact:
            mat = np.array([[self.R(s,a) for a in A] for s in S])
        else:
            mat = np.array([[[self.R(s,a,s_dash) for s_dash in S] for a in A] for s in S])
        return mat        


class TransitionProbability:
    def __init__(self):
        self.e = 10**(-2)                       # Δt
        self.BR = lambda s: 7 * min(1, 0.07*s)  # Birth Rate
        self.DR = lambda s: 7/50 * s            # Death Rate
        
    def P(self, s, a, s_dash):
        """
        Probability of transitoning from s to s' when taking action a.
        """
        # terminating state
        if s == 0:
            return 1 if s_dash == 0 else 0
        
        # do nothing
        elif a == 0:
            if s == max(S):
                if s_dash == s:
                    return 1 - self.e*self.DR(s)
                elif s_dash == s - 1:
                    return self.e*self.DR(s)
                else:
                    return 0
            elif s_dash == s + 1:
                return self.e*self.BR(s)
            elif s_dash == s - 1:
                return self.e*self.DR(s)
            elif s_dash == s:
                return 1 - self.e*(self.DR(s) + self.BR(s))
            else:
                return 0
            
        # small catch
        elif a == 1:
            if s <= 1:
                return 1/2 if s_dash <= 1 else 0
            elif s - np.ceil(s/5) <= s_dash <= s:
                return 1 / (np.ceil(s/5) + 1)
            else:
                return 0
            
        # big catch
        elif a == 2:
            if s <= 1:
                return 1 if s_dash == 0 else 0
            elif s - np.ceil(4*s/5) <= s_dash <= s - np.ceil(s/5):
                return 1 / (np.ceil(4*s/5) - np.ceil(s/5) + 1)
            else:
                return 0
            
    def as_matrix(self):
        mat = np.array([[[self.P(s,a,s_dash) for s_dash in S] for a in A] for s in S])
        return mat
            

class ObservationProbability:
    def __init__(self, b=None):
        self.P = TransitionProbability().as_matrix()
        if b:
            self.b = b
        else:
            # Assuming a uniform prior
            self.b = 1/len(S) * np.ones(len(S))
    
    def Obs(self, s_dash, a, o, b=None):
        """
        Probability of observing o after taking action a and transitioning to s'.
        """
        # dummy observation probability if no interference
        if a == 0:
            return 1 / len(O)
        
        # impossible observations
        elif o + s_dash >= len(S):
            return 0
        
        # observations after interference 
        b = self.b if type(b) == type(None) else b
        if a == 1:
            x = self.P[o + s_dash][a][s_dash] * b[o + s_dash]
            y = (self.P.T @ b)[s_dash][a]
            return x / y
        
    def simulated(self):
        """
        Probability matrix of observing o after taking action a and transitioning to s'.
        Calculated via simulation.
        """
        Obs = {}
        for s in S:
            for _ in range(10**5):
                s_dash = np.random.choice(S, p=self.P[s][1])
                o = s - s_dash
                if not O.get((s_dash,1,o)):
                    Obs[(s_dash,1,o)] = 1
                else:
                    Obs[(s_dash,1,o)] += 1
        
        mat = np.ones((len(S), len(A), len(O)))
        for s_dash in S:
            prob = np.array([Obs.get((s_dash,1,o),0) for o in O])
            mat[s_dash][1] = 1/np.sum(prob) * prob
            mat[s_dash][0] = 1/len(O) * np.ones(len(O))
            
        return mat
        
    def as_matrix(self):
        mat = np.array([[[self.Obs(s_dash,a,o) for o in O] for a in A] for s_dash in S])
        return mat
