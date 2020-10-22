import numpy as np


class POMDPThresholdPolicy:
    def __init__(self, A, η_thresholds, q):
        self.A = A.copy()
        if type(thresholds) == int:
            self.thresholds = [thresholds]
        else:
            self.thresholds = thresholds
        self.q = q
        
        self.ψ = {}
        
    def update(self, η=None, q=None, η_index=-1):
        self.ψ.clear()
        if η != None:
            self.η_thresholds[η_index] = η
        if q != None:
            self.q = q
        
    def add_threshold(self, η=None):
        self.ψ.clear()
        if η != None:
            self.η_thresholds.append(η)
            
    def add_action(self):
        self.A.append(self.A[-1]+1)
        
    def get_mean(self, b):
        return np.sum(np.arange(len(b)) * b)
    
    def get_sd(self, b):
        m2 = np.sum(np.arange(len(b))**2 * b)
        return np.sqrt(m2 - self.get_mean(b)**2)
        
    def get_action(self, b):
        if self.ψ.get(tuple(b)) is None:
            for i in reversed(range(len(self.η_thresholds))):
                η = int(self.η_thresholds[i])
                val = self.get_mean(b) - self.q * self.get_sd(b)
                if val >= η:
                    self.ψ[tuple(b)] = self.A[i+1]
                    break
            else:
                self.ψ[tuple(b)] = self.A[0]
        return self.ψ[tuple(b)]                   
            
    def copy(self):
        A = self.A.copy()
        η_thresholds = self.η_thresholds.copy()
        q = self.q
        return POMDPThresholdPolicy(A, η_thresholds, q)
