class POMDPThresholdPolicy:
    def __init__(self, A, η_thresholds, q):
        self.A = A.copy()
        self.η_thresholds = η_thresholds
        self.q = q
        
        self.τ = {}
        
    def update(self, η=None, q=None, η_index=-1):
        self.τ.clear()
        if η != None:
            self.η_thresholds[η_index] = η
        if q != None:
            self.q = q
        
    def add_threshold(self, η=None):
        self.τ.clear()
        if η != None:
            self.η_thresholds.append(η)
            
    def add_action(self):
        self.A.append(self.A[-1]+1)
        
    def get_action(self, b):
        if self.τ.get(tuple(b)) == None:
            for i in range(len(self.η_thresholds)):
                η = int(self.η_thresholds[i])
                p = np.sum(b[η+1:])
                if p >= self.q:
                    self.τ[tuple(b)] = self.A[i+1]
                else:
                    self.τ[tuple(b)] = self.A[0]
        return self.τ[tuple(b)]                   
            
    def copy(self):
        A = self.A.copy()
        η_thresholds = self.η_thresholds.copy()
        q = self.q
        return MDPThresholdPolicy(A, η_thresholds, q)
