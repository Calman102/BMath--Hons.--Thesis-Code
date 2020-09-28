class MDPThresholdPolicy:
    def __init__(self, S, A, thresholds):
        self.S = S.copy()
        self.A = A.copy()
        if type(thresholds) == int:
            self.thresholds = [thresholds]
        else:
            self.thresholds = thresholds
        
    def asarray(self):
        arr = np.zeros(len(self.S))
        for i in range(len(self.thresholds)):
            η = self.thresholds[i]
            arr[η+1:] = i+1
        return arr
        
    def update(self, η, index=-1):
        self.thresholds[index] = η
        
    def add_threshold(self, η):
        self.thresholds.append(η)
        
    def add_action(self):
        self.A.append(self.A[-1]+1)
        
    def copy(self):
        S = self.S.copy()
        A = self.A.copy()
        thresholds = self.thresholds.copy()
        return MDPThresholdPolicy(S, A, thresholds)
     
    def __getitem__(self, key):
        for i in reversed(range(len(self.thresholds))):
            if key > self.thresholds[i]:
                return self.A[i+1]
        return self.A[0]
        
    def __iter__(self):
        return MDPThresholdPolicyIterator(self)
        

class MDPThresholdPolicyIterator:
    def __init__(self, policy):
        self._policy = policy
        self._index = 0
        self._current_a = self._policy.A[0]
        self._end = False
    
    def __next__(self):
        if not self._end:
            result = self._current_a
            self._index += 1
            if self._index < len(self._policy.S):
                current_s = self._policy.S[self._index]
                if current_s + 1 in self._policy.thresholds:
                    self._current_a += 1 
            else:
                self._end = True
            return result
        raise StopIteration