class LinearEpsilon:
    def __init__(self,start=.7,end=.01,delta=.00001):
        self.epsilon=start
        self.end_epsilon = end
        self.delta = delta
    def get_epsilon(self):
        self.epsilon -= self.delta
        if self.epsilon<=self.end_epsilon:
            return self.end_epsilon
        else:
            return self.epsilon