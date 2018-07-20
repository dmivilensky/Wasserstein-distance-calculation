import numpy as np

class SinkhornMethod:
    def __init__(self, gamma, n=5, epsilon=0.001, epsilon_prox=0.001, log=False):
        """
        Sinkhorn Method for Transport Problem
        :param gamma: regularizer multiplier
        :param dim: transport vector dimension
        :param epsilon: desired accuracy
        """
        
        # dual func variables for indicator functions
        # self.lambda_ = np.zeros(n)
        self.lambda_ = np.zeros(n)
        self.my = np.zeros(n)
        
        # constants
        self.gamma = gamma
        self.n = n
        self.epsilon = epsilon
        self.epsilon_prox = epsilon_prox
        
        self.log = log
        if self.log:
            print("–––––––––––––––––––––––––––––")
            print("Algorithm configuration:")
            print("gamma = " + str(gamma))
            print("eps = " + str(epsilon))
            print("eps prox = " + str(self.epsilon_prox))
            print("–––––––––––––––––––––––––––––\n")
    
    def _new_dual_variables(self, C, p, q, xk):
        """
        Calculates Lagrange equation variables
        """
        
        a = np.min(self.gamma + C + self.my.repeat(self.n).reshape((-1, self.n)))
        
        for i in range(self.n):
            try:
                self.lambda_[i] = self.gamma * (-a + np.log(1/p[i] * np.sum([xk[i, j] * np.exp(-(self.gamma + C[i, j] + self.my[j] - a)/self.gamma) for j in range(self.n)])))
            except:
                self.lambda_[i] = -np.inf


            
        a = np.min(self.gamma + C + self.lambda_.repeat(self.n).reshape((-1, self.n)).T)

        for j in range(self.n):
            try:
                self.my[j] = self.gamma * (-a + np.log(1/q[j] * np.sum([xk[i, j] * np.exp(-(self.gamma + C[i, j] + self.lambda_[i] - a)/self.gamma) for i in range(self.n)])))
            except:
                self.my[j] = -np.inf
    
    def _new_x(self, C, p, q, xk):
        x = np.zeros((self.n, self.n))
        
        a = np.min(self.gamma + C + self.lambda_.repeat(self.n).reshape((-1, self.n)) + self.my.repeat(self.n).reshape((-1, self.n)).T)
        
        for i in range(self.n):
            for j in range(self.n):
                try:
                    x[i, j] = xk[i, j] * np.exp(-(self.gamma + C[i, j] + self.lambda_[i] + self.my[j] - a)/self.gamma)
                except:
                    x[i, j] = 0
        return x
    
    def _new_phi(self, C, p, q, xk):
        x_sum = 0
        
        a = np.min(self.gamma + C + self.lambda_.repeat(self.n).reshape((-1, self.n)) + self.my.repeat(self.n).reshape((-1, self.n)).T)
        
        for i in range(self.n):
            for j in range(self.n):
                try:
                    x_sum += xk[i, j] * np.exp(-(C[i,j] + self.lambda_[i] + self.my[j] + self.gamma - a) / self.gamma)
                except:
                    x_sum += 0
                
        return - np.sum(self.lambda_ * p) - np.sum(self.my * q) - self.gamma * x_sum
    
    def _new_f(self, C, x, xk):
        return np.sum(C * x) + self.gamma * np.sum((x + 1e-16) * np.log((x + 1e-16) / xk))
    
    def fit(self, C, p, q, with_prox=True):
        T = 0
        k = 0
        x = 1/self.n**2 * np.ones((self.n, self.n))
        while True:
            xk = x.copy() / np.sum(x)
            
            t = 0
            while True:
                self._new_dual_variables(C, p, q, xk)
                x = self._new_x(C, p, q, xk)
                     
                t += 1  
                T += 1
                
                self.phi = self._new_phi(C, p, q, xk)
                self.f = self._new_f(C, x, xk) 
                    
                c = 1 / (2 * self.n) * (np.sum(self.my) - np.sum(self.lambda_))
                self.lambda_ += c
                self.my -= c
                
                self.epsilon_ = self.epsilon / np.sqrt(np.linalg.norm(self.lambda_, 2) ** 2 + np.linalg.norm(self.my, 2) ** 2)
                if self.log:
                    print("Inner iteration " + str(t) + ":", "metric (one) = " + str(round((((p - x.sum(1))**2).sum() + ((q - x.sum(0))**2).sum())**(1/2), 6)), "> " + str(self.epsilon_), "or metric (two) = " + str(round(self.f - self.phi, 6)), "> " + str(self.epsilon))
                    
                if (((p - x.sum(1))**2).sum() + ((q - x.sum(0))**2).sum())**(1/2) < self.epsilon_ and self.f - self.phi < self.epsilon:
                    if self.log:
                        print("Inner iteration " + str(t) + ":", "metric (one) = " + str(round((((p - x.sum(1))**2).sum() + ((q - x.sum(0))**2).sum())**(1/2), 6)), "< " + str(self.epsilon_), "and metric (two) = " + str(round(self.f - self.phi, 6)), "< " + str(self.epsilon))
                    break
                
            if not with_prox:
                return x, T, k
            
            if self.log:
                print("– Outer iteration " + str(k) + ":", "metric = " + str(round(np.linalg.norm(x - xk, 2), 4)), "> " + str(self.epsilon_prox))
            
            k += 1
            if np.linalg.norm(x - xk, 2) < self.epsilon_prox:
                if self.log:
                    print("– Outer iteration " + str(k) + ":", "metric = " + str(round(np.linalg.norm(x - xk, 2), 4)), "< " + str(self.epsilon_prox))
                return x, T, k

