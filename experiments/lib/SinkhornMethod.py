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
        self.lambda_ = np.ones(n)
        self.my = np.ones(n)
        
        # constants
        self.gamma = gamma
        self.n = n
        self.epsilon = epsilon
        self.epsilon_ = round(epsilon / np.sqrt(np.linalg.norm(self.lambda_, 2) ** 2 + np.linalg.norm(self.my, 2) ** 2), 4)
        self.epsilon_prox = epsilon_prox
        
        self.log = log
        if self.log:
            print("–––––––––––––––––––––––––––––")
            print("Algorithm configuration:")
            print("gamma = " + str(gamma))
            print("eps = " + str(epsilon))
            print("eps with ~ = " + str(self.epsilon_))
            print("eps prox = " + str(self.epsilon_prox))
            print("–––––––––––––––––––––––––––––\n")
    
    def _new_dual_variables(self, C, p, q, xk):
        """
        Calculates Lagrange equation variables
        """
        for i in range(self.n):
            self.lambda_[i] = self.gamma * np.log(1/(p[i] + 1e-6) * np.sum([xk[i, j] * np.exp(-(self.gamma + C[i, j] + self.my[j])/self.gamma) for j in range(self.n)]) + 1e-5)
        
        for j in range(self.n):
            self.my[j] = self.gamma * np.log(1/(q[j] + 1e-6) * np.sum([xk[i, j] * np.exp(-(self.gamma + C[i, j] + self.lambda_[i])/self.gamma) for i in range(self.n)]) + 1e-5)
    
    def _new_x(self, C, p, q, xk):
        x = np.zeros((self.n, self.n))
        
        for i in range(self.n):
            for j in range(self.n):
                x[i, j] = xk[i, j] * np.exp(-(self.gamma + C[i, j] + self.lambda_[i] + self.my[j])/self.gamma)
        return x
    
    def _new_phi(self, C, p, q, xk):
        x_sum = 0
        
        for i in range(self.n):
            for j in range(self.n):
                x_sum += xk[i, j] * np.exp(-(C[i,j] + self.lambda_[i] + self.my[j] + self.gamma) / self.gamma)
                
        return - np.sum(self.lambda_ * p) - np.sum(self.my * q) - self.gamma * x_sum
    
    def _new_f(self, C, x, xk):
        return np.sum(C * x) + self.gamma * np.sum(x * np.log(x / xk))
    
    def fit(self, C, p, q):
        T = 0
        k = 0
        x = 1/self.n**2 * np.ones((self.n, self.n))
        while True:
            xk = x.copy()
            
            t = 0
            while True:
                self._new_dual_variables(C, p, q, xk)
                x = self._new_x(C, p, q, xk)
                     
                t += 1  
                T += 1
                if T % 20 == 0:
                    self.phi = self._new_phi(C, p, q, xk)
                    self.f = self._new_f(C, x, xk)                 
                    self.epsilon_ = round(self.epsilon / np.sqrt(np.linalg.norm(self.lambda_, 2) ** 2 + np.linalg.norm(self.my, 2) ** 2), 4)
                    if self.log:
                        print("Inner iteration " + str(t) + ":", "metric (one) = " + str(round((((p - x.sum(1))**2).sum() + ((q - x.sum(0))**2).sum())**(1/2), 4)), "> " + str(self.epsilon), "or metric (two) = " + str(round(self.f - self.phi, 4)), "> " + str(self.epsilon_))
                    if (((p - x.sum(1))**2).sum() + ((q - x.sum(0))**2).sum())**(1/2) < self.epsilon and self.f - self.phi < self.epsilon_:
                        if self.log:
                            print("Inner iteration " + str(t) + ":", "metric (one) = " + str(round((((p - x.sum(1))**2).sum() + ((q - x.sum(0))**2).sum())**(1/2), 4)), "< " + str(self.epsilon), "and metric (two) = " + str(round(self.f - self.phi, 4)), "< " + str(self.epsilon_))
                        break
                
            
            if self.log and k % 1 == 0:
                print("– Outer iteration " + str(k) + ":", "metric = " + str(round(np.linalg.norm(x - xk, 2), 4)), "> " + str(self.epsilon_prox))
            
            k += 1
            if np.linalg.norm(x - xk, 2) < self.epsilon_prox:
                if self.log:
                    print("– Outer iteration " + str(k) + ":", "metric = " + str(round(np.linalg.norm(x - xk, 2), 4)), "< " + str(self.epsilon_prox))
                return x, T, k

