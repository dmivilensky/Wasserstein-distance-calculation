import numpy as np
import matplotlib.pyplot as plt

class FastGradientMethod:
    def __init__(self, gamma, n=5, epsilon=0.001, log=False):
        self.alpha = 0
        self.a     = 0
        
        # constants
        self.epsilon = epsilon
        self.gamma = gamma
        self.l     = 1 / gamma
        self.n     = n
        
        self.u_lambda = np.ones(n) / n**2
        self.u_mu     = np.ones(n) / n**2
         
        self.y_lambda = np.zeros(n)
        self.y_mu     = np.zeros(n)
        
        self.x_lambda   = np.zeros([n])
        self.x_mu       = np.zeros([n])
        
        self.x_0 = np.ones([n, n]) / (n**2)
        
        self.log = log
        if self.log:
            print("–––––––––––––––––––––––––––––")
            print("Algorithm configuration:")
            print("gamma = " + str(gamma))
            print("eps = " + str(epsilon))
            print("–––––––––––––––––––––––––––––\n")
        
    def __get_x(self, c):
        a = self.x_0 * np.exp(-(self.gamma + c + self.x_lambda.repeat(self.n).reshape(-1, self.n) +\
                                self.x_mu.repeat(self.n).reshape(-1, self.n).T))
        
        return a / a.sum()
    
    def __new_alpha(self):
        return 1 / (2 * self.l) + np.sqrt(1 / (4 * (self.l**2)) + self.alpha**2)
    
    def __new_a(self):
        return self.a + self.__new_alpha()
    
    def __new_y(self):
        return (self.__new_alpha() * self.u_lambda + self.a * self.x_lambda) / self.__new_a(),\
               (self.__new_alpha() * self.u_mu + self.a * self.x_mu) / self.__new_a()
        
        
    def __new_u(self, c, p, q):
        x_hat = self.__get_x(c)
        
        return self.u_lambda - self.alpha * (p - x_hat.sum(1)),\
               self.u_mu - self.alpha * (q - x_hat.sum(0))
    
    def __new_x(self):
        return (self.alpha * self.u_lambda + self.a * self.x_lambda) / self.__new_a(),\
                (self.alpha * self.u_mu + self.a * self.x_mu) / self.__new_a()
    
    def fit(self, c, p, q):
        k = 0
        while True:
            k+=1
            self.y_lambda, self.y_mu = self.__new_y()
            self.u_lambda, self.u_mu = self.__new_u(c, p, q)
            self.x_lambda, self.x_mu = self.__new_x()
            
            self.alpha = self.__new_alpha()
            self.a     = self.__new_a()
            
            x_hat = self.__get_x(c)
            
            if self.log and k % 100 == 0:
                print("Iteration " + str(k) + ":", "metric = " + str(round((((p - x_hat.sum(1))**2).sum() + ((q - x_hat.sum(0))**2).sum())**(1/2), 4)), "> " + str(self.epsilon))
                
            if (((p - x_hat.sum(1))**2).sum() + ((q - x_hat.sum(0))**2).sum())**(1/2) < self.epsilon: 
                if self.log:
                    print("Iteration " + str(k) + ":", "metric = " + str(round((((p - x_hat.sum(1))**2).sum() + ((q - x_hat.sum(0))**2).sum())**(1/2), 4)), "< " + str(self.epsilon))
                return self.__get_x(c), k