import numpy as np

class DualGradientDescent:
    def __init__(self, gamma, epsilon, n):
        self.gamma   = gamma
        self.epsilon = epsilon
        self.n       = n
        self.small     = 1e-20
        
        self.lam = np.zeros(n)
        self.mu     = np.zeros(n)
        self.x_sum  = np.zeros([n, n])
        self.x_0    = np.ones([n, n]) / (n**2)
    
    def f(self, x):
        return (self.c * x).sum() + self.gamma * ((x + self.small) * np.log((x + self.small) / self.x_0)).sum()
    
    def phi(self, lam, mu, n):
        return (lam * self.p).sum() + (mu * self.q).sum() + \
                self.gamma * np.log(1/np.e * (self.x_0 * np.exp(
                    -(self.gamma + self.c + lam.repeat(n).reshape(-1, n) + mu.repeat(n).reshape(-1, n).T) / self.gamma
                )).sum())
        
    def x_hat(self, lam, mu, n):
        x_hat = self.x_0 * np.exp(
            -(self.gamma + self.c + lam.repeat(n).reshape(-1, n) + mu.repeat(n).reshape(-1, n).T) / self.gamma
        )
        return x_hat / x_hat.sum()
    
    def _new_lm(self, p, q):
        x_hat = self.x_hat(self.lam, self.mu, self.n)
        return self.lam - self.gamma * (p - x_hat.sum(1)),\
               self.mu - self.gamma * (q - x_hat.sum(0))
        
    def x_sum_update(self):
        self.x_sum += self.x_hat(self.lam, self.mu, self.n)
        
    def _new_x_wave(self, k):
        return self.x_sum * 1/k
    
    def deviation_p_q(self, x, p, q):
        return np.sqrt(np.sum((x.sum(1) - p)**2) + np.sum((x.sum(0) - q)**2))
    
    def fit(self, c, p, q):
        self.c, self.p, self.q = c, p, q
        
        k = 1
        while True:
            self.lam, self.mu = self._new_lm(self.p, self.q)
            self.x_sum_update()
            self.x_wave = self._new_x_wave(k)
            R = np.sqrt(np.linalg.norm(self.lam) + np.linalg.norm(self.mu))
            epsilon_wave = self.epsilon / R
            
            criteria_a = self.deviation_p_q(self.x_wave, self.p, self.q) < epsilon_wave
            criteria_b = self.f(self.x_wave) + self.phi(self.lam, self.mu, self.n) < self.epsilon
            
            if k % 5000 == 0:
                print(f'iteration {k}:   criteria 1 = {round(self.deviation_p_q(self.x_wave, self.p, self.q), 7)}, ' + \
                                       f'criteria 2 = {round(self.f(self.x_wave) + self.phi(self.lam, self.mu, self.n), 7)}')
            
            if criteria_a and criteria_b:
                return self.x_wave, k
            
            k += 1