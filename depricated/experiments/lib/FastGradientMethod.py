import numpy as np

class FastGradientMethod:
    def __init__(self, gamma, epsilon, n, log=False):
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.inf = 1e+6
        self.l = 1 / gamma

        self.a = self.alpha = 0
        
        self.u_lambda = self.u_mu = np.ones([n])
        self.y_lambda = self.y_mu = np.zeros(n)
        self.x_lambda = self.x_mu = np.ones([n])
        
        self.x_0 = np.ones([n, n]) / (n**2)
        
        self.log = log
        if self.log:
            print("–––––––––––––––––––––––––––––")
            print("Algorithm configuration:")
            print("gamma = " + str(gamma))
            print("eps = " + str(epsilon))
            print("–––––––––––––––––––––––––––––\n")

    
    def x_hat(self, c, lambda_y, mu_y):
        n = self.n
        x_hat = self.x_0 * np.exp(
            -(self.gamma + c + lambda_y.repeat(n).reshape(-1, n) + mu_y.repeat(n).reshape(-1, n).T) / self.gamma
        )
        return x_hat / (x_hat.sum() + 1e-16)
    
    def _new_alpha(self):
        return 1 / (2 * self.l) + np.sqrt(1 / (4 * (self.l**2)) + self.alpha**2)
    
    def _new_a(self, new_alpha):
        return self.a + new_alpha
    
    def _new_y(self, u_lambda, u_mu, x_lambda, x_mu):
        new_alpha = self._new_alpha()
        new_a = self._new_a(new_alpha)
        
        return (new_alpha * u_lambda + self.a * x_lambda) / new_a,\
               (new_alpha * u_mu     + self.a * x_mu)     / new_a
    
    def _new_u(self, c, p, q, u_lambda, u_mu, y_lambda_new, y_mu_new):
        x_hat = self.x_hat(c, y_lambda_new, y_mu_new)
        new_alpha = self._new_alpha()
        
        return u_lambda - new_alpha * (p - x_hat.sum(1)),\
               u_mu     - new_alpha * (q - x_hat.sum(0))
    
    def _new_x(self, u_lambda_new, u_mu_new, x_lambda, x_mu):
        new_alpha = self._new_alpha()
        new_a = self._new_a(new_alpha)
        
        return (new_alpha * u_lambda_new + self.a * x_lambda) / new_a,\
               (new_alpha * u_mu_new     + self.a * x_mu)     / new_a
    
    def f(self, c, x):
        return (c * x).sum() + self.gamma * (x * np.log(x / self.x_0 + 1e-16)).sum()
    
    def phi(self, c, p, q, lambda_x, mu_x):
        a = np.min(self.gamma + c + lambda_x.repeat(self.n).reshape(-1, self.n) + mu_x.repeat(self.n).reshape(-1, self.n).T)
        return -(lambda_x * p).sum() - (mu_x * q).sum() - \
                self.gamma * (-a + np.log(1/np.e * (self.x_0 * np.exp(
                    -((self.gamma + c + lambda_x.repeat(self.n).reshape(-1, self.n) + mu_x.repeat(self.n).reshape(-1, self.n).T) - a) / self.gamma
                )).sum()))
    
    def deviation_p_q(self, x, p, q):
        return np.sqrt(np.sum((x.sum(1) - p)**2) + np.sum((x.sum(0) - q)**2))
    
    def fit(self, c, p, q):
        k = 0
        
        while True:
            y_lambda_new, y_mu_new = self._new_y(self.u_lambda, self.u_mu, self.x_lambda, self.x_mu)
            u_lambda_new, u_mu_new = self._new_u(c, p, q, self.u_lambda, self.u_mu, y_lambda_new, y_mu_new)
            x_lambda_new, x_mu_new = self._new_x(u_lambda_new, u_mu_new, self.x_lambda, self.x_mu)
            
            self.y_lambda, self.y_mu = (y_lambda_new, y_mu_new)
            self.u_lambda, self.u_mu = (u_lambda_new, u_mu_new)
            self.x_lambda, self.x_mu = (x_lambda_new, x_mu_new)
            
            self.alpha = self._new_alpha()
            self.a     = self._new_a(self.alpha)
            
            x_wave = self.x_hat(c, self.y_lambda, self.y_mu)
            r = np.sqrt((self.x_lambda**2).sum() + (self.x_mu**2).sum())
            epsilon_wave = self.epsilon / r
            
            
            criteria_a = self.deviation_p_q(x_wave, p, q) <= epsilon_wave
            criteria_b = self.f(c, x_wave) - self.phi(c, p, q, self.x_lambda, self.x_mu) <= self.epsilon
            
            if self.log and k % 100 == 0:
                print(f'iteration {k}:   criteria 1 = {round(self.deviation_p_q(x_wave, p, q), 7)}, ' + \
                                       f'criteria 2 = {round(self.f(c, x_wave) + self.phi(c, p, q, self.x_lambda, self.x_mu), 7)}')
            
            if criteria_a and criteria_b:
                return x_wave, k
            
            k += 1