import numpy as np

# nesterovs=True works bad
class LinearCouplingMethod:
    def __init__(self, gamma, epsilon, n, nesterovs=True, log=False):
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.inf = 1e+6
        self.lambda_x_new = self.lambda_u = np.ones(n)
        self.mu_x_new     = self.mu_u     = np.ones(n)
        self.A = 0
        self.x_sum = 0
        self.x_0 = 1 / n**2
        
        self.nesterovs = nesterovs
        if self.nesterovs:
            print("Method Nesterov's linear coupling now works not very good. We recommend not to use it now.")
        self.log = log
        if self.log:
            print("–––––––––––––––––––––––––––––")
            print("Algorithm configuration:")
            print("gamma = " + str(gamma))
            print("eps = " + str(epsilon))
            print("–––––––––––––––––––––––––––––\n")
    
    def f(self, x):
        return (self.c * x).sum() + self.gamma * ((x + 1e-20) * np.log((x + 1e-20) / self.x_0)).sum()
    
    def phi(self, lambda_x, mu_x):
        a_min = np.min(self.gamma + self.c + lambda_x.repeat(self.n).reshape(-1, self.n) + mu_x.repeat(self.n).reshape(-1, self.n).T)
        exp_ = -(self.gamma + self.c + lambda_x.repeat(self.n).reshape(-1, self.n) + mu_x.repeat(self.n).reshape(-1, self.n).T - a_min) / self.gamma
        exp_[exp_ < -100] = -100
        return (lambda_x * self.p).sum() + (mu_x * self.q).sum() + \
                self.gamma * np.log(1/np.e * (self.x_0 * np.exp(
                    exp_
                )).sum())
        
    @staticmethod
    def argmin(func, epsilon, var_range):
        def find_upper_bound(cap):
                l, r = 0, 2**(-2)
                while func(l) >= func(r) and r <= 1:
                    l = r
                    r *= 2
                return r
        l, r = np.max(0, var_range[0]), find_upper_bound(var_range[1])
        phi = (1 + np.sqrt(5)) / 2
        x1 = r - (r - l) / phi
        x2 = l + (r - l) / phi

        while r - l > epsilon:
            if func(x1) < func(x2):
                r = x2
                x1, x2 = r - (r - l) / phi, x1
            else:
                l = x1
                x1, x2 = x2, l + (r - l) / phi
        return r
    
    def x_hat(self, lambda_x, mu_x):
        a_min = np.min(self.gamma + self.c + lambda_x.repeat(self.n).reshape(-1, self.n) + mu_x.repeat(self.n).reshape(-1, self.n).T)
        exp_ = -(self.gamma + self.c + lambda_x.repeat(self.n).reshape(-1, self.n) + mu_x.repeat(self.n).reshape(-1, self.n).T - a_min) / self.gamma
        exp_[exp_ < -100] = -100
        x_hat = self.x_0 * np.exp(
            exp_
        )
        return x_hat / x_hat.sum()
    
    def x_update(self):
        self.lambda_x = self.lambda_x_new
        self.mu_x     = self.mu_x_new
        
    def _new_beta(self, k=None):
        if k is not None:
            return k / (k + 2)
        return LinearCouplingMethod.argmin(lambda beta: self.phi(
            self.lambda_u + beta * (self.lambda_x - self.lambda_u),
            self.mu_u     + beta * (self.mu_x     - self.mu_u)
        ), 1e-4, [0, 1])
    
    def _new_y(self):
        return self.lambda_u + self.beta * (self.lambda_x - self.lambda_u), \
               self.mu_u     + self.beta * (self.mu_x     - self.mu_u)
        
    def grad_phi(self, lambda_y, mu_y):
        return np.array([
            self.p - self.x_hat(lambda_y, mu_y).sum(1), self.q - self.x_hat(lambda_y, mu_y).sum(0)
        ])
    
    def _new_h(self, L=None):
        if L is not None:
            return 1 / L
        return LinearCouplingMethod.argmin(lambda h: self.phi(
            self.lambda_x - h * self.grad_phi(self.lambda_y, self.mu_y)[0],
            self.mu_x     - h * self.grad_phi(self.lambda_y, self.mu_y)[1]
        ), 1e-4, [0, self.inf])
    
    def _new_x(self):
        return self.lambda_y - self.h * self.grad_phi(self.lambda_y, self.mu_y)[0], \
               self.mu_y     - self.h * self.grad_phi(self.lambda_y, self.mu_y)[1]
    
    def correct_lambda_mu(self):
        c = (np.linalg.norm(self.mu_x, 1) - np.linalg.norm(self.lambda_x, 1)) * 1/(2*self.n)
        self.lambda_x += c
        self.mu_x -= c
    
    def deviation_p_q(self, x):
        return np.sqrt(np.sum((x.sum(1) - self.p)**2) + np.sum((x.sum(0) - self.q)**2))
    
    def _new_alpha(self, k=None, L=None):
        if k is not None and L is not None:
            return (k + 2) / (2 * L)
        delta_phi = self.phi(self.lambda_y, self.mu_y) - self.phi(self.lambda_x, self.mu_x)
        D = max(delta_phi * (delta_phi - 2 * self.A * self.deviation_p_q(self.x_hat(self.lambda_y, self.mu_y))**2), 0)
        return (-delta_phi + np.sqrt(D)) / self.deviation_p_q(self.x_hat(self.lambda_y, self.mu_y))**2
    
    def _new_u(self):
        return self.lambda_u - self.alpha * self.grad_phi(self.lambda_y, self.mu_y)[0], \
               self.mu_u     - self.alpha * self.grad_phi(self.lambda_y, self.mu_y)[1]
    
    def _new_A(self):
        return self.A + self.alpha
    
    def x_sum_update(self):
        self.x_sum += self.alpha * self.x_hat(self.lambda_y, self.mu_y)
        
    def x_wave_(self):
        return self.x_sum * 1/(self.A + 1e-16)
    
    def fit(self, c, p, q):
        self.c, self.p, self.q = c, p, q
        
        k = 1
        while True:
            self.x_update()
            
            if self.nesterovs:
                self.beta = self._new_beta()
            else:
                self.beta = self._new_beta(k = k)
            self.lambda_y, self.mu_y = self._new_y()
            
            if self.nesterovs:
                self.h = self._new_h()
            else:
                self.h = self._new_h(L = 1/self.gamma)
            self.lambda_x_new, self.mu_x_new = self._new_x()
            self.correct_lambda_mu()
            
            if self.nesterovs:
                self.alpha = self._new_alpha()
            else:
                self.alpha = self._new_alpha(k = k, L = 1/self.gamma)
            self.x_sum_update()
            self.lambda_u, self.mu_u = self._new_u()
            
            self.A = self._new_A()
            self.x_wave = self.x_wave_()
            x_hat = self.x_hat(self.lambda_x, self.mu_x)
            
            R = np.sqrt(np.linalg.norm(self.lambda_x) + np.linalg.norm(self.mu_x))
            epsilon_wave = self.epsilon / R
            
            # criteria_a = self.deviation_p_q(self.x_wave) < epsilon_wave
            criteria_a = self.deviation_p_q(x_hat) < epsilon_wave
            criteria_b = self.f(self.x_wave) + self.phi(self.lambda_x_new, self.mu_x_new) < self.epsilon
            
            if self.log and k % 100 == 0:
                print(f'iteration {k}:   criteria 1 = {round(self.deviation_p_q(self.x_wave), 7)}, ' + \
                                     f'criteria 2 = {round(self.f(self.x_wave) + self.phi(self.lambda_x_new, self.mu_x_new), 7)}')
            
            if criteria_a and criteria_b:
                return self.x_wave, k
            
            k += 1