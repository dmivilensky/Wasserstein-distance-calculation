import numpy as np

# Convergence tests for regularized optimization methods

def sample_batch(n, size=5):
    """Sample batch for trasport problem 
    
    Arguments:
        n {int} -- NxX matrix size 
    
    Keyword Arguments:
        size {int} -- number of experiments sampled (default: {5})
    """
    C = np.random.uniform(0, 10, size=[n, n])
    p = np.random.dirichlet(np.ones(5), size=1).ravel()
    q = np.random.dirichlet(np.ones(5), size=1).ravel()
    return C, p, q

def convergence_test(method, gammas, n=5, n_exp=5):
    """Convergence tests for regularized optimization methods
    
    Arguments:
        gammas iterable of float -- regularization constants
    
    Keyword Arguments:
        n matrix size -- NxX matrix size
        n_exp {int} -- number of experiments sampled per gamma (default: {5})
    """
    n_iterations = list()
    for gamma in gammas:
        for c, p, q in sample_batch(n, n_exp):
            n_iterations.append(method(c, p, q, gamma))

    return np.array(n_iterations).reshape(-1, n_exp).sum(-1)
    