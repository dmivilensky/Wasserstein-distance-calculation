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

def sample_img_batch(n, max_int=10):
    """Sample image-based batch for trasport problem 
    
    Arguments:
        n {int} -- NxX image size 
    
    Keyword Arguments:
        max_int {int} -- maximum color saturation (default: {10})
    """
    img1, img2 = np.random.randint(1, high=(max_int+1), size=(2, n, n))
    C = np.zeros((n ** 2, n ** 2))
    for i in range(n ** 2):
        for j in range(n ** 2):
            C[i, j] = norm(np.array([i // n, i % n]) - np.array([j // n, j % n]), 2)
    p = img1.reshape((n ** 2, )) / np.sum(img1)
    q = img2.reshape((n ** 2, )) / np.sum(img2)
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
    