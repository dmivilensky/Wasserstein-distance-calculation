import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

class Experiments:
    @staticmethod
    def load_image(filename):
        basewidth = 10
        img = Image.open(filename)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        return np.asarray(img, dtype="int32")

    @staticmethod
    def load_data(img1_path='1.png', img2_path='2.png'):
        img1 = Experiments.load_image(img1_path)
        img2 = Experiments.load_image(img2_path)

        C = np.zeros((img1.shape[0] * img1.shape[1], img1.shape[0] * img1.shape[1]))

        for i in range(img1.shape[0] * img1.shape[1]):
            for j in range(img1.shape[0] * img1.shape[1]):
                C[i, j] = np.linalg.norm(np.array([i // img1.shape[0], i % img1.shape[1]]) - np.array([j // img1.shape[0], j % img1.shape[1]]), 2)

        img1 += 1
        img2 += 1
        
        p = img1.reshape((img1.shape[0] * img1.shape[1], )) / np.sum(img1)
        q = img2.reshape((img2.shape[0] * img2.shape[1], )) / np.sum(img2)

        plt.figure()
        plt.subplot(121)
        plt.imshow(img1, cmap='gray')
        plt.subplot(122)
        plt.imshow(img2, cmap='gray')
        plt.show()
        
        return C, p, q
    
    @staticmethod
    def visualize_x(x, n):
        plt.imshow(x.reshape(int(math.sqrt(n)), int(math.sqrt(n)), n).sum(2))
    
    @staticmethod
    def test(C, p, q, config={'eps': (0.01, 0.01, 1), 'gamma': (10, 0.01, 2)}, methods=[]):
        epsilons = [config['eps'][0] / config['eps'][2]**i for i in range(int(math.log(config['eps'][0] / config['eps'][1], config['eps'][2] + 1e-5)) + 1)]
        gammas = [config['gamma'][0] / config['gamma'][2]**i for i in range(int(math.log(config['gamma'][0] / config['gamma'][1], config['gamma'][2] + 1e-5)) + 1)]

        iterations = [[] for i in range(len(methods))]

        with tqdm(total=len(epsilons) * len(gammas) * len(methods)) as ph:
            for eps in epsilons:
                for gamma in gammas:
                    for i in range(len(methods)):
                        x, iterations_num, _ = methods[i](C, p, q, gamma, eps)
                        iterations[i].append(iterations_num)
                        ph.update(1)
            
        return epsilons, gammas, np.array(iterations).reshape((len(epsilons), len(methods), len(gammas)))