import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from multiprocessing.pool import ThreadPool


def task_grandient(x):
    return np.array([(i + 1) * 2 * x[i] for i in range(len(x))])


def task(x):
    return sum([(i + 1) * x[i] ** 2 for i in range(len(x))])


def nesterov_fastgrad(x, eps):
    k = 0
    last_x = x
    while True:
        k += 1

        argument = (x + ((k - 1)/(k + 2)) * (x - last_x))
        grad = task_grandient(argument)
        new_x = x - (1/L) * grad + ((k - 1)/(k + 2)) * (x - last_x)

        last_x = copy.copy(x)
        x = copy.copy(new_x)
        if task(x) < eps:
        # if sum([(i + 1) * argument[i]**2 for i in range(n)]) < eps:
            return x, k


def kim_fessler_gradient(x, eps):
    t = 0
    teta = 0
    k = 0
    x0 = copy.copy(x)
    x_sum = np.zeros(len(x))
    while True:
        k += 1
        last_t = t
        grad = task_grandient(x)
        x_sum += last_t * grad

        teta = (1 + math.sqrt(8 * teta ** 2 + 1))/2

        _t = teta
        _y = (1 - 1/_t) * x + (1/_t) * x0
        _d = (1 - 1/_t) * grad + (2/_t) * x_sum
        _x = _y - 1/L * _d

        if task(_x) < eps:
            return x, k

        t = (1 + np.sqrt(4 * t ** 2 + 1))/2
        y = (1 - 1/t) * x + (1/t) * x0
        d = (1 - 1/t) * grad + (2/t) * x_sum
        x = y - 1/L * d


def nesterov_triangle_method(x, eps):
    k = 0
    a = 0
    A = 0
    u = copy.copy(x)
    while True:
        k += 1

        a = 1 / (2 * L) + math.sqrt(1/(4 * L**2) + a**2)
        last_A = A
        A += a

        y = (a * u + last_A * x) / A
        grad = task_grandient(y)
        u = u - a * grad

        x = (a * u + last_A * x) / A

        if task(x) < eps:
            return x, k


n = 30
L = 2 * n
x0 = np.array([i for i in range(n, 0, -1)], dtype="float64")
results = [{}, {}, {}]
dots = np.linspace(0.9, 0.00001, 1000)
pool = ThreadPool(processes=3)

for i in tqdm.tqdm(range(len(dots))):
    async_results = [pool.apply_async(nesterov_fastgrad, (x0, dots[i])),
                     pool.apply_async(kim_fessler_gradient, (x0, dots[i])),
                     pool.apply_async(nesterov_triangle_method, (x0, dots[i]))]
    for j in range(3):
        # results[j].append((math.log(1/dots[i]), math.log(async_results[j].get()[1])))
        # results[j].append((dots[i], async_results[j].get()[1]))
        results[j][math.log(1/dots[i])] = math.log(async_results[j].get()[1])

print(results)


plt.title('Сравнение эффективности методов минимизации')
plt.ylabel('$\ln_{}(N) $')
plt.xlabel('$\ln_{}(\\frac{1}{E})$')
plt.grid(color='b', linestyle=':', linewidth=0.5)
# plt.xscale('log'), plt.xticks(fontsize=15)
# plt.yscale('log'), plt.yticks(fontsize=15)

# plt.xticks(np.linspace(math.log(1/dots[0]), math.log(1/dots[-1]), 10))
plt.plot(results[0].keys(), results[0].values(), color="blue", label="Nesterov 1983")
plt.plot(results[1].keys(), results[1].values(), color="green", label="Kim-Fessler 2016")
plt.plot(results[2].keys(), results[2].values(), color="orange", label="Nesterov 2016")
plt.legend()

plt.show()
