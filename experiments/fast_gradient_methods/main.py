import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from multiprocessing.pool import ThreadPool


def task_gradient(x):
    return np.array([(i + 1) * 2 * x[i] for i in range(len(x))])


def task(x):
    return sum([(i + 1) * x[i] ** 2 for i in range(len(x))])


def nesterov_fastgrad(x, eps):
    k = 0
    last_x = x
    while True:
        k += 1

        argument = (x + ((k - 1)/(k + 2)) * (x - last_x))
        grad = task_gradient(argument)
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
        grad = task_gradient(x)
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
        grad = task_gradient(y)
        u = u - a * grad

        x = (a * u + last_A * x) / A

        if task(x) < eps:
            return x, k


def find_r(y):
    x = 0
    j = 2
    px = 0
    while 1:
        x = j
        j *= 2
        if task(y - px * task_gradient(y)) <= task(y - x * task_gradient(y)):
            break
        px = x
    return x


def ternary_search_beta(eps, l, r, u, x):
    while(r - l > eps):
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        if(task(u + m1 * (x - u)) < task(u + m2 * (x - u))):
            r = m2
        else:
            l = m1
    return r


def ternary_search_h(eps, l, r, y):
    while(r - l > eps):
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        if(task(y - m1 * task_gradient(y)) < task(y - m2 * task_gradient(y))):
            r = m2
        else:
            l = m1
    return r


def linear_coupling(x, eps):
    alphprev = 0
    xres = np.zeros(n)
    xprev = x
    uprev = xprev[:]
    k = 0
#        bcur = np.array([i / (i + 2) for i in range(n)])
#        hcur = np.array([1 / L for i in range(n)])
    Acur = 0
    while(1):
        bcur = ternary_search_beta(eps, 0, 1, uprev, xprev)
        ycur = uprev + bcur * (xprev - uprev)
        r = find_r(ycur)
        hcur = ternary_search_h(eps, 0, r, ycur)
        xcur = ycur - hcur * task_gradient(ycur)
        Acur += alphprev
        alphcur = ((task(ycur) - task(xcur)) + np.sqrt((task(ycur) - task(xcur)) * ((task(ycur) - task(xcur)) + 2 * Acur * np.linalg.norm(task_gradient(ycur)) ** 2)) )/ (np.linalg.norm(task_gradient(ycur)) ** 2)
        ucur = uprev - alphcur * task_gradient(ycur)
        # Here break criteria
        if(task(xprev) - task(xres) <= eps):
          break
        xprev = xcur
        uprev = ucur
        alphprev = alphcur
        k += 1

    return xprev, k


n = 10
L = 2 * n
x0 = np.array([i for i in range(n, 0, -1)], dtype="float64")

results = [{}, {}, {}, {}]
dots = np.linspace(0.01, 0.00000001, 1000)
pool = ThreadPool(processes=len(results))

for i in tqdm.tqdm(range(len(dots))):
    async_results = [pool.apply_async(nesterov_fastgrad, (x0, dots[i])),
                     pool.apply_async(kim_fessler_gradient, (x0, dots[i])),
                     pool.apply_async(nesterov_triangle_method, (x0, dots[i])),
                     pool.apply_async(linear_coupling, (x0, dots[i]))]
    for j in range(len(async_results)):
        # results[j].append((math.log(1/dots[i]), math.log(async_results[j].get()[1])))
        # results[j].append((dots[i], async_results[j].get()[1]))
        if math.log(1/dots[i]) > 11:
            results[j][math.log(1/dots[i])] = math.log(async_results[j].get()[1])


plt.title('Сравнение эффективности методов оптимизации')
plt.ylabel('$\ln_{}(N) $')
plt.xlabel('$\ln_{}(\\frac{1}{eps})$')
plt.grid(color='b', linestyle=':', linewidth=0.5)
# plt.xscale('log'), plt.xticks(fontsize=15)
# plt.yscale('log'), plt.yticks(fontsize=15)
x = [list(i.keys()) for i in results]
y = [list(i.values()) for i in results]
angles = [round((y[i][-1] - y[i][0])/(x[i][-1] - x[i][0]), 3) for i in range(len(results))]
# plt.xticks(np.linspace(math.log(1/dots[0]), math.log(1/dots[-1]), 10))

plt.plot(x[0], y[0], color="#00693e", label="Nesterov 1983, k={}".format(angles[0]))
plt.plot(x[1], y[1], color="#d2691e", label="Kim-Fessler 2016, k={}".format(angles[1]))
plt.plot(x[2], y[2], color="#00677e", label="Nesterov 2016, k={}".format(angles[2]))
plt.plot(x[3], y[3], color="#ed4830", label="LC with LS 2018, k={}".format(angles[3]))

plt.legend()

plt.show()
