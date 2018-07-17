import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rc

gammas, epss, ts = np.load("1_gammas.npy"), np.load("1_epss.npy"), np.load("3_ts.npy")

na = 22
epss_, ts_, gammas_  = np.ones((10,na)), np.ones((10,na)), np.ones((10,na))

for i in range(10):
    for j in range(na):
        epss_[i, j], ts_[i, j], gammas_[i, j] = epss[i, j], ts[i, j], gammas[i, j] 

fig = plt.figure(figsize=(12, 9))
axes = Axes3D(fig)
xLabel = axes.set_xlabel("γ",fontsize=21)
yLabel = axes.set_ylabel("C_c", fontsize=21)
zLabel = axes.set_zlabel("N(γ, C_c)", fontsize=21)
axes.plot_surface(gammas_, epss_, ts_, cmap = cm.viridis)

angle = 0
while True:
    axes.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
    angle += 1
