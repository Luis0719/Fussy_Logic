# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D

# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# print("X")
# print(X)
# print("Y")
# print(Y)

# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# print("R")
# print(R)
# Z = np.sin(R)

# print("X")
# print(X)
# print("Y")
# print(Y)
# print("Z")
# print(Z)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)

# plt.show()

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return np.cos(2*np.pi*t) * np.exp(-t)

X = np.arange(-5, 5, 1)
Y = np.arange(-5, 5, 1)
X, Y = np.meshgrid(X, Y)

# Set up a figure twice as tall as it is wide
fig = plt.figure(figsize=plt.figaspect(0.3))
fig.suptitle('A tale of 2 subplots')

# Second subplot
ax = fig.add_subplot(1, 3, 1, projection='3d')
Z = np.sin(X + Y)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       linewidth=0, antialiased=False)

ax = fig.add_subplot(1, 3, 2, projection='3d')
Z = np.sqrt(X + Y)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       linewidth=0, antialiased=False)

ax = fig.add_subplot(1, 3, 3, projection='3d')
Z = np.power(X + Y, 2)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       linewidth=0, antialiased=False)

plt.show()