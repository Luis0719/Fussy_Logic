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


# Set up a figure twice as tall as it is wide
fig = plt.figure(figsize=plt.figaspect(.4))
fig.suptitle('A tale of 2 subplots')

# First subplot
ax = fig.add_subplot(1, 2, 1)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
t3 = np.arange(0.0, 2.0, 0.01)

ax.plot(t1, f(t1), 'bo',
        t2, f(t2), 'k--', markerfacecolor='green')
ax.grid(True)
ax.set_ylabel('Damped oscillation')

# Second subplot
ax = fig.add_subplot(1, 2, 2, projection='3d')

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1, 1)

plt.show()