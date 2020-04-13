import numpy as np
import matplotlib.pyplot as plt

import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from MultidimensionalOptimization import MultidimensionalOptimization

a, b, c, d = 11, -0.4, 1, 0.21

func = lambda x: (x[0] - a) ** 2 + (x[1] - b) ** 2 + np.exp(c * x[0] ** 2 + d * x[1])

if __name__ == "__main__":
    x0 = np.array([-1, 0], dtype=np.float)
    multidimensional_optimization = MultidimensionalOptimization(func=func, x0=x0)

    print(multidimensional_optimization.Hook_Jeeves_method(0.1, 0.0001))
    print(multidimensional_optimization.Nelder_method(0.0001))

    x, y = np.arange(-1, 0, 0.1), np.arange(-1, 0, 0.1)

    z = func(np.array([x, y]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()