import numpy as np
import matplotlib.pyplot as plt

import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from MultidimensionalOptimization import MultidimensionalOptimization

a, b, c, d = 3, -1.2, 0.02, 1.3

func = lambda x: (x[0] - a) ** 2 + (x[1] - b) ** 2 + np.exp(c * x[0] ** 2 + d * x[1])

if __name__ == "__main__":
    x0 = np.array([-1, 0], dtype=np.float)
    multidimensional_optimization = MultidimensionalOptimization(func=func, x0=x0)

    # print(func(np.array([-0.1, 0])))

    point1 = multidimensional_optimization.Hook_Jeeves_method(0.1, 0.0001)
    print("Точки: " + str(point1))
    print("Значение функции: " + str(func(point1)))

    print(multidimensional_optimization.Simplex_method(0.0001))
    print(multidimensional_optimization.Nelder_method(0.0001))
    print(multidimensional_optimization.gradient_method_const_step())
    print(multidimensional_optimization.gradient_method_crushing_step())
    print(multidimensional_optimization.gradient_method_steepest_descent())
    print(multidimensional_optimization.coordinate_descent_const_step())
    print(multidimensional_optimization.Gauss_Seidel_algorithm())
    print(multidimensional_optimization.ravine_method())
    print(multidimensional_optimization.Gelfand_method4())

    # x, y = np.arange(-1, 0, 0.1), np.arange(-1, 0, 0.1)

    # z = func(np.array([x, y]))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot(x, y, z)

    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()
