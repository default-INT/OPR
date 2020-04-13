import numpy as np


class Vector(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "({0}, {1})".format(self.x, self.y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Vector(x, y)

    def __rmul__(self, other):
        x = self.x * other
        y = self.y * other
        return Vector(x, y)

    def __truediv__(self, other):
        x = self.x / other
        y = self.y / other
        return Vector(x, y)

    def c(self):
        return self.x, self.y


class MultidimensionalOptimization:

    def __init__(self, func, x0) -> None:
        super().__init__()
        self.func = func
        self.x0 = x0

    @staticmethod
    def _qu_point(point1, point2):
        """

        :param point1: {np.array}
        :param point2: {np.array}
        :return:
        """
        qu_res = point1 == point2
        result = True
        for i in qu_res:
            result &= i
        return result

    def _find_max_point(self, _points):
        max_point = _points[0]
        index_point = 0
        max_func_val = self.func(max_point)
        for _i in range(len(_points)):
            if self.func(_points[_i]) > max_func_val:
                index_point = _i
                max_point = _points[_i]
                max_func_val = self.func(max_point)
        return max_point, index_point

    def _find_min_point(self, _points):
        min_point = _points[0]
        index_point = 0
        min_func_val = self.func(min_point)
        for _i in range(len(_points)):
            if self.func(_points[_i]) < min_func_val:
                index_point = _i
                min_point = _points[_i]
                min_func_val = self.func(min_point)
        return min_point, index_point

    def _get_center_w(self, points, max_point):
        c = 0
        len_points = len(points)
        for i in range(len_points):
            if not self._qu_point(c, max_point):
                c += points[i]
        c /= len_points
        return c

    def Nelder_method(self, eps=0.001, alpha=1, beta=2, gamma=0.5, maxiter=100):

        points = np.zeros(shape=(self.x0.shape[0] + 1, 2))
        points[0] = self.x0.copy()

        func_values = np.zeros(shape=(self.x0.shape[0] + 1, 2))
        func_values[0] = self.func(self.x0)

        for i in range(1, len(self.x0)):
            ones_vector = np.ones(shape=self.x0.shape)
            ones_vector[i] = 1

            points[i] = self.x0 + alpha * ones_vector
            func_values[i] = self.func(points[i])

        for _ in range(maxiter):
            old_points = points
            max_point, max_index = self._find_max_point(points)
            min_point, min_index = self._find_min_point(points)
            center_w = self._get_center_w(points, max_point)

            point_new = center_w + alpha * (center_w - max_point)

            if self.func(point_new) < self.func(min_point):
                point_expansion = center_w + beta * (point_new - center_w)    # растягиваем

                if self.func(point_expansion) < self.func(point_new):
                    points[max_index] = point_expansion
                else:
                    points[max_index] = point_new

            elif self.func(point_new) > self.func(min_point):
                point_contract = None
                if self.func(max_point) <= self.func(point_new):
                    point_contract = center_w + gamma * (max_point - center_w)
                else:
                    point_contract = center_w + gamma * (point_new - center_w)
                if self.func(point_contract) < self.func(point_new):
                    points[max_index] = point_contract
                elif self.func(point_contract) < self.func(max_point):
                    points[max_index] = point_new
                else:
                    points = (points + min_point * np.ones(shape=(1, len(min_point)))) / 2
            # if np.sqrt(np.sum((self.func(points) - self.func(old_points)) ** 2) / len(points)) <= eps:
            #     return self._find_min_point(points)[0]
        return self._find_min_point(points)[0]

    def Hook_Jeeves_method(self, h, delta):
        """

        :param h: начальный шаг
        :param delta: точность решения (предельное значения для шага h)
        :return:
        """

        def _exploratory_search(xprev, operation):
            x = xprev
            fprev = self.func(xprev)
            for i in range(len(x)):
                if operation == '+':
                    x[i] += h
                elif operation == '-':
                    x[i] -= h
                if self.func(x) > fprev:
                    if operation == '+':
                        x[i] -= h
                    elif operation == '-':
                        x[i] += h
            return x

        k = 0
        xk = self.x0.copy()
        fxk = self.func(self.x0)
        # z = self.x0 + h * np.ones(shape=self.x0.shape)
        while h > delta:
            x = _exploratory_search(xk, '+')
            if self.func(x) == fxk:
                x = _exploratory_search(xk, '-')

            if self.func(x) != fxk:
                xk = x + (x - xk)
                fxk = self.func(xk)
            else:
                h /= 2
        return xk
