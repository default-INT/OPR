import numpy as np
import matplotlib.pyplot as plt

from unconditional_optimization import FuncAnalytics

fig = plt.figure()


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


def distance_Euclid(point1, point2):
    return np.sum((point1 - point2) ** 2)


def dichotomy(f, a=0, b=1, n=100, e=1e-3, delta=1e-5):
    while b - a > 2 * e:
        x = (a + b) / 2

        y1 = f(x - delta)
        y2 = f(x + delta)
        if (y1 <= y2).any():
            b = x + delta
        else:
            a = x - delta
    return (a + b) / 2


def distance_array(points):
    length = 0
    for i in range(len(points) - 1):
        length += distance_Euclid(points[i], points[i + 1])
    return length


class MultidimensionalOptimization:

    def __init__(self, func, x0) -> None:
        super().__init__()
        self.func = func
        self.x0 = x0
        h = 0.01
        self.df = lambda x, nd: MultidimensionalOptimization.__dy(self.func, x, nd, h)
        self.gradient = lambda x: self.__gradient_calculate(x)
        self.length = lambda points: distance_array(points)
        # self.df = np.zeros()

    def __gradient_calculate(self, x):
        gradient = []
        for i in range(len(x)):
            gradient.append(self.df(x, i))
        return np.array(gradient)

    @staticmethod
    def __dy(func, x, i, h):
        ones_vector = np.zeros(shape=x.shape)
        ones_vector[i] = h

        f1 = func(x + ones_vector)
        f2 = func(x - ones_vector)
        # if i == 0:
        #     return 2 * (x[0] - 11) + 2 * x[0] * np.exp(x[0] ** 2 + 0.21 * x[1])
        # else:
        #     return 2 * (x[1] + 0.4) + 0.21 * np.exp(x[0] ** 2 + 0.21 * x[1])
        return (f1 - f2) / (2 * h)

    @staticmethod
    def _qu_point(point1, point2):
        """
        Сравнение точек point1 и point2
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
        """
        Поиск точки с максимальным значением функции.
        :param _points:
        :return:
        """
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
        """
        Поиск точки с минимальным значением функции.
        :param _points:
        :return:
        """
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
        """
        Поиск центра относительно точки max_point.
        Применяется в Simplex_method, Nelder_method
        :param points:
        :param max_point:
        :return:
        """
        c = 0
        len_points = len(points)
        for i in range(len_points):
            if not self._qu_point(points[i], max_point):
                c += points[i]
        c /= len_points - 1
        return c

    def _create_new_simplex(self, points, alpha):
        """
        Создание нового симплекса относительно базисной точки (точки с наменьшим значением функции).
        Метод использкуется в Simplex_method.
        Метод уменьшает старый симплекс в alpha раз.
        :param points:
        :param alpha:
        :return:
        """
        min_point, min_index = self._find_min_point(points)

        for i in range(len(points)):
            vector = min_point - points[i]  # определяется вектор направления для предыдущей точки

            points[i] = min_point + vector * alpha

    def Simplex_method(self, eps=0.001, alpha=2, maxiter=100):
        points = np.zeros(shape=(self.x0.shape[0] + 1, 2))
        points[0] = self.x0.copy()

        func_values = np.zeros(shape=(self.x0.shape[0] + 1, 2))
        func_values[0] = self.func(self.x0)

        for i in range(1, len(points)):
            ones_vector = np.zeros(shape=self.x0.shape)
            ones_vector[i - 1] = 1

            points[i] = self.x0 + alpha * ones_vector
            func_values[i] = self.func(points[i])
            ones_vector[i - 1] = 0

        simplex_alpha = 0.5
        plt.grid(True)
        while True:
            x, y = points[:, 0].tolist(), points[:, 1].tolist()

            x.append(points[0, 0]), y.append(points[0, 1])
            plt.plot(x, y)

            max_point, max_index = self._find_max_point(points)
            center_w = self._get_center_w(points, max_point)

            point_new = max_point + alpha * (center_w - max_point)
            new_func_value, max_func_value = self.func(point_new), self.func(max_point)

            if new_func_value < max_func_value:
                points[max_index] = point_new
            else:
                # simplex_alpha /= 2
                self._create_new_simplex(points, simplex_alpha)
            X0 = np.zeros(shape=(self.x0.shape[0], 2))
            for i in range(len(X0)):
                X0[i] = points[0]

            R = np.sqrt(np.sum((self.func(points[1:]) - self.func(X0)) ** 2) / len(points))
            if R <= eps:
                plt.show()
                return points[0]

    def Nelder_method(self, eps=0.001, alpha=1, beta=2, gamma=0.5, maxiter=100):

        points = np.zeros(shape=(self.x0.shape[0] + 1, 2))
        points[0] = self.x0.copy()

        func_values = np.zeros(shape=(self.x0.shape[0] + 1, 2))
        func_values[0] = self.func(self.x0)

        for i in range(1, len(points)):
            ones_vector = np.zeros(shape=self.x0.shape)
            ones_vector[i - 1] = 1

            points[i] = self.x0 + alpha * ones_vector
            func_values[i] = self.func(points[i])
            ones_vector[i - 1] = 0

        plt.grid(True)
        while True:
            x, y = points[:, 0].tolist(), points[:, 1].tolist()

            x.append(points[0, 0]), y.append(points[0, 1])
            plt.plot(x, y)

            max_point, max_index = self._find_max_point(points)
            min_point, min_index = self._find_min_point(points)
            center_w = self._get_center_w(points, max_point)

            point_new = center_w + alpha * (center_w - max_point)

            if self.func(point_new) < self.func(min_point):
                point_expansion = center_w + beta * (point_new - center_w)  # растягиваем

                if self.func(point_expansion) < self.func(point_new):
                    points[max_index] = point_expansion
                else:
                    points[max_index] = point_new

            elif self.func(point_new) > self.func(min_point):
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

            X0 = np.zeros(shape=(self.x0.shape[0], 2))
            for i in range(len(X0)):
                X0[i] = points[0]

            R = np.sqrt(np.sum((self.func(points[1:]) - self.func(X0)) ** 2) / len(points))
            if R <= eps:
                plt.show()
                return points[0]

        # return self._find_min_point(points)[0]

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
        x_plot = [xk]
        while h > delta:
            x = _exploratory_search(xk, '+')
            if self.func(x) == fxk:
                x = _exploratory_search(xk, '-')

            if self.func(x) != fxk:
                xk = x + (x - xk)
                fxk = self.func(xk)
            else:
                h /= 2
            x_plot.append(x)
        plt.grid(True)
        x_plot = np.array(x_plot)
        plt.plot(x_plot[:, 0], x_plot[:, 1])
        plt.show()
        return xk

    def gradient_method_const_step(self, alpha=0.01, eps=0.0001):
        x = self.x0
        xk = [x]
        while True:
            x = x - alpha * self.gradient(x)
            xk.append(x)
            if self.length(self.gradient(x)) <= eps:
                plt.grid(True)
                xk = np.array(xk)
                plt.plot(xk[:, 0], xk[:, 1])
                plt.show()
                return x

    def gradient_method_crushing_step(self, alpha=0.5, eps=0.001, delta=0.1):
        x = self.x0
        xk = [x]
        while True:
            grad = self.gradient(x)
            tmp_x = x - alpha * grad
            if (self.func(tmp_x) - self.func(x)) <= -delta * alpha * self.length(grad):
                x = tmp_x
                xk.append(x)
                if self.length(self.gradient(x)) <= eps:
                    plt.grid(True)
                    xk = np.array(xk)
                    plt.plot(xk[:, 0], xk[:, 1])
                    plt.show()
                    return x
            else:
                alpha /= 2

    def gradient_method_steepest_descent(self, eps=0.001):
        x = self.x0
        xk = [x]
        while True:
            grad = self.gradient(x)
            alpha = dichotomy(lambda a: x - a * grad)
            x = x - alpha * grad

            xk.append(x)
            if self.length(self.gradient(x)) <= eps:
                plt.grid(True)
                xk = np.array(xk)
                plt.plot(xk[:, 0], xk[:, 1])
                plt.show()
                return x

    def coordinate_descent_const_step(self, eps=0.001, alpha=0.02):
        x = self.x0
        xk = [x]
        while True:
            prev_x = x.copy()
            ones_vector = np.zeros(shape=self.x0.shape)
            for i in range(len(self.x0)):
                ones_vector[i] = 1
                x = x - alpha * self.gradient(x) * ones_vector
                ones_vector[i] = 0
                xk.append(x)
            if self.length(self.gradient(x)) <= eps:
                plt.grid(True)
                xk = np.array(xk)
                plt.plot(xk[:, 0], xk[:, 1])
                plt.show()
                return x

    def Gauss_Seidel_algorithm(self, eps=0.001):
        x = self.x0
        xk = [x]
        while True:
            ones_vector = np.zeros(shape=self.x0.shape)
            for i in range(len(self.x0)):
                ones_vector[i] = 1

                grad = self.gradient(x)
                alpha = dichotomy(lambda a: x - a * grad * ones_vector)
                x = x - alpha * grad * ones_vector

                ones_vector[i] = 0
                xk.append(x)
            if self.length(self.gradient(x)) <= eps:
                plt.grid(True)
                xk = np.array(xk)
                plt.plot(xk[:, 0], xk[:, 1])
                plt.show()
                return x

    def ravine_method(self, eps1=0.001, eps3=0.001, delta1=0.01, delta2=2, alpha1=0.001, alpha2=0.01):
        """
        alpha1 < alpha2
        delta1 << 1
        delta2 >> 1
        :param eps1:
        :param eps3:
        :param delta1:
        :param delta2:
        :param alpha1:
        :param alpha2:
        :return:
        """
        x = self.x0
        xk = [x]
        while True:
            prev_x = x.copy()
            # этап 1
            g = []
            grad = self.gradient(x)
            for dfi in grad:
                if np.fabs(dfi) > delta1:
                    g.append(dfi)
                else:
                    g.append(0)
            g = np.array(g)
            x = x - alpha1 * g
            xk.append(x)

            # этап 2
            g = []
            grad = self.gradient(x)
            for dfi in grad:
                if np.fabs(dfi) < delta2:
                    g.append(dfi)
                else:
                    g.append(0)
            g = np.array(g)
            x = x - alpha2 * g
            xk.append(x)

            if self.length(self.gradient(x)) <= eps3 and self.length(prev_x - x) <= eps1:
                plt.grid(True)
                xk = np.array(xk)
                plt.plot(xk[:, 0], xk[:, 1])
                plt.show()
                return x

    def Gelfand_method(self, eps1=0.001, eps3=0.001, alpha=0.001, lamda=0.001):
        x = self.x0
        sign = 1
        if np.random.rand() > 0.5:
            sign = -1
        _x = x + np.ones(shape=x.shape) * np.random.rand() * sign

        xk = [x]
        _xk = [_x]
        while True:
            prev_x = x.copy()
            _prev_x = _x.copy()

            x = x - alpha * self.gradient(x)
            _x = _x - alpha * self.gradient(_x)
            xk.append(x), _xk.append(_x)

            new_x = None
            if self.func(_x) < self.func(x):
                new_x = x + lamda * (_x - x) / self.length(_x - x)
            else:
                new_x = _x + lamda * (x - _x) / self.length(_x - _x)

            u = new_x - alpha * self.gradient(new_x)

            if self.func(u) > self.func(x):
                x, _x = new_x, u
                xk.append(x), _xk.append(_x)
                if self.length(_x - _prev_x) < eps1 and self.length(prev_x - x) <= eps3:
                    plt.grid(True)
                    xk, _xk = np.array(xk), np.array(_xk)
                    plt.plot(xk[:, 0], xk[:, 1])
                    plt.plot(_xk[:, 0], _xk[:, 1])
                    plt.show()
                    return _x
            else:
                lamda /= 2

    def Gelfand_method2(self, eps1=0.001, eps3=0.001, alpha=0.001, lamda=0.001):
        x = self.x0
        u = x - alpha * self.gradient(x)

        xk = [x]
        uk = [u]

        while True:
            sign = 1
            if np.random.rand() > 0.5:
                sign = -1

            prev_u = u.copy()
            _x = x + sign * 0.1 * np.ones(shape=x.shape)
            _u = _x - alpha * self.gradient(_x)

            xk.append(_x), uk.append(_u)

            iter_lamda = lamda
            while True:
                x_new = None
                if self.func(_u) < self.func(u):
                    x_new = u + iter_lamda * (_u - u) / self.length(_u - u)
                else:
                    x_new = _u + iter_lamda * (u - _u) / self.length(u - _u)
                u_new = x_new - alpha * self.gradient(x_new)
                if self.func(u_new) > self.func(u):
                    x = x_new
                    u = u_new

                    xk.append(x), uk.append(u)
                    break
                else:
                    iter_lamda /= 2

            if self.length(_x - prev_u) < eps1 and self.length(self.gradient(u)) <= eps3:
                plt.grid(True)
                xk, uk = np.array(xk), np.array(uk)
                plt.plot(xk[:, 0], xk[:, 1])
                plt.plot(uk[:, 0], uk[:, 1])
                plt.show()
                return u

    def Gelfand_method3(self, eps1=0.001, eps3=0.001, alpha=0.008, lamda=2):
        xk = self.x0
        xk1 = xk + np.ones(shape=xk.shape) * 0.1

        x, u = [xk], [xk1]
        i = 0
        while True:

            uk = xk + alpha * self.gradient(xk)
            uk1 = xk1 + alpha * self.gradient(xk1)

            x.append(uk), u.append(uk1)

            iter_lamda = lamda
            while True:
                xk_new = None
                if self.func(uk1) < self.func(uk):
                    xk_new = uk + iter_lamda * (uk1 - uk) / self.length(uk1 - uk)
                else:
                    xk_new = uk + iter_lamda * (uk - uk1) / self.length(uk - uk1)
                uk_new = xk_new - alpha * self.length(xk_new)
                if self.func(uk_new) < self.func(uk):
                    xk, xk1 = xk_new, uk_new

                    x.append(xk), u.append(xk1)
                    break
                else:
                    iter_lamda /= 2

            if i <= 1000:
                plt.grid(True)
                x, u = np.array(x), np.array(u)
                plt.plot(x[:, 0], x[:, 1])
                plt.plot(u[:, 0], u[:, 1])
                plt.show()
                return xk, xk1
            i += 1

    def Gelfand_method4(self, eps1=0.001, eps3=0.001, alpha=0.01, lamda=2):
        x = self.x0
        x1 = self.x0 + np.ones(shape=self.x0.shape) * 0.5

        graph_x, graph_u = [x], [x1]
        while True:
            prev_x, prev_x1 = x, x1
            _u0 = x - alpha * self.gradient(x)
            _u1 = x1 - alpha * self.gradient(x1)

            graph_x.append(_u0), graph_u.append(_u1)

            u0, u1 = 0, 0
            if self.func(_u0) < self.func(_u1):
                u0, u1 = _u1, _u0
            else:
                u0, u1 = _u0, _u1
            tmp_lamda = lamda
            while True:
                xk_new = u0 + tmp_lamda * (u1 - u0) / self.length(u1 - u0)
                uk_new = xk_new - alpha * self.gradient(xk_new)

                if self.func(uk_new) < self.func(u1):
                    x, x1 = xk_new, uk_new
                    graph_x.append(x), graph_u.append(x1)
                    break
                else:
                    tmp_lamda /= 2

            if self.length(x1 - prev_x1) <= eps1 and self.length(self.gradient(prev_x1)) <= eps3:
                plt.grid(True)
                graph_x, graph_u = np.array(graph_x), np.array(graph_u)
                plt.plot(graph_x[:, 0], graph_x[:, 1])
                plt.plot(graph_u[:, 0], graph_u[:, 1])
                plt.show()
                return x1
