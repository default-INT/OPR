import numpy as np


class FuncAnalytics:

    def __init__(self, a, b, func=None, dfunc=None) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.func = func
        self.dfunc = dfunc

    def set_interval(self, a, b):
        self.a = a
        self.b = b

    def get_func_value(self, dx):
        ax = self.a
        bx = self.b

        _x = []
        _y = []
        x = ax
        while x < bx:
            _y.append(self.func(x))
            _x.append(x)
            x += dx
        return _x, _y

    def passive_algorithm(self, N):

        x, y, k = [], [], 0
        delta = (self.b - self.a) / N
        a, b = self.a, self.b

        if N % 2 == 0:
            k = N / 2
        else:
            k = (N - 1) / 2

        def _get_func_values(ax, bx):
            _x = []
            _y = []
            for i in range(N):
                if N % 2 == 0:
                    _x.append(ax + (bx - ax) / (N + 1) * i)
                else:
                    tmp = ax + (bx - ax) / (k + 1) * i
                    _x.append(tmp - delta)
                    _x.append(tmp)
            for i in _x:
                _y.append(self.func(i))
            return _x, _y

        LN = 0
        if N % 2 == 0:
            LN = 2 * (self.b - self.a) / (N + 1)
        else:
            LN = 2 * (self.b - self.a) / (k + 1) + delta
        eps = LN / 2

        x, y = _get_func_values(self.a, self.b)

        min_y = np.min(y)
        j = y.index(min_y)
        L = x[j + 1] - x[j - 1]

        exact_min_y = 0
        exact_min_x = 0
        tmp_x, tmp_y = x, y
        while np.fabs(exact_min_y - min_y) > eps:
            tmp_x, tmp_y = _get_func_values(tmp_x[j - 1], tmp_y[j + 1])
            exact_min_y = np.min(tmp_y)
            j = tmp_y.index(exact_min_y)
            exact_min_x = tmp_x[j]

        return exact_min_x, exact_min_y

    def bisection_algorithm(self, eps):

        def _find_min_value(a, b):
            get_middle = lambda left, right: \
                ((left + right) / 2,
                 self.func((left + right) / 2))
            x2, y2 = get_middle(a, b)
            x1, y1 = get_middle(a, x2)
            x3, y3 = get_middle(x2, b)
            x = [a, x1, x2, x3, b]
            y = [self.func(a),
                 y1, y2, y3,
                 self.func(b)
                 ]
            min_y = np.min(y)
            j = y.index(min_y)

            return x[j], min_y, x[j - 1], x[j + 1]

        min_x, min_y, a, b = _find_min_value(self.a, self.b)

        while b - a > 2 * eps:
            min_x, min_y, a, b = _find_min_value(a, b)

        return min_x, min_y

    def dichotomy_method(self, eps):
        delta = eps / (1 / eps ** 2)

        def _find_min_len(a, b):
            x = (a + b) / 2
            y1 = self.func(x - delta)
            y2 = self.func(x + delta)
            if y1 <= y2:
                return a, x + delta
            else:
                return x - delta, b

        a, b = _find_min_len(self.a, self.b)

        while b - a > 2 * eps:
            a, b = _find_min_len(a, b)

        return (a + b) / 2, self.func((a + b) / 2)

    def fibonacci_method(self, N):

        def _get_number_fibonacci(n):
            F = [1, 1]
            for i in range(2, n):
                F.append(F[i - 1] + F[i - 2])
            return F

        F = _get_number_fibonacci(N)

        x1 = self.a + (self.b - self.a) * (F[N - 3] / F[N - 1])
        x2 = self.a + (self.b - self.a) * (F[N - 2] / F[N - 1])

        y1, y2 = self.func(x1), self.func(x2)

        a, b = self.a, self.b

        for i in range(N - 2):
            if y1 <= y2:
                b, x2, y2 = x2, x1, y1
                x1 = a + b - x2
                y1 = self.func(x1)
            else:
                a, x1, y1 = x1, x2, y2
                x2 = a + b - x1
                y2 = self.func(x2)

        if y1 < y2:
            return x1, y1
        else:
            return x2, y2

    def tangent_method(self, eps):
        a, b = self.a, self.b
        y1, y2 = self.func(a), self.func(b)
        z1, z2 = self.dfunc(a), self.dfunc(b)

        while (b - a) > 2 * eps:
            c = ((b * z2 - a * z1) - (y2 - y1)) / (z2 - z1)
            y, z = self.func(c), self.dfunc(c)
            if z == 0:
                return c, y
            elif z < 0:
                a, y1, z1 = c, y, z
            else:
                b, y2, z2 = c, y, z
        return c, y
