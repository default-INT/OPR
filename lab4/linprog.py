import numpy as np


func = lambda x: 2 * x[0] + x[1] + x[2] + x[3]


class LinearOptimization:

    def __init__(self, equation_coef=None, basis=np.array([0, 1, 0, 1])) -> None:
        super().__init__()
        self._equation_coef = equation_coef
        self._conditions = []
        self._basis = basis

    def add_condition(self, equation_coef=None):
        self._conditions.append(equation_coef)

    def set_basis(self, basis):
        self._basis = basis

    def _get_matrix_table(self):
        z_line = [1] + (self._equation_coef * -1).tolist() + [0]
        bias_lines = []
        for condition in self._conditions:
            bias_lines.append([0] + condition)
        return np.array([z_line] + bias_lines, dtype=np.float)

    @staticmethod
    def __get_index_lead_row(matrix_table, indx_lead_col):
        min_positive_val, indx_lead_row = None, None
        for i in range(1, matrix_table.shape[0]):
            rel = matrix_table[i][-1] / matrix_table[i][indx_lead_col]
            if min_positive_val is None or min_positive_val > rel >= 0:
                min_positive_val, indx_lead_row = rel, i
        return indx_lead_row

    @staticmethod
    def __gauss_jordan_iteration(matrix_table):
        pass

    def optimize(self):
        matrix_table = self._get_matrix_table()
        basis = self._basis
        equations = np.zeros(shape=len(self._conditions))
        k = 0
        for i, val in enumerate(basis):
            if val == 1:
                equations[k] = i
                k += 1
        while True:
            for indx, basis_value in enumerate(basis):
                if basis_value == 0:
                    indx_lead_col = indx + 1
                    indx_lead_row = self.__get_index_lead_row(matrix_table, indx_lead_col)
                    koef = matrix_table[indx_lead_row][indx_lead_col]
                    for i in range(1, matrix_table.shape[1]):
                        matrix_table[indx_lead_row][i] /= koef
                    for i in range(matrix_table.shape[0]):
                        if i != indx_lead_row:
                            k = matrix_table[i][indx_lead_col]
                            for j in range(matrix_table.shape[1]):
                                matrix_table[i][j] -= k * matrix_table[indx_lead_row][j]
                    equations[indx_lead_row - 1] = indx_lead_col - 1
                    k = 0
                    basis = np.zeros(shape=len(self._equation_coef))
                    for val in equations:
                        for j, _ in enumerate(basis):
                            if j == val:
                                basis[j] = 1
                    if (matrix_table[0] >= 0).all():
                        result = np.zeros(shape=self._equation_coef.shape)
                        for i in range(1, len(matrix_table)):
                            for j, value in enumerate(basis):
                                if equations[i - 1] == j:
                                    result[j] = matrix_table[i][-1]
                        return result, matrix_table


if __name__ == "__main__":
    line_opt = LinearOptimization(
        np.array([2, 1, 1, 1])
    )
    line_opt.add_condition([1, 2, 2, 1, 2])
    line_opt.add_condition([1, 0, -1, 1, 1])

    result, matrix = line_opt.optimize()

    print(result)
    print(func(result))