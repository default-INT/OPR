import sys

import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QVBoxLayout

from main_window import Ui_MainWindow
from linprog import LinearOptimization


func = lambda x: 2 * x[0] + x[1] + x[2] + x[3]


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.calcButton.clicked.connect(lambda: self._calc())

    def _calc(self):
        x1, x2, x3, x4 = float(self.ui.x1LineEdit.text()), float(self.ui.x2LineEdit.text()), \
                         float(self.ui.x3LineEdit.text()), float(self.ui.x4LineEdit.text())

        b1, b2, b3, b4 = float(self.ui.bLineEdit_1.text()), float(self.ui.bLineEdit_2.text()), \
                         float(self.ui.bLineEdit_3.text()), float(self.ui.bLineEdit_4.text())

        condition1 = [
            float(self.ui.xLineEdit_11.text()),
            float(self.ui.xLineEdit_12.text()),
            float(self.ui.xLineEdit_13.text()),
            float(self.ui.xLineEdit_14.text()),
            float(self.ui.result1LineEdit.text())
        ]

        condition2 = [
            float(self.ui.xLineEdit_21.text()),
            float(self.ui.xLineEdit_22.text()),
            float(self.ui.xLineEdit_23.text()),
            float(self.ui.xLineEdit_24.text()),
            float(self.ui.result2LineEdit.text())
        ]

        line_opt = LinearOptimization(
            np.array([x1, x2, x3, x4]),
            np.array([b1, b2, b3, b4])
        )
        line_opt.add_condition(condition1)
        line_opt.add_condition(condition2)

        result, _ = line_opt.optimize()

        result = np.round(result, 3)

        self.ui.xLabel_1.setText(str(result[0]))
        self.ui.xLabel_2.setText(str(result[1]))
        self.ui.xLabel_3.setText(str(result[2]))
        self.ui.xLabel_4.setText(str(result[3]))

        self.ui.zLabel.setText(str(func(result)))


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    application = MainWindow()
    application.show()

    sys.exit(app.exec())
