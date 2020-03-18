from unconditional_optimization import FuncAnalytics
from interface import *

func = lambda x: 1.4 * x + np.exp(np.fabs(x - 2))
dfunc = lambda x: 1.4 + ((x - 2) * np.exp(np.fabs(x - 2))) / np.fabs(x - 2)

# пассивный оптимальный алгоритм +
# алгоритм деления интервала пополам +
# метод дихотомии +
# метод Фибоначчи +
# метод касательных +

if __name__ == "__main__":
    func_analytics = FuncAnalytics(-5, 5, func, dfunc)
    print(func_analytics.passive_algorithm(500))
    print(func_analytics.bisection_algorithm(0.001))
    print(func_analytics.dichotomy_method(0.001))
    print(func_analytics.fibonacci_method(100))
    print(func_analytics.tangent_method(0.0001))

    root = tk.Tk()
    root.geometry('700x550+140+90')
    root.config(bg='#292f3f')
    root.title('Методы оптимизации')
    MainWindow(root)
    root.resizable(False, False)
    root.mainloop()
