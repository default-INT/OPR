from tkinter import filedialog as fd
from tkinter import ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
from tkinter import messagebox as mb
from unconditional_optimization import FuncAnalytics

func = lambda x: 1.4 * x + np.exp(np.fabs(x - 2))
dfunc = lambda x: 1.4 + ((x - 2) * np.exp(np.fabs(x - 2))) / np.fabs(x - 2)


class MainWindow(tk.Frame):
    btn_color = '#1b1b1b'
    text_color = 'white'
    bg_color = '#292f3f'
    body_color = '#292f3f'
    toolbar_color = '#1b1b1b'
    _body = None

    def __init__(self, root):
        super().__init__(root)
        self.info_img = tk.PhotoImage(file='info.png')
        self.graph_img = tk.PhotoImage(file='graph.png')
        self.edit_img = tk.PhotoImage(file='edit2.png')
        self.add_img = tk.PhotoImage(file='add.png')

        self.a_var_str = tk.StringVar()
        self.b_var_str = tk.StringVar()
        self.eps_var_str = tk.StringVar()
        self.n_var_str = tk.StringVar()

        self.init_window()

    def init_window(self):
        self._body = tk.Frame(bg=self.body_color, bd=2, height=305)
        self._body.pack(side=tk.TOP, fill=tk.X)

        toolbar = tk.Frame(bg=self.toolbar_color, bd=2, height=140)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar.pack_propagate(False)

        in_pad_x = 35
        in_pad_y = 3

        self._main_page()

        self.add_button(toolbar, 'Главное\nменю', self._main_page, img=self.edit_img, column=0,
                        ipadx=in_pad_x, ipady=in_pad_y)
        self.add_button(toolbar, 'Построить\nграфик', method=self._plot_histogram, img=self.graph_img, column=1,
                        ipadx=in_pad_x, ipady=in_pad_y)

    def add_label(self, root, text, font="Arial 12", justify="left"):
        label = tk.Label(root, bg=self.bg_color, fg=self.text_color, text=text, font=font, justify=justify,
                         wraplength=600)
        label.pack(expand=True, fill=tk.BOTH)

    def _add_entry_grid(self, root, textvariable=None, width=10, row=0, column=0, padx=0, pady=0, ipadx=0,
                        ipady=0):
        entry = tk.Entry(root, textvariable=textvariable, width=width)
        entry.grid(row=row, column=column, padx=padx, pady=pady, ipady=ipady, ipadx=ipadx)

    def _add_label_grid(self, root, text, font="Arial 12", justify="left", row=0, column=0, padx=0, pady=0, ipadx=0,
                        ipady=0, sticky="w"):
        label = tk.Label(root, bg=self.bg_color, fg=self.text_color, text=text, font=font, justify=justify)
        label.grid(row=row, column=column, padx=padx, pady=pady, ipady=ipady, ipadx=ipadx, sticky=sticky)

    def _add_table_item(self, root, text, font="Arial 12", justify="left", width=0, row=0, column=0, padx=0, pady=0,
                        ipadx=0, ipady=0, border=0):
        label = tk.Label(root, bg=self.bg_color, fg=self.text_color, text=text, font=font, justify=justify,
                         bd=2, width=width)
        label.grid(row=row, column=column, padx=padx, pady=pady, ipady=ipady, ipadx=ipadx)

    def add_button(self, root, text_content="undefined", method=None, img=None, row=0, column=0, ipadx=0, ipady=0):
        btn = tk.Button(root, text=text_content, command=method, bg=self.btn_color, activebackground=self.btn_color,
                        fg=self.text_color, bd=0, compound=tk.TOP, image=img)
        btn.grid(row=row, column=column, ipadx=ipadx, ipady=ipady)

    def read_file(self):
        for widget in self._body.winfo_children():
            widget.destroy()
        pass

    def write_file(self):
        pass

    def _main_page(self):
        for widget in self._body.winfo_children():
            widget.destroy()

        self._add_label_grid(self._body, text='Методы одномерной оптимизации',
                       font='Arial 16', justify='center', column=2)

        self._add_label_grid(self._body, text='Введите a:', justify='left', row=1)
        self._add_entry_grid(self._body, textvariable=self.a_var_str, row=1, column=1)

        self._add_label_grid(self._body, text='Введите b:', justify='left', row=2)
        self._add_entry_grid(self._body, textvariable=self.b_var_str, row=2, column=1)

        self._add_label_grid(self._body, text='Введите Eps:', justify='left', row=3)
        self._add_entry_grid(self._body, textvariable=self.eps_var_str, row=3, column=1)

        self._add_label_grid(self._body, text='Введите N:', justify='left', row=4)
        self._add_entry_grid(self._body, textvariable=self.n_var_str, row=4, column=1)

        self.add_button(self._body, text_content='Высчитать', row=5, column=1, method=self._calculate)

    def _calculate(self):
        for widget in self._body.winfo_children():
            widget.destroy()

        self._add_label_grid(self._body, text='Результаты вычисления',
                             font='Arial 20', justify='center', column=1, sticky='w')

        self.func_analytics = FuncAnalytics(float(self.a_var_str.get()), float(self.b_var_str.get()), func, dfunc)

        N = int(self.n_var_str.get())
        eps = float(self.eps_var_str.get())

        x, y = self.func_analytics.passive_algorithm(N)
        self._add_label_grid(self._body, text='Пассивный оптимальный алгоритм: ', justify='left', row=1)
        self._add_label_grid(self._body, text='x = ' + str(x) + ', y = ' + str(y),
                             justify='left', row=1, column=1)

        x, y = self.func_analytics.bisection_algorithm(eps)
        self._add_label_grid(self._body, text='Алгоритм деления интервала пополам: ', justify='left', row=2)
        self._add_label_grid(self._body, text='x = ' + str(x) + ', y = ' + str(y),
                             justify='left', row=2, column=1)

        x, y = self.func_analytics.bisection_algorithm(eps)
        self._add_label_grid(self._body, text='Метод дихотомии: ',
                             justify='left', row=3)
        self._add_label_grid(self._body, text='x = ' + str(x) + ', y = ' + str(y),
                             justify='left', row=3, column=1)

        x, y = self.func_analytics.fibonacci_method(N)
        self._add_label_grid(self._body, text='Метод Фибоначчи: ', justify='left', row=4)
        self._add_label_grid(self._body, text='x = ' + str(x) + ', y = ' + str(y), justify='left',
                             row=4, column=1)

        x, y = self.func_analytics.tangent_method(eps)
        self._add_label_grid(self._body, text='Метод касательных: ', justify='left', row=5)
        self._add_label_grid(self._body, text='x = ' + str(x) + ', y = ' + str(y), justify='left', row=5, column=1)

    def _analytics_companys(self):
        pass

    def _plot_histogram(self):
        if self.func_analytics is None:
            mb.showinfo('Ошибка', 'Данные неинициализированы')
            return
        for widget in self._body.winfo_children():
            widget.destroy()

        N = int(self.n_var_str.get())
        fig = plt.figure(figsize=(6, 4), dpi=100)

        fig.patch.set_facecolor('grey')

        canvas = FigureCanvasTkAgg(fig, master=self._body)
        plot_widget = canvas.get_tk_widget()
        plt.xlabel(u'Компании')
        plt.ylabel(u'Сред. заработная плата')

        eps = float(self.eps_var_str.get())
        x, y = self.func_analytics.get_func_value(eps)
        graph1 = plt.plot(x, y)
        grid1 = plt.grid(True)
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    @staticmethod
    def _create_table(root, array, headings=tuple()):
        di = 1
        table = Table(root, headings=headings)
        for company in array:
            values = [array[0], array[1], array[2]]
            table.add_row(values)
            # di += i
        table.pack(expand=tk.YES, fill=tk.BOTH)


class Table(tk.Frame):
    def __init__(self, master=None, headings=tuple()):
        super().__init__(master)
        self._table = ttk.Treeview(self, show="headings", selectmod='browse')
        self._table['columns'] = headings
        self._table['displaycolumns'] = headings

        for head in headings:
            self._table.heading(head, text=head, anchor=tk.CENTER)
            self._table.column(head, anchor=tk.CENTER, width=8)

        scroll_table = tk.Scrollbar(self, command=self._table.yview)
        self._table.configure(yscrollcommand=scroll_table.set)
        scroll_table.pack(side=tk.RIGHT, fill=tk.Y)
        self._table.pack(expand=tk.YES, fill=tk.BOTH)

    def add_row(self, values):
        self._table.insert('', tk.END, values=tuple(values))
