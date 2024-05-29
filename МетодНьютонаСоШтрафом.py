import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Определяем входные параметры
a = -6
b = -7
c = 1
d = -3
f_coef = 2
e = -3
x0 = np.array([-6.0, -6.0])
tol = 0.01

# Определяем целевую функцию
def f(x, R=None):
    x1, x2 = x
    if R is not None:
        return (x1 - a) ** 2 + (x2 - b) ** 2 + c * x1 * x2 + (1 / R) * (d * x1 + f_coef * x2 + e) ** 2
    else:
        return (x1 - a) ** 2 + (x2 - b) ** 2 + c * x1 * x2

# Определяем градиент целевой функции
def grad_f(x, R=None):
    x1, x2 = x
    if R is not None:
        df_dx1 = 2 * (x1 - a) + c * x2 + (2 / R) * (d * x1 + f_coef * x2 + e) * d
        df_dx2 = 2 * (x2 - b) + c * x1 + (2 / R) * (d * x1 + f_coef * x2 + e) * f_coef
    else:
        df_dx1 = 2 * (x1 - a) + c * x2
        df_dx2 = 2 * (x2 - b) + c * x1
    return np.array([df_dx1, df_dx2])

# Определяем гессиан целевой функции
def hessian_f(x, R=None):
    x1, x2 = x
    if R is not None:
        return np.array([[2 + (2 / R) * d ** 2, c + (2 / R) * d * f_coef],
                         [c + (2 / R) * d * f_coef, 2 + (2 / R) * f_coef ** 2]])
    else:
        return np.array([[2, c], [c, 2]])

# Функция для минимизации методом Сильвестра
def minimize_sylvester(x0, f, grad_f, hessian_f):
    res = minimize(f, x0, method='Newton-CG', jac=grad_f, hess=hessian_f, options={'xtol': 1e-6, 'maxiter': 1000, 'disp': False})
    if res.success:
        return res.x
    else:
        print(res.message)
        return None

# Функция метода Ньютона
def newton_method(f, grad_f, hessian_f, x0, R_values, tol=1e-6, max_iter=100):
    optimums = []  # Хранение найденных оптимумов
    x = x0
    optimums.append(x)  # Добавляем начальную точку
    for R in R_values:
        damping_factor = 1.0
        for iteration in range(max_iter):
            grad = grad_f(x, R)
            hessian = hessian_f(x, R)
            delta_x = -np.linalg.inv(hessian) @ grad * damping_factor
            new_x = x + delta_x
            # Проверка условия достижения оптимума
            if np.linalg.norm(grad) < tol:
                break
            # Проверка условия, при котором необходимо увеличить коэффициент
            if f(new_x, R) >= f(x, R):
                damping_factor *= 0.5  # Уменьшаем коэффициент демпфирования вдвое
            x = new_x
        optimums.append(x)  # Добавляем найденную точку
        x0 = x  # Используем найденную точку как начальную для следующей итерации
    return np.array(optimums)

# Функция для объединения кодов и построения графика
def optimize_and_plot(x0):
    R_values = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    # Генерируем сетку точек для построения линий уровня
    x1_values = np.linspace(-15, 10, 400)
    x2_values = np.linspace(-15, 5, 400)
    X1, X2 = np.meshgrid(x1_values, x2_values)
    Z = np.zeros_like(X1)
    for i in range(len(x1_values)):
        for j in range(len(x2_values)):
            Z[i, j] = f([X1[i, j], X2[i, j]], R_values[0])  # Вычисляем значение функции для каждой точки с R = R_values[0]
    # Построение графика
    plt.figure(figsize=(10, 8))
    plt.contour(X1, X2, Z, levels=[10, 20, 30, 70, 100], colors='gray', linestyles='dashed')  # Линии уровня функции
    # Запуск метода Ньютона для нахождения оптимума
    optimums = newton_method(f, grad_f, hessian_f, x0, R_values)
    # Сохранение первых двух координат в отдельный массив
    first_two_optimums = optimums[:2]
    # Отображаем найденные точки и соединяем их линиями
    for i in range(1, len(optimums) - 1):  # Начинаем с индекса 1
        plt.plot([optimums[i][0], optimums[i + 1][0]], [optimums[i][1], optimums[i + 1][1]], color='red')  # Линии между точками
        plt.plot(optimums[i][0], optimums[i][1], 'ro')  # Точки оптимума
    # Отображаем первые две точки как красную линию
    plt.plot([first_two_optimums[0][0], first_two_optimums[1][0]], [first_two_optimums[0][1], first_two_optimums[1][1]], color='red')
    # Отображаем линию функции -3*x1 + 2*x2 - 3
    plt.contour(X1, X2, -3 * X1 + 2 * X2 - 3, levels=[0], colors='blue')
    plt.axhline(0, color='black')  # Горизонтальная линия по центру
    plt.axvline(0, color='black')  # Вертикальная линия по центру
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.show()
    tol = 1e-6
    print("Входные параметры:", "x0 =", x0, "tol =", tol)
    for i, R in enumerate(R_values):
        print(f"Значения оптимума для R={R} методом Ньютона:", optimums[i])
        # Решение оптимума методом Сильвестра
        sylvester_optimal_point = minimize_sylvester(x0, lambda x: f(x, R), lambda x: grad_f(x, R),
                                                     lambda x: hessian_f(x, R))
        if sylvester_optimal_point is not None:
            print(f"Проверка методом Сильвестра найдена точка оптимума для R={R} при (x1, x2) =", sylvester_optimal_point)
        else:
            print(f"Проверить методом Сильвестра для R={R} не удалось найти точку оптимума.")

# Вызываем функцию для оптимизации и построения графика
optimize_and_plot(x0)
