import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def func(omega, target_omega):
    return omega**2 - target_omega**2

def simulate_search(initial_omega, domg, target_omega, max_iter=50, improved=False, newton=False):
    omega = initial_omega
    omega_history = [omega]
    old_dEr = func(omega, target_omega)
    found_solution = False  # 标记是否找到解
    zomg = omega

    for _ in range(max_iter):

        if improved:
            omega = omega + domg
            dEr = func(omega, target_omega)
            if np.real(dEr) * np.real(old_dEr) < 0:
                omega = omega - np.real(domg)
                domg = complex(np.real(domg) / 2, np.imag(domg))
            if np.imag(dEr) * np.imag(old_dEr) < 0:
                omega = omega - complex(0.0, np.imag(domg))
                domg = complex(np.real(domg), np.imag(domg) / 2)
        else:
            if newton:
                omega = zomg
                dEr = func(omega, target_omega)
                omgtmp = omega + domg
                old_dEr = func(omgtmp, target_omega)
                domg = -domg * dEr / (old_dEr - dEr)
                zomg = zomg + domg
            else:
                omega = omega + domg
                dEr = func(omega, target_omega)
                if np.real(dEr) * np.real(old_dEr) < 0:
                    omega = omega - domg
                    domg = domg / 2
                if np.imag(dEr) * np.imag(old_dEr) < 0:
                    omega = omega - domg
                    domg = domg / 2

        omega_history.append(omega)

        if abs(dEr) < 1e-6:
            found_solution = True
            break  # 找到解后退出循环

    return omega_history, found_solution

# 设置参数
initial_omega = 2 + 1j
initial_domg = 0.5 + 0.2j
target_omega = 9.5 + 2.5j

# 运行搜索算法
omega_history_original, found_original = simulate_search(initial_omega, initial_domg, target_omega, improved=False, newton=False)
omega_history_improved, found_improved = simulate_search(initial_omega, initial_domg, target_omega, improved=True, newton=False)
omega_history_newton, found_newton = simulate_search(initial_omega, initial_domg, target_omega, improved=False, newton=True)

all_omega = omega_history_original + omega_history_improved + omega_history_newton

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('Real(omega)')
ax.set_ylabel('Imag(omega)')
ax.set_title('Search Path in Complex Plane (Dynamic Axes)')
ax.grid(True)

padding = 0.5
ax.set_xlim(np.min(np.real(all_omega)) - padding, np.max(np.real(all_omega)) + padding)
ax.set_ylim(np.min(np.imag(all_omega)) - padding, np.max(np.imag(all_omega)) + padding)

ax.plot(np.real(initial_omega), np.imag(initial_omega), 'gs', markersize=10, label='Start')
ax.plot(np.real(target_omega), np.imag(target_omega), 'g*', markersize=15, label='Target')

line_original, = ax.plot([], [], 'bo-', label='Original Algorithm')
line_improved, = ax.plot([], [], 'ro-', label='Improved Algorithm')
line_newton, = ax.plot([], [], 'go-', label='Newton Algorithm')

# 添加找到解后的标记 (初始时不可见)
solution_marker_original, = ax.plot([], [], 'bX', markersize=12, visible=False)  # 蓝色 X
solution_marker_improved, = ax.plot([], [], 'rP', markersize=12, visible=False)  # 红色 P
solution_marker_newton, = ax.plot([], [], 'gD', markersize=12, visible=False)  # 绿色 D

ax.legend()

def update(frame):
    x_original = np.real(omega_history_original[:frame+1])
    y_original = np.imag(omega_history_original[:frame+1])
    line_original.set_data(x_original, y_original)

    x_improved = np.real(omega_history_improved[:frame+1])
    y_improved = np.imag(omega_history_improved[:frame+1])
    line_improved.set_data(x_improved, y_improved)

    x_newton = np.real(omega_history_newton[:frame+1])
    y_newton = np.imag(omega_history_newton[:frame+1])
    line_newton.set_data(x_newton, y_newton)
    
    # 显示/隐藏找到解的标记
    if found_original and frame >= len(omega_history_original) -1 :
          solution_marker_original.set_data(x_original[-1], y_original[-1])
          solution_marker_original.set_visible(True)

    if found_improved and frame >= len(omega_history_improved) -1:
          solution_marker_improved.set_data(x_improved[-1], y_improved[-1])
          solution_marker_improved.set_visible(True)

    if found_newton and frame >= len(omega_history_newton) -1:
        solution_marker_newton.set_data(x_newton[-1], y_newton[-1])
        solution_marker_newton.set_visible(True)


    all_x = np.concatenate((x_original, x_improved, x_newton))
    all_y = np.concatenate((y_original, y_improved, y_newton))

    if len(all_x) > 0:
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)

        if x_min < ax.get_xlim()[0] or x_max > ax.get_xlim()[1]:
            ax.set_xlim(x_min - padding, x_max + padding)
        if y_min < ax.get_ylim()[0] or y_max > ax.get_ylim()[1]:
            ax.set_ylim(y_min - padding, y_max + padding)

        ax.figure.canvas.draw()

    return line_original, line_improved, line_newton, solution_marker_original, solution_marker_improved, solution_marker_newton

ani = animation.FuncAnimation(
    fig, update, frames=max(len(omega_history_original), len(omega_history_improved), len(omega_history_newton)),
    interval=100, blit=True
)
plt.show()
