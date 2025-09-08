import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4(f, y0, t):
    """
    四阶龙格库塔方法求解常微分方程
    
    参数:
    f: 函数，表示微分方程 dy/dt = f(t, y)
    y0: 初始条件，y(t0) = y0
    t: 时间点数组，用于计算解
    
    返回:
    y: 在时间点t上的解数组
    """
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return y

# 示例：求解微分方程 dy/dt = -2y + t, y(0) = 1
if __name__ == "__main__":
    # 定义微分方程
    def f(t, y):
        return -2 * y + t
    
    # 初始条件
    y0 = 1
    
    # 时间点
    t = np.linspace(0, 2, 21)  # 从0到2，21个点
    
    # 使用四阶龙格库塔方法求解
    y_rk4 = runge_kutta_4(f, y0, t)
    
    # 精确解（用于比较）
    y_exact = 0.25 * (2 * t + 1 + 3 * np.exp(-2 * t))
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_exact, 'r-', label='精确解')
    plt.plot(t, y_rk4, 'bo--', label='四阶龙格库塔')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.title('四阶龙格库塔方法求解微分方程')
    plt.grid(True)
    plt.show()
    
    # 计算并打印最大误差
    max_error = np.max(np.abs(y_rk4 - y_exact))
    print(f"最大误差: {max_error:.6e}")