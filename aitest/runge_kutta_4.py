import numpy as np
import matplotlib.pyplot as plt

def rk4(func, y0, t_span, dt):
    """
    四阶龙格库塔法求解常微分方程
    
    参数:
    func : callable
        微分方程函数 dy/dt = func(t, y)
    y0 : float or array-like
        初始条件
    t_span : tuple
        时间区间 (t_start, t_end)
    dt : float
        时间步长
    
    返回:
    t : ndarray
        时间点数组
    y : ndarray
        对应时间点的解
    """
    t_start, t_end = t_span
    # 创建时间数组
    t = np.arange(t_start, t_end + dt, dt)
    # 初始化解数组
    y = np.zeros((len(t), len(y0) if hasattr(y0, '__len__') else 1))
    y[0] = y0
    
    # RK4迭代
    for i in range(len(t) - 1):
        ti = t[i]
        yi = y[i]
        
        # 计算四个斜率
        k1 = func(ti, yi)
        k2 = func(ti + dt/2, yi + dt/2 * k1)
        k3 = func(ti + dt/2, yi + dt/2 * k2)
        k4 = func(ti + dt, yi + dt * k3)
        
        # 更新解
        y[i+1] = yi + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y

# 示例：求解简单的一阶ODE
def example_ode1(t, y):
    """
    示例1: dy/dt = -2*y + 1
    解析解: y(t) = 0.5 + (y0 - 0.5)*exp(-2*t)
    """
    return -2 * y + 1

# 示例：求解二阶ODE转换的系统（如简谐振动）
def example_ode2_system(t, y):
    """
    示例2: 简谐振动 d^2x/dt^2 + x = 0
    转换为一阶系统:
    dy[0]/dt = y[1]  (位置的变化率 = 速度)
    dy[1]/dt = -y[0] (速度的变化率 = -位置)
    
    其中 y[0] 是位置，y[1] 是速度
    解析解: x(t) = A*cos(t) + B*sin(t)
    """
    dydt = np.zeros_like(y)
    dydt[0] = y[1]      # dx/dt = v
    dydt[1] = -y[0]     # dv/dt = -x
    return dydt

# 示例：洛伦兹系统（混沌系统）
def lorenz_system(t, y, sigma=10, rho=28, beta=8/3):
    """
    洛伦兹系统:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    
    参数:
    sigma, rho, beta: 系统参数
    """
    x, y_val, z = y
    dydt = np.zeros_like(y)
    dydt[0] = sigma * (y_val - x)
    dydt[1] = x * (rho - z) - y_val
    dydt[2] = x * y_val - beta * z
    return dydt

def main():
    """主函数：演示不同类型的ODE求解"""
    
    print("四阶龙格库塔法求解微分方程示例")
    print("=" * 40)
    
    # 示例1: 一阶线性ODE
    print("\n示例1: 求解 dy/dt = -2*y + 1, y(0) = 0")
    y0_1 = 0
    t_span_1 = (0, 5)
    dt_1 = 0.01
    
    t1, y1 = rk4(example_ode1, y0_1, t_span_1, dt_1)
    
    # 绘制结果
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(t1, y1, label='RK4解')
    # 解析解
    y_analytical = 0.5 * (1 - np.exp(-2 * t1))
    plt.plot(t1, y_analytical, '--', label='解析解')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('一阶线性ODE')
    plt.legend()
    plt.grid(True)
    
    # 示例2: 简谐振动系统
    print("\n示例2: 简谐振动系统 d^2x/dt^2 + x = 0, x(0)=1, v(0)=0")
    y0_2 = [1, 0]  # 初始位置=1, 初始速度=0
    t_span_2 = (0, 10)
    dt_2 = 0.01
    
    t2, y2 = rk4(example_ode2_system, y0_2, t_span_2, dt_2)
    
    plt.subplot(1, 2, 2)
    plt.plot(t2, y2[:, 0], label='位置 x')
    plt.plot(t2, y2[:, 1], label='速度 v')
    plt.xlabel('t')
    plt.ylabel('值')
    plt.title('简谐振动系统')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rk4_examples.png')
    plt.show()
    
    # 示例3: 洛伦兹系统（如果需要查看三维图）
    print("\n示例3: 洛伦兹系统")
    y0_3 = [1, 1, 1]  # 初始条件
    t_span_3 = (0, 20)
    dt_3 = 0.01
    
    t3, y3 = rk4(lorenz_system, y0_3, t_span_3, dt_3)
    
    # 绘制洛伦兹吸引子
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y3[:, 0], y3[:, 1], y3[:, 2], lw=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('洛伦兹吸引子')
    plt.savefig('lorenz_attractor.png')
    plt.show()
    
    print("\n所有示例完成！结果已保存为图片。")

if __name__ == "__main__":
    main()
