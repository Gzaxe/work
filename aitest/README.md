# 四阶龙格库塔法求解微分方程

这个项目实现了使用四阶龙格库塔法（RK4）求解常微分方程的Python代码。

## 文件说明

- `runge_kutta_4.py`: 包含RK4算法实现和示例
- `rk4_examples.png`: 一阶ODE和简谐振动系统的求解结果图
- `lorenz_attractor.png`: 洛伦兹吸引子的3D图

## 算法说明

四阶龙格库塔法是一种常用的数值方法，用于求解形如 dy/dt = f(t, y) 的常微分方程。其计算公式为：

k1 = f(t_n, y_n)
k2 = f(t_n + h/2, y_n + h*k1/2)
k3 = f(t_n + h/2, y_n + h*k2/2)
k4 = f(t_n + h, y_n + h*k3)

y_{n+1} = y_n + h*(k1 + 2*k2 + 2*k3 + k4)/6

其中 h 是步长。

## 使用方法

```python
# 导入必要的库
import numpy as np

# 定义微分方程 dy/dt = f(t, y)
def ode_function(t, y):
    return -2 * y + 1  # 示例: dy/dt = -2*y + 1

# 设置初始条件和参数
y0 = 0          # 初始值
t_span = (0, 5) # 时间区间
dt = 0.01       # 时间步长

# 调用RK4函数求解
t, y = rk4(ode_function, y0, t_span, dt)
```

## 示例

程序包含了三个示例：

1. 一阶线性ODE: dy/dt = -2*y + 1
2. 简谐振动系统: d^2x/dt^2 + x = 0
3. 洛伦兹系统（混沌系统）

## 运行代码

```bash
python runge_kutta_4.py
```

这将生成两个图像文件显示求解结果。
