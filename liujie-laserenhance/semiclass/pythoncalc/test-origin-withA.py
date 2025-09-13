#测试计算原始参数，fig3-a-phase1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams
import matplotlib.font_manager as fm
import pandas as pd

# 设置中文字体支持
try:
    # 尝试设置中文字体
    font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi', 'Arial Unicode MS']
    available_fonts = []
    
    for font_name in font_list:
        try:
            font = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family=font_name)))
            available_fonts.append(font_name)
        except:
            continue
    
    if available_fonts:
        plt.rcParams['font.sans-serif'] = available_fonts + ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"已设置中文字体: {available_fonts[0]}")
    else:
        print("警告: 未找到中文字体，可能会显示乱码")
except Exception as e:
    print(f"字体设置错误: {e}")

class CoulombPotentialParticle:
    """库伦势场中粒子运动的数值模拟"""
    
    def __init__(self, k=1.0, m=1.0, q=1.0, Q=1.0):
        self.k = k  # 库伦常数
        self.m = m  # 粒子质量
        self.q = q  # 粒子电荷
        self.Q = Q  # 中心电荷
        
    def potential(self, x):
        """计算库伦势能"""
        return self.k * self.q * self.Q / np.abs(x)
    
    def force(self, x):
        """计算库伦力"""
        return self.k * self.q * self.Q / (x * x)
    
    def hamiltonian(self, x, p):
        """计算哈密顿量 (总能量)"""
        return p**2 / (2 * self.m) + self.potential(x)
    
    def equations_of_motion(self, t, y):
        """哈密顿方程"""
        x, p = y
        dxdt = p / self.m + 28.9176 * np.sin(1.51859 *10**(18) * t) * np.sin(1.26549 *10**17 * t)**2
        dpdt = self.force(x)
        return np.array([dxdt, dpdt])
    
    def runge_kutta_4(self, y0, t):
        """四阶龙格库塔方法求解微分方程"""
        n = len(t)
        y = np.zeros((n, len(y0)))
        y[0] = y0
        
        for i in range(n - 1):
            h = t[i+1] - t[i]
            k1 = self.equations_of_motion(t[i], y[i])
            k2 = self.equations_of_motion(t[i] + h/2, y[i] + h/2 * k1)
            k3 = self.equations_of_motion(t[i] + h/2, y[i] + h/2 * k2)
            k4 = self.equations_of_motion(t[i] + h, y[i] + h * k3)
            y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return y
    
    def simulate(self, x0, p0, t_start, t_max, num_points=1000):
        """运行模拟"""
        y0 = np.array([x0, p0])
        t = np.linspace(t_start, t_max, num_points)
        y = self.runge_kutta_4(y0, t)
        
        x = y[:, 0]
        p = y[:, 1]
        energy = np.array([self.hamiltonian(x_i, p_i) for x_i, p_i in zip(x, p)])
        
        return t, x, p, energy
    
    def plot_results(self, t, x, p, energy, save_path=None):
        """绘制结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 位置随时间变化
        axes[0, 0].plot(t, x, 'b-')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('位置')
        axes[0, 0].set_title('位置随时间变化')
        axes[0, 0].grid(True)
        
        # 动量随时间变化
        axes[0, 1].plot(t, p, 'r-')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('动量')
        axes[0, 1].set_title('动量随时间变化')
        axes[0, 1].grid(True)
        
        # 相图 (位置 vs 动量)
        axes[1, 0].plot(x, p, 'g-')
        axes[1, 0].set_xlabel('位置')
        axes[1, 0].set_ylabel('动量')
        axes[1, 0].set_title('相图')
        axes[1, 0].grid(True)
        
        # 能量随时间变化
        axes[1, 1].plot(t, energy, 'm-')
        axes[1, 1].set_xlabel('时间')
        axes[1, 1].set_ylabel('能量')
        axes[1, 1].set_title('能量随时间变化')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def animate_motion(self, t, x, p, save_path=None):
        """创建粒子运动的动画"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 设置坐标轴范围
        x_min, x_max = min(x) - 0.1, max(x) + 0.1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.1, 0.1)
        
        # 绘制势能曲线
        x_range = np.linspace(x_min, x_max, 1000)
        potential = self.potential(x_range)
        potential_norm = 0.1 * potential / np.max(potential)
        ax.plot(x_range, potential_norm, 'r-', alpha=0.5, label='势能 (归一化)')
        
        # 创建粒子
        particle, = ax.plot([], [], 'bo', markersize=10, label='粒子')
        
        # 添加文本显示时间、位置和动量
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        pos_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        mom_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        
        ax.legend(loc='upper right')
        ax.set_xlabel('位置')
        ax.set_title('库伦势场中粒子的运动')
        ax.grid(True)
        
        def init():
            particle.set_data([], [])
            time_text.set_text('')
            pos_text.set_text('')
            mom_text.set_text('')
            return particle, time_text, pos_text, mom_text
        
        def animate(i):
            particle.set_data([x[i]], [0])
            time_text.set_text(f'时间 = {t[i]:.2f}')
            pos_text.set_text(f'位置 = {x[i]:.4f}')
            mom_text.set_text(f'动量 = {p[i]:.4f}')
            return particle, time_text, pos_text, mom_text
        
        # 创建动画
        ani = FuncAnimation(fig, animate, frames=len(t),
                            init_func=init, blit=True, interval=20)
        
        if save_path:
            ani.save(save_path, writer='pillow', fps=30)
        
        plt.show()
        return ani

# 示例使用
if __name__ == "__main__":
    # 创建模拟器实例
    simulator = CoulombPotentialParticle(k=2.3019*10**(-28), m=2*10**(-27), q=1.0, Q=1.0)
    
    # 设置初始条件
    x0 = 5.0*10**(-11)  # 初始位置
    p0 = -1.79*10**(-21 ) # 初始动量
    
    # 运行模拟
    t, x, p, energy = simulator.simulate(x0, p0, t_start=0, t_max=1.24125*10**(-17), num_points=2000)
    
    # 绘制结果
    simulator.plot_results(t, x, p, energy, save_path="coulomb_potential_results.png")
    
    # 打印能量守恒情况
    energy_change = np.max(energy) - np.min(energy)
    print(f"初始能量: {energy[0]:.6f}")
    print(f"最终能量: {energy[-1]:.6f}")
    print(f"能量变化: {energy_change:.6e}")
    print(f"相对能量变化: {energy_change/energy[0]:.6e}")

 # 创建DataFrame
df = pd.DataFrame({
    'time': t,
    'position_x': x,
    'momentum_z': p,
})
#指定路径，默认路径奇怪
save_dir = "D:/work/liujie-laserenhance/data/semiclass/"
# 保存为CSV
df.to_csv(save_dir + 'fig3-a-phase1-origin.csv', index=False)

# 保存为Excel（需要安装openpyxl）
# df.to_excel('simulation_results.xlsx', index=False)

print("结果已使用Pandas保存")