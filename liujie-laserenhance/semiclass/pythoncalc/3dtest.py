#测试三维的计算
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm

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

class CoulombPotentialParticle3D:
    """三维直角坐标下库伦势场中粒子运动的数值模拟"""
    
    def __init__(self, k=1.0, m=1.0, q=1.0, Q=1.0):
        self.k = k  # 库伦常数
        self.m = m  # 粒子质量
        self.q = q  # 粒子电荷
        self.Q = Q  # 中心电荷
        
    def distance(self, r):
        """计算位置向量的模"""
        return np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    
    def potential(self, r):
        """计算库伦势能"""
        r_mag = self.distance(r)
        # 避免除以零错误
        epsilon = 1e-16
        return self.k * self.q * self.Q / (r_mag + epsilon)
    
    def force(self, r):
        """计算库伦力（三维向量）"""
        r_mag = self.distance(r)
        # 避免除以零错误
        epsilon = 1e-16
        force_magnitude = self.k * self.q * self.Q / (r_mag**2 + epsilon)
        # 力的方向与位置向量方向相反
        force_vector = force_magnitude * r / (r_mag + epsilon)
        return force_vector
    
    def hamiltonian(self, r, p):
        """计算哈密顿量 (总能量)"""
        # 动能 + 势能
        kinetic_energy = np.sum(p**2) / (2 * self.m)
        potential_energy = self.potential(r)
        return kinetic_energy + potential_energy
    
    def equations_of_motion(self, t, y):
        """哈密顿方程 - 三维形式"""
        # y 包含6个分量：[x, y, z, px, py, pz]
        r = y[:3]  # 位置向量
        p = y[3:]  # 动量向量
        
        # dr/dt = p/m
        drdt = p / self.m
        
        # dp/dt = F = -∇V
        dpdt = self.force(r)
        
        return np.concatenate((drdt, dpdt))
    
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
    
    def simulate(self, r0, p0, t_max, num_points=1000):
        """运行模拟"""
        # 初始条件：位置向量和动量向量
        y0 = np.concatenate((r0, p0))
        t = np.linspace(0, t_max, num_points)
        
        # 使用RK4求解
        y = self.runge_kutta_4(y0, t)
        
        # 提取结果
        r = y[:, :3]  # 位置向量
        p = y[:, 3:]  # 动量向量
        
        # 计算位置模长和动量模长
        r_mag = np.sqrt(np.sum(r**2, axis=1))
        p_mag = np.sqrt(np.sum(p**2, axis=1))
        
        # 计算能量
        energy = np.array([self.hamiltonian(r_i, p_i) for r_i, p_i in zip(r, p)])
        
        return t, r, p, r_mag, p_mag, energy
    
    def plot_results(self, t, r_mag, p_mag, energy, save_path=None):
        """绘制结果：坐标模大小-时间和动量模大小-时间图像"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 位置模长随时间变化
        axes[0, 0].plot(t, r_mag, 'b-')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('位置模长')
        axes[0, 0].set_title('位置模长随时间变化')
        axes[0, 0].grid(True)
        
        # 动量模长随时间变化
        axes[0, 1].plot(t, p_mag, 'r-')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('动量模长')
        axes[0, 1].set_title('动量模长随时间变化')
        axes[0, 1].grid(True)
        
        # 能量随时间变化
        axes[1, 0].plot(t, energy, 'm-')
        axes[1, 0].set_xlabel('时间')
        axes[1, 0].set_ylabel('能量')
        axes[1, 0].set_title('能量随时间变化')
        axes[1, 0].grid(True)
        
        # 相图 (位置模长 vs 动量模长)
        axes[1, 1].plot(r_mag, p_mag, 'g-')
        axes[1, 1].set_xlabel('位置模长')
        axes[1, 1].set_ylabel('动量模长')
        axes[1, 1].set_title('相图 (模长)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# 示例使用
if __name__ == "__main__":
    # 创建模拟器实例
    simulator = CoulombPotentialParticle3D(k=0.23, m=0.2, q=1.0, Q=1.0)
    
    # 设置初始条件
    # 初始位置向量 (x, y, z)
    r0 = np.array([5.0, 0.0, 0.0])
    # 初始动量向量 (px, py, pz)
    p0 = np.array([-1.8, 0.0, 0.0])
    
    # 运行模拟
    t, r, p, r_mag, p_mag, energy = simulator.simulate(
        r0, p0, t_max=1, num_points=2000
    )
    
    # 绘制结果
    simulator.plot_results(t, r_mag, p_mag, energy, save_path="coulomb_3d_results.png")
    
    # 打印能量守恒情况
    energy_change = np.max(energy) - np.min(energy)
    print(f"初始能量: {energy[0]:.6f}")
    print(f"最终能量: {energy[-1]:.6f}")
    print(f"能量变化: {energy_change:.6e}")
    print(f"相对能量变化: {energy_change/energy[0]:.6e}")
    
    # 打印轨迹的一些统计信息
    print(f"最小位置模长: {np.min(r_mag):.4f}")
    print(f"最大位置模长: {np.max(r_mag):.4f}")
    print(f"最小动量模长: {np.min(p_mag):.4f}")
    print(f"最大动量模长: {np.max(p_mag):.4f}")