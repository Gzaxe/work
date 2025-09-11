import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

def box_muller_transform(n_samples):
    """
    使用Box-Muller变换生成标准正态分布随机数
    """
    # 生成均匀分布的随机数
    u1 = np.random.rand(n_samples)
    u2 = np.random.rand(n_samples)
    
    # Box-Muller变换
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)
    
    return z0, z1

def generate_phase_space_particles(n_particles, mu_x, sigma_x, mu_px, sigma_px):
    """
    生成相空间粒子分布
    
    参数:
    n_particles: 粒子数量
    mu_x, sigma_x: x方向的均值和标准差
    mu_px, sigma_px: px方向的均值和标准差
    
    返回:
    x, px: 粒子的位置和动量数组
    """
    # 生成标准正态分布随机数
    z_x, z_px = box_muller_transform(n_particles)
    
    # 转换为指定均值和标准差的高斯分布
    x = mu_x + sigma_x * z_x
    px = mu_px + sigma_px * z_px
    
    return x, px

def verify_distribution(x, px, mu_x, sigma_x, mu_px, sigma_px):
    """
    验证生成的分布是否符合预期
    """
    print("验证结果:")
    print(f"x的理论均值: {mu_x}, 实际均值: {np.mean(x):.6f}")
    print(f"x的理论标准差: {sigma_x}, 实际标准差: {np.std(x):.6f}")
    print(f"px的理论均值: {mu_px}, 实际均值: {np.mean(px):.6f}")
    print(f"px的理论标准差: {sigma_px}, 实际标准差: {np.std(px):.6f}")
    
    # 计算相关系数（理论上应该接近0）
    correlation = np.corrcoef(x, px)[0, 1]
    print(f"x和px的相关系数: {correlation:.6f}")

def plot_results(x, px, mu_x, sigma_x, mu_px, sigma_px):
    """
    绘制结果图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # x分布直方图
    axes[0, 0].hist(x, bins=50, density=True, alpha=0.7, color='blue')
    x_range = np.linspace(mu_x - 4*sigma_x, mu_x + 4*sigma_x, 100)
    axes[0, 0].plot(x_range, norm.pdf(x_range, mu_x, sigma_x), 'r-', lw=2)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('概率密度')
    axes[0, 0].set_title('x分布直方图')
    axes[0, 0].grid(True, alpha=0.3)
    
    # px分布直方图
    axes[0, 1].hist(px, bins=50, density=True, alpha=0.7, color='green')
    px_range = np.linspace(mu_px - 4*sigma_px, mu_px + 4*sigma_px, 100)
    axes[0, 1].plot(px_range, norm.pdf(px_range, mu_px, sigma_px), 'r-', lw=2)
    axes[0, 1].set_xlabel('px')
    axes[0, 1].set_ylabel('概率密度')
    axes[0, 1].set_title('px分布直方图')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 相空间散点图
    axes[1, 0].scatter(x, px, s=1, alpha=0.5, color='purple')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('px')
    axes[1, 0].set_title('相空间分布 (x-px)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 二维直方图（相空间密度）
    hist, x_edges, px_edges = np.histogram2d(x, px, bins=50, density=True)
    extent = [x_edges[0], x_edges[-1], px_edges[0], px_edges[-1]]
    im = axes[1, 1].imshow(hist.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('px')
    axes[1, 1].set_title('相空间密度分布')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()

def main():
    # 设置参数
    n_particles = 100000  # 粒子数量
    mu_x, sigma_x = 0.0, 1.0  # x方向的均值和标准差
    mu_px, sigma_px = 0.0, 0.5  # px方向的均值和标准差
    
    print(f"生成 {n_particles} 个粒子...")
    start_time = time.time()
    
    # 生成相空间粒子
    x, px = generate_phase_space_particles(n_particles, mu_x, sigma_x, mu_px, sigma_px)
    
    end_time = time.time()
    print(f"生成完成，耗时: {end_time - start_time:.4f} 秒")
    
    # 验证分布
    verify_distribution(x, px, mu_x, sigma_x, mu_px, sigma_px)
    
    # 绘制结果
    plot_results(x, px, mu_x, sigma_x, mu_px, sigma_px)
    
    # 保存部分数据（前1000个粒子）
    np.savetxt('phase_space_particles.csv', 
               np.column_stack((x[:1000], px[:1000])), 
               delimiter=',', 
               header='x,px', 
               comments='')

if __name__ == "__main__":
    main()