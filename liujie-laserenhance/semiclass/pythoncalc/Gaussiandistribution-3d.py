import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

def generate_6d_phase_space_particles(n_particles, mu_r, sigma_r, mu_p, sigma_p):
    """
    生成6维相空间粒子分布（3D位置 + 3D动量）
    """
    # 生成位置分量
    x = np.random.normal(mu_r[0], sigma_r[0], n_particles)
    y = np.random.normal(mu_r[1], sigma_r[1], n_particles)
    z = np.random.normal(mu_r[2], sigma_r[2], n_particles)
    positions = np.column_stack((x, y, z))
    
    # 生成动量分量
    px = np.random.normal(mu_p[0], sigma_p[0], n_particles)
    py = np.random.normal(mu_p[1], sigma_p[1], n_particles)
    pz = np.random.normal(mu_p[2], sigma_p[2], n_particles)
    momenta = np.column_stack((px, py, pz))
    
    return positions, momenta

def plot_distributions(positions, momenta, mu_r, sigma_r, mu_p, sigma_p):
    """
    绘制位置和动量的分布图像
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 位置分布直方图
    axes[0, 0].hist(positions[:, 0], bins=50, density=True, alpha=0.7, color='blue')
    x_range = np.linspace(mu_r[0] - 4*sigma_r[0], mu_r[0] + 4*sigma_r[0], 100)
    axes[0, 0].plot(x_range, norm.pdf(x_range, mu_r[0], sigma_r[0]), 'r-', lw=2)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('概率密度')
    axes[0, 0].set_title('x分布')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(positions[:, 1], bins=50, density=True, alpha=0.7, color='green')
    y_range = np.linspace(mu_r[1] - 4*sigma_r[1], mu_r[1] + 4*sigma_r[1], 100)
    axes[0, 1].plot(y_range, norm.pdf(y_range, mu_r[1], sigma_r[1]), 'r-', lw=2)
    axes[0, 1].set_xlabel('y')
    axes[0, 1].set_ylabel('概率密度')
    axes[0, 1].set_title('y分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].hist(positions[:, 2], bins=50, density=True, alpha=0.7, color='red')
    z_range = np.linspace(mu_r[2] - 4*sigma_r[2], mu_r[2] + 4*sigma_r[2], 100)
    axes[0, 2].plot(z_range, norm.pdf(z_range, mu_r[2], sigma_r[2]), 'r-', lw=2)
    axes[0, 2].set_xlabel('z')
    axes[0, 2].set_ylabel('概率密度')
    axes[0, 2].set_title('z分布')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 动量分布直方图
    axes[1, 0].hist(momenta[:, 0], bins=50, density=True, alpha=0.7, color='blue')
    px_range = np.linspace(mu_p[0] - 4*sigma_p[0], mu_p[0] + 4*sigma_p[0], 100)
    axes[1, 0].plot(px_range, norm.pdf(px_range, mu_p[0], sigma_p[0]), 'r-', lw=2)
    axes[1, 0].set_xlabel('px')
    axes[1, 0].set_ylabel('概率密度')
    axes[1, 0].set_title('px分布')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(momenta[:, 1], bins=50, density=True, alpha=0.7, color='green')
    py_range = np.linspace(mu_p[1] - 4*sigma_p[1], mu_p[1] + 4*sigma_p[1], 100)
    axes[1, 1].plot(py_range, norm.pdf(py_range, mu_p[1], sigma_p[1]), 'r-', lw=2)
    axes[1, 1].set_xlabel('py')
    axes[1, 1].set_ylabel('概率密度')
    axes[1, 1].set_title('py分布')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(momenta[:, 2], bins=50, density=True, alpha=0.7, color='red')
    pz_range = np.linspace(mu_p[2] - 4*sigma_p[2], mu_p[2] + 4*sigma_p[2], 100)
    axes[1, 2].plot(pz_range, norm.pdf(pz_range, mu_p[2], sigma_p[2]), 'r-', lw=2)
    axes[1, 2].set_xlabel('pz')
    axes[1, 2].set_ylabel('概率密度')
    axes[1, 2].set_title('pz分布')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_particle_data(positions, momenta, filename, num_particles_to_save=100):
    """
    保存粒子数据到CSV文件
    """
    # 创建DataFrame
    data = {
        'x': positions[:num_particles_to_save, 0],
        'y': positions[:num_particles_to_save, 1],
        'z': positions[:num_particles_to_save, 2],
        'px': momenta[:num_particles_to_save, 0],
        'py': momenta[:num_particles_to_save, 1],
        'pz': momenta[:num_particles_to_save, 2]
    }
    
    df = pd.DataFrame(data)
    
    # 保存到CSV文件
    df.to_csv(filename, index=False, float_format='%.6f')
    
    # 打印前几个粒子的数据
    print("前10个粒子的坐标和动量数据:")
    print(df.head(10).to_string(index=False))
    
    return df

def main():
    # 设置参数
    n_particles = 10000  # 粒子数量
    
    # 位置参数
    mu_r = [0.0, 0.0, 0.0]  # x, y, z的均值
    sigma_r = [1.0, 0.8, 0.5]  # x, y, z的标准差
    
    # 动量参数
    mu_p = [0.0, 0.0, 0.0]  # px, py, pz的均值
    sigma_p = [0.5, 0.4, 0.3]  # px, py, pz的标准差
    
    print(f"生成 {n_particles} 个6维相空间粒子...")
    
    # 生成6维相空间粒子
    positions, momenta = generate_6d_phase_space_particles(n_particles, mu_r, sigma_r, mu_p, sigma_p)
    
    # 绘制分布图像
    plot_distributions(positions, momenta, mu_r, sigma_r, mu_p, sigma_p)
    
    # 保存粒子数据
    df = save_particle_data(positions, momenta, 'particle_data.csv')
    
    # 打印统计信息
    print("\n统计信息:")
    print(f"x: 均值 = {np.mean(positions[:, 0]):.6f}, 标准差 = {np.std(positions[:, 0]):.6f}")
    print(f"y: 均值 = {np.mean(positions[:, 1]):.6f}, 标准差 = {np.std(positions[:, 1]):.6f}")
    print(f"z: 均值 = {np.mean(positions[:, 2]):.6f}, 标准差 = {np.std(positions[:, 2]):.6f}")
    print(f"px: 均值 = {np.mean(momenta[:, 0]):.6f}, 标准差 = {np.std(momenta[:, 0]):.6f}")
    print(f"py: 均值 = {np.mean(momenta[:, 1]):.6f}, 标准差 = {np.std(momenta[:, 1]):.6f}")
    print(f"pz: 均值 = {np.mean(momenta[:, 2]):.6f}, 标准差 = {np.std(momenta[:, 2]):.6f}")

if __name__ == "__main__":
    main()