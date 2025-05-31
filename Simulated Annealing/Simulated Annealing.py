"""
0-1背包问题模拟退火算法解决方案
作者: 巫海洲
日期: 2025-05-25
版本: 3.0

本程序使用模拟退火算法(Simulated Annealing)解决0-1背包问题：
1. 随机生成背包问题实例（物品重量、价值、背包容量）
2. 使用模拟退火算法寻找最优解
3. 可视化算法过程和结果

算法特点：
- 自适应初始解生成
- 邻域操作：随机翻转+修复不可行解
- 指数降温策略
- 内循环优化
- 进度条实时监控

关键依赖库：
- numpy
- matplotlib
- tqdm
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import random
import time
from tqdm import tqdm  # 进度条库

# ======================
# 可视化设置
# ======================
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'SimSun']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子保证结果可复现
np.random.seed(42)


# ======================
# 问题生成函数
# ======================
def generate_knapsack_problem(n_items, weight_range=(1, 20), value_range=(1, 50), capacity_ratio=0.5):
    """
    生成随机背包问题实例
    
    参数:
    :param n_items: 物品数量
    :param weight_range: 物品重量范围(最小值, 最大值)
    :param value_range: 物品价值范围(最小值, 最大值)
    :param capacity_ratio: 背包容量占总重量的比例
    
    返回:
    :return: 
        weights: 物品重量列表(一维数组)
        values: 物品价值列表(一维数组)
        capacity: 背包容量
    """
    # 随机生成物品重量 (均匀分布)
    weights = np.random.randint(weight_range[0], weight_range[1] + 1, size=n_items)
    # 随机生成物品价值 (均匀分布)
    values = np.random.randint(value_range[0], value_range[1] + 1, size=n_items)
    # 计算所有物品总重量
    total_weight = np.sum(weights)
    # 按比例计算背包容量
    capacity = int(total_weight * capacity_ratio)
    return weights, values, capacity


# ======================
# 模拟退火算法核心实现
# ======================
def simulated_annealing_knapsack(weights, values, capacity, 
                                 T0=1000, alpha=0.95, 
                                 max_iter=1000, min_temp=1e-5,
                                 inner_loop=50):
    """
    模拟退火算法求解0-1背包问题
    
    参数:
    :param weights: 物品重量列表
    :param values: 物品价值列表
    :param capacity: 背包容量
    :param T0: 初始温度
    :param alpha: 降温系数(0 < alpha < 1)
    :param max_iter: 最大迭代次数(外循环次数)
    :param min_temp: 最低温度(终止条件)
    :param inner_loop: 每个温度下的内循环次数
    
    返回:
    :return: 
        best_solution: 最优解(二进制数组，1表示选择对应物品)
        best_value: 最优解的总价值
        history: 包含算法过程数据的字典
    """
    # 初始化历史记录字典
    history = {
        'best_value': [],     # 每次迭代的最优值
        'current_value': [],  # 当前解的值
        'temperature': [],    # 温度变化
        'accept_rate': []     # 接受率变化
    }
    
    n = len(weights)  # 物品数量
    
    # ----------------------
    # 辅助函数定义
    # ----------------------
    
    def initialize_solution():
        """初始化可行解（保证不超重）"""
        solution = np.zeros(n, dtype=int)  # 初始化为全0
        total_weight = 0  # 当前总重量
        
        # 随机排列物品索引
        indices = list(range(n))
        random.shuffle(indices)
        
        # 随机添加物品直到背包满
        for i in indices:
            if total_weight + weights[i] <= capacity:
                solution[i] = 1  # 选择该物品
                total_weight += weights[i]  # 更新总重量
        return solution
    
    def evaluate(solution):
        """评估解的质量（总价值）"""
        total_value = np.dot(solution, values)  # 计算总价值
        total_weight = np.dot(solution, weights)  # 计算总重量
        # 如果超重则返回负无穷（表示不可行解）
        return total_value if total_weight <= capacity else -np.inf
    
    def get_neighbor(current_solution):
        """生成邻域解（通过随机翻转一个物品的选择状态）"""
        new_solution = current_solution.copy()  # 复制当前解
        idx = np.random.randint(0, n)  # 随机选择一个物品
        # 翻转选择状态（0变1，1变0）
        new_solution[idx] = 1 - new_solution[idx]
        
        # 修复不可行解（如果超重）
        total_weight = np.dot(new_solution, weights)
        while total_weight > capacity:
            # 获取当前已选物品的索引
            selected_indices = np.where(new_solution == 1)[0]
            if len(selected_indices) == 0:
                break  # 没有物品可选，直接退出
            
            # 随机移除一个已选物品
            remove_idx = np.random.choice(selected_indices)
            new_solution[remove_idx] = 0  # 取消选择
            total_weight -= weights[remove_idx]  # 更新总重量
            
        return new_solution
    
    # ----------------------
    # 算法初始化
    # ----------------------
    
    # 生成初始可行解
    current_solution = initialize_solution()
    # 评估初始解
    current_value = evaluate(current_solution)
    # 初始化最优解
    best_solution = current_solution.copy()
    best_value = current_value
    
    # 初始化算法参数
    T = T0  # 当前温度
    total_accept = 0  # 总接受次数
    total_trials = 0  # 总尝试次数
    
    # ----------------------
    # 模拟退火主循环
    # ----------------------
    
    # 使用tqdm创建进度条
    with tqdm(total=max_iter, desc="模拟退火进度") as pbar:
        for iter_count in range(max_iter):
            # 温度终止检查
            if T < min_temp:
                break
                
            inner_accept = 0  # 内循环接受次数
            
            # 内循环：每个温度下进行多次邻域搜索
            for _ in range(inner_loop):
                # 生成邻域解
                new_solution = get_neighbor(current_solution)
                new_value = evaluate(new_solution)
                
                # 接受准则：Metropolis准则
                if new_value > current_value:
                    # 新解更优，直接接受
                    accept = True
                else:
                    # 计算能量差（价值差）
                    delta = new_value - current_value
                    if delta == 0:
                        # 价值相同，50%概率接受
                        accept = random.random() < 0.5
                    else:
                        # 计算接受概率（指数函数）
                        accept_prob = np.exp(delta / T)
                        # 按概率接受劣质解
                        accept = random.random() < accept_prob
                
                # 更新当前解
                if accept:
                    current_solution = new_solution
                    current_value = new_value
                    inner_accept += 1  # 更新内循环接受计数
                    total_accept += 1  # 更新总接受计数
                    
                    # 更新最优解
                    if current_value > best_value:
                        best_solution = current_solution.copy()
                        best_value = current_value
                
                total_trials += 1  # 更新总尝试次数
            
            # ----------------------
            # 记录历史数据
            # ----------------------
            # 计算当前温度下的接受率
            accept_rate = inner_accept / inner_loop if inner_loop > 0 else 0
            # 记录各项指标
            history['best_value'].append(best_value)
            history['current_value'].append(current_value)
            history['temperature'].append(T)
            history['accept_rate'].append(accept_rate)
            
            # ----------------------
            # 降温操作
            # ----------------------
            T *= alpha  # 指数降温
            
            # ----------------------
            # 更新进度条
            # ----------------------
            pbar.update(1)  # 更新进度
            # 设置进度条后缀信息
            pbar.set_postfix({
                '温度': f"{T:.2f}",
                '当前值': current_value,
                '最优值': best_value,
                '接受率': f"{accept_rate:.2%}"
            })
    
    return best_solution, best_value, history


# ======================
# 结果可视化函数
# ======================
def visualize_results(history, weights, values, solution, capacity):
    """
    可视化算法过程和结果
    
    参数:
    :param history: 算法历史记录字典
    :param weights: 物品重量数组
    :param values: 物品价值数组
    :param solution: 最终解（二进制数组）
    :param capacity: 背包容量
    """
    # 创建大画布
    plt.figure(figsize=(15, 10))
    
    # ----------------------
    # 子图1: 价值变化曲线
    # ----------------------
    plt.subplot(2, 2, 1)
    plt.plot(history['best_value'], 'g-', label='最优值')
    plt.plot(history['current_value'], 'b-', alpha=0.5, label='当前值')
    plt.xlabel('迭代次数')
    plt.ylabel('背包价值')
    plt.title('价值变化曲线')
    plt.legend()
    plt.grid(True)
    
    # ----------------------
    # 子图2: 温度变化曲线
    # ----------------------
    plt.subplot(2, 2, 2)
    # 使用半对数坐标（Y轴对数）
    plt.semilogy(history['temperature'], 'r-')
    plt.xlabel('迭代次数')
    plt.ylabel('温度(对数尺度)')
    plt.title('温度衰减曲线')
    plt.grid(True)
    
    # ----------------------
    # 子图3: 接受率变化
    # ----------------------
    plt.subplot(2, 2, 3)
    plt.plot(history['accept_rate'], 'm-')
    plt.xlabel('迭代次数')
    plt.ylabel('接受率')
    plt.title('邻域解接受率')
    plt.grid(True)
    
    # ----------------------
    # 子图4: 物品价值-重量分布
    # ----------------------
    plt.subplot(2, 2, 4)
    # 区分选中和未选中的物品
    selected = solution == 1
    not_selected = solution == 0
    
    # 绘制选中的物品（绿色圆圈）
    plt.scatter(weights[selected], values[selected], 
                c='g', marker='o', s=50, label='选中物品')
    # 绘制未选中的物品（红色叉号）
    plt.scatter(weights[not_selected], values[not_selected], 
                c='r', marker='x', s=50, label='未选物品')
    
    # 绘制背包容量参考线（蓝色虚线）
    plt.axvline(x=capacity, color='b', linestyle='--', 
                linewidth=2, label='背包容量')
    
    # 计算价值密度（价值/重量）
    density = values / weights
    # 按价值密度排序
    sorted_indices = np.argsort(density)
    sorted_weights = weights[sorted_indices]
    sorted_values = values[sorted_indices]
    
    # 绘制价值密度参考线（灰色虚线）
    for i in range(1, len(weights)):
        plt.plot([sorted_weights[i-1], sorted_weights[i]], 
                 [sorted_values[i-1], sorted_values[i]], 
                 'k--', alpha=0.1)
    
    plt.xlabel('物品重量')
    plt.ylabel('物品价值')
    plt.title('物品分布（颜色表示是否被选中）')
    plt.legend()
    plt.grid(True)
    
    # 调整子图布局
    plt.tight_layout()
    plt.show()


# ======================
# 主函数
# ======================
def main():
    """主函数：执行算法流程"""
    # ======================
    # 参数设置
    # ======================
    n_items = 100          # 物品数量
    weight_range = (1, 20)  # 物品重量范围
    value_range = (1, 50)   # 物品价值范围
    capacity_ratio = 0.5    # 背包容量占总重比例
    
    # 模拟退火算法参数（参考教材和文献）
    T0 = 1000              # 初始温度（基于目标函数值范围）
    alpha = 0.95           # 降温系数（文献常用值）
    max_iter = 500         # 最大迭代次数
    min_temp = 1e-5        # 终止温度
    inner_loop = 100       # 每个温度下的内循环次数
    
    # ======================
    # 问题生成
    # ======================
    weights, values, capacity = generate_knapsack_problem(
        n_items, weight_range, value_range, capacity_ratio
    )
    
    # 打印问题信息
    print("="*50)
    print(f"0-1背包问题参数（物品数: {n_items}, 背包容量: {capacity})")
    print(f"物品总重量: {np.sum(weights)}")
    print(f"物品总价值: {np.sum(values)}")
    print("="*50)
    
    # ======================
    # 算法执行
    # ======================
    start_time = time.time()  # 记录开始时间
    # 运行模拟退火算法
    best_solution, best_value, history = simulated_annealing_knapsack(
        weights, values, capacity, T0, alpha, max_iter, min_temp, inner_loop
    )
    runtime = time.time() - start_time  # 计算运行时间
    
    # ======================
    # 结果分析
    # ======================
    total_weight = np.dot(best_solution, weights)  # 计算选中物品总重量
    selected_items = np.sum(best_solution)  # 计算选中物品数量
    
    # 打印结果摘要
    print("\n" + "="*50)
    print("算法结果:")
    print(f"最优解价值: {best_value}")
    print(f"背包使用重量: {total_weight}/{capacity} ({total_weight/capacity:.2%})")
    print(f"选中物品数量: {selected_items}/{n_items}")
    print(f"运行时间: {runtime:.2f}秒")
    print("="*50)
    
    # ======================
    # 结果可视化
    # ======================
    visualize_results(history, weights, values, best_solution, capacity)


# 程序入口
if __name__ == "__main__":
    main()