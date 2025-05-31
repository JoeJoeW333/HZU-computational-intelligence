import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import pandas as pd
from matplotlib.ticker import MaxNLocator

# 生成500个城市的坐标
def generate_cities(n_cities=500):
    np.random.seed(42)
    cities = np.random.rand(n_cities, 2) * 100
    return cities

# 计算距离矩阵
def compute_distance_matrix(cities):
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dist_matrix

# 计算路径长度
def path_length(path, dist_matrix):
    total = 0
    n = len(path)
    for i in range(n - 1):
        total += dist_matrix[path[i]][path[i + 1]]
    total += dist_matrix[path[-1]][path[0]]  # 回到起点
    return total

# 生成初始解
def generate_initial_solution(n_cities):
    path = list(range(n_cities))
    random.shuffle(path)
    return path

# 生成邻域解（交换两个城市）
def generate_neighbors(path, tabu_list, tabu_tenure, best_value, dist_matrix, n_neighbors=1000):
    neighbors = []
    n = len(path)
    for _ in range(n_neighbors):
        i, j = random.sample(range(n), 2)
        if i > j:
            i, j = j, i
        new_path = path.copy()
        new_path[i], new_path[j] = new_path[j], new_path[i]  # 交换两个城市
        
        # 计算移动的特征值（用于禁忌判断）
        move = (min(path[i], path[j]), max(path[i], path[j]))
        
        # 计算新路径长度
        new_length = path_length(new_path, dist_matrix)
        
        # 检查是否满足渴望准则
        aspiration = new_length < best_value
        
        # 检查是否在禁忌表中
        is_tabu = move in tabu_list and tabu_list[move] > 0
        
        neighbors.append((new_path, move, new_length, is_tabu and not aspiration))
    
    return neighbors

# 更新禁忌表
def update_tabu_list(tabu_list, tabu_tenure):
    expired_moves = []
    for move, tenure in tabu_list.items():
        tabu_list[move] = tenure - 1
        if tabu_list[move] <= 0:
            expired_moves.append(move)
    for move in expired_moves:
        del tabu_list[move]
    return tabu_list

# 禁忌搜索主函数
def tabu_search(dist_matrix, max_iter=1000, tabu_tenure=20, n_neighbors=1000):
    n_cities = dist_matrix.shape[0]
    current_path = generate_initial_solution(n_cities)
    current_length = path_length(current_path, dist_matrix)
    best_path = current_path.copy()
    best_length = current_length
    
    tabu_list = {}  # 禁忌表：键为移动元组(city_i, city_j)，值为禁忌剩余代数
    history = []    # 记录每次迭代的最优值
    current_history = []  # 记录每次迭代的当前值
    tabu_size_history = []  # 记录禁忌表大小变化
    improvement_history = []  # 记录每次改进的幅度
    
    start_time = time.time()
    
    for iteration in tqdm(range(max_iter), desc="Tabu Search Progress"):
        # 生成邻域解
        neighbors = generate_neighbors(current_path, tabu_list, tabu_tenure, best_length, dist_matrix, n_neighbors)
        
        # 筛选非禁忌解和满足渴望准则的解
        candidates = [n for n in neighbors if not n[3]]
        
        # 如果没有候选解，选择所有解中最好的
        if not candidates:
            candidates = neighbors
            
        # 选择最佳候选解
        best_candidate = min(candidates, key=lambda x: x[2])
        new_path, move, new_length, _ = best_candidate
        
        # 更新当前解
        current_path = new_path
        current_length = new_length
        
        # 更新禁忌表
        tabu_list[move] = tabu_tenure
        tabu_list = update_tabu_list(tabu_list, tabu_tenure)
        
        # 更新全局最优解
        if new_length < best_length:
            improvement = best_length - new_length
            improvement_history.append((iteration, improvement))
            best_path = new_path.copy()
            best_length = new_length
        
        # 记录历史数据
        history.append(best_length)
        current_history.append(current_length)
        tabu_size_history.append(len(tabu_list))
    
    runtime = time.time() - start_time
    
    # 创建性能数据报告
    performance_data = {
        "best_length": best_length,
        "initial_length": history[0],
        "improvement_percent": (history[0] - best_length) / history[0] * 100,
        "runtime": runtime,
        "iterations": max_iter,
        "final_tabu_size": len(tabu_list),
        "improvements_count": len(improvement_history),
        "avg_improvement": np.mean([imp[1] for imp in improvement_history]) if improvement_history else 0,
        "best_iteration": improvement_history[-1][0] if improvement_history else 0,
        "tabu_tenure": tabu_tenure,
        "n_neighbors": n_neighbors
    }
    
    return best_path, best_length, history, current_history, tabu_size_history, improvement_history, performance_data

# 可视化函数 - 最优路径
def plot_optimal_path(cities, path, length):
    plt.figure(figsize=(10, 8))
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', s=15, alpha=0.7)
    
    # 绘制路径
    for i in range(len(path)):
        start = cities[path[i]]
        end = cities[path[(i + 1) % len(path)]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'r-', lw=0.8, alpha=0.5)
    
    # 标记起点
    plt.scatter(cities[path[0]][0], cities[path[0]][1], c='green', s=100, marker='*', edgecolors='black', label='Start/End')
    
    plt.title(f"Optimal TSP Path for {len(cities)} Cities\nPath Length: {length:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 可视化函数 - 收敛曲线
def plot_convergence(history, current_history):
    plt.figure(figsize=(12, 6))
    
    # 历史最优值
    plt.plot(history, 'b-', label='Best Solution', lw=1.5)
    
    # 当前解值
    plt.plot(current_history, 'g-', alpha=0.5, label='Current Solution', lw=0.8)
    
    plt.title("Convergence History")
    plt.xlabel("Iteration")
    plt.ylabel("Path Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加最后值标记
    plt.annotate(f'Final: {history[-1]:.2f}', 
                xy=(len(history)-1, history[-1]), 
                xytext=(len(history)*0.7, history[0]*0.9),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.show()

# 可视化函数 - 禁忌表大小变化
def plot_tabu_size(tabu_size_history):
    plt.figure(figsize=(12, 6))
    
    plt.plot(tabu_size_history, 'm-', label='Tabu List Size', lw=1.5)
    plt.title("Tabu List Size Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Tabu Moves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加平均大小标记
    avg_size = np.mean(tabu_size_history)
    plt.axhline(y=avg_size, color='r', linestyle='--', label=f'Average Size: {avg_size:.1f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 可视化函数 - 改进历史
def plot_improvement_history(improvement_history):
    if not improvement_history:
        print("No improvement history to plot")
        return
        
    plt.figure(figsize=(12, 6))
    
    iterations, improvements = zip(*improvement_history)
    plt.bar(iterations, improvements, color='c', alpha=0.7, label='Improvement')
    
    # 添加改进趋势线
    if len(improvements) > 1:
        z = np.polyfit(iterations, improvements, 1)
        p = np.poly1d(z)
        plt.plot(iterations, p(iterations), 'r--', lw=2, label='Improvement Trend')
    
    plt.title("Solution Improvement History")
    plt.xlabel("Iteration")
    plt.ylabel("Improvement Amount")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加总改进标记
    total_improvement = sum(improvements)
    plt.annotate(f'Total Improvement: {total_improvement:.2f}', 
                xy=(iterations[-1], improvements[-1]), 
                xytext=(iterations[-1]*0.7, max(improvements)*0.8),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.show()

# 可视化函数 - 算法参数
def plot_algorithm_parameters(performance_data):
    labels = ['Initial Length', 'Best Length', 'Improvement %', 'Runtime (s)']
    values = [
        performance_data['initial_length'],
        performance_data['best_length'],
        performance_data['improvement_percent'],
        performance_data['runtime']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=['blue', 'green', 'red', 'purple'])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.set_title('Algorithm Performance Summary')
    ax.set_ylabel('Value')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    print("Generating 500 cities...")
    cities = generate_cities(500)
    print("Computing distance matrix...")
    dist_matrix = compute_distance_matrix(cities)
    
    # 算法参数
    max_iter = 500
    tabu_tenure = 20
    n_neighbors = 1000
    
    print(f"\nStarting Tabu Search with parameters:")
    print(f"- Max iterations: {max_iter}")
    print(f"- Tabu tenure: {tabu_tenure}")
    print(f"- Neighbors per iteration: {n_neighbors}")
    
    # 运行禁忌搜索
    results = tabu_search(
        dist_matrix, 
        max_iter=max_iter, 
        tabu_tenure=tabu_tenure,
        n_neighbors=n_neighbors
    )
    best_path, best_length, history, current_history, tabu_size_history, improvement_history, performance_data = results
    
    # 打印性能摘要
    print("\n=== Algorithm Performance Summary ===")
    print(f"Initial path length: {performance_data['initial_length']:.2f}")
    print(f"Best path length: {performance_data['best_length']:.2f}")
    print(f"Improvement: {performance_data['improvement_percent']:.2f}%")
    print(f"Runtime: {performance_data['runtime']:.2f} seconds")
    print(f"Number of improvements: {performance_data['improvements_count']}")
    print(f"Average improvement: {performance_data['avg_improvement']:.4f}")
    print(f"Final tabu list size: {performance_data['final_tabu_size']}")
    
    # 生成可视化图表
    print("\nGenerating visualizations...")
    plot_optimal_path(cities, best_path, best_length)
    plot_convergence(history, current_history)
    plot_tabu_size(tabu_size_history)
    plot_improvement_history(improvement_history)
    plot_algorithm_parameters(performance_data)
    
    # 保存性能数据
    df = pd.DataFrame({
        'Iteration': range(len(history)),
        'Best_Length': history,
        'Current_Length': current_history,
        'Tabu_Size': tabu_size_history
    })
    df.to_csv('tabu_search_performance.csv', index=False)
    print("Performance data saved to 'tabu_search_performance.csv'")