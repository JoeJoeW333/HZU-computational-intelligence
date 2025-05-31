import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random

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
def generate_neighbors(path, tabu_list, tabu_tenure, best_value, n_neighbors=1000):
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
    
    for iteration in tqdm(range(max_iter), desc="Tabu Search Progress"):
        # 生成邻域解
        neighbors = generate_neighbors(current_path, tabu_list, tabu_tenure, best_length, n_neighbors)
        
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
            best_path = new_path.copy()
            best_length = new_length
        
        # 记录历史最优值
        history.append(best_length)
    
    return best_path, best_length, history

# 可视化结果
def plot_results(cities, path, history):
    plt.figure(figsize=(15, 5))
    
    # 绘制路径图
    plt.subplot(1, 2, 1)
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', s=10)
    for i in range(len(path)):
        start = cities[path[i]]
        end = cities[path[(i + 1) % len(path)]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'r-', lw=0.5)
    plt.title(f"Optimal Path (Length: {history[-1]:.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    # 绘制收敛曲线
    plt.subplot(1, 2, 2)
    plt.plot(history, 'b-')
    plt.title("Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Path Length")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 生成城市和距离矩阵
    cities = generate_cities(500)
    dist_matrix = compute_distance_matrix(cities)
    
    # 运行禁忌搜索
    start_time = time.time()
    best_path, best_length, history = tabu_search(
        dist_matrix, 
        max_iter=500, 
        tabu_tenure=20,
        n_neighbors=1000
    )
    end_time = time.time()
    
    # 打印结果
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
    print(f"Best path length: {best_length:.2f}")
    
    # 可视化结果
    plot_results(cities, best_path, history)