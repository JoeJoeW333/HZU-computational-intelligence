import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import time
import os

# 创建输出目录
if not os.path.exists('Baldwinian_tsp_visualizations'):
    os.makedirs('Baldwinian_tsp_visualizations')

class TSPProblem:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.dist_matrix = self._compute_distance_matrix()
        
    def _compute_distance_matrix(self):
        """计算城市间的距离矩阵"""
        dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i+1, self.num_cities):
                dist = np.linalg.norm(self.cities[i] - self.cities[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix
    
    def get_distance(self, i, j):
        """获取两个城市间的距离"""
        return self.dist_matrix[i, j]
    
    def path_length(self, path):
        """计算路径长度"""
        total = 0
        for i in range(len(path) - 1):
            total += self.get_distance(path[i], path[i+1])
        total += self.get_distance(path[-1], path[0])  # 回到起点
        return total

class ACOWithLocalSearch:
    def __init__(self, problem, num_ants=25, evaporation_rate=0.5, alpha=1, beta=2, 
                 local_search_ratio=0.8, max_iter=100):
        self.problem = problem
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha  # 信息素指数
        self.beta = beta    # 启发式信息指数
        self.local_search_ratio = local_search_ratio
        self.max_iter = max_iter
        
        # 初始化信息素矩阵
        self.pheromone = np.ones((problem.num_cities, problem.num_cities)) * 0.1
        
        # 历史记录
        self.best_path_history = []
        self.avg_path_history = []
        self.worst_path_history = []
        self.best_path = None
        self.best_length = float('inf')
        
    def _select_next_city(self, current_city, unvisited):
        """根据信息素和启发式信息选择下一个城市"""
        pheromone = self.pheromone[current_city, unvisited]
        heuristic = 1 / (self.problem.dist_matrix[current_city, unvisited] + 1e-10)
        
        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
        probabilities /= probabilities.sum()
        
        next_idx = np.random.choice(len(unvisited), p=probabilities)
        return unvisited[next_idx]
    
    def construct_solution(self):
        """构建一个解"""
        path = []
        unvisited = list(range(self.problem.num_cities))
        
        # 随机选择起始城市
        current = np.random.choice(unvisited)
        path.append(current)
        unvisited.remove(current)
        
        while unvisited:
            next_city = self._select_next_city(current, unvisited)
            path.append(next_city)
            unvisited.remove(next_city)
            current = next_city
            
        return path
    
    def two_opt_swap(self, path, i, k):
        """执行2-opt交换"""
        new_path = path.copy()
        new_path[i:k+1] = path[i:k+1][::-1]
        return new_path
    
    def two_opt(self, path):
        """2-opt局部搜索"""
        improved = True
        best_path = path
        best_length = self.problem.path_length(path)
        
        while improved:
            improved = False
            for i in range(1, len(path) - 2):
                for k in range(i+1, len(path)):
                    if k - i == 1: continue  # 相邻边，跳过
                    
                    new_path = self.two_opt_swap(best_path, i, k)
                    new_length = self.problem.path_length(new_path)
                    
                    if new_length < best_length:
                        best_path = new_path
                        best_length = new_length
                        improved = True
                        break  # 找到一个改进就重新开始搜索
                if improved:
                    break
                    
        return best_path, best_length
    
    def update_pheromone(self, paths, path_lengths):
        """更新信息素"""
        # 信息素挥发
        self.pheromone *= (1 - self.evaporation_rate)
        
        # 精英策略：只让最好的蚂蚁更新信息素
        best_idx = np.argmin(path_lengths)
        best_path = paths[best_idx]
        
        # 更新信息素
        for i in range(len(best_path) - 1):
            city1, city2 = best_path[i], best_path[i+1]
            self.pheromone[city1, city2] += 1.0 / path_lengths[best_idx]
            self.pheromone[city2, city1] += 1.0 / path_lengths[best_idx]
        
        # 回到起点
        city1, city2 = best_path[-1], best_path[0]
        self.pheromone[city1, city2] += 1.0 / path_lengths[best_idx]
        self.pheromone[city2, city1] += 1.0 / path_lengths[best_idx]
    
    def run(self):
        """运行算法"""
        start_time = time.time()
        convergence_data = []
        
        for iteration in tqdm(range(self.max_iter), desc="Running Baldwinian ACO"):
            paths = []
            path_lengths = []
            improved_paths = []  # 存储改进后的路径（仅用于评估）
            
            # 每只蚂蚁构建路径
            for ant in range(self.num_ants):
                path = self.construct_solution()
                original_length = self.problem.path_length(path)
                
                # 对部分蚂蚁应用局部搜索（仅用于评估，不改变实际路径）
                if np.random.rand() < self.local_search_ratio:
                    improved_path, improved_length = self.two_opt(path)
                    # Baldwinian模式：使用改进后的长度进行评估，但保留原始路径
                    evaluation_length = improved_length
                    improved_paths.append(improved_path)
                else:
                    evaluation_length = original_length
                    improved_paths.append(path)
                
                paths.append(path)  # 保留原始路径（基因型不变）
                path_lengths.append(evaluation_length)  # 使用评估长度（可能经过局部搜索改进）
                
                # 更新全局最优（使用评估长度）
                if evaluation_length < self.best_length:
                    self.best_length = evaluation_length
                    # Baldwinian模式：保存改进后的路径（表现型）作为最优解
                    self.best_path = improved_paths[-1]
            
            # 更新信息素（使用原始路径，但基于评估长度）
            self.update_pheromone(paths, path_lengths)
            
            # 记录历史数据（使用评估长度）
            self.best_path_history.append(self.best_length)
            self.avg_path_history.append(np.mean(path_lengths))
            self.worst_path_history.append(np.max(path_lengths))
            
            # 每5次迭代保存一次路径可视化
            if iteration % 5 == 0:
                self.visualize_path(iteration)
        
        # 生成收敛曲线图
        self.plot_convergence()
        
        # 生成最终结果图
        self.visualize_path('final')
        
        print(f"\nAlgorithm completed in {time.time()-start_time:.2f} seconds")
        print(f"Best path length: {self.best_length:.2f}")
        
        return self.best_path, self.best_length
    
    def visualize_path(self, iteration):
        """可视化当前最优路径"""
        plt.figure(figsize=(10, 8))
        
        # 绘制城市点
        cities = self.problem.cities
        plt.scatter(cities[:, 0], cities[:, 1], c='red', s=50, zorder=5)
        
        # 绘制路径
        path = self.best_path
        for i in range(len(path)):
            start = cities[path[i]]
            end = cities[path[(i+1) % len(path)]]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=1.5)
        
        # 添加箭头表示方向
        for i in range(0, len(path), max(1, len(path)//10)):
            start = cities[path[i]]
            end = cities[path[(i+1) % len(path)]]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            plt.arrow(start[0], start[1], dx*0.9, dy*0.9, 
                      head_width=0.5, head_length=0.7, 
                      fc='green', ec='green', length_includes_head=True)
        
        plt.title(f'Baldwinian TSP Solution (Iteration: {iteration}, Length: {self.best_length:.2f})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.savefig(f'Baldwinian_tsp_visualizations/tsp_iteration_{iteration}.png')
        plt.close()
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(12, 8))
        
        iterations = range(len(self.best_path_history))
        
        plt.plot(iterations, self.best_path_history, 'g-', linewidth=2, label='Best Path')
        plt.plot(iterations, self.avg_path_history, 'b-', linewidth=2, label='Average Path')
        plt.plot(iterations, self.worst_path_history, 'r-', linewidth=2, label='Worst Path')
        
        plt.title('Baldwinian ACO with 2-opt Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Path Length')
        plt.legend()
        plt.grid(True)
        
        # 标注最终结果
        plt.annotate(f'Final Best: {self.best_path_history[-1]:.2f}', 
                    xy=(0.98, 0.05), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
        
        plt.savefig('Baldwinian_tsp_visualizations/convergence_curve.png')
        plt.close()

def generate_cities(num_cities, seed=42):
    """随机生成城市坐标"""
    np.random.seed(seed)
    return np.random.rand(num_cities, 2) * 100

if __name__ == "__main__":
    # 生成城市数据
    num_cities = 50
    cities = generate_cities(num_cities)
    
    # 创建TSP问题实例
    problem = TSPProblem(cities)
    
    # 初始化并运行Baldwinian ACO算法
    aco = ACOWithLocalSearch(problem, 
                             num_ants=30, 
                             evaporation_rate=0.5, 
                             alpha=1, 
                             beta=3,
                             local_search_ratio=0.8,
                             max_iter=100)
    
    best_path, best_length = aco.run()
    
    print("\nVisualizations saved to 'Baldwinian_tsp_visualizations' folder.")