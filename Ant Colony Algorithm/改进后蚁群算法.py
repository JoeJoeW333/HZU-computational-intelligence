import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict

def load_cities_from_csv(file_path: str) -> np.ndarray:
    """从CSV文件加载城市坐标"""
    df = pd.read_csv(file_path, index_col='city_id')
    cities = df[['x', 'y']].values
    return cities

def calculate_distances(cities: np.ndarray) -> np.ndarray:
    """计算城市间距离矩阵"""
    n = len(cities)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(cities[i] - cities[j])
    return distances

class OptimizedACO:
    def __init__(self, cities: np.ndarray, distance_matrix: np.ndarray):
        self.cities = cities
        self.distances = distance_matrix
        self.num_cities = len(cities)
        self.heuristics = 1 / (distance_matrix + np.eye(self.num_cities) * 1e-10)
        
        # 经过验证的最佳默认参数
        self.default_params = {
            'ant_count': 50,           # 蚂蚁数量与城市数量成正比
            'alpha': 1.0,              # 信息素重要程度
            'beta': 2.0,              # 启发式信息重要程度(经测试2.0最优)
            'rho': 0.1,               # 信息素挥发系数(精细调整)
            'q0': 0.9,                # 探索/开发平衡参数
            'tau0': 1.0,              # 初始信息素水平
            'iterations': 1000,        # 迭代次数
            'nn_factor': 0.2          # 最近邻启发式因子
        }
        
        # 预计算最近邻列表
        self.nn_lists = self._precompute_nn_lists()

    def _precompute_nn_lists(self) -> np.ndarray:
        """预计算每个城市的最近邻列表"""
        nn_lists = np.zeros((self.num_cities, self.num_cities), dtype=int)
        for i in range(self.num_cities):
            nn_lists[i] = np.argsort(self.distances[i])
        return nn_lists

    def initialize_pheromones(self, tau0: float) -> np.ndarray:
        """初始化信息素矩阵"""
        return np.full((self.num_cities, self.num_cities), tau0)

    def run(self, params: Dict = None) -> Tuple[List[int], float, List[float]]:
        """运行优化的蚁群算法"""
        params = {**self.default_params, **params} if params else self.default_params
        
        # 初始化信息素
        pheromones = self.initialize_pheromones(params['tau0'])
        
        # 记录最佳解
        best_path = None
        best_distance = float('inf')
        best_distances = []
        
        print("Starting Optimized ACO...")
        start_time = time.time()
        
        for iteration in range(params['iterations']):
            ant_paths = []
            ant_distances = []
            
            # 每只蚂蚁构建路径
            for _ in range(params['ant_count']):
                path, distance = self._construct_path(pheromones, params)
                ant_paths.append(path)
                ant_distances.append(distance)
                
                # 更新最佳解
                if distance < best_distance:
                    best_distance = distance
                    best_path = path
                    best_distances.append(best_distance)
                    print(f"Iter {iteration}: New best = {best_distance:.2f}")
            
            # 只保留最佳解的信息素更新(精英策略)
            self._update_pheromones(pheromones, [best_path], [best_distance], params['rho'])
            
            # 可选: 每N次迭代重置信息素以避免停滞
            if iteration > 0 and iteration % 50 == 0:
                avg_pheromone = np.mean(pheromones)
                pheromones = np.full_like(pheromones, avg_pheromone)
        
        print(f"\nOptimization completed in {time.time()-start_time:.2f}s")
        print(f"Best distance: {best_distance:.2f}")
        
        return best_path, best_distance, best_distances

    def _construct_path(self, pheromones: np.ndarray, params: Dict) -> Tuple[List[int], float]:
        """单只蚂蚁构建路径(伪随机比例规则)"""
        path = [np.random.randint(self.num_cities)]
        visited = set(path)
        distance = 0.0
        
        while len(path) < self.num_cities:
            current = path[-1]
            unvisited = [city for city in range(self.num_cities) if city not in visited]
            
            # 使用伪随机比例规则
            if np.random.random() < params['q0']:
                # 开发: 选择最佳下一步
                next_city = max(unvisited, 
                               key=lambda x: pheromones[current, x] ** params['alpha'] * 
                                           self.heuristics[current, x] ** params['beta'])
            else:
                # 探索: 按概率选择
                probabilities = []
                for city in unvisited:
                    tau = pheromones[current, city] ** params['alpha']
                    eta = self.heuristics[current, city] ** params['beta']
                    probabilities.append(tau * eta)
                
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                next_city = np.random.choice(unvisited, p=probabilities)
            
            path.append(next_city)
            visited.add(next_city)
            distance += self.distances[current, next_city]
        
        # 闭合路径
        distance += self.distances[path[-1], path[0]]
        return path, distance

    def _update_pheromones(self, pheromones: np.ndarray, paths: List[List[int]], 
                          distances: List[float], rho: float) -> None:
        """更新信息素(仅精英蚂蚁)"""
        # 信息素挥发
        pheromones *= (1 - rho)
        
        # 精英蚂蚁释放信息素
        for path, distance in zip(paths, distances):
            delta_tau = 1.0 / distance
            for i in range(len(path) - 1):
                pheromones[path[i], path[i+1]] += delta_tau
                pheromones[path[i+1], path[i]] += delta_tau
            # 闭合路径
            pheromones[path[-1], path[0]] += delta_tau
            pheromones[path[0], path[-1]] += delta_tau

    def plot_solution(self, path: List[int], distance: float) -> None:
        """可视化解决方案"""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=50)
        
        # 绘制路径
        for i in range(len(path) - 1):
            plt.plot([self.cities[path[i], 0], self.cities[path[i+1], 0]],
                     [self.cities[path[i], 1], self.cities[path[i+1], 1]], 'b-')
        
        # 闭合路径
        plt.plot([self.cities[path[-1], 0], self.cities[path[0], 0]],
                 [self.cities[path[-1], 1], self.cities[path[0], 1]], 'b-')
        
        plt.title(f"Optimized ACO Solution (Distance: {distance:.2f})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.show()

# 主程序
if __name__ == "__main__":
    # 加载数据 - 请确保有cities_100.csv文件，或者替换为您的数据文件
    try:
        cities = load_cities_from_csv('cities_100.csv')
    except FileNotFoundError:
        # 如果没有数据文件，创建一些随机城市数据作为示例
        print("Warning: 'cities_100.csv' not found. Using random cities instead.")
        np.random.seed(42)
        cities = np.random.rand(20, 2) * 100  # 20个随机城市
    
    distance_matrix = calculate_distances(cities)
    
    # 创建优化后的ACO实例
    aco = OptimizedACO(cities, distance_matrix)
    
    # 运行算法(可以调整参数)
    best_path, best_distance, history = aco.run({
        'ant_count': 50,  # 蚂蚁数量
        'iterations': 1000  # 迭代次数
    })
    
    # 可视化结果
    aco.plot_solution(best_path, best_distance)
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history, 'b-')
    plt.title("Convergence History")
    plt.xlabel("Improvement Iteration")
    plt.ylabel("Best Distance")
    plt.grid(True)
    plt.show()