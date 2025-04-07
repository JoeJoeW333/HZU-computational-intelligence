import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List

def load_cities_from_csv(file_path: str) -> np.ndarray:
    """从CSV文件加载城市坐标"""
    df = pd.read_csv(file_path, index_col='city_id')
    return df[['x', 'y']].values

def calculate_distances(cities: np.ndarray) -> np.ndarray:
    """计算城市间距离矩阵"""
    n = len(cities)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(cities[i] - cities[j])
    return distances

class ACO_TSP:
    def __init__(self, cities: np.ndarray, distance_matrix: np.ndarray):
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.num_cities = len(cities)

    def run(self, params: dict) -> (List[int], float, List[float]):
        """执行蚁群算法"""
        # 参数初始化
        ant_count = params.get('ant_count', 50)
        alpha = params.get('alpha', 1)
        beta = params.get('beta', 3)
        rho = params.get('rho', 0.5)
        Q = params.get('Q', 100)
        iterations = params.get('iterations', 100)

        # 算法变量初始化
        pheromones = np.ones((self.num_cities, self.num_cities)) / self.num_cities
        best_path = None
        best_distance = float('inf')
        history = []
        
        print("Starting Ant Colony Optimization...")
        start_time = time.time()

        for iteration in range(iterations):
            iter_start = time.time()
            paths = []
            distances = []

            # 蚂蚁路径构建
            for _ in range(ant_count):
                current = np.random.randint(self.num_cities)
                path = [current]
                distance = 0.0
                
                while len(path) < self.num_cities:
                    unvisited = [c for c in range(self.num_cities) if c not in path]
                    probabilities = []
                    
                    for city in unvisited:
                        pheromone = pheromones[current, city]
                        heuristic = 1 / (self.distance_matrix[current, city] + 1e-10)
                        probabilities.append((pheromone ** alpha) * (heuristic ** beta))
                    
                    probabilities = np.array(probabilities)
                    if np.sum(probabilities) == 0:
                        probabilities = np.ones_like(probabilities) / len(probabilities)
                    else:
                        probabilities /= probabilities.sum()
                    
                    next_city = np.random.choice(unvisited, p=probabilities)
                    path.append(next_city)
                    distance += self.distance_matrix[current, next_city]
                    current = next_city

                # 闭合路径
                distance += self.distance_matrix[path[-1], path[0]]
                paths.append(path)
                distances.append(distance)

            # 信息素更新
            pheromones *= (1 - rho)
            for path, dist in zip(paths, distances):
                for i in range(len(path)-1):
                    pheromones[path[i], path[i+1]] += Q / dist
                    pheromones[path[i+1], path[i]] += Q / dist
                pheromones[path[-1], path[0]] += Q / dist
                pheromones[path[0], path[-1]] += Q / dist

            # 更新最佳路径
            current_best = np.argmin(distances)
            if distances[current_best] < best_distance:
                best_distance = distances[current_best]
                best_path = paths[current_best]
            
            history.append(best_distance)
            
            # 进度输出
            if iteration % 50 == 0 or iteration == iterations-1:
                print(f"Iter {iteration+1:4d}/{iterations} | Best: {best_distance:.2f} | "
                      f"Time: {time.time()-iter_start:.2f}s")

        print(f"\nTotal optimization time: {time.time()-start_time:.2f} seconds")
        return best_path, best_distance, history

    def plot_solution(self, path: List[int], distance: float) -> None:
        """可视化解决方案"""
        plt.figure(figsize=(12, 7))
        
        # 绘制城市点
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=60, 
                   edgecolors='black', zorder=2)
        
        # 绘制路径
        for i in range(-1, len(path)-1):
            start = path[i]
            end = path[i+1]
            plt.plot([self.cities[start, 0], self.cities[end, 0]],
                     [self.cities[start, 1], self.cities[end, 1]], 
                     'b-', linewidth=1, alpha=0.8, zorder=1)
        
        # 添加标注
        for i, (x, y) in enumerate(self.cities):
            plt.text(x, y, str(i), fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.title(f"ACO TSP Solution (Distance: {distance:.2f})", fontsize=14)
        plt.xlabel("X Coordinate", fontsize=12)
        plt.ylabel("Y Coordinate", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 数据加载
    try:
        cities = load_cities_from_csv('cities_100.csv')
    except FileNotFoundError:
        print("Warning: Using random cities as 'cities_100.csv' not found")
        np.random.seed(42)
        cities = np.random.rand(100, 2) * 100  # 生成100个随机城市

    # 计算距离矩阵
    distance_matrix = calculate_distances(cities)
    
    # 初始化并运行算法
    aco = ACO_TSP(cities, distance_matrix)
    best_path, best_dist, history = aco.run({
        'ant_count': 100,
        'alpha': 1.2,
        'beta': 4.0,
        'rho': 0.4,
        'Q': 150,
        'iterations': 10000
    })
    
    # 结果可视化
    aco.plot_solution(best_path, best_dist)
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history, 'b-', linewidth=1.5)
    plt.title("Convergence History", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Best Distance", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
