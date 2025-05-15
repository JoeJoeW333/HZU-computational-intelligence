
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict


# 数据生成函数
def generate_data(num_customers=20, vehicle_capacity=30, seed=None):
    if seed is not None:
        np.random.seed(seed)
    depot = np.array([50, 50])
    customers = np.random.randint(0, 100, size=(num_customers, 2))
    coords = np.vstack([depot, customers])
    demands = np.zeros(num_customers + 1)
    demands[1:] = np.random.randint(1, 6, size=num_customers)
    return coords, demands, vehicle_capacity

# 距离矩阵计算
def calculate_distance_matrix(coords):
    n = len(coords)
    return np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)

# 路径评估函数
def evaluate_route(route, coords, demands, vehicle_capacity, dist_matrix):
    current_load = 0
    total_dist = 0
    current_route = [0]
    
    for customer in route:
        demand = demands[customer]
        if current_load + demand > vehicle_capacity:
            total_dist += dist_matrix[current_route[-1]][0]
            current_route = [0]
            current_load = 0
        total_dist += dist_matrix[current_route[-1]][customer]
        current_route.append(customer)
        current_load += demand
    
    total_dist += dist_matrix[current_route[-1]][0]
    return total_dist

class EnhancedImmuneAlgorithm:
    def __init__(self, coords, demands, vehicle_capacity, 
                 pop_size=100, mutation_rate=0.15, max_generations=200):
        self.coords = coords
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.num_customers = len(coords) - 1
        self.pop_size = pop_size
        self.base_mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.dist_matrix = calculate_distance_matrix(coords)
        self.similarity_cache = defaultdict(dict)  # 相似度缓存

    def initialize_population(self):
        return [np.random.permutation(range(1, self.num_customers+1)).tolist() 
                for _ in range(self.pop_size)]

    def affinity(self, individual):
        return 1 / (evaluate_route(individual, self.coords, self.demands, 
                                  self.vehicle_capacity, self.dist_matrix) + 1e-6)

    def calculate_similarity(self, ind1, ind2):
        # 使用汉明距离计算相似度
        key = tuple(ind1), tuple(ind2)
        if key not in self.similarity_cache:
            matches = sum(g1 == g2 for g1, g2 in zip(ind1, ind2))
            self.similarity_cache[key] = matches / len(ind1)
        return self.similarity_cache[key]

    def calculate_concentration(self, population):
        concentrations = np.zeros(len(population))
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                similarity = self.calculate_similarity(population[i], population[j])
                if similarity > 0.6:  # 优化后的相似度阈值
                    concentrations[i] += 1
                    concentrations[j] += 1
        return concentrations / len(population)

    def dynamic_selection(self, population, affinities, concentrations, gen):
        # 动态调整选择权重
        alpha = 0.7 * (1 - gen/self.max_generations)  # 前期侧重亲和度
        fitness = alpha * affinities + (1 - alpha) * (1 - concentrations)
        fitness = np.clip(fitness, 1e-6, None)  # 避免零概率
        probs = fitness / fitness.sum()
        selected_indices = np.random.choice(len(population), size=len(population), 
                                            p=probs, replace=True)
        return [population[i] for i in selected_indices]

    def enhanced_crossover(self, parent1, parent2):
        # 改进的OX交叉
        start, end = sorted(np.random.choice(len(parent1), 2, replace=False))
        child = [-1] * len(parent1)
        child[start:end+1] = parent1[start:end+1]
        
        ptr = 0
        for gene in parent2:
            if gene not in child:
                while ptr >= start and ptr <= end:
                    ptr += 1
                if ptr < len(parent1):
                    child[ptr] = gene
                    ptr += 1
        return child

    def adaptive_mutate(self, individual, gen):
        # 自适应变异操作
        if np.random.rand() < self.base_mutation_rate + 0.2*(gen/self.max_generations):
            mutation_type = np.random.choice(['swap', 'reverse', 'insert'])
            
            if mutation_type == 'swap':
                i, j = np.random.choice(len(individual), 2, replace=False)
                individual[i], individual[j] = individual[j], individual[i]
                
            elif mutation_type == 'reverse':
                i, j = sorted(np.random.choice(len(individual), 2, replace=False))
                individual[i:j+1] = individual[i:j+1][::-1]
                
            elif mutation_type == 'insert':
                i = np.random.randint(len(individual))
                gene = individual.pop(i)
                j = np.random.randint(len(individual))
                individual.insert(j, gene)
                
        return individual

    def run(self):
        start_time = time.time()
        population = self.initialize_population()
        best_history = []
        best_solution = None
        best_fitness = float('inf')
        
        for gen in range(self.max_generations):
            # 计算亲和度
            affinities = np.array([self.affinity(ind) for ind in population])
            
            # 计算浓度
            concentrations = self.calculate_concentration(population)
            
            # 记录最优解
            current_best_idx = np.argmax(affinities)
            current_best = 1 / affinities[current_best_idx]
            if current_best < best_fitness:
                best_fitness = current_best
                best_solution = population[current_best_idx]
            best_history.append(best_fitness)
            
            # 选择
            population = self.dynamic_selection(population, affinities, concentrations, gen)
            
            # 交叉与变异
            new_pop = []
            for i in range(0, len(population), 2):
                p1 = population[i]
                p2 = population[i+1] if i+1 < len(population) else population[i]
                new_pop.append(self.enhanced_crossover(p1, p2))
                new_pop.append(self.enhanced_crossover(p2, p1))
            
            # 自适应变异
            population = [self.adaptive_mutate(ind, gen) for ind in new_pop[:self.pop_size]]
            
            # 精英保留
            population[0] = best_solution.copy()
        
        return best_solution, best_fitness, time.time()-start_time, best_history

# 遗传算法（保持原样）
class GeneticAlgorithm:
    def __init__(self, coords, demands, vehicle_capacity, pop_size=50, mutation_rate=0.1, max_generations=100):
        self.coords = coords
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.num_customers = len(coords) - 1
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.dist_matrix = calculate_distance_matrix(coords)

    def initialize_population(self):
        return [np.random.permutation(range(1, self.num_customers+1)).tolist() 
                for _ in range(self.pop_size)]

    def fitness(self, individual):
        return 1 / (evaluate_route(individual, self.coords, self.demands, 
                                  self.vehicle_capacity, self.dist_matrix) + 1e-6)

    def select(self, population, fitnesses):
        probs = fitnesses / fitnesses.sum()
        return [population[i] for i in np.random.choice(len(population), size=len(population), p=probs)]

    def crossover(self, parent1, parent2):
        start, end = sorted(np.random.choice(len(parent1), 2, replace=False))
        child = [-1] * len(parent1)
        child[start:end+1] = parent1[start:end+1]
        ptr = 0
        for gene in parent2:
            if gene not in child:
                while ptr >= start and ptr <= end:
                    ptr += 1
                if ptr < len(parent1):
                    child[ptr] = gene
                    ptr += 1
        return child

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            i, j = np.random.choice(len(individual), 2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def run(self):
        start_time = time.time()
        population = self.initialize_population()
        best_history = []
        best_solution = None
        best_fitness = float('inf')
        
        for gen in range(self.max_generations):
            fitnesses = np.array([self.fitness(ind) for ind in population])
            current_best = 1 / fitnesses.max()
            if current_best < best_fitness:
                best_fitness = current_best
                best_solution = population[np.argmax(fitnesses)]
            best_history.append(best_fitness)
            
            selected = self.select(population, fitnesses)
            new_pop = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[i+1] if i+1 < len(selected) else selected[i]
                new_pop.extend([self.crossover(p1, p2), self.crossover(p2, p1)])
            population = [self.mutate(ind) for ind in new_pop[:self.pop_size]]
        
        return best_solution, best_fitness, time.time()-start_time, best_history

# 粒子群算法（改进版）
class EnhancedPSO:
    def __init__(self, coords, demands, vehicle_capacity, 
                 pop_size=50, w=0.7, c1=1.4, c2=1.4, max_generations=100):
        self.coords = coords
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.num_customers = len(coords) - 1
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_generations = max_generations
        self.dist_matrix = calculate_distance_matrix(coords)

    def decode(self, particle):
        return list(np.argsort(particle) + 1)

    def initialize_swarm(self):
        swarm = np.random.rand(self.pop_size, self.num_customers)
        velocities = np.zeros_like(swarm)
        pbest = swarm.copy()
        pbest_fitness = np.array([self.fitness(self.decode(p)) for p in swarm])
        gbest = swarm[np.argmax(pbest_fitness)].copy()
        return swarm, velocities, pbest, pbest_fitness, gbest

    def fitness(self, route):
        return 1 / (evaluate_route(route, self.coords, self.demands, 
                                  self.vehicle_capacity, self.dist_matrix) + 1e-6)

    def run(self):
        start_time = time.time()
        swarm, velocities, pbest, pbest_fitness, gbest = self.initialize_swarm()
        best_history = []
        best_fitness = 0
        
        for gen in range(self.max_generations):
            # 动态调整惯性权重
            current_w = self.w * (1 - gen/self.max_generations)
            
            for i in range(self.pop_size):
                current_fitness = self.fitness(self.decode(swarm[i]))
                if current_fitness > pbest_fitness[i]:
                    pbest_fitness[i] = current_fitness
                    pbest[i] = swarm[i].copy()
            
            current_gbest_fitness = pbest_fitness.max()
            if current_gbest_fitness > best_fitness:
                best_fitness = current_gbest_fitness
                gbest = pbest[np.argmax(pbest_fitness)].copy()
            
            best_history.append(1 / best_fitness)
            
            # 更新速度和位置
            r1, r2 = np.random.rand(2)
            velocities = current_w * velocities + \
                        self.c1 * r1 * (pbest - swarm) + \
                        self.c2 * r2 * (gbest - swarm)
            swarm = np.clip(swarm + velocities, 0, 1)
        
        return self.decode(gbest), 1 / best_fitness, time.time()-start_time, best_history

# 可视化函数
def plot_convergence(histories, labels):
    plt.figure(figsize=(10, 6))
    for hist, label in zip(histories, labels):
        plt.plot(hist, label=label)
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.title('Algorithm Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_route(coords, route, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(coords[0][0], coords[0][1], c='black', s=100, marker='s', label='Depot')
    plt.scatter(coords[1:, 0], coords[1:, 1], c='blue', s=50)
    
    current_route = [0]
    for customer in route:
        current_route.append(customer)
        if len(current_route) > 1:
            pts = coords[current_route[-2:]]
            plt.plot(pts[:, 0], pts[:, 1], color='gray', alpha=0.5)
    current_route.append(0)
    pts = coords[current_route[-2:]]
    plt.plot(pts[:, 0], pts[:, 1], color='gray', alpha=0.5)
    
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    coords, demands, v_cap = generate_data(seed=42)
    
    # 运行算法
    print("Running Enhanced Immune Algorithm...")
    eia = EnhancedImmuneAlgorithm(coords, demands, v_cap, pop_size=100, max_generations=200)
    eia_sol, eia_dist, eia_time, eia_hist = eia.run()
    
    print("Running Genetic Algorithm...")
    ga = GeneticAlgorithm(coords, demands, v_cap)
    ga_sol, ga_dist, ga_time, ga_hist = ga.run()
    
    print("Running Enhanced PSO...")
    epso = EnhancedPSO(coords, demands, v_cap, max_generations=200)
    epso_sol, epso_dist, epso_time, epso_hist = epso.run()

    # 打印结果
    print("\n=== 算法性能对比 ===")
    print(f"改进免疫算法 | 最短距离: {eia_dist:.2f} | 时间: {eia_time:.2f}s")
    print(f"遗传算法    | 最短距离: {ga_dist:.2f} | 时间: {ga_time:.2f}s")
    print(f"改进PSO     | 最短距离: {epso_dist:.2f} | 时间: {epso_time:.2f}s")

    # 绘制收敛曲线
    plot_convergence([eia_hist, ga_hist, epso_hist], 
                    ['Enhanced Immune', 'Genetic', 'Enhanced PSO'])
    
    # 绘制最优路径
    plot_route(coords, eia_sol, 'Enhanced Immune Algorithm Best Route')
    plot_route(coords, ga_sol, 'Genetic Algorithm Best Route')
    plot_route(coords, epso_sol, 'Enhanced PSO Best Route')

if __name__ == "__main__":
    main()