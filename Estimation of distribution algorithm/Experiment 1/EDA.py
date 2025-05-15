import numpy as np

# 问题参数设置
n_items = 10  # 物品数量
np.random.seed(42)  # 随机种子，确保结果可复现
values = np.random.randint(1, 20, size=n_items)  # 物品价值
weights = np.random.randint(1, 10, size=n_items)  # 物品重量
max_weight = np.sum(weights) // 2  # 背包最大承重

# 算法参数设置
pop_size = 100  # 种群大小
m = 50  # 选择的较优个体数量
max_iter = 100  # 最大迭代次数

# 初始化概率向量（每个物品被选中的初始概率为0.5）
prob = np.ones(n_items) * 0.5

# 生成初始种群
population = (np.random.rand(pop_size, n_items) < prob).astype(int)

# 记录全局最优解
best_solution = None
best_fitness = -np.inf

for iteration in range(max_iter):
    # 计算适应度（考虑背包重量约束）
    fitness = []
    for ind in population:
        total_value = np.dot(ind, values)
        total_weight = np.dot(ind, weights)
        if total_weight > max_weight:
            fitness.append(-np.inf)  # 超重则适应度为负无穷
        else:
            fitness.append(total_value)
    fitness = np.array(fitness)
    
    # 更新全局最优解
    current_best_idx = np.argmax(fitness)
    current_best_fitness = fitness[current_best_idx]
    if current_best_fitness > best_fitness:
        best_fitness = current_best_fitness
        best_solution = population[current_best_idx].copy()
    
    # 选择适应度最高的可行解
    feasible_mask = fitness != -np.inf
    feasible_pop = population[feasible_mask]
    feasible_fitness = fitness[feasible_mask]
    
    if len(feasible_fitness) == 0:
        # 无可行解时重新初始化种群
        population = (np.random.rand(pop_size, n_items) < 0.5).astype(int)
        continue
    else:
        # 按适应度降序选择前m个可行解
        sorted_indices = np.argsort(-feasible_fitness)
        selected_count = min(m, len(sorted_indices))
        selected_pop = feasible_pop[sorted_indices[:selected_count]]
    
    # 估计概率模型（拉普拉斯平滑防止概率为0）
    prob = (np.sum(selected_pop, axis=0) + 1) / (selected_count + 2)
    
    # 生成新种群
    population = (np.random.rand(pop_size, n_items) < prob).astype(int)

# 输出结果
print("物品价值:", values)
print("物品重量:", weights)
print("背包承重上限:", max_weight)
print("\n最优解（选中物品的索引）:", np.where(best_solution == 1)[0])
print("总价值:", np.dot(best_solution, values))
print("总重量:", np.dot(best_solution, weights))