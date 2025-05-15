# 分布估计算法实验报告

## 实验一：求解0-1背包问题



### 1. 问题描述

**0-1背包问题**：给定一组物品（每个物品有价值和重量）和一个背包，选择物品的子集装入背包，使得总重量不超过背包容量且总价值最大。

### 2. 参数设置

#### 2.1 问题参数

- 物品数量：`10`
- 物品价值：`[7, 15, 11, 8, 7, 19, 11, 11, 4, 8]`
- 物品重量：`[8, 3, 6, 5, 2, 8, 6, 2, 5, 1]`
- 背包承重上限：`23`（总重量的一半取整）

#### 2.2 算法参数

- 种群大小：`100`
- 较优个体数量：`50`
- 最大迭代次数：`100`
- 初始概率向量：`[0.5, 0.5, ..., 0.5]`

### 3. 算法流程

1. **初始化种群**：按初始概率随机生成二进制个体。
2. **适应度评估**：计算每个个体的总价值和重量，超重个体适应度为负无穷。
3. **选择较优个体**：保留适应度最高的前`m`个可行解。
4. **更新概率模型**：统计较优个体中各物品被选中的频率（使用拉普拉斯平滑防止概率为0）。
5. **生成新种群**：根据更新后的概率采样生成新个体。
6. **终止条件**：达到最大迭代次数后停止。

### 4. 代码实现

```
import numpy as np

# 问题参数设置
n_items = 10
np.random.seed(42)
values = np.random.randint(1, 20, size=n_items)
weights = np.random.randint(1, 10, size=n_items)
max_weight = np.sum(weights) // 2

# 算法参数设置
pop_size = 100
m = 50
max_iter = 100
prob = np.ones(n_items) * 0.5

# 初始化种群
population = (np.random.rand(pop_size, n_items) < prob).astype(int)

# 全局最优解
best_solution = None
best_fitness = -np.inf

# 迭代优化
for iteration in range(max_iter):
    # 计算适应度
    fitness = []
    for ind in population:
        total_value = np.dot(ind, values)
        total_weight = np.dot(ind, weights)
        fitness.append(total_value if total_weight <= max_weight else -np.inf)
    fitness = np.array(fitness)
    
    # 更新最优解
    current_best_idx = np.argmax(fitness)
    if fitness[current_best_idx] > best_fitness:
        best_fitness = fitness[current_best_idx]
        best_solution = population[current_best_idx].copy()
    
    # 选择较优个体
    feasible_mask = fitness != -np.inf
    feasible_pop = population[feasible_mask]
    feasible_fitness = fitness[feasible_mask]
    
    if len(feasible_fitness) == 0:
        population = (np.random.rand(pop_size, n_items) < 0.5).astype(int)
    else:
        sorted_indices = np.argsort(-feasible_fitness)
        selected_count = min(m, len(sorted_indices))
        selected_pop = feasible_pop[sorted_indices[:selected_count]]
    
    # 更新概率模型（拉普拉斯平滑）
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
```

------

### 5. 运行结果

```
物品价值: [ 7 15 11  8  7 19 11 11  4  8]
物品重量: [8 3 6 5 2 8 6 2 5 1]
背包承重上限: 23

最优解（选中物品的索引）: [1 2 4 5 7 9]
总价值: 71
总重量: 22
```

------

### 6. 结果分析

1. **最优解**：选中的物品索引为 `[1, 2, 4, 5, 7, 9]`，对应总价值 **71**，总重量 **22**（未超过上限23）。
2. 关键物品：
   - 索引5（价值19）和索引1（价值15）为高价值物品，且重量分别为8和3，显著提升总价值。
   - 索引9（重量1）和索引4（重量2）为轻量物品，充分利用背包容量。
3. **算法有效性**：分布估计算法通过概率模型迭代优化，成功找到接近重量上限的高价值解。

