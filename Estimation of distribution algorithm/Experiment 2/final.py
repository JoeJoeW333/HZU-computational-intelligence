import numpy as np
import matplotlib.pyplot as plt


#解决可视化展示的时候中文乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题



# 中文字体配置（覆盖scienceplots的字体设置）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
alpha = 0.5  # 缺陷率权重
beta = 0.5    # 能耗权重
population_size = 100  # 种群大小
K = 20       # 优势个体数量
max_generations = 1000  # 最大迭代次数

# 变量范围约束
var_ranges = np.array([
    [150, 300],   # T_melt
    [50, 150],    # P_inj
    [5, 20],      # t_inj
    [5, 20]       # t_cool
])

def initialize_population(N):
    """初始化种群，保证生产周期约束"""
    population = np.zeros((N, 4))
    population[:, 0] = np.random.uniform(150, 300, N)  # T_melt
    population[:, 1] = np.random.uniform(50, 150, N)   # P_inj
    
    for i in range(N):
        # 生成满足 t_inj + t_cool <= 30 的参数
        t_inj = np.random.uniform(5, 20)
        max_t_cool = 30 - t_inj
        t_cool = np.random.uniform(5, min(20, max_t_cool))
        population[i, 2] = t_inj
        population[i, 3] = t_cool
    
    return population

def defect_rate(X):
    """计算缺陷率"""
    T_melt, P_inj, t_inj, t_cool = X
    term1 = ((T_melt - 220)/80)**2
    term2 = 100 / P_inj
    term3 = 0.05 * (t_cool - 10)**2
    return term1 + term2 + term3

def energy_cost(X):
    """计算能耗"""
    T_melt, P_inj, t_inj, t_cool = X
    return 0.1 * P_inj * t_inj + 0.02 * T_melt * t_cool

def fitness_function(X, alpha, beta):
    """适应度计算（带约束惩罚）"""
    # 参数范围约束检查
    if not (var_ranges[0][0] <= X[0] <= var_ranges[0][1] and
            var_ranges[1][0] <= X[1] <= var_ranges[1][1] and
            var_ranges[2][0] <= X[2] <= var_ranges[2][1] and
            var_ranges[3][0] <= X[3] <= var_ranges[3][1]):
        return float('inf')
    
    # 生产周期约束检查
    if X[2] + X[3] > 30:
        return float('inf')
    
    return alpha * defect_rate(X) + beta * energy_cost(X)

# 初始化种群
population = initialize_population(population_size)

# 记录收敛过程
best_fitness_history = []
avg_fitness_history = []
mu_history = []

for gen in range(max_generations):
    # 计算适应度
    fitness_values = [fitness_function(ind, alpha, beta) for ind in population]
    
    # 记录统计信息
    best_idx = np.argmin(fitness_values)
    best_fitness = fitness_values[best_idx]
    best_fitness_history.append(best_fitness)
    avg_fitness_history.append(np.mean(fitness_values))
    
    # 选择优势个体
    sorted_indices = np.argsort(fitness_values)
    selected = population[sorted_indices[:K]]
    
    # 更新概率模型参数
    mu = np.mean(selected, axis=0)
    sigma = np.cov(selected, rowvar=False)
    sigma += 1e-5 * np.eye(4)  # 防止奇异矩阵
    
    mu_history.append(mu)
    
    # 生成新种群
    new_population = np.random.multivariate_normal(mu, sigma, population_size)
    
    # 应用变量范围约束
    for i in range(4):
        new_population[:, i] = np.clip(new_population[:, i], 
                                      var_ranges[i][0], 
                                      var_ranges[i][1])
    
    population = new_population

# 可视化结果
plt.figure(figsize=(12, 8))

# 收敛曲线
plt.subplot(2, 2, 1)
plt.plot(best_fitness_history, label='最佳适应度')
plt.plot(avg_fitness_history, label='平均适应度')
plt.xlabel('迭代次数')
plt.ylabel('适应度')
plt.title('收敛曲线')
plt.legend()
plt.grid(True)

# 参数均值变化
mu_history = np.array(mu_history)
plt.subplot(2, 2, 2)
plt.plot(mu_history[:, 0], label='熔融温度')
plt.plot(mu_history[:, 1], label='注射压力')
plt.plot(mu_history[:, 2], label='注射时间')
plt.plot(mu_history[:, 3], label='冷却时间')
plt.xlabel('迭代次数')
plt.ylabel('参数值')
plt.title('参数均值演变')
plt.legend()
plt.grid(True)

# 最优参数分布
best_solution = population[np.argmin([fitness_function(ind, alpha, beta) for ind in population])]
plt.subplot(2, 2, 3)
parameters = ['熔融温度', '注射压力', '注射时间', '冷却时间']
plt.bar(parameters, best_solution)
plt.ylabel('值')
plt.title('最优参数')

# 生产周期验证
plt.subplot(2, 2, 4)
plt.bar(['注射时间', '冷却时间'], [best_solution[2], best_solution[3]])
plt.ylabel('时间 (s)')
plt.title(f'生产时间: {best_solution[2] + best_solution[3]:.1f}s')

plt.tight_layout()
plt.show()

# 输出最优解
print("最优参数:")
print(f"熔融温度: {best_solution[0]:.2f} ℃")
print(f"注射压力: {best_solution[1]:.2f} MPa")
print(f"注射时间: {best_solution[2]:.2f} s")
print(f"冷却时间: {best_solution[3]:.2f} s")
print(f"\n缺陷率: {defect_rate(best_solution):.2f}%")
print(f"能耗成本: {energy_cost(best_solution):.2f}")
print(f"总生产时间: {best_solution[2] + best_solution[3]:.1f} s")