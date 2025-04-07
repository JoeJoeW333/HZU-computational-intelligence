import numpy as np
import pandas as pd

# 生成随机城市坐标
np.random.seed(42)  # 设置随机种子确保可重复性
num_cities = 100
cities = np.random.rand(num_cities, 2) * 100  # 生成0-100范围内的坐标

# 创建DataFrame并保存为CSV
df = pd.DataFrame(cities, columns=['x', 'y'])
df.index.name = 'city_id'
df.to_csv('cities_100.csv')

print("城市坐标已保存到 cities_100.csv")