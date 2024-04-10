import numpy as np

# 生成在-π到π区间内均匀分布的数据集
num_points = 1250
x_values = np.linspace(-np.pi, np.pi, num_points)
# 对x_values应用sin函数以获得y_values
y_values = np.sin(x_values)

with open('sin_data_1000.txt', 'w') as f:
    for x, y in zip(x_values, y_values):
        f.write(f"{x} {y}\n")