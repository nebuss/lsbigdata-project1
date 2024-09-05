import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

penguins = load_penguins()

df = penguins.dropna()
df = df[['bill_length_mm', 'bill_depth_mm']]
df = df.rename(columns={'bill_length_mm' : 'y',
                        'bill_depth_mm' : 'x'})

group1 = df.query("x < 16.4")# 1번 그룹
group2 = df.query("x >= 16.4")  # 2번 그룹

# depth2 
def my_mse(data, x):
    n1 = data.query(f"x < {x}").shape[0]  # 1번 그룹
    n2 = data.query(f"x >= {x}").shape[0]  # 2번 그룹 

    y_hat1 = data.query(f"x < {x}")['y'].mean() # 1번 그룹 예측값
    y_hat2 = data.query(f"x >= {x}")['y'].mean() # 2번 그룹 예측값

      # 각 그룹의 MSE는 얼마인가요?
    mse1 = np.mean((data.query(f"x < {x}")['y'] - y_hat1)**2)
    mse2 = np.mean((data.query(f"x >= {x}")['y'] - y_hat2)**2)

    return (mse1*n1 + mse2*n2) / (n1+n2) 



x_values1 = np.arange(group1['x'].min()+0.01, group1['x'].max(), 0.01)
result1 = np.repeat(0.0, len(x_values1))
for i in range(0, len(x_values1)):
    result1[i] = my_mse(group1, x_values1[i])
x_values1[np.argmin(result1)] # 14.01
result1.min()

x_values2 = np.arange(group2['x'].min() + 0.01, group2['x'].max(), 0.01)
result2 = np.repeat(0.0, len(x_values2))
for i in range(0, len(x_values2)):
    result2[i] = my_mse(group2, x_values2[i])
x_values2[np.argmin(result2)] # 19.4

# x, y 산점도 그래프 & 평행선 4개
thresholds = [14.01, 16.42, 19.4]
df['group'] = np.digitize(df['x'], thresholds)
y_mean = df.groupby('group').mean()['y']

k1 = np.linspace(13, 14.01, 100)
k2 = np.linspace(14.01, 16.42, 100)
k3 = np.linspace(16.42, 19.4, 100)
k4 = np.linspace(19.4, 22, 100)

df.plot(kind='scatter', x='x', y='y', color = 'black', s=5)
plt.axvline(x=16.4, color='b', linestyle=':')
plt.axvline(x=14.01, color='r', linestyle=':')
plt.axvline(x=19.4, color='r', linestyle=':')

plt.scatter(k1, np.repeat(y_mean[0], 100), color='r', s=5)
plt.scatter(k2, np.repeat(y_mean[1], 100), color='r', s=5)
plt.scatter(k3, np.repeat(y_mean[2], 100), color='r', s=5)
plt.scatter(k4, np.repeat(y_mean[3], 100), color='r', s=5)