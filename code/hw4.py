import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform


x = uniform.rvs(loc=3, scale=4, size=20*10000)
x
x.shape
x=x.reshape(-1, 20)
x
x.shape

pop_var=uniform.var(loc=3, scale=4)

# n-1 나누는 케이스
s_2 = np.var(x, ddof=1, axis=1)
s_2.shape
sns.histplot(s_2, stat='density', color='green')
plt.axvline(x = pop_var, color = 'red', linestyle = '-', linewidth = 2)
plt.xlim([0, 2.5])
plt.show()
# plt.clf()

k_2 = np.var(x, ddof=0, axis=1)
k_2.shape
sns.histplot(k_2, stat='density')
plt.axvline(x = pop_var, color = 'red', linestyle = '-', linewidth = 2)
plt.xlim([0, 2.5])
plt.show()
plt.clf()



s_2 = np.var(x, ddof=1, axis=1)
k_2 = np.var(x, ddof=0, axis=1)

result = []
for i in range(10000):
    if (s_2[i] - pop_var)**2 < (k_2[i] - pop_var)**2:
        result.append("n-1")
    elif (s_2[i] - pop_var)**2 > (k_2[i] - pop_var)**2:
        result.append("n")
# 10000개의 분산값을 각각 비교해서 이론적인 분산에 더 가까우면 그 방법을 1점 주는 반복문.

sns.countplot(data = result)
plt.show()
plt.clf()
# n-1이 더 효과 있따!!!



