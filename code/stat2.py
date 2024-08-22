# 자격에 대하여
# 어떤 일에 대한 본인의 자격은 자신의 생각이 그렇게 
# 중요하지 않음.

# 진짜 중요한 것은 타인이 생각하는 그 자리에 대한
# 본인의 자격이 중요함.

# numpy, pandas, matplotlib 설치해볼것!

import numpy as np
import matplotlib.pyplot as plt

# 예제 넘파이 배열 생성
data = np.random.rand(10)
# sum(data < 0.18)

# 히스토그램 그리기
plt.clf()
plt.hist(data, bins=4, alpha=0.7, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# x=np.random.rand(50000) \
#     .reshape(-1, 5) \
#     .mean(axis=1)
x=np.random.rand(10000, 5).mean(axis=1)
plt.hist(x, bins=30, alpha=0.7, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()


import numpy as np

x=np.arange(33)
sum(x)/33
sum((x - 16) * 1/33)
(x - 16)**2
np.unique((x - 16)**2)

np.unique((x - 16)**2) * (2/33)
sum(np.unique((x - 16)**2) * (2/33))

# E[X^2]
sum(x**2 * (1/33))

# Var(X) = E[X^2] - (E[X])^2
sum(x**2 * (1/33)) - 16**2


## Example X: 0, 1, 2, 3
x=np.arange(4)
x
pro_x=np.array([1/6, 2/6, 2/6, 1/6])
pro_x

# 기대값
Ex=sum(x * pro_x)
Exx=sum(x**2 * pro_x)

# 분산
Exx - Ex**2
sum((x - Ex)**2 * pro_x)

## Example
x=np.arange(99)
x

# 1-50-1 벡터
x_1_50_1=np.concatenate((np.arange(1, 51), np.arange(49, 0, -1)))
pro_x=x_1_50_1/2500

# 기대값
Ex=sum(x * pro_x)
Exx=sum(x**2 * pro_x)

# 분산
Exx - Ex**2
sum((x - Ex)**2 * pro_x)

sum(np.arange(50))+sum(np.arange(51))

## Example3 X: 0, 2, 4, 6
x=np.arange(4)*2
x
pro_x=np.array([1/6, 2/6, 2/6, 1/6])
pro_x

# 기대값
Ex=sum(x * pro_x)
Exx=sum(x**2 * pro_x)

# 분산
Exx - Ex**2
sum((x - Ex)**2 * pro_x)

4 * 0.916

np.sqrt(9.52**2 / 10)
