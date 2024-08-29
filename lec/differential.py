import numpy as np
import random  # random 모듈을 임포트합니다

import matplotlib.pyplot as plt
k=2
x = np.linspace(-4, 8, 100)
y = ((x-2) **2 )+ 1
# y=4x - 11 그래프

plt.plot(x, y)
plt.xlim(-4, 8)
plt.ylim(0, 15)

# f'(x)=2x-4
# k=4의 기울기

l_slope=2*k - 4

#  k 를 넣었을 때의 함수값
f_k=(k-2)**2 + 1
l_intercept=f_k - l_slope * k

#  y=slope*x+intercept 그래프
line_y= l_slope * x + l_intercept
plt.plot(x, line_y, color='red')

-16*0.01
#  y=x^2  초기값 10, 델타는 0.9일때 x100은?
x = 10
delta = 0.9

for i in range(100):
    x = x-delta *( 2 * x)

x # 0.000000002037035976334498

# 경사하강법
# 학습률(learning rate)을 매 반복마다 다르게 설정하여 𝑥 값을 업데이트하는 방식입니다. 이 방법을 통해 학습률을 점차 줄여나가며 목표 함수의 최소값을 찾는 과정
x = 10
lstep = np.arange(100, 0, -1) * 0.01
for i in range(100):
    x = x - lstep[i]*(2*x)
x

x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)
z = (x-3)**2 + (y-4)**2 + 3

plt.figure()
cp= plt.contour(x, y, z, levels=20)
plt.colorbar(cp) # 등고선 레벨값이 같은 애들끼리 같은 색으로 묶어줌.
plt.plot(9 , 2, 'bo')

x=9
y=2
lstep=0.1

x
y

for i in range(100):
    random_color = (random.random(), random.random(), random.random())
    x, y= np.array([x, y])- lstep * np.array([2 *x -6, 2*y-8])    
    plt.scatter(float(x), float(y), color=random_color, s=50)
print(x, y)


#=========================회귀 직선 베타 찾기
b0 = np.linspace(-10, 10, 400)
b1 = np.linspace(-10, 10, 400)
b0, b1 = np.meshgrid(b0, b1)

# 함수 f(x, y)를 계산합니다.
z = (1-(b0+b1))**2 + (4-(b0+2*b1))**2 + (1.5-(b0+3*b1))**2 + (5-(b0+4*b1))**2

plt.figure()
cp = plt.contour(b0, b1, z, levels=200)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# f(b0, b1) = (1-(b0+b1))**2 + (4-(b0+2*b1))**2 + (1.5-(b0+3*b1))**2 + (5-(b0+4*b1))**@
b0 = 10
b1 = 10
delta = 0.01
for i in range(1000):
    gradient_b0 = 8*b0 + 20*b1 -23
    gradient_b1 = 20*b0 + 60*b1 -67
    b0, b1 = np.array([b0, b1]) - delta * np.array([gradient_b0, gradient_b1])
    plt.scatter(b0, b1, color = 'red', s=10)
print(b0, b1)



# =========모델 fit으로 베타 구하기
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    'x':np.array([1,2,3,4]),
    'y' : np.array([1, 4, 1.5, 5])
})

model = LinearRegression()
model.fit(df[['x']], df[['y']])

model.intercept_
model.coef_