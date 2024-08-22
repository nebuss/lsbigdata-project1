import pandas as pd
import numpy as np

old_seat=np.arange(1, 29)


np.random.seed(20240729)
# 1~28 숫자 중에서 중복 없이 28개 숫자를 뽑는 방법
new_seat=np.random.choice(old_seat, 28, replace=False)

result=pd.DataFrame(
    {"old_seat": old_seat,
     "new_seat": new_seat}
)

result.to_csv(result, "result.csv")


# y=2x 그래프 그리기
# 점을 직선으로 이어서 표현
import matplotlib.pyplot as plt

x = np.linspace(0, 8, 2)
y = 2 * x
# plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")
plt.show()
plt.clf()

# y = x^2 를 점 3개 사용해서 그리기
x = np.linspace(-8, 8, 100)
y = x**2
# plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")

# x, y 축 범위 설정
plt.xlim(-10, 10)
plt.ylim(0, 40)
# 비율 맞추기
# plt.axis('equal')는 xlim, ylim과 같이 사용 x
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
plt.clf()

from scipy.stats import norm
import numpy as np

x=np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
x.mean()
len(x)

z_005=norm.ppf(0.95, loc=0, scale=1)
z_005
# 신뢰구간
x.mean() + z_005 * 6 / np.sqrt(16)
x.mean() - z_005 * 6 / np.sqrt(16)

# 데이터로부터 E[X^2] 구하기
x=norm.rvs(loc=3, scale=5, size=100000)

np.mean(x**2)
# sum(x**2) / (len(x) - 1)
np.mean((x - x**2) / (2*x))

np.random.seed(20240729)
x=norm.rvs(loc=3, scale=5, size=100000)
x_bar = x.mean()
s_2 = sum((x - x_bar)**2) / (100000-1)
s_2
# np.var(x) 사용하면 안됨 주의! # n으로 나눈 값
np.var(x, ddof=1) # n-1로 나눈 값 (표본 분산)

# n-1 vs. n
x=norm.rvs(loc=3, scale=5, size=20)
np.var(x)
np.var(x, ddof=1)

# 숙제 표본 분산 n-1 vs. n
# 표본 분산 계산 시 왜 n-1로 나누는지 알아보도록 하겠습니다.
# 균일분포 (3, 7)에서 20개의 표본을 뽑아서
# 분산을 2가지 방법으로 추정해보세요.
# n-1로 나눈 것을 s_2, n으로 나눈 것을 k_2로 정의하고,
# s_2의 분포와 k_2의 분포를 그려주세요! (10000개 사용)
# 각 분포 그래프에 모분산의 위치에 녹색 막대를 그려주세요.

