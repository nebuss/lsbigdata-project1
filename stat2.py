import numpy as np

#Y 확률변수
np.arange(-16,16).sum()/33



sum(np.unique((np.arange(33) - 16)**2 )*(2/33))

x=np.arange(33)
sum(x)/33
sum((x - 16) * 1/33)
(x - 16) **2

np.unique((x-16)**2) * (2/33)
sum(np.unique((x-16)**2) * (2/33))

# E[X^2]
sum(x**2 * (1/33))

#Var(x) = E[X^2] - (E[X]^2)
sum(x**2 *(1/33)) - 16**2

#Example
x=np.arange(4)
x
pro_x = np.array([1/6, 2/6, 2/6, 1/6])
pro_x

# 기대값
Ex= sum(x * pro_x)
# x^2 의 기대값
Exx = sum(x **2 * pro_x)

#분산
var = Exx - Ex**2


# 0~100Rkwl 정수

#Var(x) = E[X^2] - (E[X]^2)
x = np.arange(99)

# 벡터를 1-50까지, 50-1까지 만들기
x_1_50_1 = np.concatenate((np.arange(1, 51),np.arange(49, 0, -1)))
pro_x = x_1_50_1 / 2500


ex = sum(x * pro_x)
exx = sum(x**2 * por_x)

# 분산
exx - ex*2
sum((x - ex)**2 * pro_x)


# Y확률변수의 분포 구하기 Y:0, 2, 4, 6
y = np.arange(4) *2
pro_y = np.array([1/6, 2/6, 2/6, 1/6])

ex = sum(y * pro_y)
exx = sum(y**2 * pro_y)

# 분산
exx - ex*2

#표준편차 구하기


np.sqrt(9.52**2 / 10)

from scipy.stats import bernoulli
from scipy.stats import binom

#  확률질량함수 pmf / 확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
# bernoulli.pmf(k, p)   

# P(X=1)
bernoulli.pmf(1, 0.3) 
# P(X=0)
bernoulli.pmf(0, 0.3)   


# x가 k일확률 모수가 n, p일때
# 이항 분포 P(X = k | n, p)
# n: 베르누이 확률변수 더한 갯수, p: 1이 나올 확률
#binom.pmf(k, n, p)
binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3) 
binom.pmf(2, n=2, p=0.3) 


# 이항분포. X ~ B(n, p) n,p 는 모수가 두개라는 뜻이다. 베르누이 확률변수를 두번 더하면 0, 1, 2를 가짐.
[binom.pmf(x, n=30, p=0.3) for x in range(31)]
 
import numpy as np
binom.pmf(np.arange(31), n=30, p=0.3)


# 콤비네이션 (54, 26) 구해보기.
import math
a = math.factorial(54)
b = math.factorial(26)
c = math.factorial(28)

a / (b * c)


math.comb(54, 26)

# 54! 구하기
np.cumprod(np.arange(1, 55))


# (2, 0)  콤비네이션
math.comb(2, 0) * 0.3 ** 0 * (1 - 0.3)**2
# (2, 1)  콤비네이션
math.comb(2, 1) * 0.3 ** 1 * (1 - 0.3)**1



binom.pmf(0, 2, 0.3)


# X ~ B(n=10, p = 0.36)일때 X가 4가 나올 확률
binom.pmf(4, n=10, p=0.36)

# X ~ B(n=10, p = 0.36)일때 X가 4보다 작거나 같게 나올 확률
binom.pmf(np.arange(5), n=10, p=0.36).sum()

# X ~ B(n=10, p = 0.36)일때 X가 2보다 크면서 8보다 작거나 같게 나올 확률
binom.pmf(np.arange(3, 9), n=10, p=0.36).sum()


# X ~ B(n=30, p = 0.2)일때 X가 4보다 작으면서 25보다 크거나 같게 나올 확률


# X가 4보다 작은 확률 (X = 0, 1, 2, 3)
prob_less_than_4 = binom.pmf(np.arange(4), 30, 0.2).sum()

# X가 25보다 크거나 같은 확률 (X = 25, 26, ..., 30)
prob_greater_or_equal_25 = binom.pmf(np.arange(25, 31), 30, 0.2).sum()

total_prob = prob_less_than_4 + prob_greater_or_equal_25


# 또다른 방법

other = binom.pmf(np.arange(4, 25), 30, 0.2).sum()
other
1  - other


# rvs함수(random variates sample) : 표본추출함수

bernoulli.rvs(0.3)
# 베르누이 확률변수를 더해서 n=1, p=0.3인 이항분포
bernoulli.rvs(0.3) + bernoulli.rvs(0.3) 
binom.rvs(n=2, p=0.3, size=3) 

# X ~ B(30, 0.26)따르는 확률변수에서 표본 30개를 뽑아보세요 
# -> 베르누이 확률변수를 30개 더한것이 이 이항분포.
binom.rvs(n=30, p=0.26, size=30)  

# 위의 이항분포의 기대값은?
# 베르누이 확률변수의 기대값은 p 이다. 
# 이항분포의 기대값은  n x p  이다.
0.26 * 30


# X ~ B(30, 0.26) 시각화해보기 - > 가질 수 있는 값을 막대그래프로 그리기
 #x가 나올수있는 확률값  

import matplotlib.pyplot as plt
import seaborn
seaborn.barplot(prob_x)
plt.show()
plt.clf()

#교재 207p 활용
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x= np.arnage(31)
prob_x =binom.pmf(x, n=30, p=0.26)

df = pd.DataFrame( {"x": x, "prob": prob_x})
df

sns.barplot(data=df, x='x', y='prob')
plt.show()
plt.clf()



# F_X(x) = X <= p
binom.cdf(4, n=30, p=0.26)


# cdf : cumulative dist. function (누적확률분포 함수)
# p(4<= x < 18)일때 누적확률분포 구하기 
binom.cdf(18, 30, 0.26) - binom.cdf(4, 30, 0.26)
# p( 13< x < 20)일때 누적확률분포 구하기 => p(x<=19) - p(x<=13)
binom.cdf(19, 30, 0.26) - binom.cdf(13, 30, 0.26)



# 확률질량함수 그려보세요.

from scipy.stats import bernoulli
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


x_1 = binom.rvs(n=30, p=0.26, size=10)  # 이항분포의 확률질량함수를 구하고 샘플 하나를 뽑아 그래프를 그린것.
 
n= 30
p=0.26
# 기대값은 7.8
x= np.arange(31)
prob_x =binom.pmf(x, n=30, p=0.26) # 
sns.barplot(prob_x, color='blue')

#기대값표현
plt.axvline(x=n*p, color='green', linestyle='--', linewidth =2)

# add a point at (2,0)
plt.scatter(x_1, np.repeat(0.002, 10), zorder=100, color='red')

plt.show()                                    
plt.clf()

binom.ppf(0.5, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
binom.cdf(7, n=30, p=0.26)

# p(x< ?)= 0.7
binom.ppf(0.7, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
binom.cdf(9, n=30, p=0.26)


from scipy.stats import norm

norm.pdf(5, loc=3, scale=4)


#정규분포 그려보기
# mu:분포의 중심 결정하는 모수
z = np.linspace(-5, 5, 100)
y = norm.pdf(z, loc=2, scale=1)

plt.plot(z, y, color='black')
plt.show()
plt.clf()

# 정규분포
# mu:분포의 중심 결정하는 모수
#  sigma(scale) :  분포이 퍼짐 결정하는 모수
z = np.linspace(-5, 5, 100)
y = norm.pdf(z, loc=0, scale=1)
y2 = norm.pdf(z, loc=0, scale=2)
y3 = norm.pdf(z, loc=0, scale=3)
plt.plot(z, y, color='black')
plt.scatter(z, y2, color='red', s=3)
plt.plot(z, y3, color='green')
plt.show()
plt.clf()

norm.cdf(0, loc=0, scale=1)

# mu=0, sigma=1, p(-2<x<0.54)
norm.cdf(0.5, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)

# p(1< x or x > 3)
norm.cdf(1 , loc=0, scale=1) 
1- norm.cdf(3, loc=0, scale=1)

# X ~ N(3, 5^2)
# P(3 < X < 5) = ? 1 5 .5 4 % 따른다. 
norm.cdf(5, loc=3, scale=5) - norm.cdf(3, loc=3, scale=5)

# 확률변수 표본 1000개 뽑아보자
x=norm.rvs(loc=3, scale=5, size=1000)
sum((x > 3) & (x <5))/1000



# 평균0, 표준편차 1 확률변수에서 표본 1000개 뽑아서 0보다 작은 비율 구하기
x=norm.rvs(loc=0, scale=1, size=1000)
sum(x<0)/1000




x =norm.rvs(3, 2, 1000)
sns.histplot(x, stat='density') #stat='density' 인자. PDF로 그렸을때 y축의 범위를 맞춰준다.

xmin, xmax = (x.min(). x.max())  # x축 범위
x = np.linspace(xmin, xmax, 100)  # x축 값
p = norm.pdf(x, 3, 2)  # 정규분포 PDF 계산

plt.plot(x, p, 'k', linewidth=2. color='red')  # PDF 그래프 그리기
plt.title("Histogram with Normal Distribution PDF")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

plt.clf()
