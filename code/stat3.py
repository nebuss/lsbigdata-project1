# 한렬님의 생각 (무엇이 옳은가)
# 여행을 많이 가는 집단에게 광고를 하는 것: (회사, 돈을 벌어야하는 집단)
# 여행을 가지 않는 집단에게 광고를 하는 것: (정부, 공공의 이익 추구)

# 그로스 해킹: 한정된 자원을 활용할 경우 단골을 만드는 것이 더 효과적
# 비용 측면에서 기존 고객층을 유지시키는 것이 효율적

# !pip install scipy
from scipy.stats import bernoulli

# 확률질량함수(pmf)
# 확률변수가 갖는 값에 해당하는 확률을 저장하고
# 있는 함수
# bernoulli.pmf(k, p)
# P(X=1)
bernoulli.pmf(1, 0.3)
# P(X=0)
bernoulli.pmf(0, 0.3)

# 이항분포 X ~ P(X = k | n, p)
# n: 베르누이 확률변수 더한 갯수
# p: 1이 나올 확률
# binom.pmf(k, n, p)
from scipy.stats import binom

binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)


# X ~ B(n, p)
# list comp.
result=[binom.pmf(x, n=30, p=0.3) for x in range(31)]
result
# numpy
import numpy as np
binom.pmf(np.arange(31), n=30, p=0.3)


import math
math.factorial(54) / (math.factorial(26) * math.factorial(28))
math.comb(54, 26)

# ======몰라도 됨====================================
# 1*2*3*4
# np.cumprod(np.arange(1, 5))[-1]
# fact_54=np.cumprod(np.arange(1, 55))[-1]
# ln
log(a * b) = log(a) + log(b)
log(1 * 2 * 3 * 4) = log(1) + log(2) + log(3) + log(4)

np.log(24)
sum(np.log(np.arange(1, 5)))

math.log(math.factorial(54))
logf_54=sum(np.log(np.arange(1, 55)))
logf_26=sum(np.log(np.arange(1, 27)))
logf_28=sum(np.log(np.arange(1, 29)))
# math.comb(54, 26)
np.exp(logf_54 - (logf_26 + logf_28))
# =================================================

math.comb(2, 0) * 0.3**0 * (1-0.3)**2
math.comb(2, 1) * 0.3**1 * (1-0.3)**1
math.comb(2, 2) * 0.3**2 * (1-0.3)**0

# pmf: probability mass function (확률질량함수)
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

# X ~ B(n=10, p=0.36)
# P(X = 4) =?
binom.pmf(4, n=10, p=0.36)
# P(X <= 4) =?
binom.pmf(np.arange(5), n=10, p=0.36).sum()
binom.pmf(np.arange(3, 9), n=10, p=0.36).sum()

X~B(30, 0.2)
binom.pmf(np.arange(31), n=30, p=0.2).sum()
# 1
a=binom.pmf(np.arange(4), n=30, p=0.2).sum()
# 2
b=binom.pmf(np.arange(25, 31), n=30, p=0.2).sum()
a+b

# 4
1 - binom.pmf(np.arange(4, 25), n=30, p=0.2).sum()

# rvs 함수 (random variates sample)
# 표본 추출 함수
# X1 ~ Bernulli(p=0.3)
bernoulli.rvs(p=0.3)
# X2 ~ Bernulli(p=0.3)
bernoulli.rvs(p=0.3)
# X ~ B(n=2, p=0.3)
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)
binom.rvs(n=2, p=0.3, size=1)
binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

# X ~ B(30, 0.26)
# 기대값 30*0.26 = 7.8
# 표본 30개를 뽑아보세요!
binom.rvs(n=30, p=0.26, size=30)


import seaborn as sns
import matplotlib.pyplot as plt
x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)

sns.barplot(prob_x)
plt.show()
plt.clf()

# 교재 p.207
import pandas as pd

x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)

df = pd.DataFrame({"x": x, "prob": prob_x})
df

sns.barplot(data = df, x = "x", y = "prob")
plt.show()


# cdf: cumulative dist. function 
# (누적확률분포 함수)
# F_X(x) = P(X <= x)
binom.cdf(4, n=30, p=0.26)
binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26)
binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)


# 
import numpy as np
import seaborn as sns

x_1 = binom.rvs(n=30, p=0.26, size=1)
x_1
x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x, color="blue")

# Add a point at (2, 0)
plt.scatter(x_1, 0.002, color='red', zorder=100, s=10)

# 기대값 표현
plt.axvline(x=7.8, color='green', 
            linestyle='--', linewidth=2)

plt.show()
plt.clf()



import numpy as np
import seaborn as sns

x_1 = binom.rvs(n=30, p=0.26, size=10)
x_1
x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x, color="blue")

# Add a point at (2, 0)
plt.scatter(x_1, np.repeat(0.002, 10), 
            color='red', zorder=10, s=10)

# 기대값 표현
plt.axvline(x=7.8, color='green', 
            linestyle='--', linewidth=2)

plt.show()
plt.clf()

binom.ppf(0.5, n=30, p=0.26)
binom.ppf(0.7, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
binom.cdf(9, n=30, p=0.26)


1/np.sqrt(2 * math.pi)
from scipy.stats import norm

norm.pdf(0, loc=0, scale=1)
norm.pdf(5, loc=3, scale=4)


# 정규분포 pdf 그리기
k=np.linspace(-5, 5, 100)
y=norm.pdf(k, loc=0, scale=1)

plt.plot(k, y, color="black")
plt.show()
plt.clf()

## mu (loc): 분포의 중심 결정하는 모수
k=np.linspace(-5, 5, 100)
y=norm.pdf(k, loc=0, scale=1)

plt.plot(k, y, color="black")
plt.show()
plt.clf()

## sigma (scale): 분포의 퍼짐 결정하는 모수(표준편차)
k=np.linspace(-5, 5, 100)
y=norm.pdf(k, loc=0, scale=1)
y2=norm.pdf(k, loc=0, scale=2)
y3=norm.pdf(k, loc=0, scale=0.5)
plt.plot(k, y, color="black")
plt.plot(k, y2, color="red")
plt.plot(k, y3, color="blue")
plt.show()
plt.clf()


norm.cdf(0, loc=0, scale=1)
norm.cdf(100, loc=0, scale=1)
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)

# 정규분포: Normal distribution
# X ~ N(3, 5^2)
# P(3 < X < 5) =? 15.54%
norm.cdf(5, 3, 5) - norm.cdf(3, 3, 5)
# 위 확률변수에서 표본 1000개 뽑아보자!
x=norm.rvs(loc=3, scale=5, size=1000)
sum((x > 3) & (x < 5))/1000


# 밸런스 게임
# 이상형과 평생 친구(연애불가) vs.
# 이상형과 1년 연애 헤어지기 (5년 연애불가)
# 결론: 이상형과의 1년 연애 == 일반 연애 5년 (True)


# 평균:0, 표준편차: 1
# 표본 1000개 뽑아서 0보다 작은 비율 확인
x=norm.rvs(loc=0, scale=1, size=1000)
np.mean(x < 0)
sum(x < 0)/1000


x=norm.rvs(loc=3, scale=2, size=1000)
x

sns.histplot(x, stat="density")

# Plot the normal distribution PDF
xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=3, scale=2)
plt.plot(x_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()



# 숙제 1번 from scipy.stat import norm 사용금지

# # 여자친구 남자친구의 이성친구는 어디까지 가능한가?
# # (여사친 / 남사친)
# 
# 단둘이 밥 먹기 (아침 세모, 점심 o, 저녁 8-10시 o)
# 
# 술먹기 
# - 단둘이 (아침 x, 점심 x, 저녁 x)
# - 단체 (아침 o, 점심 o, 저녁 o)
# 
# 여행 
# - 단둘이 (당일치기 x, 1박)
# - 단체 (당일치기 o, 1박 6인이상 x)
# 
# 영화
# - 단둘이 (집 x, 영화관 조조 x, 일반 x, 심야 x)
# - 단체 (집 o, 영화관 조조 o, 일반 o, 심야 x)
# 
# 무엇을 하고 놀아? 
# 1. 대화를 한다!
# 2. 롤 o, 보드게임 x
# 3. 단체로 놀거나, 나를 끼워서 놀아라.





