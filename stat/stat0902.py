import numpy as np

x_values = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

probabilities = np.array([1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36])

# 기대값 계산
expected_value = np.sum(x_values * probabilities)

# 분산 계산
variance = np.sum((x_values - expected_value)**2 * probabilities)

# 결과 출력
print(f"기대값: {expected_value}")
print(f"분산: {variance}")

# 2X+3은?

expected_value2 = 2 * expected_value + 3

variance2 = np.sqrt(4 * variance)  # (2X + 3)의 분산은 2^2 * X의 분산

print(f"기대값2: {expected_value2}")
print(f"분산2: {variance2}")


# =======이항분포 Y~B가 (3, 0.7) 따를 때, y가 갖는 값에 대응하는 확률 구하는 법.
import scipy.stats as stats
from scipy.stats import binom

n = 3  
p = 0.7
binom_dist = stats.binom(n, p)

# PMF 값을 계산할 성공 횟수 k
k_values = [0, 1, 2, 3]

pmf_values = [binom_dist.pmf(k) for k in k_values]

for k, pmf in zip(k_values, pmf_values):
    print(f"P(Y = {k}) = {pmf:.4f}")

# =========또는 
binom.pmf(np.array([0, 1, 2, 3]), 3, 0.7)

# Y~B(20, 0.45)
# P( 6< Y <= 14) = ? 

sum(binom.pmf(np.array([7, 8, 9, 10, 11, 12, 13, 14]), 20, 0.45))
binom.cdf(14, 20, 0.45) - binom.cdf(6, 20, 0.45)

# X ~ N(30, 4^2)
# P(X > 24) = ?
from scipy.stats import norm
# 표준편차:  scale, 
# P (X < 24)일 확률
norm.cdf(24, 30, 4)
# P(X > 24) 일 확률은
1 - norm.cdf(24, 30, 4)

# X ~ N(30, 4^2) 표본을 8개를 뽑아서 표본평균 X_bar 뽑았다.
# P(28 < X_bar < 29.7) = ? 


mu = 30
sigma = 4

# 표본 크기
n = 8

# 표본평균의 표준편차
sigma_x_bar = sigma / np.sqrt(n)


# P(28 < X_bar < 29.7) 계산

# 각각의 확률 계산
prob_lower = stats.norm.cdf(28, mu, sigma_x_bar)
prob_upper = stats.norm.cdf(29.7, mu, sigma_x_bar)

# 두 확률의 차이가 원하는 범위 내의 확률
probability = prob_upper - prob_lower

print(f"P(28 < X_bar < 29.7) = {probability:.4f}")

# ===선생님의 코드
# X_bar ~ N(30, 4^2/8)
a = norm.cdf(29.7, loc=30, scale=np.sqrt(4**2/8)) 
b = norm.cdf(28, loc=30, scale=np.sqrt(4**2/8)) 
a-b

## 또는 (같은 코드)

mean = 30
s_var = 4/np.sqrt(8)
right_x = (29.7 - mean) / s_var
left_x = (28 - mean) / s_var

a=norm.cdf(right_x, 0, 1)
b = norm.cdf(left_x, 0, 1)
a- b


# 자유도 7인 카이제곱분포 확률밀도 함수 그리기
from scipy.stats import chi2
import matplotlib.pyplot as plt
k = np.linspace(0, 20, 100)
y = chi2.pdf(k, df=7)
plt.plot(k, y, color='black')

# 독립성 검정
#  귀무가설: 두 변수(운동선수 유무랑 흡연 유무) 독립
# 대립가설: 두 변수 독립 아님.
import numpy as np

mat_a=np.array([14, 4, 0, 10]).reshape(2, 2)

from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a)
chi2.round(3) # 검정 통계량
p.round(4) # p value

# 유의수준 0.05라면 p값이 0.05보다 작으므로 귀무가설 기각. 즉, 두 변수는 독립이 아니다.
# 운동선수에게 흡연유무가 관련있다.

# X~chi2(1)일때, P(X > 12.6) = ?
from scipy.stats import chi2
1 - chi2.cdf(12.6, df=1)