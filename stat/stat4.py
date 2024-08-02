# 신뢰구간 구하기 연습.
# 다음은 한 고등학교의 3학년 학생들 중 16명을 무작위로 선별하여 몸무게를 측정한 데이터이다. 이
# 데이터를 이용하여 해당 고등학교 3학년 전체 남학생들의 몸무게 평균을 예측하고자 한다.
# 79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8
# 단, 해당 고등학교 3학년 남학생들의 몸무게 분포는 정규분포를 따른다고 가정한다.



import numpy as np
from scipy.stats import norm

# 주어진 데이터
weights = [79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8]

# 데이터의 샘플 크기
n = len(weights)

# 표본평균
sample_mean = np.mean(weights)

# 표준편차 (작년 남학생 전체 분포의 표준편차)
sigma = 6

# 신뢰수준 90%에 해당하는 z 값 (양쪽이므로 1 - 0.9 = 0.1, 각쪽은 0.05)
z = norm.ppf(0.95, loc=0, scale=1) # 정규분포의 95번째 백분위수

# 표본오차
margin_of_error = z * (sigma / np.sqrt(n))

# 신뢰구간 계산
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print(f"90% 신뢰구간: {confidence_interval}")

# 정규분포 𝑋∼𝑁(𝜇,𝜎2) X∼N(μ,σ 2)의 제곱 𝑋^2의 기대값을 구하는 문제
# 데이터로부터 E[X^2] 구하기
x =norm.rvs(loc=3, scale=5, size=100000)

np.mean(x**2)
sum(x**2) / (len(x)) - 1


# E[(X - X^2) / (2X)] = ? 구하기

x =norm.rvs(loc=3, scale=5, size=100000)

np.mean(x**2)
np.mean((x - x**2) / 2*x)

# 
np.random.seed(20240729)
x =norm.rvs(loc=3, scale=5, size=100000)
x_bar = x.mean()
x - x_bar


sample_variance = np.var(x)
sample_variance


# n-1  vs n 으로나눈것. 데이터개수작음
x = norm.rvs(loc=3, scale=5, size=20)
np.var(x)
np.var(x, ddof=1)
