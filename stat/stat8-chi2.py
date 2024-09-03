import numpy as np
mat_a = np.array([14, 4, 0, 10]).reshape(2, 2)

from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a)
chi2.round(2) # 카이제곱검정통계량 15.5가 아닌 12.6나온 이유: 소수점으로 떨어지면 구하기 애매한데, 자동으로 correction 해주는 기능이 있는 것이다. False로 하면 우리가 아는 검정통계량 나옴.
chi2, p, df, expected = chi2_contingency(mat_a, correction=False)
chi2.round(2)
p.round(4)
expected #Eij들

np.sum((mat_a - expected)** 2 / expected)
1 - chi2.cdf(12.6, df=1)

mat_b = np.array([[50, 30, 20],
                  [45, 30, 20]])

chi2, p, df, expected = chi2_contingency(mat_b, correction=False)
chi2.round(3)
p.round(4)
expected

import pandas as pd
data = [[49,47],[15,27],[32,30]]
columns = ["핸드폰", "유선전화"]
index = ["진보", "중도", "보수"]
phone_data = pd.DataFrame(data, columns=columns, index=index)
phone_data

from scipy.stats import chi2_contingency
result = chi2_contingency(phone_data)
result[3]

x_squared, p_value, df, expected = result
print('x_squared:',x_squared)

print('pavalue:',p_value)
expected
# 카이제곱 통계량 3.219에 대응하는 p‑value 0.199는 유의수준 5%보다 크므로, 귀무가설을 기각하지 못한다. 따라서, 휴대폰 사용여부는 정당지지와는 관련이 없다 (독립이다) 라고 판단한다.

from scipy.stats import chisquare
import numpy as np

observed = np.array([13, 23, 24, 20, 27, 18, 15])
expected = np.repeat(20, 7)
statstic, p_value = chisquare(observed, f_exp = expected)

print("Test statistic:", statstic.round(3))

from statsmodels.stats.proportion import proportions_ztest
z_score, p_value = proportions_ztest(45, 82, 0.5, alternative='larger')
print("x􀼡sqaured:",z_score**2)

import numpy as np
from scipy.stats import chi2 # 독립성
from scipy.stats import chi2_contingency # 동질성
from scipy.stats import chisquare # 적합도
from statsmodels.stats.proportion import proportions_ztest # 비율

##### 귀무가설 : 동일하다/현재 결과에 변함이 없다. #####
##### p-value가 특정 유의 수준보다 작으면 귀무가설 기각 #####

# 112p 동질성 검정
## 문제 1
## 귀무가설 : 정당 지지와 핸드폰 사용 유무는 독립이다. (관련없다.)
## 대립가설 : 정당 지지와 핸드폰 사용 유무는 독립이 아니다.

mat_a = np.array([[49, 47], [15, 27], [32, 30]])|

chi2, p, df, expected = chi2_contingency(mat_a, correction=False)
chi2.round(3)
p.round(4)

### 결론 : 유의수준 0.05보다 p값이 크므로, 귀무가설을 기각할 수 없다.
expected

# 104p 적합도 검정
observed = np.array([13, 23, 24, 20, 27, 18, 15])
expected = np.repeat(20, 7)
statistic, p_value = chisquare(observed, f_exp=expected)

print("Test statistic: ", statistic.round(3))
print("p༡value: ", p_value.round(3))
print("Expected: ", expected)

# 106p 비율 검정
z_score, p_value = proportions_ztest(45, 82, 0.5, alternative='larger')
z_score2, p_value2 = proportions_ztest(45, 82, 0.7, alternative='larger')

print("x༡sqaured:",z_score**2)
print("x༡sqaured2:",z_score2**2)

# 112p
## 귀무가설 : 선거구별 후보A의 지지율이 동일하다.
## 대립가설 : 선거구별 후보A의 지지율이 동일하지 않다.

mat_b = np.array([[176, 124], [193, 107], [159, 141]])

chi2, p, df, expected = chi2_contingency(mat_b, correction=False)
chi2.round(3)
p.round(4)

### 결론 : 유의수준 0.05보다 p값이 작으므로, 귀무가설을 기각한다.