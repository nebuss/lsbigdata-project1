# y=2x+3
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


y = 2*x+3
x= np.linspace(0, 100, 400)
# 
# np.random.seed(20240805)
obs_x = np.random.choice(np.arange(100), 20)
epsilon_i = norm.rvs(loc=0, scale=20, size=20) # 노이즈/ 분산이 적을수록 예측 점과 실제 데이터가 비슷해짐.
obs_y= 2*obs_x + 3 + epsilon_i

# 모델 학습
obs_x = obs_x.reshape(-1, 1) 
model = LinearRegression()
new =model.fit(obs_x, obs_y)


plt.plot(x,  y, color='black')
plt.scatter(obs_x, obs_y, color="blue", s=3)


plt.show()

# 회귀 직선의 기울기와 절편
model.coef_ # a_hat
model.intercept_ # b_hat



# 회귀 직선 그리기
x= np.linspace(0, 100, 400)
y = model.coef_[0] * x + model.intercept_
plt.xlim([0, 100])
plt.ylim([0, 300])
plt.plot(x, y, color='red')


plt.show()
plt.clf()


# 라이브러리로 선형회귀 분석 수행하는 코드
!pip install statsmodels
import statsmodels.api as sm
obs_x_reshaped = sm.add_constant(obs_x)
model = sm.OLS(obs_y, obs_x).fit()
print(model.summary())


# 예측한 값이 실제랑 일치하지 않을 수도 있으니 검정을 해야함.


import numpy as np


(8.79)/ str(20)


1 - norm.cdf(18, loc=10, scale=1.96)








import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
n = len(x)
x = np.arange(-3, 3, 0.1)
y = t.pdf(x, df=n-1)
reject_range = t.ppf(0.975, df=n-1)
plt.plot(x, y, color='k')
plt.axhline(0, color='black', linewidth=0.5)
plt.fill_between(x, y,
where=(x < -reject_range) | (x > reject_range),
color='yellow')
plt.plot(t_value, 0, 'ro') # 검정통계량 표현
plt.show()
