# 정규분포

from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np

uniform.rvs(loc=2, scale=4, size=1)

z = np.linspace(0, 8, 100)
y = uniform.pdf(z, loc=0, scale=1)

plt.plot(z, y, color='black')

plt.show()
plt.clf()

# p(X<3.25) 일 확률은?

uniform.cdf(3.25, loc=2, scale=4)

# p (5<X<8.39)일 확률은?
uniform.cdf(8.39, loc=2, scale=4) - uniform.cdf(5, loc=2, scale=4)

uniform.ppf(0.93, loc=2, scale=4)



# 표본 20개를 뽑아서 표본평균도 계산해보세요
x = uniform.rvs(loc=2, scale=4, size = 20*1000, random_state=42) #20개씩 천개
x = x.reshape(1000, 20)
x.shape
blue_x = x.mean(axis=1)
blue_x

import seaborn as sns
sns.histplot(blue_x, stat = "density")
plt.show()
plt.clf()


## X bar ~ N(mu, sigma^2/n)
## X bar ~ N(4, 1.33333/20)
uniform.var(loc=2, scale=4)  #분산 #loc:최솟값, scale: 분포의 범위
uniform.expect(loc=2, scale=4) #기댓값


#Plot the normal distribution PDF
from scipy.stats import norm
x_values = np.linspace(2, 6, 100) # xmin,xmax 사이를 100개의 균등한 간격으로 나눈 값

blue_x=uniform.rvs(loc=2, scale=4, size=20).mean()

a = blue_x +0.665
b = blue_x - 0.665

pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.33333/20))
plt.plot(x_values,pdf_values, color='red', linewidth=2)


# 검정색 벽돌이 나올 확률의 기대값의 평균이 파란색벽돌 평균 
plt.axvline(x=a, color='blue', linestyle='--', linewidth =1)
plt.axvline(x=b, color='blue', linestyle='--', linewidth =1)

# 파란벽돌(표본평균) 점찍기
plt.scatter(blue_x, 0.002, s=10, zorder=10, color='blue')
 
plt.show()
plt.clf()

# 95% 분포
4 -norm.ppf(0.025, loc=4, scale=np.sqrt(1.33333/20))
4 -norm.ppf(0.975, loc=4, scale=np.sqrt(1.33333/20))


# 99%
4 -norm.ppf(0.005, loc=4, scale=np.sqrt(1.33333/20))
norm.ppf(0.005, loc=4, scale=np.sqrt(1.33333/20))

import numpy as np
import pandas as pd


old_seat = np.arange(1, 29)

new_seat = np.random.choice(old_seat, 28, replace=False)

result = pd.DataFrame(
        {
          "old_seat" : old_seat,
          "new_seat" : new_seat})

pd.to_csv(result, "result.csv")
