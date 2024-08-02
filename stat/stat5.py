# X~N(3, 7^2)
from scipy.stats import norm


x=norm.ppf(0.25, loc=3, scale=7)

z=norm.ppf(0.25, loc=0, scale=1)

3 + z * 7



# X~N(3, 7^2) 5이하가 나올확률
x=norm.cdf(5, loc=3, scale=7)
x
z=norm.cdf(2/7, loc=0, scale=1)
z

#정규분포 0, 1을 따르는  표본1000개
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

z=norm.rvs(loc=0, scale=1, size=1000)
z

x=z*np.sqrt(2) + 3
sns.histplot(z, stat="density", color="grey")
sns.histplot(x, stat="density", color="green")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 500)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
pdf_values2 = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.plot(z_values, pdf_values2, color='blue', linewidth=2)

plt.show()
plt.clf()



# X~N(5, 3^2) 
# Z=X-5/3  가 표준정규분포를 따르는지?
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

x=norm.rvs(loc=5, scale=3, size=1000)
z= (x-5) / 3
sns.histplot(z, stat="density", color="grey")
zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()



# x에서 표본 10 뽑아서 표본분산계산
x=norm.rvs(loc=5, scale=3, size=10)

sample_var = np.var(x, ddof=1)
# 또는그냥 분산 구하기 위해 s=np.std(x, ddof=1)  로 적어줘도 된다.

x = norm.rvs(loc=5, scale=3, size=1000)

z = (x - 5)/np.sqrt(sample_var)
#우리가 만든 z가 표준정규분포가 아님. 

zmin, zmax = (z.min(), x.max())
sns.histplot(z, stat="density", color="green")
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.show()


#  t  분포에 대해 알아보자
# 자유도가 4인 t분포의 pdf는?
from scipy.stats import t


t_values= np.linspace(-4, 4, 100)
pdf_t= t.pdf(t_values, df=1)
plt.plot(t_values, pdf_t)
plt.show()

plt.clf()



# 표준정규분포는 빨간색
pdf_values = norm.pdf(t_values, loc=0, scale=1)
plt.plot(t_values, pdf_values, color='red', linewidth=2)
plt.show()


#  자유도가 n-1 인 t분포

x=norm.rvs(loc=15, scale=3, size=16, random_state=42)
n=len(x)

x_bar = x.mean()
# 모분산을 모를때:모평균에 대한 95% 신뢰구간 구하기. 
x_bar + t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)

# 모분산을 알때:모평균에 대한 95% 신뢰구간 구하기. 
x_bar + norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)
x_bar - norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)

