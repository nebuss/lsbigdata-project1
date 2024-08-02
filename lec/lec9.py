import numpy as np
import matplotlib.pyplot as plt
#E[X]
sum(np.arange(4) * np.array([1, 2, 2, 1]) / 6)

data = np.random.rand(10)
#히스토그램 그리기
plt.clf()
plt.hist(data, bins=4, alpha=0.7, color='pink')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


data = np.random.rand(5)



## 0~1사이 숫자 5개발생
## 표폰평균 구하기
## 10000번 반복
## 히스토그램 구하기
x = np.random.rand(50000).reshape(-1, 5).mean(axis=1) #5번이 만번 반복
#혹은 np.random.(10000, 5).reshape(-1, 5).mean(axis=1)

plt.hist(x, bins=30, alpha=0.7, color='blue')
plt.show()
