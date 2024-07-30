# y=2x 그래프 그리기
#  파이썬은 점을 직선으로 이어서 표현
import matplotlib.pyplot as plt
import numpy as np

# x 범위를 설정합니다. 예를 들어 -10에서 10까지.
x = np.linspace(0, 10, 10)
# y=2x 함수를 정의합니다.
y = 2 * x

# 그래프를 그립니다.
plt.plot(x, y, label='y=2x')

# 그래프에 제목과 축 라벨을 추가합니다.
plt.title('Graph of y=2x')
plt.xlabel('x')
plt.ylabel('y')

# 범례를 추가합니다.
plt.legend()
plt.grid()

# 그래프를 보여줍니다.
plt.show()
plt.clf()


# y =x^2 점 세개를 사용해서 그려보자.
x = np.linspace(0, 10, 3)
y= x**2

plt.scatter(x, y, s=3)
plt.plot(x, y, color='green')
plt.show()

# y = x^2 그래프가 꺾여서 그려진다.
x = np.linspace(-8, 8, 100)
y= x**2

plt.scatter(x, y, s=3)
plt.plot(x, y, color='green')

plt.show()

# x축과 y축의 비율을 맞추는 함수
plot.axis('equal')
plt.clf()

# 범위를 지정하는 또다른 방법
# x축의 범위를 -10에서 10 까지 설정,  y축의 범위를 0에서 40까지 설정
plt.xlim(-10, 10)
plt.ylim(0, 40)

plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# economics 데이터 불러오기 
import pandas as pd
import matplotlib.pyplot as plt

economimatplotlibeconomics =pd.read_csv('data/economics.csv')
economics.head()


economics.info()

import seaborn as sns
sns.lineplot(data= economics, x = 'date', y = 'unemploy')
plt.show()
plt.clf()
economics['date2'] = pd.to_datetime(economics['date'])
economics.info()


economics[['date', 'date2']]
economics['date2'].dt.year
economics['date2'].dt.quarter 
economics['quarter'] = economics['date2'].dt.quarter
economics[['date2' , 'quarter']]

# 각 날짜는 무슨 요일인가?
economics['date2'].dt.day_name()
economics['date2'] + pd.DateOffset(days=3)
economics['date2'] + pd.DateOffset(months=1)


# 연도 변수 만들기
economics['year'] =  economics['date2'].dt.year 
economics.head()

 
sns.lineplot(data= economics, x = 'year', y='unemploy')
plt.show()

# as_index=False해야 year도 새로운 칼럼이된다.
my_df=economics.groupby('year', as_index=False)\ 
         .agg(
           mon_mean=('unemploy', 'mean'),
           mon_std = ('unemploy', 'std'),
           mon_n = ('unemploy', 'count')
         )
my_df

mon_mean + 1.96*std/sqrt(12)
my_df['left_ci'] =  my_df['mon_mean'] - (1.96 * my_df['mon_std'] / np.sqrt(my_df['mon_n']))
my_df['right_ci'] = my_df['mon_mean'] + (1.96* (my_df['mon_std'] / np.sqrt(my_df['mon_n']))
my_df

# 각행의 1월~12월 실업자 수 평균을 구해서 표준편차도 구해서, line 그래프는 표본평균을 그려준다.
import matplotlib.pyplot as plt
x = my_df['year']
y = my_df['mon_mean']


plt.plot(x, y)
plt.scatter(x, my_df['left_ci'], s=3) 
plt.scatter(x, my_df['right_ci'], s=3) 

plt.show()
plt.clf()
