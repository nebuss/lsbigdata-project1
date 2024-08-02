import numpy as np
import pandas as pd

a=2
b=0
x= np.linspace(-5, 5, 50)

y = a*x + b
plt.plot(x, y)
plt.axvline(0, color='black')
plt.axhline(0, color='black')\
plt.xlim(-5, 5)
plt.ylim(-5, 5)

plt.show()
plt.clf()

a= 80
b = 5
x= np.linspace(0, 5, 100)
y=a*x + b

house_df = pd.read_csv("./data/train.csv")
my_df=house_df[['BedroomAbvGr', 'SalePrice']]
my_df['SalePrice']=my_df['SalePrice']/1000
plt.scatter(x=my_df['BedroomAbvGr'], y =my_df['SalePrice'], color='orange')
plt.plot(x,  y)
plt.show()
plt.clf()


# 테스트 집 정보 가져오기
house_test=pd.read_csv("./data/houseprice/test.csv")
a=70; b=10
(a * house_test["BedroomAbvGr"] + b) * 1000

# sub 데이터 불러오기
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

# SalePrice 바꿔치기
sub_df["SalePrice"] = (a * house_test["BedroomAbvGr"] + b) * 1000
sub_df


# 직선을 구하는 방법


sub_df.to_csv("./data/houseprice/sample_submission3.csv", index=False)


# 직선 성능 평가
a=70
b=10

# y_hat은 어떻게 구할까?

y_hat = (a* my_df['BedroomAbvGr']+ b) * 1000

# y는 어디에 있는가?
y=  my_df['SalePrice']

np.sum(abs(y- y_hat))


!pip install scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편값 구해줌                                                                   


model.coef_ # 계수(기울기 값)
model.intercept_ # 절편(b 값)
# 회귀 직선의 기울기와 절편
slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 
y_pred = model.predict(x) 

plt.rcParams.update({'font.family' : 'Malgun Gothic'})
# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()
