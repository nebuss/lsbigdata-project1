# 직선의 방정식
# y = ax + b
# y=x+2
# 예: y = 2x + 3 의 그래프를 그려보세요!
a = 1
b = 0

x = np.linspace(-5, 5, 100)
y = a * x + b

import numpy as np
import matplotlib.pyplot as plt

plt.plot(x, y, color="blue")
plt.axvline(0, color="black")
plt.axhline(0, color="black")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
plt.clf()

import pandas as pd

a = 80
b = 5

x = np.linspace(0, 5, 100)
y = a * x + b

house_train=pd.read_csv("./data/houseprice/train.csv")
my_df=house_train[["BedroomAbvGr", "SalePrice"]].head(10)
my_df["SalePrice"]=my_df["SalePrice"] /1000
plt.scatter(x=my_df["BedroomAbvGr"], y=my_df["SalePrice"])
plt.plot(x, y, color="blue")
plt.show()
plt.clf()

y = 70 * x + 10
방 1개 집값: 8천만
방 2개 집값: 1억 5천
방 3개 집값: 2억 2천
방 4개 집값: 2억 9천
방 5개 집값: 3억 6천
# 1조: (70, 10) - 0.52186
# 2조: (12, 170) - 0.46138
# 3조: (53, 45) - 0.46803 a방법 1등 값
# 4조: (36, 68) - 0.42609 b방법 1등 값값
# 5조: (80, -30) - 0.69897
# 6조: (47, 63) - 0.46078
# 7조: (63, 100) - 1.376
# 회귀: (16.38, 133.966)

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
a=16
b=117

# y_hat 어떻게 구할까?
y_hat=(a * house_train["BedroomAbvGr"] + b) * 1000
# y는 어디에 있는가?
y=house_train["SalePrice"]

np.abs(y - y_hat)  # 절대거리
np.sum(np.abs(y - y_hat)) # 절대값 합
np.sum((y - y_hat)**2) # 제곱합

# 1조: 106021410
# 2조: 94512422
# 3조: 93754868 1등
# 4조: 81472836
# 5조: 103493158

# 1조: 106021410
# 2조: 94512422
# 3조: 93754868 
# 4조: 81472836 1등
# 5조: 103493158
# 9459298301338
# 회귀(절대값): 79617742
# 회귀(제곱합): 82373710


# !pip install scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()


# 회귀모델을 통한 집값 예측

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

## 회귀분석 적합(fit)하기 
x = np.array(house_train["BedroomAbvGr"]).reshape(-1, 1)
y = house_train["SalePrice"]/1000

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

test_x = np.array(house_test["BedroomAbvGr"]).reshape(-1, 1)
test_x

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y*1000
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission4.csv", index=False)

# ====================
# =====   옵션   =====
# ====================

import numpy as np
from scipy.optimize import minimize

# 최소값을 찾을 다변수 함수 정의
def my_f(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# 회귀직선 구하기

import numpy as np
from scipy.optimize import minimize

def line_perform(par):
    y_hat=(par[0] * house_train["BedroomAbvGr"] + par[1]) * 1000
    y=house_train["SalePrice"]
    return np.sum(np.abs((y-y_hat)))

line_perform([36, 68])

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# y=x^2 + 3의 최소값이 나오는 입력값 구하기
def my_f(x):
    return x**2+3

my_f(3)

import numpy as np
from scipy.optimize import minimize

# 초기 추정값
initial_guess = [-10]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# z = x^2 + y^2 + 3
def my_f2(x):
    return x[0]**2 + x[1]**2 + 3

my_f2([1, 3])

# 초기 추정값
initial_guess = [-10, 3]

# 최소값 찾기
result = minimize(my_f2, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# f(x, y, z) = (x-1)^2 + (y-2)^2 + (z-4)^2 + 7
def my_f3(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-4)**2 + 7

my_f3([1, 2, 3])

# 초기 추정값
initial_guess = [-10, 3, 4]

# 최소값 찾기
result = minimize(my_f3, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

