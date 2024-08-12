import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#필요한 데이터 불러오기
house_test = pd.read_csv('houseprice/data/test.csv')
house_train = pd.read_csv('houseprice/data/train.csv')
sub_df = pd.read_csv('houseprice/data/sample_submission.csv')

#이상치 탐색 및 제거 
# house_train.query("GrLivArea > 4500") #탐색
house_train = house_train.query("GrLivArea <= 4500") #4500보다 작거나 같은 것만 할당해줌
house_train["GrLivArea"]

# 회귀분석 적합(FIT)하기 
# np.arrray  - reshape( , ) 사용해서 array 배열을 세로벡터 형태로 만들어 줘도 되지만,
# []두개로 리스트형태로 불러오면 판다스 데이터프레임 형태가 돼서, np.array로 불러오지 않아도 된다.
x = house_train[["GrLivArea", "GarageArea"]]
y = np.array(house_train["SalePrice"])
# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) #fit함수가 자동으로 기울기, 절편 값을 구해줌.

# 회귀 직선의 기울기와 절편
model.coef_         #기울기 a
model.intercept_    #절편 b


slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산, x는 방 개수
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()


test_x = np.array(house_test["GrLivArea"]).reshape(-1,1)
test_x


pred_y = model.predict(test_x) #test셋에 대한 집값
pred_y



#SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

==================================================================
# 원하는 변수를 사용해서 회귀모델을 만들고, 제출할것!
# 원하는 변수 2개
# 회귀모델을 통한 집값 예측

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("houseprice/data/train.csv")
house_test=pd.read_csv("houseprice/data/test.csv")
sub_df=pd.read_csv("houseprice/data/sample_submission.csv")

## 이상치 탐색
house_train=house_train.query("GrLivArea <= 4500")

## 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임
x = house_train[["GrLivArea", "GarageArea"]]
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

def my_houseprice(x, y):
    return model.coef_[0]*x + model.coef_[1]*y + model.intercept_

my_houseprice(300, 55)

my_houseprice(house_test["GrLivArea"], house_test["GarageArea"])

test_x = house_test[["GrLivArea", "GarageArea"]]
test_x

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("houseprice/data/sample_submission7.csv", index=False)

