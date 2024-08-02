# 선형회귀 모델이용해서 내가원하는 변수로 집값 예측해보기
# 리모델링 년도로 예측!

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

house_train = pd.read_csv("data/train.csv")
my_df = house_train[["MSSubClass", "SalePrice"]]

# 선형 회귀 모델 생성
model = LinearRegression()



x = np.array(my_df["MSSubClass"]).reshape(-1,1)
y = my_df["SalePrice"]

# 모델 학습
model.fit(x, y) #자동으로 기울기, 절편 값을 구해줌


# test셋의 x를 꺼내서 predict에 넣는다.
house_test = pd.read_csv("data/test.csv")
test_x = np.array(house_test["MSSubClass"]).reshape(-1,1)


pred_y = model.predict(test_x)    # test 셋에 대한 집값(모델학습 완료)
 
sub = pd.read_csv("data/sample_submission.csv")
#SalePrice 바꿔치기
sub["SalePrice"] = pred_y


sub.to_csv("sub_predictionMSSubClass.csv", index=False)

## GrLivArea로 해보기 

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#필요한 데이터 불러오기
house_test = pd.read_csv('data/houseprice/test.csv')
house_train = pd.read_csv('data/houseprice/train.csv')
sub_df = pd.read_csv('data/houseprice/sample_submission.csv')

#이상치 탐색 및 제거 
house_train.query("GrLivArea > 4500") #탐색
house_train = house_train.query("GrLivArea <= 4500") #4500보다 작거나 같은 것만 할당해줌


# 회귀분석 적합(FIT)하기 
x = np.array(house_train["GrLivArea"]).reshape(-1, 1)
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

#csv로 바꿔치기
# sub_df.to_csv("data/houseprice/sample_submission9.csv", index = False)
