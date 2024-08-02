# 선형회귀 모델이용해서 내가원하는 변수로 집값 예측해보기
# 리모델링 년도로 예측!

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

house_train = pd.read_csv("data/train.csv")
my_df = house_train[["YearRemodAdd", "SalePrice"]]

# 선형 회귀 모델 생성
model = LinearRegression()

x = np.array(my_df["YearRemodAdd"]).reshape(-1,1)
y = my_df["SalePrice"]

# 모델 학습
model.fit(x, y) #자동으로 기울기, 절편 값을 구해줌


# test셋의 x를 꺼내서 predict에 넣는다.
house_test = pd.read_csv("data/test.csv")
test_x = np.array(house_test["YearRemodAdd"]).reshape(-1,1)


pred_y = model.predict(test_x)    # test 셋에 대한 집값(모델학습 완료)
 
sub = pd.read_csv("data/submission.csv")
#SalePrice 바꿔치기
sub["SalePrice"] = pred_y


sub.to_csv("sub_predictionREMO.csv", index=False)

