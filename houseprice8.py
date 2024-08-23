# 검증데이터, 훈련데이터 나눠서 성능 회귀 분석 측정

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

house_train.shape
house_test.shape

# 더미변수 생성
neighborhood_dummies = pd.get_dummies(
    combine_df["Neighborhood"],
    drop_first=True)



## 이상치 탐색 여기다 하면안됨. 이상치가 제거되면서 인덱스가 변하기 때문이다.
#  validation set 나누고 나서 이상치 제거 할것
# house_train=house_train.query("GrLivArea <= 4500")


# 더미변수 데이터를 train, test으로 분리
train_dummies = neighborhood_dummies.iloc[:1460,]

test_dummies = neighborhood_dummies.iloc[1460:,]
test_dummies = test_dummies.reset_index(drop=True)

# 필요한 변수들만 골라서 더미 데이터 합치기
my_train = pd.concat([house_train[["SalePrice", "GrLivArea", "GarageArea"]],
               train_dummies], axis=1)

my_test_x = pd.concat([house_test[["GrLivArea", "GarageArea"]],
               test_dummies], axis=1)

# train 데이터의 길이
train_n = len(my_train) # 1460

## Validation 셋(모의고사 셋) 만들기
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size = 438,
                 replace = False) #30% 정도의 갯수를 랜덤으로 고르기.

new_valid = my_train.loc[val_index]  # 30% 438개
new_train = my_train.drop(val_index) # 70% 1022개

# 이상치 탐색 및 삭제 (validation을 나눠서 여기서 이상치 지워도 상관없다.변함이 없다)
new_train = new_train.query("GrLivArea <= 4500")

# train 데이터에서 가격 분리
train_x = new_train.iloc[:,1:]
train_y = new_train[["SalePrice"]]

valid_x = new_valid.iloc[:,1:]
valid_y = new_valid[["SalePrice"]]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)

# 성능 측정
y_hat = model.predict(valid_x)
np.mean(np.sqrt((valid_y-y_hat)**2)) #26265