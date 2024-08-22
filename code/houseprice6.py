# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

## 이상치 탐색
# house_train=house_train.query("GrLivArea <= 4500")


## 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임
# 숫자형 변수만 선택하기
x = house_train.select_dtypes(include=[int, float])
# 필요없는 칼럼 제거하기
x = x.iloc[:,1:-1]
y = house_train["SalePrice"]


# 변수별로 결측값 채우기
fill_values = {
    'LotFrontage': x["LotFrontage"].mean(),
    'MasVnrArea': x["MasVnrArea"].mean(),
    'GarageYrBlt': x["GarageYrBlt"].mean()
}
x = x.fillna(value=fill_values)
x.isna().sum()
x.mean()

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 테스트 데이터 예측
test_x = house_test.select_dtypes(include=[int, float])
test_x = test_x.iloc[:,1:]

# 결측치 채우기
test_x=test_x.fillna(test_x.mean())


# 결측치 확인
test_x.isna().sum()
test_x.mean()
# 테스트 데이터 집값 예측
pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission9.csv", index=False)
