## 0805 숫자형 변수 모두가져와서 가격 예측하기(다항 회귀 분석)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


## 필요한 데이터 불러오기
house_train=pd.read_csv("houseprice/data/train.csv")
house_test=pd.read_csv("houseprice/data/test.csv")
sub_df=pd.read_csv("houseprice/data/sample_submission.csv")

house_train.info()




## 회귀분석 적합(fit)하기
# 숫자형 데이터를 가진 열만 불러오기

x = house_train.select_dtypes(include = [int, float])
x.info()
test_x.info()
# 필요없는 ID, SalePrice 열 빼고 불러오기
x = x.iloc[:, 1:-1]
y = house_train["SalePrice"]


# x결측치 처리하기
# x.isna().sum()
# x['LotFrontage'].fillna(x['LotFrontage'].mean(), inplace= True)
# x['GarageYrBlt'].fillna(x['GarageYrBlt'].mean(), inplace= True)
# x['MasVnrArea'].fillna(x['MasVnrArea'].mean(), inplace= True)



fill_values = {
    'LotFrontage': x["LotFrontage"].mean(),  
    'MasVnrArea': x["MasVnrArea"].mean()[0], 
    'GarageYrBlt': x["GarageYrBlt"].mean()[0]
}
x = x.fillna(value=fill_values)
x.isna().sum()

# 선형 회귀 모델 생성
model = LinearRegression()


# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 테스트 데이터 예측
test_x = house_test.select_dtypes(include = [int, float])
test_x

# 결측치 확인


# test_x = test_x.iloc[:, 1:]
# test_x['LotFrontage'].fillna(test_x['LotFrontage'].mean(), inplace= True)
# test_x['MasVnrArea'].fillna(test_x['MasVnrArea'].mean(), inplace= True)
# test_x['BsmtFinSF1'].fillna(test_x['BsmtFinSF1'].mean(), inplace= True)
# test_x['BsmtFinSF2'].fillna(test_x['BsmtFinSF2'].mean(), inplace= True)
# test_x['BsmtUnfSF'].fillna(test_x['BsmtUnfSF'].mean(), inplace= True)
# test_x['BsmtFullBath'].fillna(test_x['BsmtFullBath'].mean(), inplace= True)
# test_x['BsmtHalfBath'].fillna(test_x['BsmtHalfBath'].mean(), inplace= True)
# test_x['GarageYrBlt'].fillna(test_x['GarageYrBlt'].mean(), inplace= True)
# test_x['GarageCars'].fillna(test_x['GarageCars'].mean(), inplace= True)
# test_x['GarageArea'].fillna(test_x['GarageArea'].mean(), inplace= True)
# test_x['TotalBsmtSF'].fillna(test_x['TotalBsmtSF'].mean(), inplace= True)

# 쌤코드 / 결측치
test_x = house_test.select_dtypes(include=[int, float])
test_x = test_x.iloc[:,1:]

# fill_values = {
#     'LotFrontage': test_x["LotFrontage"].mean(),
#     'MasVnrArea': test_x["MasVnrArea"].mode()[0],
#     'GarageYrBlt': test_x["GarageYrBlt"].mode()[0]
# }
# test_x = test_x.fillna(value=fill_values)
test_x=test_x.fillna(test_x.mean())

# 테스트 데이터 집값 예측
pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("houseprice/data/many_variables6.csv", index=False)
