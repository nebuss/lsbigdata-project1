<<<전체 코드>>>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 워킹 디렉토리 설정
import os
cwd=os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)

## 필요한 데이터 불러오기
house_train=pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LSbigdata-project1/houseprice/data/train.csv")
house_test=pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LSbigdata-project1/houseprice/data/test.csv")
sub_df=pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LSbigdata-project1/houseprice/data/sample_submission.csv")


# 1. 데이터 전처리 개선
# SalePrice 로그 변환
house_train['SalePrice'] = np.log1p(house_train['SalePrice'])

# NaN 채우기
# 숫자형 변수는 평균으로, 범주형 변수는 최빈값으로 채우기
for col in house_train.select_dtypes(include=[np.number]).columns:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
    
for col in house_train.select_dtypes(include=[object]).columns:
    house_train[col].fillna(house_train[col].mode()[0], inplace=True)

for col in house_test.select_dtypes(include=[np.number]).columns:
    house_test[col].fillna(house_test[col].mean(), inplace=True)
    
for col in house_test.select_dtypes(include=[object]).columns:
    house_test[col].fillna(house_test[col].mode()[0], inplace=True)

# 이상치 제거 (SalePrice가 500000 이상인 데이터 제거)
house_train = house_train[house_train['SalePrice'] < np.log1p(500000)]

# 데이터 병합 및 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)
df = pd.get_dummies(df, drop_first=True)

# train/test 분리
train_n = len(house_train)
train_df = df.iloc[:train_n, :]
test_df = df.iloc[train_n:, :]

# train_x, train_y, test_x 생성
train_x = train_df.drop(["Id", "SalePrice"], axis=1)
train_y = train_df["SalePrice"]
test_x = test_df.drop(["Id", "SalePrice"], axis=1)

# 2. StackingRegressor 사용
# 모델 정의
eln_model = ElasticNet(random_state=42)
rf_model = RandomForestRegressor(random_state=42)

# 그리드 서치 for ElasticNet
param_grid_eln = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}

grid_search_eln = GridSearchCV(
    estimator=eln_model, 
    param_grid=param_grid_eln, 
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_eln.fit(train_x, train_y)
best_eln_model = grid_search_eln.best_estimator_

# 그리드 서치 for RandomForest
param_grid_rf = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [20, 10, 5],
    'min_samples_leaf': [5, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None]
}

grid_search_rf = GridSearchCV(
    estimator=rf_model, 
    param_grid=param_grid_rf, 
    scoring='neg_mean_squared_error',
    cv=10
)
grid_search_rf.fit(train_x, train_y)
best_rf_model = grid_search_rf.best_estimator_

# 스택킹 모델 설정
estimators = [
    ('eln', best_eln_model),
    ('rf', best_rf_model)
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(random_state=42)
)

# 스택킹 모델 학습
stacking_model.fit(train_x, train_y)

# 3. 테스트 데이터 예측
pred_y = stacking_model.predict(test_x)

# 로그 역변환
pred_y = np.expm1(pred_y)

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y


sub_df.to_csv("C:/Users/USER/Documents/LS빅데이터스쿨/LSbigdata-project1/houseprice/data/submission0911_4.csv", index=False)
