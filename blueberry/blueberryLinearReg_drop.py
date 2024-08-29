import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 로드
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
sub_df = pd.read_csv("data/sample_submission.csv")

# 결측치 처리 (수치형)
quantitative = train_data.select_dtypes(include=[int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    train_data[col].fillna(train_data[col].mean(), inplace=True)

# 결측치 처리 (범주형)
categorical = train_data.select_dtypes(include=[object])
cate_selected = categorical.columns[categorical.isna().sum() > 0]

for col in cate_selected:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)

# 필요한 열만 선택
selected_features = ['fruitset', 'seeds', 'fruitmass']

train_x = train_data[selected_features]
train_y = train_data["yield"]

test_x = test_data[selected_features]

# 교차 검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

# 단순 선형 회귀 모델 학습 및 평가
model = LinearRegression()
rmse_score = np.sqrt(-cross_val_score(model, 
                                      train_x, 
                                      train_y, 
                                      cv=kf,
                                      n_jobs=-1, 
                                      scoring="neg_mean_squared_error").mean())

print("RMSE of Linear Regression with selected features:", rmse_score)

# 모델 학습
model.fit(train_x, train_y)

# 예측 수행
pred_y = model.predict(test_x)

# 예측 결과를 제출 파일에 추가
sub_df["yield"] = pred_y

# 결과를 CSV 파일로 저장
sub_df.to_csv("./블루베리_drop_단순선형회귀.csv", index=False)
