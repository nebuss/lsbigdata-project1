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

# 데이터 통합 및 원핫 인코딩
train_n = len(train_data)
df = pd.concat([train_data, test_data], ignore_index=True)

df = pd.get_dummies(
    df,
    columns=df.select_dtypes(include=[object]).columns,
    drop_first=True
)

train_df = df.iloc[:train_n, :]
test_df = df.iloc[train_n:, :]

# 독립 변수와 종속 변수 분리
train_x = train_df.drop("yield", axis=1)
train_y = train_df["yield"]

test_x = test_df.drop("yield", axis=1, errors='ignore')

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

print("RMSE of Linear Regression:", rmse_score)

# 모델 학습
model.fit(train_x, train_y)

# 예측 수행
pred_y = model.predict(test_x)

# 예측 결과를 제출 파일에 추가
sub_df["yield"] = pred_y

# 결과를 CSV 파일로 저장
sub_df.to_csv("./블루베리_단순선형회귀.csv", index=False)
