import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 로드
train_data= pd.read_csv("data/train.csv")
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

# 모델 평가 함수
def rmse(model):
    score = np.sqrt(-cross_val_score(model, 
                                     train_x, 
                                     train_y, 
                                     cv=kf,
                                     n_jobs=-1, 
                                     scoring="neg_mean_squared_error").mean())
    return score

# 알파 값 탐색
alpha_values = np.arange(0.01, 1.01, 0.001) 
mean_scores = np.zeros(len(alpha_values))

for i, alpha in enumerate(alpha_values):
    ridge = Ridge(alpha=alpha)
    mean_scores[i] = rmse(ridge)

# 최적의 알파 값 찾기
optimal_alpha = alpha_values[np.argmin(mean_scores)]
print("Optimal alpha:", optimal_alpha)

# 최적의 알파 값으로 모델 학습
model = Ridge(alpha=optimal_alpha)
model.fit(train_x, train_y)

# 예측 수행
pred_y = model.predict(test_x)

# 예측 결과를 제출 파일에 추가
sub_df["yield"] = pred_y

# 결과를 CSV 파일로 저장
sub_df.to_csv("./yield_prediction_ridge2.csv", index=False)

# 결과 시각화
plt.plot(alpha_values, mean_scores, label='Validation Error', color='red')
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.legend()
plt.title('Ridge Regression - Validation Error vs Alpha')
plt.show()