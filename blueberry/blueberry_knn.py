import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
train_data = pd.read_csv("blueberry/data/train.csv")
test_data = pd.read_csv("blueberry/data/test.csv")
sub_df = pd.read_csv("blueberry/data/sample_submission.csv")
train_data.columns
# 결측치 처리
# 수치형 데이터 결측치 처리
quantitative = train_data.select_dtypes(include=[int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    train_data[col].fillna(train_data[col].mean(), inplace=True)

# 범주형 데이터 결측치 처리
Categorical = train_data.select_dtypes(include=[object])
Cate_selected = Categorical.columns[Categorical.isna().sum() > 0]

for col in Cate_selected:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)

# 데이터 결합 후 원-핫 인코딩
train_n = len(train_data)
df = pd.concat([train_data, test_data], ignore_index=True)
df = pd.get_dummies(df, columns=df.select_dtypes(include=[object]).columns, drop_first=True)

# 다시 train과 test 데이터로 분리
train_df = df.iloc[:train_n, :]
test_df = df.iloc[train_n:, :]

# 피처와 타겟 변수 분리
train_x = train_df.drop("yield", axis=1)
train_y = train_df["yield"]
test_x = test_df.drop("yield", axis=1, errors='ignore')

# 데이터 스케일링
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# KFold 설정
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

# RMSE 계산 함수
def rmse(model):
    score = np.sqrt(-cross_val_score(model, 
                                     train_x, 
                                     train_y, 
                                     cv=kf,
                                     n_jobs=-1, 
                                     scoring="neg_mean_squared_error").mean())
    return(score)

# k 값의 범위 설정 및 최적의 k 값 찾기
k_values = np.arange(1, 20)
mean_scores = np.zeros(len(k_values))

for i, k in enumerate(k_values):
    knn = KNeighborsRegressor(n_neighbors=k)
    mean_scores[i] = rmse(knn)

# 결과 시각화
df = pd.DataFrame({
    'k': k_values,
    'validation_error': mean_scores
})   

plt.plot(df['k'], df['validation_error'], label='Validation Error', color='blue')
plt.xlabel('k')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('KNN Regression Train vs Validation Error')
plt.show()

# 최적의 k 값 찾기
optimal_k = df['k'][np.argmin(df['validation_error'])]
print("Optimal k:", optimal_k)

# 최적의 k 값으로 KNN 모델 학습
model = KNeighborsRegressor(n_neighbors=int(optimal_k))
model.fit(train_x, train_y)

# 예측
pred_y = model.predict(test_x)

# 결과 저장
sub_df["yield"] = pred_y
sub_df.to_csv("./KNN_최적k.csv", index=False)
