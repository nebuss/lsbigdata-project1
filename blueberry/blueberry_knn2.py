import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
berry_train = pd.read_csv("blueberry/data/train.csv")
berry_test= pd.read_csv("blueberry/data/test.csv")
sub_df = pd.read_csv("blueberry/data/sample_submission.csv")

# 결측치 확인
berry_train.isna().sum()
berry_test.isna().sum()

berry_train.info()

# 피처와 타겟 변수 분리
X = berry_train.drop(["yield", "id"], axis=1)
y = berry_train["yield"]
berry_test = berry_test.drop(["id"], axis=1)

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_X_scaled = scaler.transform(berry_test)

# 정규화된 데이터를 DataFrame으로 변환
X = pd.DataFrame(X_scaled, columns=X.columns)
test_X = pd.DataFrame(test_X_scaled, columns=berry_test.columns)

# 다항 특성 생성 (차수 3)
polynomial_transformer = PolynomialFeatures(3)

polynomial_features = polynomial_transformer.fit_transform(X.values)
features = polynomial_transformer.get_feature_names_out(X.columns)
X = pd.DataFrame(polynomial_features, columns=features)

polynomial_features = polynomial_transformer.fit_transform(test_X.values)
features = polynomial_transformer.get_feature_names_out(test_X.columns)
test_X = pd.DataFrame(polynomial_features, columns=features)

# 교차 검증 설정
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X, y, cv=kf,
                                     n_jobs=-1, scoring="neg_mean_squared_error").mean())
    return(score)

# 각 k 값에 대한 교차 검증 점수 저장
k_values = np.arange(1, 21)
mean_scores = np.zeros(len(k_values))

for i, k in enumerate(k_values):
    knn = KNeighborsRegressor(n_neighbors=k)
    mean_scores[i] = rmse(knn)

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'k': k_values,
    'validation_error': mean_scores
})

# 최적의 k 값 찾기
optimal_k = df['k'][np.argmin(df['validation_error'])]
print("Optimal k:", optimal_k)

# 결과 시각화
plt.plot(df['k'], df['validation_error'], label='Validation Error', color='blue')
plt.xlabel('k')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('KNN Regression Train vs Validation Error')
plt.show()

# 최적의 k 값으로 KNN 모델 학습
model = KNeighborsRegressor(n_neighbors=int(optimal_k))

# 모델 학습
model.fit(X, y)

# 예측
pred_y = model.predict(test_X)

# 결과 저장
sub_df["yield"] = pred_y
sub_df.to_csv("KNN_optimal_k_submission.csv", index=False)
