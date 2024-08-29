import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 데이터 로드
berry_train = pd.read_csv("blueberry/data/train.csv")
berry_test= pd.read_csv("blueberry/data/test.csv")
sub_df = pd.read_csv("blueberry/data/sample_submission.csv")

# 데이터 전처리
X = berry_train.drop(["yield", "id"], axis=1)
y = berry_train["yield"]
berry_test = berry_test.drop(["id"], axis=1)

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_X_scaled = scaler.transform(berry_test)

# 다항 특성 생성
polynomial_transformer = PolynomialFeatures(3)
X_poly = polynomial_transformer.fit_transform(X_scaled)
test_X_poly = polynomial_transformer.transform(test_X_scaled)

# 교차 검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_poly, y, cv=kf,
                                     n_jobs=-1, scoring="neg_mean_squared_error").mean())
    return(score)

# 각 모델 학습 및 예측
models = {
    "Ridge": Ridge(alpha=0.03),  # 알파 값을 미세 조정
    "Lasso": Lasso(alpha=0.005),  # 알파 값을 미세 조정
    "LinearRegression": LinearRegression()  # 단순 선형 회귀 추가
}

predictions = pd.DataFrame()

for name, model in models.items():
    model.fit(X_poly, y)
    pred = model.predict(test_X_poly)
    predictions[name] = pred

# 가중치 합을 사용한 배깅
# 가중치 조정: 성능을 기반으로 가중치를 미세 조정
weights = {
    "Ridge": 0.4,  # Ridge의 가중치를 조금 더 높게 설정
    "Lasso": 0.3,
    "LinearRegression": 0.3  # Linear Regression 가중치를 추가
}

pred_weighted = sum(predictions[name] * weight for name, weight in weights.items())

# 최종 예측값 저장
sub_df["yield"] = pred_weighted
sub_df.to_csv("./submission_berry_bagging_with_linear.csv", index=False)

# 모델별 성능 평가
for name, model in models.items():
    score = rmse(model)
    print(f"{name} RMSE: {score}")
