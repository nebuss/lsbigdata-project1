import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
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

# 모델 평가 함수
def rmse(model):
    score = np.sqrt(-cross_val_score(model, 
                                     train_x, 
                                     train_y, 
                                     cv=kf,
                                     n_jobs=-1, 
                                     scoring="neg_mean_squared_error").mean())
    return score

# ElasticNet 하이퍼파라미터 탐색
alpha_values = np.arange(0.001, 1, 0.01)
l1_ratios = np.arange(0.1, 1.0, 0.1)
best_score = float('inf')
best_alpha = None
best_l1_ratio = None

for alpha in alpha_values:
    for l1_ratio in l1_ratios:
        elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=2024)
        score = rmse(elastic_net)
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_l1_ratio = l1_ratio

print(f"Optimal alpha: {best_alpha}, Optimal l1_ratio: {best_l1_ratio}")

# 최적의 하이퍼파라미터로 모델 학습
model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, random_state=2024)
model.fit(train_x, train_y)

# 예측 수행
pred_y = model.predict(test_x)

# 예측 결과를 제출 파일에 추가
sub_df["yield"] = pred_y

# 결과를 CSV 파일로 저장
sub_df.to_csv("./블루베리_ElasticNet도전.csv", index=False)
