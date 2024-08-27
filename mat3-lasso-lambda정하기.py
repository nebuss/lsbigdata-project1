import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자능
np.random.seed(42)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i # 독립변수 20개 y가 1개인 데이터 셋
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

from sklearn.linear_model import Lasso

# 결과 받기 위한 벡터 만들기. 0 백번 써놓음.
val_result=np.repeat(0.0,100)  
tr_result=np.repeat(0.0,100)


for i in np.arange(0, 100):
        model= Lasso(alpha=i*(0.1)) #lambda가 alpha로 표현됨. I가 0이면 일반 회귀모델(모든 데이터 사용)
        model.fit(train_x, train_y)

        y_hat_train = model.predict(train_x)
        y_hat_val = model.predict(valid_x)

        pref_train=sum((train_df["y"] - y_hat_train)**2) # 잔차제곱합을 통해서 성능 측정
        pref_val=sum((valid_df["y"] - y_hat_val)**2)
        val_result[i]=pref_val
        tr_result[i]=pref_train

model.coef_

val_result
tr_result




### 용규오빠 코드
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자능
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

valid_df = df.loc[20:]
valid_df

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_x
valid_y = valid_df["y"]
valid_y

from sklearn.linear_model import Lasso

val_result=np.repeat(0.0, 100)
tr_result=np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train=sum((train_df["y"] - y_hat_train)**2)
    perf_val=sum((valid_df["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val
    tr_result
    val_result

import seaborn as sns

df = pd.DataFrame({
    'l': np.arange(0, 1, 0.01), 
    'tr': tr_result,
    'val': val_result
})

#

val_result[0]
val_result[1]
np.min(val_result)

# alpha를 0.03로 선택!
np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]



model= Lasso(alpha=0.03)
model.fit(train_x,train_y)
#간격 0.01 에 대한 예측값 계산
k = np.arange(-4,4,0.01)
df2 = pd.DataFrame({
    "x" : k
})
df2
for i in range(2, 21):
    df2[f"x{i}"] = df2["x"] ** i
df2_y = model.predict(df2)


## valid set , valid set 에 대한 y
# expect_y = model.predict(valid_x)
plt.plot(df2["x"],df2_y,color="red")
plt.scatter(valid_x["x"], valid_y)


## 수업중 퀴즈. 추정된 라쏘(lambda=0.03)모델 사용해서 간격이 -4, 4 까지 간격 0.01 x에 대하여 예측값 계산, 산점도 valid set 파란색으로 그린 다음, -4, 4까지 예측값을 빨간선으로 겹쳐서 그릴것 . 선생님 코드



k = np.linspace(-4, 4, 800) # 그대로 쓸 수 없음.


k_df = pd.DataFrame({
    "x" : k
})
k_df



for i in range(2, 21):
    train_df[f"x{i}"] = k_df["x"] ** i

k_df


## 5개 교차검증해서 최적값 평균, 표준편차 구하기
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    "x": x,
    "y": y
})

# 다항식 특성 생성
for i in range(2, 21):  # 예시로 2차부터 5차 항까지 생성
    df[f"x{i}"] = df["x"] ** i

# KFold 설정
kf = KFold(n_splits=5)

# 모델 설정
model = Lasso(alpha=0.03)

# 교차 검증 수행 및 MSE 계산
mse_scores = []

for train_index, valid_index in kf.split(df):
    train_df = df.iloc[train_index]
    valid_df = df.iloc[valid_index]
    
    train_x = train_df.drop(columns=["y"])
    train_y = train_df["y"]
    valid_x = valid_df.drop(columns=["y"])
    valid_y = valid_df["y"]
    
    # 모델 학습
    model.fit(train_x, train_y)
    
    # 예측
    reg_line = model.predict(valid_x)
    # MSE 계산
    mse = mean_squared_error(valid_y, reg_line)
    mse_scores.append(mse)
    
    # 그래프 그리기 (원하는 경우 생략 가능)
    plt.plot(valid_x["x"], reg_line, color="red")
    plt.scatter(valid_x["x"], valid_y, color="blue")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Lasso Regression with 5-Fold Cross Validation")
plt.show()

# MSE 평균 및 표준편차 계산
mse_mean = np.mean(mse_scores)
mse_std = np.std(mse_scores)

print(f"MSE 평균: {mse_mean:.4f}")
print(f"MSE 표준편차: {mse_std:.4f}")

## 교차검증 용규뀨 코드
# 초기 설정
example_list = list(range(0, 30))
np.random.shuffle(example_list)
groups = [example_list[i:i + 6] for i in range(0, len(example_list), 6)]

np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    "y": y,
    "x": x
})

# 폴드별 예측 결과를 저장할 리스트
all_predictions = []

# 폴드 순회 및 cross-validation 진행
for fold in range(5):
    # 검증 세트와 훈련 세트 구분
    valid_idx = groups[fold]
    train_idx = [idx for group in groups if group != valid_idx for idx in group]
    
    train_df = df.loc[train_idx].copy()
    valid_df = df.loc[valid_idx].copy()
    
    # 다항식 특성 생성
    for i in range(2, 21):
        train_df[f"x{i}"] = train_df["x"] ** i
        valid_df[f"x{i}"] = valid_df["x"] ** i
    
    # 훈련 데이터와 검증 데이터 분리
    train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
    train_y = train_df["y"]
    
    valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
    valid_y = valid_df["y"]

    # Lasso 모델 학습
    model = Lasso(alpha=0.08)
    model.fit(train_x, train_y)
    
    # 새로운 x 값에 대한 예측
    k = np.arange(-4, 4, 0.01)
    df2 = pd.DataFrame({"x": k})
    for i in range(2, 21):
        df2[f"x{i}"] = df2["x"] ** i
    df2_y = model.predict(df2)
    
    # 예측 결과를 리스트에 추가
    all_predictions.append(df2_y)

# 폴드별 예측 값의 평균 계산
mean_prediction = np.mean(all_predictions, axis=0)

# 그래프 그리기
plt.figure(figsize=(10, 8))
plt.plot(k, mean_prediction, color="red", label="Mean Prediction")
plt.scatter(df["x"], df["y"], color="blue", edgecolor='black', label="Original Data")
plt.title("Lasso Regression - Mean Prediction Across 5 Folds")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")
plt.show()


df=pd.read_clipboard()
df