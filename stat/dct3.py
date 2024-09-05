# 펭귄데이터 부리길이 예측 모형 만들기
# 엘라스틱넷 & 의사결정나무 회귀모델 사용
# 모든 변수 자유롭게 사용.
# 종속변수: bill_length_mm

from palmerpenguins import load_penguins
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler


penguins = load_penguins()

penguins = penguins.dropna()

numeric_features = ['bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = penguins[numeric_features]  # 독립변수: 나머지 변수들
y = penguins['bill_length_mm']  # 종속변수: 부리 길이 (bill_length_mm)

X.isna().sum()
y.isna().sum()

# 결측치 처리
X["bill_depth_mm"] =X["bill_depth_mm"].fillna(X["bill_depth_mm"].mean())
X["flipper_length_mm"] = X["flipper_length_mm"].fillna(X["flipper_length_mm"].mean())
X["body_mass_g"] =X["body_mass_g"].fillna(X["body_mass_g"].mean())

X.isna().sum()

y = y.fillna(y.mean())
y.isna().sum()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
####의사결정나무
model = DecisionTreeRegressor(random_state=42, max_depth=2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred)
print("의사결정나무 MSE:", mse_tree)

# 예측 결과 출력
print(y_pred)

######### 엘라스틱 넷
from sklearn.metrics import mean_squared_error
elasticnet_model = ElasticNet()

param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}              # 하이퍼 파라미터 튜닝

from sklearn.model_selection import GridSearchCV

grid_search=GridSearchCV(
    estimator=elasticnet_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(X, y)
elasticnet_model.fit(X_train, y_train)

print("최적의 하이퍼파라미터:", grid_search.best_params_)


y_pred_elasticnet = elasticnet_model.predict(X_test)

# 성능 
mse_elasticnet = mean_squared_error(y_test, y_pred_elasticnet)
print("ElasticNet 예측값:", y_pred_elasticnet)
print("ElasticNet MSE:", mse_elasticnet)


#######===지원 언니 코드
# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

df=sns.load_dataset("penguins")

df.isnull().sum()
df.iloc[:,2:6]=df.iloc[:,2:6].fillna(df.iloc[:,2:6].mean())
df.sex=df.sex.fillna("Male")
df = pd.get_dummies(
    df,
    columns = df.select_dtypes(include=[object]).columns,
    drop_first = True
)
df

X=df.drop(columns="bill_length_mm")
Y=df.bill_length_mm


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(X, Y, test_size=0.3)


# ElasticNet / 그리드서치
from sklearn.linear_model import ElasticNet
ela = ElasticNet()
param_grid={
    "alpha": [0.1 , 1.0 , 10.0 , 100.0], 
    "l1_ratio":[0, 0.1,0.5,1.0]}

from sklearn.model_selection import  GridSearchCV
grid_search = GridSearchCV(
    estimator=ela,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3, 
    refit=True
)


grid_search.fit(x_train, y_train)

grid_search.best_params_ # 0.1,1
grid_search.cv_results_
grid_search.best_score_# -5.9
best_model=grid_search.best_estimator_
#-----------------------------------------
#  DecisionRegressor/ 그리드서치
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=2024)
param_grid={
    'max_depth': [0,5,10,50],
    'min_samples_split': [0,5,10,50]
}

grid_search=GridSearchCV(
    estimator=dtr,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    refit=True
)

grid_search.fit(x_train, y_train)

grid_search.best_params_ #max_depth:50, mss:10
grid_search.cv_results_
grid_search.best_score_ #-7.7
best_model=grid_search.best_estimator_


### ========상후네 조 

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins=load_penguins()
penguins.head()

## Nan 채우기
quantitative = penguins.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    penguins[col].fillna(penguins[col].mean(), inplace=True)
penguins[quant_selected].isna().sum()

## 범주형 채우기
qualitative = penguins.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    penguins[col].fillna(penguins[col].mode()[0], inplace=True)
penguins[qual_selected].isna().sum()

df = penguins
df = pd.get_dummies(
    df,
    columns = df.select_dtypes(include=[object]).columns,
    drop_first = True
)
df

x=df.drop("bill_length_mm", axis=1)
y=df[['bill_length_mm']]
x
y

## 모델 생성
from sklearn.linear_model import ElasticNet
model = ElasticNet()

## 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV

param_grid={
    'alpha': np.arange(0, 0.2, 0.01),
    'l1_ratio': np.arange(0.8, 1, 0.01)
}

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(x,y)

grid_search.best_params_ #alpha=0.19, l1_ratio=0.99
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

##
# 모델 생성
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)
param_grid={
    'max_depth': np.arange(7, 20, 1),
    'min_samples_split': np.arange(10, 30, 1)
}

# 하이퍼파라미터 튜닝
grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(x,y)

grid_search.best_params_ #8, 22
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

### 트리 시각화
model = DecisionTreeRegressor(random_state=42,
                              max_depth=2,
                              min_samples_split=22)
model.fit(x,y)

from sklearn import tree
tree.plot_tree(model)