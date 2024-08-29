import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

train_data= pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
sub_df = pd.read_csv("data/sample_submission.csv")

train_data.isna().sum()
test_data.isna().sum()


# 수치형만
quantitative = train_data.select_dtypes(include = [int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    train_data[col].fillna(train_data[col].mean(), inplace=True)

# 범주형만
Categorical = train_data.select_dtypes(include = [object])
Cate_selected = Categorical.columns[Categorical.isna().sum() > 0]

for col in Cate_selected:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)

# 올바른 데이터 분리
train_n = len(train_data)

df = pd.concat([train_data, test_data], ignore_index=True)

df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
)

train_df = df.iloc[:train_n,]
test_df = df.iloc[train_n:,]

train_x = train_df.drop("yield", axis=1)
train_y = train_df["yield"]

test_x = test_df.drop("yield", axis=1, errors='ignore')

kf = KFold(n_splits=10, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, 
                                     train_x, 
                                     train_y, 
                                     cv = kf,
                                     n_jobs=-1, 
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

alpha_values = np.arange(0, 1, 0.001)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})   

plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)
model = Lasso(alpha=0.00001)

model.fit(train_x, train_y)

pred_y = model.predict(test_x)

sub_df["yield"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("./라쏘_알파마지막도전.csv", index=False)
