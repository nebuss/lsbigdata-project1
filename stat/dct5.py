import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
import seaborn as sns
import matplotlib.pyplot as plt

penguins = load_penguins()
penguins.head()


df=penguins.dropna()
df=df[["species","bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={
                   'species': 'y',
                   'bill_length_mm' : 'x1',
                   'bill_depth_mm': 'x2'})
df

# x1, x2 산점도를 그리되, 점 색깔은 펭귄 종별 다르게 그리기

sns.scatterplot(x='x1', y='x2', hue='y', data=df, palette='deep')
plt.axvline(x = 45) 

# Q. 나누기 전 현재의 엔트로피는?
# 45로 나눴을 때 엔트로피 평균은 얼마인가?

# 함수사용 현재 엔트로피
from scipy.stats import entropy
import numpy as np

species_counts = df['y'].value_counts(normalize=True) # normalize => 고유값의 비율 계산

# 현재의 엔트로피 계산
current_entropy = entropy(species_counts, base=2)
current_entropy


# 45로 나눴을 때 엔트로피
# 데이터 나누기: 임의로 첫 45개 데이터와 나머지로 나누기
group1 = df.query("x1 < 45")
group2 = df.query("x1 >= 45")
# 각 그룹의 엔트로피 계산
group1_counts = group1['y'].value_counts(normalize=True)
group2_counts = group2['y'].value_counts(normalize=True)

group1_entropy = entropy(group1_counts, base=2)
group2_entropy = entropy(group2_counts, base=2)

# 전체 가중평균 엔트로피 계산
total_entropy = (len(group1) * group1_entropy + len(group2) * group2_entropy) / len(df)
total_entropy

## ====함수없이 직접 해보기
# Q. 나누기 전 현재의 엔트로피?
# Q. 45로 나눴을때, 엔트로피 평균은 얼마인가요?
# 입력값이 벡터 -> 엔트로피!
p_i=df['y'].value_counts() / len(df['y'])
entropy_curr=-sum(p_i * np.log2(p_i))

# x1=45 기준으로 나눈 후, 평균 엔트로피 구하기!
# x1=45 기준으로 나눴을때, 데이터포인트가 몇개 씩 나뉘나요?
n1=df.query("x1 < 45").shape[0]  # 1번 그룹
n2=df.query("x1 >= 45").shape[0] # 2번 그룹

# 1번 그룹은 어떤 종류로 예측하나요?
# 2번 그룹은 어떤 종류로 예측하나요?
y_hat1=df.query("x1 < 45")['y'].mode()
y_hat2=df.query("x1 >= 45")['y'].mode()

# 각 그룹 엔트로피는 얼마 인가요?
p_1=df.query("x1 < 45")['y'].value_counts() / len(df.query("x1 < 45")['y'])
entropy1=-sum(p_1 * np.log2(p_1))

p_2=df.query("x1 >= 45")['y'].value_counts() / len(df.query("x1 >= 45")['y'])
entropy2=-sum(p_2 * np.log2(p_2))

entropy_x1_45=(n1 * entropy1 + n2 * entropy2)/(n1 + n2)
entropy_x1_45




# 원래 MSE는?
np.mean((df["y"] - df["y"].mean())**2)
29.81

# x=15 기준으로 나눴을때, 데이터포인트가 몇개 씩 나뉘나요?
# 57, 276
n1=df.query("x < 15").shape[0]  # 1번 그룹
n2=df.query("x >= 15").shape[0] # 2번 그룹

# 1번 그룹은 얼마로 예측하나요?
# 2번 그룹은 얼마로 예측하나요?
y_hat1=df.query("x < 15").mean()[0]
y_hat2=df.query("x >= 15").mean()[0]



# X1 기준으로 최적 기준값은 얼마인가?

# 기준값 x를 넣으면 entropy 값이 나오는 함수는?

from scipy.stats import entropy
def my_entropy(x):
    group1 = df.query(f"x1 < {x}")
    group2 = df.query(f"x1 >= {x}")
    n1 = group1.shape[0]  # 1번 그룹의 크기
    n2 = group2.shape[0]  # 2번 그룹의 크
    if n1 == 0 or n2 == 0:  # 그룹이 비어있으면 엔트로피 0 반환
        return 0
    # 각 그룹의 y값 확률 분포 계산
    p1 = group1['y'].value_counts(normalize=True)
    p2 = group2['y'].value_counts(normalize=True)
    # 엔트로피 계산 (scipy.stats.entropy 사용)
    entropy1 = entropy(p1, base=2)
    entropy2 = entropy(p2, base=2)
    # 가중평균 엔트로피 계산
    total_entropy = (n1 * entropy1 + n2 * entropy2) / (n1 + n2)
    return total_entropy
my_entropy(42.3)

from scipy.optimize import minimize_scalar

# 최소화할 함수
def entropy_to_minimize(x):
    return my_entropy(x)

# 범위 내에서 최적 분할점 찾기 (X1의 최소값과 최대값 범위에서)
x1_min = df['x1'].min()
x1_max = df['x1'].max()
result = minimize_scalar(entropy_to_minimize, bounds=(x1_min, x1_max), method='bounded')

# 최적의 기준값과 최소 엔트로피
best_split = result.x
min_entropy_value = result.fun

print(f"최적의 기준값: {best_split}, 최소 엔트로피: {min_entropy_value}")


my_entropy(42.3)

# 13~22 사이 값 중 0.01 간격으로 MSE 계산을 해서
# minimize 사용해서 가장 작은 MSE가 나오는 x 찾아보세요!
x_values=np.arange(13.2, 16.4, 0.01)
nk=x_values.shape[0]
result=np.repeat(0.0, nk)
for i in range(nk):
    result[i]=my_mse(x_values[i])

result
x_values[np.argmin(result)]
# 14.01, 16.42, 19.4

# x, y 산점도를 그리고, 빨간 평행선 4개 그려주세요!
import matplotlib.pyplot as plt

df.plot(kind="scatter", x="x", y="y")
thresholds=[14.01, 16.42, 19.4]
df["group"]=np.digitize(df["x"], thresholds)
y_mean=df.groupby("group").mean()["y"]
k1=np.linspace(13, 14.01, 100)
k2=np.linspace(14.01, 16.42, 100)
k3=np.linspace(16.42, 19.4, 100)
k4=np.linspace(19.4, 22, 100)
plt.plot(k1, np.repeat(y_mean[0],100), color="red")
plt.plot(k2, np.repeat(y_mean[1],100), color="red")
plt.plot(k3, np.repeat(y_mean[2],100), color="red")
plt.plot(k4, np.repeat(y_mean[3],100), color="red")