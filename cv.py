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

for i in range(2, 21):
   df[f"x{i}"] = df["x"] ** i

df  
#cv 3

myindex=np.random.choice(30, 30, replace=False)
myindex[0:10]
myindex[10:20]
myindex[20:30]


train_set1= np.array([myindex[0:10], myindex[20:30]])
valid_set1 = myindex[20:30]

df.loc[myindex[0:10]] #
df.drop(myindex[0:10]) #나머지 애들 

fold_num=0
fold_num=1

def make_tr_val(fold_num, df, cv_num=3):
   np.random.seed(2024)
   myindex=np.random.choice(30, 30, replace=False)

   # valid index
   val_index = myindex[10 * (fold_num):(10 * fold_num+10)]

   # valid set, train set
   valid_set = df.loc[val_index]
   train_set = df.drop(val_index)
                    
   return (train_set, valid_set) 
  
train_set0, valid_set0 = make_tr_val(0, df=df) # 튜플로 train_set이 나온다
train_set0
valid_set0

## 또 다른 방법.
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

for i in range(2, 21):
   df[f"x{i}"] = df["x"] ** i

df  
#cv 3

myindex=np.random.choice(30, 30, replace=False)
myindex[0:10]
myindex[10:20]
myindex[20:30]


train_set1= np.array([myindex[0:10], myindex[20:30]])
valid_set1 = myindex[20:30]

df.loc[myindex[0:10]] #
df.drop(myindex[0:10]) #나머지 애들 

fold_num=0
fold_num=1

def make_tr_val(fold_num, df, cv_num=3):
   np.random.seed(2024)
   myindex=np.random.choice(30, 30, replace=False)

   # valid index
   val_index = myindex[10 * (fold_num):(10 * fold_num+10)]

   # valid set, train set
   valid_set = df.loc[val_index]
   train_set = df.drop(val_index)

   train_X=train_set.iloc[:, 1:]
   train_Y=train_set.iloc[:, 0]

   valid_X=valid_set.iloc[:, 1:]
   valid_Y=valid_set.iloc[:, 0]
                    
   return (train_X, train_Y, valid_X, valid_Y) 
  
train_X, train_Y, valid_X, valid_Y = make_tr_val(fold_num=0, df=df)
train_X
train_Y
valid_X
valid_Y

# 각각 람다에 
from sklearn.linear_model import Lasso
val_result_total=np.repeat(0.0, 300).reshape(3, -1)
tr_result_total=np.repeat(0.0, 300).reshape(3, -1)

for j in np.arange(0, 3):
    train_X, train_Y, valid_X, valid_Y = make_tr_val(fold_num=j, df=df)


# 결과 받기 위한 벡터 만들기. 0 백번 써놓음.
val_result=np.repeat(0.0,100)  
tr_result=np.repeat(0.0,100)


for i in np.arange(0, 100):
        model= Lasso(alpha=i*(0.1)) #lambda가 alpha로 표현됨. I가 0이면 일반 회귀모델(모든 데이터 사용)
        model.fit(train_X, train_Y)

        y_hat_train = model.predict(train_X)
        y_hat_val = model.predict(valid_X)

        pref_train=sum((train_Y - y_hat_train)**2) # 잔차제곱합을 통해서 성능 측정
        pref_val=sum((valid_Y - y_hat_val)**2)
        val_result[i]=pref_val
        tr_result[i]=pref_train


tr_result_total[j, :]= tr_result
val_result_total[j, :]=val_result
