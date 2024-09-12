# GBRT

import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X = np.random.rand(100) - 0.5
y = 3 * X**2 + 0.05*np.random.randn(100)
X=X.reshape(100,1)


import matplotlib.pyplot as plt
plt.scatter(x=X, y=y)

tree_model1=DecisionTreeRegressor(max_depth=2) # 깊이가 2인 나무를 만든다
tree_model1.fit(X, y)
y_tree1=tree_model1.predict(X) # X를 넣었을 때 예측값

# 1차 트리 예측값 시각화
import matplotlib.pyplot as plt
plt.scatter(x=X, y=y_tree1)

y2=y-tree_model1.predict(X) # tree1에서 예측값 - 새로운 y 를 새로운 y로.
tree_model2=DecisionTreeRegressor(max_depth=2)
tree_model2.fit(X, y2)
y_tree2=tree_model2.predict(X)
# 두번째 y데이터 (즉, y-1차 트리 예측값) 잔차는 어떻게 생겼을까?
plt.scatter(x=X, y=y)
plt.scatter(x=X, y=y_tree1+ y_tree2) 

y3=y2-tree_model2.predict(X)
tree_model3=DecisionTreeRegressor(max_depth=2)
tree_model3.fit(X, y3)
y_tree3=tree_model2.predict(X)

# 1차 + 2차+ 3차 트리
plt.scatter(x=X, y=y)
plt.scatter(x=X, y=y_tree1+ y_tree2+y_tree3) 

y4=y3-tree_model2.predict(X)
tree_model4=DecisionTreeRegressor(max_depth=2)
tree_model4.fit(X, y3)
y_tree4=tree_model2.predict(X)

# 1차 + 2차+ 3차 + 4차 트리
plt.scatter(x=X, y=y)
plt.scatter(x=X, y=y_tree1+ y_tree2+y_tree3+y_tree4) 


# 새로운 데이터
X_new=np.array([[0.5], [-0.7], [0.2]])


tree_model1.predict(([[0.5]])) + tree_model2.predict(([[0.5]])) + tree_model3.predict(([[0.5]]))  


# 위 내용을 scikit-learn 사용해서 구현
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, 
                                 n_estimators=3,
                                 learning_rate=1.0,
                                 random_state=42)
gbrt.fit(X, y)
gbrt.predict(X)