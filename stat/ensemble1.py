from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging_model = BaggingClassifier(DecisionTreeClassifier(), 
                                  n_estimators = 100,
                                  max_samples=100, 
                                  n_jobs=-1,
                                  random_state=42)
# n_estimators: bagging에 사용될 만들 모델 개수
# max_samples : 데이터셋 만들때 뽑을 표본 크기
# bagging_model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=50,
                                  max_leaf_nodes=16,
                                  n_jobs=-1,
                                  random_state= 42)
rf_model.fit()