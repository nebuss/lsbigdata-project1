import numpy as np

# 회귀분석 데이터행렬
x=np.array([13, 15,
           12, 14,
           10, 11,
           5, 6]).reshape(4, 2)
x
vec1=np.repeat(1, 4).reshape(4, 1)
matX=np.hstack((vec1, x))
y=np.array([20, 19, 20, 12]).reshape(4, 1)
matX

# minimize로 라쏘 베타 구하기
from scipy.optimize import minimize

def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 30*np.abs(beta[1:]).sum() #x앞에 붙여져있는 베타들만 곱해보자


line_perform_lasso([8.55,  5.96, -4.38])
line_perform_lasso([3.76,  1.36, 0]) # 손실이 더 적으므로 이 값이 성능이 더 좋음

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# 예측식: y_hat = 8.14 + 0.96 * X1 + 0 * X2
# 라쏘모델은 회귀 직선 모델보다 변수를 덜 쓰네?

#  [8.55,  5.96, -4.38] # 패널티 없었을 때. 람다가 0일 때(모든 변수 사용)
# [8.14, 0.96, 0] # 람다가 3
# [17.74, 0, 0] 람다 500

