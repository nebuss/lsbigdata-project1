import numpy as np

#벡터 * 벡터(내적)
a= np.arange(1, 4)
b= np.array([3, 6, 9])


a.dot(b)


# 행렬 * 벡터 (곱셈)

a = np.array([1, 2, 3, 4]).reshape((2, 2), 
                                    order='F')

b = np.array([5, 6]).reshape(2,1)

a.dot(b)
# dot을 @ 연산으로 대체할 수 있음.
a@b

# 행렬 * 행렬
a = np.array([1, 2, 3, 4]).reshape((2, 2), order='F')
b = np.array([5, 6, 7, 8]).reshape((2, 2), order='F')

a@b

#Q1.
a = np.array([1, 2, 1, 0, 2, 3]).reshape(2, 3)
b= np.array([1, -1, 2, 0, 1, 3]).reshape((3, 2), order='F')
# = 또는 np.array([1, 0, -1, 1, 2, 3]).reshape(3,2) 하면 order 안해도 됨.
a@b


#Q2.
a=np.array([3, 5, 7, 2, 4, 9, 3, 1, 0]).reshape(3, 3)
b= np.eye(3)

a@b

# transpose
a.transpose()
b=a[:, 0:2]
b.transpose()

# model.predict 가 하는 역할을 벡터와 행렬곱으로 구현해보기/ 회귀분석 데이터 행렬
x = np.array([13, 15,
          12, 13,
          10, 11,
           5, 6]).reshape(4,2)
x
vec1= np.repeat(1, 4).reshape(4,1)
matX=np.hstack((vec1, x)) # hstack = > 옆에 열 붙임
matX

beta_vec=np.array([2, 0, 1]).reshape(3,1)
beta_vec

matX @ beta_vec

# b0, b1, b2를 한변으로 하는 정사각형의 넓이가  최소가 되게끔 하고 싶다.
# (예측값 - 실제값) ^ 2 로 성능 측정.!

y = np.array([20, 19, 20, 12]).reshape(4,1)

(y - matX @ beta_vec).transpose() @ (y - matX @ beta_vec)

# 3x3역행렬

a = np.array([-4, -6, 2, 5, -1, 3, -2, 4, 3]).reshape(3, 3)
a_inv=np.linalg.inv(a)

a@a_inv

# 역행렬이 존재하지 않는 경우 (선형 종속)
a = np.array([1, 2, 3, 2, 4, 5 ,3, 6, 7]).reshape(3, 3)
a_inv=np.linalg.inv(a)

# 벡터 형태로 베타 구하기
XtX_inv=np.linalg.inv((matX.transpose() @ matX))
xty = matX.transpose() @ y
beta_hat = XtX_inv @ xty
beta_hat

# model.fit으로 베타 구해보기 
from sklearn.linear_model import LinearRegression

model=LinearRegression()


model.fit(matX[:, 1:], y)

model.coef_
model.intercept_

# minimize로 베타 구하기

from scipy.optimize import minimize

# 최소값을 찾을 다변수 함수 정의
def line_perform(beta):
    beta = np.array(beta).reshape(3,1)
    a= (y - matX @ beta)
    return (a.transpose() @ a)

line_perform([6, 1, 3])
# 초기 추정값
initial_guess = [1, 0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

from sklearn.linear_model import LinearRegression

model=LinearRegression()


model.fit(matX[:, 1:], y)

model.coef_
model.intercept_

# minimize로 라쏘 베타 구하기

from sklearn.linear_model import Lasso

# 최소값을 찾을 다변수 함수 정의
def line_perform_lasso(beta):
    beta = np.array(beta).reshape(3,1)
    a= (y - matX @ beta)
    return (a.transpose() @ a) + 3*np.abs(beta).sum()

line_perform_lasso([8.55, 5.96, -4.38])
line_perform_lasso([3.76, 1.36, 0])
# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# minimize로 릿지 베타 구하기

from sklearn.linear_model import Lasso

# 최소값을 찾을 다변수 함수 정의
def line_perform_ridge(beta):
    beta = np.array(beta).reshape(3,1)
    a= (y - matX @ beta)
    return (a.transpose() @ a) + 3*np.abs(beta**2).sum()

line_perform_ridge([8.55, 5.96, -4.38])
line_perform_ridge([3.76, 1.36, 0])
# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_ridge, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
