## 회귀직선 최소값을 찾는 코드
def my_f(x):
  return x **2 + 3

my_f(3)

import numpy as np
from scipy.optimize import minimize
# minimize = 최소값을 찾을 다변수 함수 정의

#초기추정값(최적화 알고리즘에 제공할 시작점)
init_guess= [0]

result = minimize(my_f, init_guess)
result
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)



# x와  y가 있는 이차함수 

def my_f2(x):
  return x[0]**2 + x[1]**2 + 3  

my_f2([1, 3])


init_guess= [0, 0]

result = minimize(my_f2, init_guess)
result
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)



# f(x, y, z) = (x-1)**2 + (y-2)**2 + (z-4)**2 + 7
def my_f3(x):
  return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-4)**2 + 7  

my_f3([1, 3, 3])


init_guess= [0, 0, 0]

result = minimize(my_f3, init_guess)
result
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
