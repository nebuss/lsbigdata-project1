#파이썬의 math 함수
import math

sqrt_val = math.sqrt(16)
print("16의 제곱근은", sqrt_val)


exp_val = math.exp(5)
print("e^5의 값은", exp_val)

log_val = math.log(10, 10)
print("10의 밑 10 로그 값은:", log_val)

fact_val = math.factorial(5)
print("5의 팩토리얼은:", fact_val)

sin_val = math.sin(math.radians(90))
print("90도의 사인 함수 값은:", sin_val)

def normal_pdf(x, mu, sigma): 
  sqrt_two_pi = math.sqrt(2 * math.pi)
  factor = 1 / (sigma * sqrt_two_pi)
  return factor * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

mu = 0
sigma = 1
x = 1

pdf_value = normal_pdf(x, mu, sigma)
print("정규분포 확률밀도 함수 값은:", pdf_value)


def my_func(x, y, z):
  return (x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)

x=2
y=9
z = math.pi /2
my_func(x, y, z)


def fname(`indent('.') ? 'self' : ''`):

# numpy

import numpy as np
# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)

type(a)
a[4]
a[2:]
a[1:4]


x = np.empty(3)
print("빈 벡터 생성하기:", x)
x[0]=1
x[1]=2
x[2]=3
x

vec1=np.arange(1, 101, 4)
vec1

linear_space2 = np.linspace(0, 1, 5, endpoint=False)
print("0부터 1까지 5개 원소, endpoint 제외:", linear_space2)

#배열반복
repeated_array = np.repeat([1, 2, 4], 2)
print("Repeated array [1, 2, 4] tow times", repeated_array)

#35672 이하 홀수들의 합
sum(np.arange(1, 35673, 2))

a=np.array([1, 2])
b=np.array([1,2,3,4])
a+b

#35672 보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 개수

vec = np.arange(1, 35673)
num = vec%7==3
count = np.sum(num)

print(count)
np.sum(np.arange(1, 10)% 7 == 3)

import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = 2.0
a*b

a.shape
b.shape

matrix = np.array([[ 0.0, 0.0, 0.0],
 [10.0, 10.0, 10.0],
 [20.0, 20.0, 20.0],
 [30.0, 30.0, 30.0]])
matrix.shape
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])
vector.shape
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)

matrix = np.array([[ 0.0, 0.0, 0.0],
 [10.0, 10.0, 10.0],
 [20.0, 20.0, 20.0],
 [30.0, 30.0, 30.0]])
# 벡터 생성
vector = np.array([1.0, 2.0, 3.0, 4.0])
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)
