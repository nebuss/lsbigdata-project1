import numpy as np

#Y 확률변수
np.arange(-16,16).sum()/33



sum(np.unique((np.arange(33) - 16)**2 )*(2/33))

x=np.arange(33)
sum(x)/33
sum((x - 16) * 1/33)
(x - 16) **2

np.unique((x-16)**2) * (2/33)
sum(np.unique((x-16)**2) * (2/33))

# E[X^2]
sum(x**2 * (1/33))

#Var(x) = E[X^2] - (E[X]^2)
sum(x**2 *(1/33)) - 16**2

#Example
x=np.arange(4)
x
pro_x = np.array([1/6, 2/6, 2/6, 1/6])
pro_x

# 기대값
Ex= sum(x * pro_x)
# x^2 의 기대값
Exx = sum(x **2 * pro_x)

#분산
var = Exx - Ex**2


# 0~100Rkwl 정수

#Var(x) = E[X^2] - (E[X]^2)
x = np.arange(99)

# 벡터를 1-50까지, 50-1까지 만들기
x_1_50_1 = np.concatenate((np.arange(1, 51),np.arange(49, 0, -1)))
pro_x = x_1_50_1 / 2500


ex = sum(x * pro_x)
exx = sum(x**2 * por_x)

# 분산
exx - ex*2
sum((x - ex)**2 * pro_x)


# Y확률변수의 분포 구하기 Y:0, 2, 4, 6
y = np.arange(4) *2
pro_y = np.array([1/6, 2/6, 2/6, 1/6])

ex = sum(y * pro_y)
exx = sum(y**2 * pro_y)

# 분산

exx - ex*2

