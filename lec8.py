import numpy as np

matrix = np.v_stack(
      (nparange(1, 5),)
)

# Q1. 0에서 99까지 수 중 랜덤하게 50개 숫자 뽑아서 5 by 10 행렬 만드세요
numbers = np.random.randint(0, 100, 50)
numbers.reshape((5, -1))

mat_a = np.arange(1, 21).reshape((4, 5), order="F")

#인덱싱
mat_a[1:3, 1:4]

#해당 행자리, 열자리 비어있는경우 전체 행, 또는 열 선택
mat_a[3, : ]
mat_a[: , 3]

#짝수행만 선택하려면?
mat_b = np.arange(1, 101).reshape((20, -1))
mat_b[1 ::2, :]

mat_b[[1, 4, 6, 14], ]

x=np.arange(1, 11).reshape((5, 2)) * 2
x[[True, True, False, False, True], 0]


mat_b[:, 1] #벡터
mat_b[:, (1,)] #행렬
mat_b[:, [1]]
mat_b[:, 1:2]

mat_b[:, 1].reshape(-1,1) #이렇게 reshape 를 하면 밑에 행렬과같이 형태가 바뀜.

#7의 배수가 있는 행  추출
mat_b[mat_b[:,1] % 7 == 0, :] # -> mat_b 행렬의 두번째 열의 원소가 7의 배수

mat_b[mat_b[:, 1] > 50,:] #mat_b행렬에서 두번째 열(인덱스 1)이 50보다 큰 행을 추출..?

import matplotlib.pyplot as plt
# 난수 생성하여 3x3 크기의 행렬 생성/ 사진은 행렬이다
np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)
plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "img/jelly.png")

import imageio
import numpy as np
# 이미지 읽기
jelly = imageio.imread("jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])

jelly[:, :, 0].shape()
jelly[:, :, 0].transpose().shape


plt.imshow(jelly)
plt.imshow(jelly[:, :, 0].transpose())
plt.imshow(jelly[:, :, 1])
plt.imshow(jelly[:, :, 2])
plt.imshow(jelly[:, :, 2]) # 투명도
plt.show()

#3차원 배열
#두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

# 3차원 배열로 합치기
my_array = np.array([mat1, mat2])
my_array.shape
print("3차원 배열 my_array:\n", my_array)

# 첫 번째 2차원 배열 선택
first_slice = my_array[0, :, :]
print("첫 번째 2차원 배열:\n", first_slice)

# 두 번째 차원의 세 번째 요소를 제외한 배열 선택
filtered_array = my_array[:, :, :-1]
my_array[:, 0, :]
my_array[0, 1, 1:3] # = my_array[0, 1, [1, 2]]


mat_x = np.arange(1, 101).reshape((5, 5, 4))
mat_x = np.arange(1, 101).reshape((10, 5, 2))

print("세 번째 요소를 제외한 배열:\n", filtered_array)

my_array2 = np.array([my_array, my_array])
my_array2.shape

a =np.array([[1, 2, 3], [4, 5, 6]])

a.mean(axis=0)
a.mean(axis=1)

mat_b =np.random.randint(0, 100, 50).reshape((5, -1))
#가장 큰 수는?
mat_b.max()

#행별로 가장 큰 수는?
mat_b.max(axis=1)
#열별로 가장 큰 수는?
mat_b.max(axis=0)

mat_b.cumsum(axis=1)
mat_b.sum()

a=np.array([1, 3, 2, 5])
a.cumsum(axis=0)
a.cumprod(axis=0)

mat_b.reshape((2, 5, 5))
mat_b.flatten()

d = np.array([1, 2, 3, 4, 5])
d.clip(2,4)


#균일확률변수

np.random.rand(1)


def X():
  return np.random.rand(3)

X()  


def X(i):
  return np.random.rand(i)

X(3)  

#베르누이 확률변수 모수:p 만들기
num=3
p=0.5
def Y(num, p):
  x=np.random.rand(num)
  return np.where(x < p, 1, 0)

sum(Y(num=5, p=0.5)) / 100
Y(num=10000, p=0.5).mean()




#새로운 확률변수
#가질 수 있는 값: 0, 1 , 2
# 20%, 50%, 30%

def Z():
  x=np.random.rand(1)
  result = np.where(x <0.2, 0, np.where(x <0.7, 1, 2))
  return result
Z()




#확률변수가 3개
def Z(p):
  x=np.random.rand(1)
  p_cumsum = p.cumsum()
  result = np.where(x < p_cumsum[0], 0, np.where(x < p_cumsum[1], 1, 2))
  return result

Z(p)
p  = np.array([0.2, 0.5, 0.3])

