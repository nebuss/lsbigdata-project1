fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]
print("과일 리스트:", fruits)
print("숫자 리스트:", numbers)
print("혼합 리스트:", mixed)


numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))

range_list[3] = 'LS'

range_list[1] = ['안', '녕']

range_list[1][0]

#리스트 comprehension
#1.  넣고싶은 수식표현을 x(x가 아니어도 됨)를 사용해서 표현
#2. for .. in ..을 사용해서 원소정보 제공

list(range(10))
squares = [x**2 for x in range(10)]

list = [3, 5, 2, 15]
squares = [x**3 for x in list]


#np.array도 가능한지
squares = [x**3 for x in np.array([3, 5, 2, 15])]

#pandas series도 가능할까?
import pandas as pd

exam = pd.read_csv('data/exam.csv')
exam ['math']

squares = [x**3 for x in exam ['math']]


numbers = [5, 2, 3]
repeated_list = [x for x in numbers for _ in range(3)] 
#x를 쓸건데, range(3) 즉 0,1,2번째 만큼 세번을 쓸것이다.
repeated_list


# for i in 범위:
# 작동방식

for x in [4, 1, 2, 3]:
  print(x)
  
  
for i in range(5):
  print(i**2)

#리스트 하나 만들어서  for  루프 사용해서 2~20 수 채워넣기

[i for i in range(2, 21, 2)]

mylist = []
for i in range(1, 11):
  mylist.append(i * 2)
print(mylist)  


mylist = []
for i in [1, 2, 3]:
  mylist.append(i * 2)
print(mylist)  

mylist = [0] * 10
for i in range(10):
  mylist[i] = (i+1) *2
mylist

# 인덱스 공유해서 카피하기
mylist_b = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
mylist = [0] * 10

for i in range(10):
  mylist[i] = mylist_b[i]
mylist

#퀴즈: mylist_b의 홀수번째 위치의 숫자만 mylist에 넣기
mylist_b = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

mylist = [0] * 5
for i in range(5):
  mylist[i]=mylist_b[i*2]
mylist


# 리스트 컴프리헨션으로 바꾸는 방법
# 리스트 반환위해 바깥은 무조건 대괄호로 묶어 줌 / for 의 : 생략
# 실행부분을 먼저 써준다.

mylist = []
for i in range(1, 11):
  mylist.append(i*2)
mylist

mylist = []
mylist = [i*2 for i in range(1, 11)]


for i in [0, 1, 2]:
  for j in [0, 1]:
    print(i, j)


numbers = [5, 2, 3]
for i in numbers:
  for j in range(4):
    print(i)
    
#를 컴프리헨션으로 바꾸면
[i for i in numbers for j in range(4)]

#원소 체크
fruits = ['apple' , 'banana', 'cherry']
fruits

'banana' in fruits

#[x == 'banana' for x in fruits]
mylist = []
for x in fruits:
  mylist.append(x == 'banana')
  
# 바나나의 위치를 뱉어내게 하려면 ?
fruits = ['apple' , 'banana', 'cherry']

if 'banana' in fruits : 
  banana_index = fruits.index('banana')
  print(banana_index)
else:
  print('banana is not in list')
  
import numpy as np
  
  
fruits = np.array(['apple', 'apple' , 'banana', 'cherry'])
np.where(fruits=='banana')  

fruits = ['apple' , 'banana', 'cherry']
fruits.reverse()

#원소 맨 긑에 붙여주기 
fruits.append('melon')
fruits.reverse()
fruits.append('peach')
fruits.reverse()

#특정 위치에 원소 삽입. 내가 삽입하고 싶은 위치에 넣어주고, 다른 원소들은 뒤로 밀림.
fruits.insert(1, 'pineapple')
fruits.pop(5)
