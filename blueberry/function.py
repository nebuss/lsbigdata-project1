import inspect
def g(x=3):
    result = x + 1
    return result

print(g)
print(inspect.getsource(g))

# 리스트 컴프리헨션
x = [1, -2, 3, -4, 5]
result = ["양수" if value > 0  else "음수" value < 0]

import numpy as np
x = np.array([1, -2, 3, -4, 0])

conditions = [
    x>0,
    x==0,
    x<0
]


choice = [
    "양수",
    "0",
    "음수"
]

result = np.select(conditions, choice, x)
print(result)

for i in range(1, 4):
    print(f"Here is {i}")

name = '남규'
age='31 (진)'
f"Hello, my name is {name}, my age is {age}"

i = 0
while i <= 10:
 i += 3
 print(i)


import pandas as pd
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
print(df)
df.apply(max, axis=0)
df.apply(max, axis=1)

def my_func(x, const=3):
 return max(x)**2 + const

df.apply(my_func, axis=1)


df.apply(my_func, axis=0)

df.apply(my_func, axis=1, const=5)


array_2d = np.arange(1, 13).reshape((3, 4), order='F')
print(array_2d)

np.apply_along_axis(max, axis=0, arr=array_2d)

y = 2
def my_func(x):
 global y
 y = 1 + y
 result = x + y
 return result
print(y)
my_func(3)
print(y)


def add_many(*args):
   result=0
   for i in args:
      result= result + i
   return result 
 
add_many(1, 2, 3)

def add_mul(choice, *args): 
     if choice == "add":   # 매개변수 choice에 "add"를 입력받았을 때
         result = 0 
         for i in args: 
             result = result + i 
     elif choice == "mul":   # 매개변수 choice에 "mul"을 입력받았을 때
         result = 1 
         for i in args: 
             result = result * i 
     return result 

add_mul("mul", 5, 4, 3, 1)
add_mul("add", 5, 4, 3, 1)



#  ** 두개는 입력값을 딕셔너리로 만들어 준다.
def print_kwargs(**kwargs):
    if choice == "first":
        return print(kwargs["age"])
    elif choice == "second":
        return print(kwargs["name"])
    else:
        return print(kwargs) 

print_kwargs("second", name='foo', age=3)
