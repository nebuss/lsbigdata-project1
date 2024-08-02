import numpy as np
# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(2024)
a = np.random.randint(1, 21, 10)
print(a)
# 두 번째 값 추출
print(a[1])

a[2:5]
a[-2]

1에서 1000사이 3의 배수의 합

sum(np.arange(3, 1001, 3))

x = np.arange(1, 1001)
sum(x[2:1000:3]) #3번째 부터

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
a[(a > 2000) & ( a < 5000)]

(a > 2000) & ( a < 5000)


!pip install pydataset
import pydataset

df=pydataset.data('mtcars')
df
np_df=np.array(df['mpg'])


#연비가 15이상 25이하인 데이터 개수

(np_df >= 15) & (np_df <= 25)

#평균 mpg 보다 높은(이상) 자동차 대수는?

np_df >= np.mean(np_df)
sum(np_df >= np.mean(np_df))

#15보다 작거나 22이상인 데이터 개수는?
(np_df <15) | (np_df <=22)
sum((np_df <15) | (np_df <=22))


np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b = np.array(["A", "B", "C", "F", "W"])

a[(a > 2000) & (a < 5000)]
b[(a > 2000) & (a < 5000)]
# a의 원소를 가지고 필터링 하지만, 실제 뽑아오는 것은 b에서 한다.
a[a> 3000] = 3000

model_names= np.array(df.index)
model_names

#15보다 작거나 20이상인 자동차 모델은?
model_names[(np_df <15) | (np_df <=20)]

#평균 mpg 보다 높은(이상) 자동차 모델은?
model_names[(np_df >= np.mean(np_df))]

#평균 mpg 보다 낮은(이하) 자동차 모델은? / 연비안좋음
model_names[(np_df <= np.mean(np_df))]


np.random.seed(2024)
a = np.random.randint(1, 100, 10)
a<50
np.where(a < 50) # a가 true인 원소의 인덱스 출력 됨.

np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
a

#처음으로 22000보다 큰 숫자가 나오는 위치는?

x=np.where(a > 22000)
type(x)
my_index = x[0][0]
a[my_index]

#처음으로 10000보다 큰 숫자가 나오는 위치는?
#50번째로 나온 숫자 위치와 그 숫자는 무엇인가요?
np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)

x=np.where(a>10000)[0]
my_index=x[49]
a[my_index]


#500보다 작은 숫자들 중 가장 마지막으로 나온 나오는 숫자 위치와 그 숫자는 무엇인가요?

y=np.where(a<500)[0]
my_index2 =y[-1]
my_index2
a[my_index2]

a= None
b = np.nan
b
a

str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]

mix_vec = np.array(["사과", 12, "배", "수박", "참외"], dtype=str)
mix_vec

combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec

#np.column_stack()과 np.row_stack()
col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.row_stack((np.arange(1, 5), np.arange(12, 16)))
row_stacked

vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
uneven_stacked = np.resize(vec1, len(vec2))


# 주어진 벡터의 홀수번째 요소만 추출
a = np.array([12, 21, 35, 48, 5])
a[1::2]

a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)

a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
x = np.empty(6)
x

#홀수
# x[[0, 2, 4]] = 0
x[0::2]=a
x

#짝수
# x[[1, 3, 5]] = b
x[1::2] = b
x


import numpy as np
import pandas as pd

df = pd.DataFrame({'name' : ['김지훈', '이유진', '박동현', '김민지'],
                    'english' : [90, 80, 60, 70],
                    'math': [50, 60 , 70, 80]})
df
df['name'] #열을 선택하는 방법
type(df['name']) #그 결과의 데이터 타입을 확인하는 방법.

sum(df["english"])


df = pd.DataFrame({'제품' : ['사과', '딸기', '수박'],
                    '가격' : [1800, 1500, 3000],
                    '판매량': [24, 38, 13]})
df

sum(df['가격'])/ 3
sum(df['판매량']/3)


!pip install openpyxl

import pandas as pd
import numpy as np
#엑셀파일 불러오기
df_exam = pd.read_excel('data/excel_exam.xlsx')
df_exam

sum(df_exam['english']) / 20
sum(df_exam['science']) / 20

df_exam.shape
len(df_exam) #행 개수 반환
df_exam.size

df_exam["total"] = df_exam["math"] + df_exam["science"] + df_exam["english"]
df_exam

df_exam["mean"] = (df_exam["math"] + df_exam["science"] + df_exam["english"])/3
df_exam

df_exam[(df_exam["math"] > 50 ) & (df_exam["english"]>50)]

df_exam[(df_exam["math"] > df_exam["math"].mean() ) & (df_exam["english"] < df_exam["english"].mean())]

class_3= df_exam[df_exam["nclass"]==3]
class_3[["math", "english", "science"]]


class_3[1:]

df_exam[0:10:2]

df_exam.sort_values('math')
df_exam.sort_values(['nclass','math'], ascending=[True, False])


np.where(a>3, "Up", "Down")
df_exam["updown"] = np.where(df_exam["math"] > 50, "Up", "Down")
df_exam
