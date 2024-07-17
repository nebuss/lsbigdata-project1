import pandas as pd
import numpy as np
import math

df = pd.DataFrame({'sex' : ['M', 'F', np.nan, 'M', 'F'],
                   'score' : [5, 4, 3, 4, np.nan]})
df

pd.isna(df)

df.dropna()
df.dropna(subset = 'score')
df.dropna(subset = ['score' , 'sex'])


exam = pd.read_csv('data/exam.csv')


#데이터 프레임 location을 사용한 인덱싱
#exam.loc[행 인덱스, 열 인덱스]

exam.loc[[2, 7,  14], ['math']] = np.nan
exam.iloc[1:2,7]


df.loc[df['score']==3, ['score']] = 4
df
df['score']
df.iloc[df['score']==3, ['score']] = 4



#수학 점수 50점 이하인 학생들 점수 50점으로 상향조정

exam.loc[exam['math']<=50, ['math']] = 50
exam


#영어 점수 90점 이상 90으로 하향 조정
#iloc은 숫자 벡터만 들어감
exam.iloc[exam['english'] >= 90, 3)] # 실행안됨

exam.iloc[exam['english'] >= 90, 3] = 90
exam.iloc[exam[exam['english'] >= 90].index, 3] = 90
exam.iloc[np.where(exam['english'] >= 90)[0], 3] #np.where도 튜플이라 [0] 사용해서 꺼내면 됨
exam.iloc[np.array(exam['english'] >= 90, 3)]
exam




----------------------------------------------

#math 50점 이하  - 로 변경
exam.loc[exam['math']<=50, ['math']] = '-'
exam.iloc[exam['english'] <=50, 2] = '-'

# - 결측치를 수학함수 평균으로 바꾸기
# 1번 방법
math_mean = exam.loc[(exam['math'] != '-'), math].mean()
exam.loc[exam['math'] == '-', 'math'] = math_mean
exam
# 2번 방법
exam.loc[exam['math'] == '-', 'math'] = exam.query('math not in ["-"]')['math'].mean()
exam
# 3번 방법
math_mean2 = exam[exam['math']!= '-']['math'].mean()
exam.loc[exam['math'] == '-', 'math'] = math_mean2
exam

#4번 방법
exam.loc[exam['math'] == '-', ['math']] = np.nan
math_mean = exam['math'].mean()
exam.loc[pd.isna(exam['math']), ['math']] = math_mean2
exam

#5버 
math_mean = exam['math'].mean()
exam['math'] = np.where(exam['math'] == '-', math_mean, exam['math'])
exam

#6번 - 에 np.nan를 넣어서 만드는 방법
vector = np.array([np.nan if x=='-' else float(x) for x in exam['math']])
np.nanmean(vector) # nan값을 제외한 값들의 평균을 구하는 함수.
vector2 = np.array([float(x) if x !='-' else np.nan for x in exam['math']]) # -가 아닌경우 float(x) 로 받아온다
vector2

#7번 mean값을 구하고 replace 함수로 -를 math_mena으로 바꿔라!
math_mean = exam.loc[(exam['math'] != '-'), math].mean()
exam['math'] = exam['math'].replace('-', math_mean)
exam
