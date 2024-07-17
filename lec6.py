import pandas as pd
import numpy as np


exam = pd.read_csv("data/exam.csv")
exam.head()

exam.info()
exam.describe()

exam2 = exam.copy()
exam2 = exam2.rename(columns = {'nclass' : 'class'})

exam2['total'] = exam2['math'] + exam2['english'] + exam2['science']
exam2

exam2['test'] = np.where(exam2['total'] > 200, 'pass', 'fail')
exam2

import matplotlib.pyplot as plt
exam2['test'].value_counts().plot.bar(rot=0)
plt.show()


#200 이상:A
#100이상: B
#100 미만 : C

exam2['test2'] = np.where(exam2['total'] >= 230, "A", 
                 np.where(exam2['total'] >= 190, "B", "C"))

exam2['score'] = np.where(exam2['test2'].isin(['A' , 'B']), 'good', 'bad')
exam2

numbers = np.random.choice(10, 5, replace=False)
print(numbers)

exam = pd.read_csv('data/exam.csv')
exam
exam.query('nclass==1')
exam.query('nclass!=1')
exam.query('math > 50')
exam.query('math < 50')
exam.query('english >= 50')
exam.query('english <= 80')
exam.query('nclass == 1 & math >= 50')
exam.query('nclass == 2 & english >= 80')
exam.query('math >= 90 | english >= 90')
exam.query('english < 90 | science < 50')
exam.query('nclass ==1 | nclass ==3 | nclass == 5')
exam.query('nclass not in [1, 2]')
# = exam[~exam['nclass'].isin([1, 2])


exam.query("nclass == 1")[["math", "english"]]
exam.query("nclass == 1") \
    [["math", "english"]] \
    .head()

#정렬하기
exam.sort_values("math")
exam.sort_values("math", ascending = False)
exam.sort_values(["nclass", "english"], ascending = [True, False])

#변수 추가
exam = exam.assign(
  total = exam['math'] + exam['english'] + exam['science'],
  mean = (exam['math'] + exam['english'] + exam['science'])/3
  ) \
  .sort_values('total', ascending=False)
  .sort_values('mean',ascending=False)
exam  

# lambda 함수 사용하기
exam2 = pd.read_csv('data/exam.csv')

exam2 = exam2.assign(
  total = lambda x: x['math'] + x['english'] + x['science']
  mean = lambda x: x['total'] / 3
) \

.sort_values('total', ascending=False)
.sort_values('mean',ascending=False)
exam2.head()


#그룹을 나눠 요약을 하는 .groupby() + .agg() 콤보
exam2.agg(mean_math = ('math', 'mean'))
exam2.groupby('nclass') \
     .agg(mean_math = ('math', 'mean'))

exam2.groupby('nclass') \
     .agg(
       mean_math = ('math', 'mean'),
       mean_english = ('english', 'mean'),
       mean_science = ('science', 'mean'),
       )
       
import pydataset

df= pydataset.data("mpg")
df
df.columns

df.query('category == '"suv"')
  .assign(total = (mpg = (df['hwy'] + df['cty'] / 2 \
  .groupby('manufacturer')\
  .)))
