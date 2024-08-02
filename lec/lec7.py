import pandas as pd


test1 = pd.DataFrame({'id' : [1,2,3,4,5],
                      'midterm' : [60, 80, 70, 90, 85]})

test2 = pd.DataFrame({'id' : [1,2,3,4,5],
                      'final' : [70, 83, 65, 95, 80]})
test1
test2

#Left Join
total = pd.merge(test1, test2, how="left", on="id")
total

name = pd.DataFrame({'nclass' : [1, 2, 3, 4, 5],
                     'teacher' : ['kim', 'lee', 'park', 'choi', 'jung']})
name

exam = pd.read_csv('data/exam.csv')
exam

total2 = pd.merge(exam, name, how="left", on="nclass")
total2

#데이터 세로로 합치기

group_a = pd.DataFrame({'id' : [1, 2, 3, 4, 5],
                      'test' : [60, 80, 70, 90, 85]})
                      
group_b = pd.DataFrame({'id' : [6, 7, 8, 9, 10],
                      'test' : [60, 80, 70, 90, 85]})
                      
group_all=pd.concat([group_a, group_b])                      
group_all
