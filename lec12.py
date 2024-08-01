import pandas as pd
import numpy as np
import seaborn as sns
import pyreadstat

raw_welfare = pd.read_spss('data/Koweps_hpwc14_2019_beta2.sav')

welfare =raw_welfare.copy()

welfare
welfare.shape
welfare.info()
welfare.describe()

welfare = welfare.rename(
  columns = {'h14_g3' : 'sex',
             'h14_g4' : 'birth',
             'h14_g10' : 'marriage_type',
             'h14_g11' : 'religion',
             'p1402_8aq1' : 'income',
             'h14_eco9' : 'code_job',
             'h14_reg7' : 'code_region'}
)

welfare.columns

welfare['sex'].dtypes
welfare['sex'].value_counts()

# 이상치 결측처리 
welfare['sex'] = np.where(welfare['sex'] ==9, np.nan, welfare['sex'])
# 결측치 확인
welfare['sex'].isna().sum()

# 숫자1 남자. 숫자2 여자

welfare['sex'] = np.where(welfare['sex'] ==1, 'male', 'female')

# 빈도 구하기
welfare['sex'].value_counts()

# 빈도 막대그래프 그리기
import matplotlib.pyplot as plt
sns.countplot(data = welfare, x = 'sex')
plt.show()



welfare['income'].describe()
welfare['income'].isna().sum() 


welfare['income'] = np.where(welfare['income'] == 9999, np.nan, welfare['income'])
welfare['income'].isna().sum()


sex_income = welfare.dropna(subset = 'income') \
                    .groupby('sex', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))
sex_income  

#막대그래프 만들기
sns.barplot(data= sex_income, x = 'sex', y ='mean_income', hue='sex')
plt.show()
plt.clf()

sns.histplot(data = welfare, x = 'birth')
plt.show()
plt.clf()

# 무응답 확인하기
welfare['birth'].describe()
welfare['birth'].isna().sum()



welfare['birth'] = np.where(welfare['birth'] == 9999, np.nan, welfare['birth'])
welfare['birth'].isna().sum()


welfare = welfare.assign(age=2019 - welfare['birth'] + 1)
welfare['age'].describe()


sns.histplot(data= welfare, x = 'age')
plt.show() 
plt.clf()


age_income = welfare.dropna(subset='income') \
                    .groupby('age') \
                    
age_income.head()

sum(welfare['income']==0)
welfare['age'].isna().sum()

# 나이별 income  칼럼 na 개수 세기(무응답자 수 그래프)

welfare['income_na'] = welfare['income'].isna()

na = welfare.groupby('age', as_index = False) \
            .agg(n = ('income_na', 'sum'))

na

sns.lineplot(data=na, x='age', y='n')
plt.show()
plt.clf()

# 연령대에 따른 월급 차이는?
welfare = welfare.assign(ageg = np.where(welfare['age'] < 30, 'young', 
                                np.where(welfare['age'] <= 59, 'middle', 'old')))

welfare['ageg'].value_counts()

sns.countplot(data= welfare, x = 'ageg')
plt.show()
plt.clf()

# 연령대별 월급 평균표 만들기

ageg_income = welfare.dropna(subset= 'income') \
                     .groupby('ageg', as_index=False) \
                     .agg(mean_income = ('income', 'mean'))
                     
sns.barplot(data= ageg_income, x= 'ageg', y='mean_income')

# 막대 정렬
sns.barplot(data = ageg_income, x= 'ageg', y = 'mean_income', 
            order = ['young', 'middle', 'old'])

plt.show()
plt.clf()


#!! 응용해서 나이가 0에서 9 , 10-19, 20-29 / 나이변수의 컷을 0에서 9까지

vec_x = np.random.randint(0, 100, 50)
bin_cut = np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
pd.cut(vec_x, bins=bin_cut)

welfare = welfare.assign(age_group = pd.cut(welfare["age"], 
                        bins=bin_cut, 
                        labels = (np.arange(12)*10).astype(str)+"대"))
welfare["age_group"]


age_income = welfare.dropna(subset='income') \
                    .groupby('age_group', as_index=False) \
                    .agg(mean_income = ('income', 'mean'))

age_income
plt.rcParams.update({'font.family' : 'Malgun Gothic'})
sns.barplot(data=age_income, x='age_group' , y='mean_income')
plt.show ()
plt.clf()


#연령대 및 성별 월급 평균표 만들기
# 내 변수가 판다스 데이터프레임을 다룰 때, 변수의 타입이 
# 카테고리로 설정되어 있는 경우, groupby + agg 안먹힘. -> 오브젝트로 교환
# welfare['age_group']=welfare['age_group'].astype('object')
# sex_income = \
#     welfare.dropna(subset='income')\
#            .groupby(['ageg', 'sex'], as_index=False) \
#            .agg(mean_income = ('income', 'mean'))
# 
# sns.barplot(data= sex_income, x='ageg', y='mean_income', hue='sex')           
# plt.show()
# plt.clf()


# 연령대별, 성별 상위 4% 수입 찾기
#quantile q에 해당하는 값 뽑아줌(norm.ppf는 이미 지정된 분포의 이론적 x값, quantile은 내가 입력한
#데이터를 기준으로 상위 n% 뽑아줌. 방금 이부분 꼭 알아두기! 함수 옵션 여러개 있을때)
#lambda의 좋은점: 속성이 여러개인 값을 넣어줄 수 있음.

# sex_age_income = \
#     welfare.dropna(subset='income')\
#            .groupby(['ageg', 'sex'], as_index=False) \
#            .agg(top4per_income = ('income', lambda x: np.quantile(x, q=0.96)))
# welfare['income']                                            # lambda함수의 x 는 income열 
# sex_age_income
      



---
## 서연 코드
bin_cut = np.array([0, 9 , 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
welfare = welfare.assign(age_group = pd.cut(welfare["age"], 
                        bins=bin_cut, 
                        labels = (np.arange(12)*10).astype(str)+"대"))
welfare["age_group"]

age_income = welfare.dropna(subset = "income")\ 
                    .groupby("age_group", as_index = False)\
                    .agg(mean_income = ("income", "mean"))
sns.barplot(data = age_income, x="age_group", y="mean_income")
plt.show()
plt.clf()

welfare["age_group"] = welfare["age_group"].astype("object")
sex_age_income_top4er = welfare.dropna(subset = "income")\ 
                    .groupby(["age_group", "sex"], as_index = False)\
                    .agg(top4per_income = ("income", lambda x: np.quantile(x, q = 0.96)))
                    
sns.barplot(data=sex_age_income_top4er, x="age_group", y="top4per_income", hue="sex")
plt.show()
plt.clf()


## 나이 별 성별 월급 차이

sex_age = welfare.dropna(subset= 'income') \
                 .groupby(['age', 'sex'], as_index=False) \
                 .agg(mean_income= ('income', 'mean'))
sns.lineplot(data= sex_age, x='age', y = 'mean_income', hue='sex')                 

plt.show() 


# 9-6장 7/31

welfare['code_job'].dtypes       
welfare['code_job'].value_counts()    
list_job = pd.read_excel('data/Koweps_Codebook_2019.xlsx', sheet_name='직종코드')  


welfare = welfare.merge(list_job, how='left', on='code_job')

welfare.dropna(subset=['job', 'income'])[['income', 'job']].head()

job_income = welfare.dropna(subset = ['job', 'income']) \
                    .groupby('job', as_index=False) \
                    .agg(mean_income=('income', 'mean'))
                    # 밑에작업을 여기서하려면
                   #.sort_values('mean_income', ascending= False) \
                   #.head(10)
job_income
# 상위 10개 직업 추출
top10 = job_income.sort_values('mean_income', ascending=False).head(10)


import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Malgun Gothic'})

sns.barplot(data=top10, y='job', x='mean_income', hue='job')
plt.show()
plt.clf()

# 임금이 적은 직업 10개


bottaom10 = job_income.sort_values('mean_income').head(10)

sns.barplot(data=bottaom10, y='job', x='mean_income', hue='job') \
   .set(xlim = [0, 8100])
plt.show()


# 여자 돈 잘버는 직업
female_income = welfare.dropna(subset = ['job', 'income']) \
                      .query("sex == 'female'") \
                      .groupby('job', as_index=False) \
                      .agg(mean_income=('income', 'mean')) \
                      .sort_values('mean_income', ascending= False) \
                      .head(10)

sns.barplot(data=female_income, y='job', x='mean_income', hue='job')
plt.show()
plt.clf()


# 9-8

# 종교 유무에 따른 이혼율 분석하기


df_div = welfare.query("marriage_type != 5") \
                    .groupby('religion', as_index=False) \
                    ["marriage_type"] \
                    .value_counts(normalize = True) # 핵심
                    # count를 세주는 거에 normalize가 proportion을 세줌
                    
df_div.query("marriage_type == 1") \
      .assign(proportion = df_div['proportion']*100) \
      .round(1)  # 100곱해서 보기 쉽게 바꿈.


sns.barplot(data=df_div, x ='religion', y='proportion')
plt.show()                    
 plt.clf()                    
                    
