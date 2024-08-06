#0806 t 검정

import pandas as pd
import numpy as np


tab3= pd.read_csv('stat/tab3.csv')
tab3

tab1 = pd.DataFrame({"id": np.arange(1, 13),
                     "score" : tab3["score"]})

tab2= tab1.assign(gender=['female'] *7 + ['male'] * 5)
tab2

# 1표본 t 검정 (그룹 1개)
# 귀무가설 vs. 대립가설
# H0: mu = 10 vs. Ha: mu != 10
# 유의수준은 5%로 설정.

from scipy.stats import ttest_1samp

result = ttest_1samp(tab1["score"], popmean=10, alternative='two-sided')
print("t statistic:", t_statistic)
t_value=result[0] # t 검정통계량
p_value=result[1] # 유의확률 (p-value)
result.pvalue
result.statistic
# 귀무가설이 참일 때, 11.53이 관찰될 확률이 6.48% 이므로 이것은 우리가 생각하는 
# 보기힘들다고 판단하는 기분인 0.05보다 (유의수준) 크므로 거짓이라고 보기 힘들다. 
# 유의확률 0.0648이 유의수준 0.05보다 크므로 귀무가설을 기각하지 못한다.
#유의수준-> 기각역이 될 확률


#95% 신뢰구간 구하기
ci =result.confidence_interval(confidence_level=0.95)


# 2표본 t 검정(그룹 2) - 분산 같고, 다를 때
분산 같은경우: 독립2표본 t검정
분산 다를경우: 웰치스 t 검정
# 귀무가설 vs. 대립가설
# H0: mu_m = mu_f vs. Ha: mu_m > mu_f
# 유의수준은 1%로 설정.


from scipy.stats import ttest_ind

m_tab2 = tab2[tab2['gender'] == 'male']
f_tab2 = tab2[tab2['gender'] == 'female']

# alternative="less"의 의미는 대립가설이 첫번째 입력그룹의 평균이 두번째 입력그룹 평균보다 작다.
result=ttest_ind(f_tab2['score'], m_tab2['score'], alternative="less", equal_var = True)

result.statistic
result.pvalue
#단측검정이어서 한쪽이 뚫려서 나옴 --> 무슨 말인가?
ci=result.confidence_interval(0.95)
ci[0]
ci[1]


# 3표본 대응표본 (짝지을 수 있는 표본)
# 귀무가설 vs. 대립가설
# H0: mu_before = mu_after vs. Ha: mu_after > mu_before
# H0: mu_d = 0 vs. Ha: mu_d > 0 -> 모평균이 0
## mu_d = mu_after - mu_before 
# 유의수준은 1%로 설정.


# mu_d에 대응하는 표본으로 변환
tab3_data = tab3.pivot_table(index='id',
                             columns = 'group',
                             values='score').reset_index()

tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test_data = tab3_data[['score_diff']]
test_data

from scipy.stats import ttest_1samp

result = ttest_1samp(test_data["score_diff"], popmean=0, alternative='greater')
t_value=result[0] # t 검정통계량
p_value=result[1] # 유의확률 (p-value)
t_value; p_value


# 연습1

df = pd.DataFrame({"id" : [1, 2, 3],
                  "A" : [10, 20, 30],
                  "B" : [40, 50, 60]})
                  
df

df.melt(id_vars="id", value_vars= ["A", "B"],
                      var_name="group",
                      value_name = "score")
                      
                      
df_long= df.melt(id_vars='id',
                 value_vars=["A", "B"],
                 var_name = "group",
                 value_name="score")
                 
df_long.pivot_table(
  columns="group",
  values="score"
)                      

df_long.pivot_table(
  index="id",
  columns="group",
  values="score"
)                       
                              
                      
# 연습2
import seaborn as sns
tips = sns.load_dataset('tips')
tips

tips = sns.load_dataset('tips')

# 'day' 열의 데이터만 추출하여 피벗 테이블 생성
tips.reset_index(drop=False) \
    .pivot_table(
       index="index",
       columns = "day",
       values ="tip").reset_index()

tips.pivot_table(columns="day",
           values ="tip")

# 요일별로 펼치고 싶은 경우
tips.columns.delete(4)
tips.reset_index(drop=False)\
    .pivot_table(
        index=index_list,
        columns= "day",
        values = "tip").reset_index()



