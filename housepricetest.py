# 전반적 품질과 리모델링 연도로 예측해보기 

import pandas as pd
import numpy as np


house_df = pd.read_csv("./data/train.csv")
house_df

#전체적인 품질별 가격 평균 묶기
my_df=house_df.groupby(['OverallQual', 'YearRemodAdd'], as_index=False)\ 
         .agg(saleprice=('SalePrice', 'mean'))
         
my_df
#테스트 데이터 셋 가져오기         
test_df = pd.read_csv("./data/test.csv")         

# 테스트 데이터에 평균 집 값을 매핑


test_df=test_df[['Id', 'OverallQual', 'YearRemodAdd']]
test_df = pd.merge(test_df, my_df, on=['OverallQual', 'YearRemodAdd'], how='left')

# 결측치 있나 확인
test_df['saleprice'].isna().sum()


# 결측치 평균으로 처리
mean = house_df["SalePrice"].mean()
test_df["saleprice"] = test_df["saleprice"].fillna(mean)

test_df['saleprice'].isna().sum()

# 필요한 열만 꺼내서 내보내기
# test_df = test_df[['Id', 'saleprice']]
#test_df.to_csv("./data/submission0731_3.csv", index=False)


# 시각화

import seaborn as sns
import matplotlib.pyplot as plt

# 리모델링 년도와 가격 관계
sns.lineplot(data=test_df, x=test_df['YearRemodAdd'], y=test_df['saleprice'], errorbar = None)
plt.show()
plt.clf()

# 전반적인 품질과 가격 관계
sns.lineplot(data=test_df, x=test_df['OverallQual'], y=test_df['saleprice'])
plt.show()
plt.clf()


