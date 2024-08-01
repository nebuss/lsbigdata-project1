import pandas as pd
import numpy as np


house_df = pd.read_csv("./data/train.csv")
house_df

price_mean = house_df['SalePrice'].mean()

sample_df = pd.read_csv("data/sample_submission.csv")
sample_df

sample_df['SalePrice'] = price_mean

sample_df.to_csv('/data/sample_submission.csv', index=False)

#같은 년도를 그룹으로 같은 해에 지어진 집들을 한 그룹으로. 이 년도별 집들의 평균을 구해서 
#test셋에 있는 집값을 예측해보자

house_df = pd.read_csv("./data/train.csv")
house_df.columns

#연도별 평균
my_df=house_df.groupby('YearBuilt', as_index=False)\ 
         .agg(
           price_mean=('SalePrice', 'mean')
         )
         
         
         
         
# 테스트 데이터 불러오기
test_df = pd.read_csv("./data/test.csv")

# 테스트 데이터에 평균 집 값을 매핑


test_df=test_df[['Id', 'YearBuilt']]
test_df = pd.merge(test_df, my_df, on='YearBuilt', how='left')
test_df = test_df.rename(columns={'price_mean' : 'SalePrice'})

#결측치 확인
test_df['SalePrice'].isna().sum()
test_df.loc[test_df['SalePrice'].isna()]

#집 값 채우기
mean = house_df["SalePrice"].mean()
test_df["SalePrice"].fillna(mean)


# SalePrice 바꿔치기
test_df['SalePrice'] = house_df['SalePrice']
test_df

test_df = test_df[['Id', 'SalePrice']]
# 예측 결과 저장
test_df.to_csv("./data/submission.csv", index=False)




#집에서 연식, 평수, 위치 등등 그룹을 나눠서  집값을 예측해보자
import pandas as pd
import numpy as np


house_df = pd.read_csv("./data/train.csv")
house_df.columns

my_df=house_df.groupby(['YearBuilt', 'OverallCond', 'GrLivArea'], as_index=False)\ 




         .agg(
           saleprice=('SalePrice', 'mean'))
         
# 테스트 데이터 로드

test_df = pd.read_csv("./data/test.csv")

predictions = test_df.merge(
    my_df,
    on=['YearBuilt', 'OverallCond', 'GrLivArea'],
    how='left'
)

overall_mean = my_df['saleprice'].mean()

# 병합 결과에서 결측치가 있는 경우 전체 평균값으로 대체
predictions['SalePrice'] = predictions['saleprice'].fillna(overall_mean)


predictions[['Id', 'SalePrice']].to_csv("./data/submission2.csv", index=False)

# 결과 확인
print(predictions[['Id', 'SalePrice']])







