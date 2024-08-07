import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



## 필요한 데이터 불러오기
house_train=pd.read_csv("houseprice/data/house_loc.csv")



want_col = ['MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF']

df = house_train[want_col]       
       

quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
exposure_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
fin_type_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
heating_map = {'Floor': 6, 'GasA': 5, 'GasW': 4, 'Grav': 3, 'OthW': 2, 'Wall': 1}
central_air_map = {'Y': 1, 'N': 0}
electrical_map = {'SBrkr': 5, 'FuseA': 4, 'FuseF': 3, 'FuseP': 2, 'Mix': 1}
foundation_map = {'BrkTil': 1, 'CBlock': 2, 'PConc': 3, 'Slab': 4, 'Stone': 5, 'Wood': 6}

df.loc[:, 'ExterQual'] = df['ExterQual'].map(quality_map)
df.loc[:, 'ExterCond'] = df['ExterCond'].map(quality_map)
df.loc[:, 'BsmtQual'] = df['BsmtQual'].map(quality_map)
df.loc[:, 'BsmtCond'] = df['BsmtCond'].map(quality_map)
df.loc[:, 'BsmtExposure'] = df['BsmtExposure'].map(exposure_map)
df.loc[:, 'BsmtFinType1'] = df['BsmtFinType1'].map(fin_type_map)
df.loc[:, 'BsmtFinType2'] = df['BsmtFinType2'].map(fin_type_map)
df.loc[:, 'HeatingQC'] = df['HeatingQC'].map(quality_map)
df.loc[:, 'CentralAir'] = df['CentralAir'].map(central_air_map)
df.loc[:, 'Electrical'] = df['Electrical'].map(electrical_map)
df.loc[:, 'Foundation'] = df['Foundation'].map(foundation_map)
df.loc[:, 'Heating'] = df['Heating'].map(heating_map)

# 상관관계 계산
corr_matrix = df.corr()

# 상관관계 히트맵 시각화
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, \
          annot_kws={"size": 2}, yticklabels={'size': 2}, xticklabels={'size': 2})
plt.title('Correlation Matrix of Selected Features')
plt.show()

plt.clf()



corr_pairs = corr_matrix.unstack()
sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
strong_pairs = sorted_pairs[sorted_pairs != 1]  # 자신과의 상관관계 제외

# 상위 10개의 상관관계 출력
print(strong_pairs.head(10))






sns.histplot(house_train['SalePrice'], kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()
plt.clf()

# 지상 생활 면적과 주택 가격 사이의 관계를 시각화합니다.
sns.scatterplot(x='GrLivArea', y='SalePrice', data=house_train)
plt.title('Sale Price vs. Ground Living Area')
plt.xlabel('Ground Living Area (sq ft)')
plt.ylabel('Sale Price')
plt.show()


# 
sns.pairplot(house_train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']])

plt.show()
plt.clf()


# 동네에 속하는 주택 수 
plt.figure(figsize=(12, 6))
sns.countplot(x='Neighborhood', data=house_train)
plt.xticks(rotation=90)
plt.title('Number of Houses by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Count')
plt.show()

# Neighborhood와 SalePrice를 사용한 박스 플롯/ 특정 지역의 가격분포
plt.figure(figsize=(14, 8))
sns.boxplot(x='Neighborhood', y='SalePrice', data=house_train)
plt.xticks(rotation=90)  # x축 라벨을 90도 회전
plt.title('Sale Price Distribution by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Sale Price')
plt.show()
plt.clf()

-----------------------------
import pandas as pd
import folium
from folium.plugins import HeatMap

# 데이터 불러오기
house_train = pd.read_csv("houseprice/data/house_loc.csv")

house_train.info()
house_loc= house_train.iloc[:, -2:]


map_sig = folium.Map(location = [42.034482, -93.642897],
                     zoom_start = 8,
                     tiles = 'cartodbpositron')


house_loc = house_loc.iloc[:, -2:]
house_loc.mean()
map_sig = folium.Map(location = [42.034482, -93.642897],
                     zoom_start = 10,
                     tiles = 'cartodbpositron')

for i in range(2930):
  folium.Marker([house_loc.iloc[i,1], house_loc.iloc[i,0]], popup = '0').add_to(map_sig)
map_sig.save('map_ames.html')



----------

# 필요한 데이터만 추출 ('Neighborhood', 'SalePrice', 'Latitude', 'Longitude'가 포함된 경우)
df = house_train[['Neighborhood', 'Sale_Price', 'Latitude', 'Longitude']]

# 결측값 제거
df = df.dropna(subset=['Latitude', 'Longitude', 'Sale_Price'])

# 지도의 중심 좌표를 설정 (예: Ames, Iowa의 중심)
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]





my_map= folium.Map(location=map_center, zoom_start=12,
                  tiles="CartoDB positron")
# 마커 찍기
folium.Marker([42.0344, -93.619], popup="0").add_to(my_map)

# 
# for idx, row in df.iterrows():
#     folium.Marker(
#         location=[row['Latitude'], row['Longitude']],
#     ).add_to(my_map)
#    



from folium.plugins import MarkerCluster
mc = MarkerCluster().add_to(my_map)

# 마커 찍기
for i in range(len(df)):
    folium.Marker(
        location=[df.iloc[i, df.columns.get_loc('Latitude')], df.iloc[i, df.columns.get_loc('Longitude')]],
        popup=str(i)  # 인덱스를 문자열로 변환
    ).add_to(mc)

# 지도 저장
my_map.save('map_ames.html')

# HeatMap 데이터 준비
# heat_data = [[row['Latitude'], row['Longitude'], row['Sale_Price']] for index, row in df.iterrows()]
# 
# # HeatMap 추가
# HeatMap(heat_data, radius=15).add_to(m)

# 지도 표시
my_map.save("heatmap.html")






