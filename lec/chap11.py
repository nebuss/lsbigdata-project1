## 0807 서울시 지도 그리기

import numpy as np
import matplotlib.pyplot as plt
import json

geo_seoul = json.load(open("lec/data/SIG_Seoul.geojson", encoding="UTF-8"))

# 데이터 탐색
type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul["features"][0]
len(geo_seoul["features"])
len(geo_seoul["features"][0])
geo_seoul["features"][0].keys()

# 숫자가 바뀌면 "구"가 바뀌는구나!
geo_seoul["features"][2]["properties"]
geo_seoul["features"][0]["geometry"]

# 리스트로 정보 빼오기
coordinate_list=geo_seoul["features"][2]["geometry"]["coordinates"]
len(coordinate_list[0][0])
coordinate_list[0][0]

coordinate_array=np.array(coordinate_list[0][0])
x=coordinate_array[:,0]
y=coordinate_array[:,1]

plt.plot(x, y)
plt.show()
plt.clf()

# 함수로 만들기
def draw_seoul(num):
    gu_name=geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]

    plt.rcParams.update({"font.family": "Malgun Gothic"})
    plt.plot(x, y)
    plt.title(gu_name)
    plt.show()
    plt.clf()
    
    return None

draw_seoul(12)


# 서울시 전체 지도 그리기
import pandas as pd
import numpy as np


gu_name | x | y
===============
종로구  | 126 | 36
종로구  | 126 | 36
종로구  | 126 | 36
......
종로구  | 126 | 36
종로구  | 126 | 36
중구  | 126 | 36
중구  | 126 | 36
......
중구  | 126 | 36

plt.plot(x, y, hue="gu_name")


# 리스트 컴프리헨션
# gu_name = [geo_seoul["features"][x]["properties"]["SIG_KOR_NM"] for x in range(len(geo_seoul["features"]))]
# gu_name

gu_name = list()
for i in range(25):
  # guname = gu_name+[geo_seoul["features"][i]["properties"]["SIG_KOR_NM"]]
  gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])
gu_name


coordinate_list = [geo_seoul["features"][x]["geometry"]['coordinates'] for x in range(len(geo_seoul["features"]))]
coordinate_list

# x, y 판다스 데이터 프레임
def draw_seoul(num):
    gu_name = geo_seoul['features'][num]['properties']['SIG_KOR_NM']
    coordinate_list = geo_seoul['features'][num]['geometry']['coordinates']
    coordinate_list = np.array(coordinate_list[0][0])
    x = coordinate_list[:, 0]
    y = coordinate_list[:, 1]
    
    return pd.DataFrame({'gu_name' : gu_name, 'x' : x, 'y' : y})
  
draw_seoul(12)

result= pd.DataFrame({})

for i in range(25):
  result=pd.concat([result, draw_seoul(i)], ignore_index=True)
result 

result.plot(x="x", y="y", kind='scatter', s=1)
plt.show()
plt.clf()

import seaborn as sns
sns.scatterplot(data=result, x='x', y='y', s=1 , hue='gu_name', legend=False, palette = "deep")
plt.show()
plt.clf()

# 강남구만 빨간색
gangnam_df= result.assign(is_gangnam = np.where(result['gu_name'] == '강남구', '강남', '안강남'))
sns.scatterplot(data=gangnam_df, x='x', y='y', s=1 , hue='is_gangnam',
                palette={"안강남":'grey', "강남": 'red'})
plt.show()
plt.clf()

gangnam_df["is_gangnam"].unique() # 이 코드 찍어보면 안강남이 0번.


------서연이네 조 코드 -------
geo_mex=[]
geo_mey=[]
geo_name=[]

for x in np.arange(0,25):
    gu_name=geo_seoul["features"][x]["properties"]['SIG_KOR_NM']
    coordinates_list=geo_seoul["features"][x]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinates_list[0][0])
    
    geo_mex.append(coordinate_array[:,0])
    geo_mey.append(coordinate_array[:,1])
    geo_name.append(gu_name)

for x in np.arange(0,25):
    plt.plot(geo_mex[x],geo_mey[x])
    plt.show()
    
plt.clf() 



geo_seoul['features'][0]['properties']
df_pop = pd.read_csv('lec/data/Population_SIG.csv')
df_seoulpop=df_pop.iloc[1:26]

#code를 문자타입으로 변경
df_seoulpop['code'] = df_seoulpop['code'].astype(str)
df_seoulpop.info()

import folium

my_map = folium.Map(location=[37.55, 126.97], 
                    zoom_start=8,
                    tiles="CartoDB positron") 
my_map.save("seoul_map.html")


folium.Choropleth(
  geo_data= geo_seoul,
  data = df_seoulpop,
  columns= ('code', 'pop'),
  key_on = 'feature.properties.SIG_CD')\
   .add_to(my_map)
my_map.save("seoul_map.html")

bins = df_seoulpop["pop"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])

folium.Choropleth(
  geo_data= geo_seoul,
  data = df_seoulpop,
  columns= ('code', 'pop'),
  key_on = 'feature.properties.SIG_CD',
  fill_color = 'YlGnBu',
  fill_opacity = 1,
  line_opacity = 0.5,
  bins = bins).add_to(my_map)

my_map.save("seoul_map.html")


# 점 찍는 법
# make_seouldf(0).iloc[:,1:3].mean()
folium.Marker([37.583744, 126.983800], popup="종로구").add_to(my_map)
my_map.save("seoul_map.html")
