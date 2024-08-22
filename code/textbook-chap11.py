import numpy as np
import matplotlib.pyplot as plt
import json

geo_seoul = json.load(open("./data/SIG_Seoul.geojson", encoding="UTF-8"))

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
    # 축 비율 1:1로 설정
    plt.axis('equal')
    plt.show()
    plt.clf()
    
    return None

draw_seoul(12)


# 서울시 전체 지도 그리기
# gu_name | x | y
# ===============
# 종로구  | 126 | 36
# 종로구  | 126 | 36
# 종로구  | 126 | 36
# ......
# 종로구  | 126 | 36
# 종로구  | 126 | 36
# 중구  | 126 | 36
# 중구  | 126 | 36
# ......
# 중구  | 126 | 36
# x, y 판다스 데이터 프레임
import pandas as pd

def make_seouldf(num):
    gu_name=geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]

    return pd.DataFrame({"gu_name":gu_name, "x": x, "y": y})

make_seouldf(1)

result=pd.DataFrame({})
for i in range(25):
    result=pd.concat([result, make_seouldf(i)], ignore_index=True)    

result

# 서울 그래프 그리기
import seaborn as sns
sns.scatterplot(data=result,
    x='x', y='y', hue='gu_name', legend=False,
    palette="viridis", s=2)
plt.show()
plt.clf()


# 서울 그래프 그리기
import seaborn as sns
gangnam_df=result.assign(is_gangnam=np.where(result["gu_name"]!="강남구", "안강남", "강남"))
sns.scatterplot(
    data=gangnam_df,
    x='x', y='y', legend=False, 
    palette={"안강남": "grey", "강남": "red"},
    hue='is_gangnam', s=2)
plt.show()
plt.clf()

# # 데이터프레임 concat 예제
# df_a = pd.DataFrame({
#     'ID': [1, 2, 3],
#     'Name': ['David', 'Eva', 'Frank'],
#     'Age': [35, 45, 55]
# })
# 
# df_b = pd.DataFrame({
#     'ID': [4, 5, 6],
#     'Name': ['David', 'Eva', 'Frank'],
#     'Age': [40, 45, 50]
# })
# df_a=pd.concat([df_a, df_b])

# 파이썬
# 가죽자켓&썬구리
# 흰티(셔츠)&청바지
## 받은 맨투맨 회색티 & 청바지 연청,중청
## 개발자 코스프레 (체크셔츠)
# 와이셔츠 슬랙스

# # 구이름 만들기
# # 방법 1
# gu_name=list()
# for i in range(25):
#     gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])
# gu_name
# 
# # 방법 2
# gu_name = [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"] for i in range(25))]
# gu_name

import numpy as np
import matplotlib.pyplot as plt
import json

geo_seoul = json.load(open("./data/SIG_Seoul.geojson", encoding="UTF-8"))
geo_seoul["features"][0]["properties"]

df_pop = pd.read_csv("data/Population_SIG.csv")
df_seoulpop=df_pop.iloc[1:26]
df_seoulpop["code"]=df_seoulpop["code"].astype(str)
df_seoulpop.info()

# 패키지 설치하기
# !pip install folium
import folium

center_x=result["x"].mean()
center_y=result["y"].mean()
# p.304
# 흰 도화지 맵 가져오기
map_sig=folium.Map(location = [37.551, 126.973],
                  zoom_start=12,
                  tiles="cartodbpositron")

# 코로플릿 사용해서 - 구 경계선 그리기
folium.Choropleth(
    geo_data=geo_seoul,
    data=df_seoulpop,
    columns=("code", "pop"),
    key_on = "feature.properties.SIG_CD").add_to(map_sig)
    
map_sig.save("map_seoul.html")

# 코로플릿 with bins
# matplotlib 팔레트
# tab10, tab20, Set1, Paired, Accent, Dark2, Pastel1, hsv 
# seaborn 팔레트
# deep, muted, bright, pastel, dark, colorblind, viridis, inferno, magma, plasma

bins = list(df_seoulpop["pop"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))
folium.Choropleth(
    geo_data=geo_seoul,
    data=df_seoulpop,
    columns=("code", "pop"),
    fill_color="viridis",
    bins=bins,
    key_on = "feature.properties.SIG_CD").add_to(map_sig)
    
map_sig.save("map_seoul.html")

# 점 찍는 법
# make_seouldf(0).iloc[:,1:3].mean()
make_seouldf(1)
folium.Marker([37.583744, 126.983800], popup="종로구").add_to(map_sig)
map_sig.save("map_seoul.html")


# House price 데이터의 lon, lat 정보
# 지도에 표시하기
