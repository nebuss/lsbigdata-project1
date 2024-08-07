import json
geo= json.load(open('data/SIG.geojson', encoding='UTF-8'))

geo['features'][0]['properties']

#위도, 경도 좌표 출력
geo['features'][0]['geometry']


#시군구별 데이터 준비하기
import pandas as pd
df_pop = pd.read_csv('data/Population_SIG.csv')
df_pop.head()
df_pop.info()

df_pop['code'] = df_pop['code'].astype(str)

!pip install folium

import folium
folium.Map(location= [35.95, 127.7],
           zoom_start= 8)
          
map_sig = folium.Map(location = [35.95, 127.7],
                     zoom_start = 8,
                     tiles = 'cartodbpositron')
