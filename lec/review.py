import pandas as pd
import matplotlib.pyplot as plt
import json

geo_seoul= json.load(open("lec/data/SIG_Seoul.geojson"))

type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul["features"][0]
len(geo_seoul["features"][0])
geo_seoul["features"][0].keys()

geo_seoul["features"][2]["properties"]

coordinate_list=geo_seoul["features"][2]["geometry"]["coordinates"]
len(coordinate_list[0][0])
coordinate_list[0][0]


import numpy as np

coordinate_array=np.array(coordinate_list[0][0])


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

gu_name = list()
for i in range(25): 
  gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])
  
gu_name


coordinate_list = [geo_seoul["features"][x]["geometry"]["coordinates"] for x in range(len(geo_seoul["features"]))]
coordinate_list


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
