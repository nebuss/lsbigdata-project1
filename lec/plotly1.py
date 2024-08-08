#0808
!pip install plotly
 
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np

df_covid19_100 = pd.read_csv('data/df_covid19_100.csv')
df_covid19_100.columns

margin_P = {'t':50, 'b':25, 'l':25, 'r':25}

fig=go.Figure(
  data = [
    {"type": "scatter",
     "mode": "markers",
     #X, Y 축에 변수 매핑
     "x" : df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", 'date'],
     "y" : df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
     "marker" : {"color" :'red'}
     },
     {"type": "scatter",
     "mode": "lines",
     #X, Y 축에 변수 매핑
     "x" : df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", 'date'],
     "y" : df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
     "line" : {"color" :'#5E88FC'}
      }
  ], 
  layout = {
    'title' : "코로나 19 발생 현황",
    'xaxis' : {'title' : '날짜', 'showgrid' : False},
    'yaxis' : {'title': '확진자 수'},
    'margin' : margin_P
  }).show()
  
  #프레임속성========
# 애니메이션 프레임 생성
frames = []
dates = df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"].unique()

for date in dates:
    frame_data = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "marker": {"color": "red"}
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "line": {"color": "blue", "dash": "dash"}
            }
        ],
        "name": str(date)
    }
    frames.append(frame_data)


x_range = ['2022-10-03', '2023-01-11']
y_range = [8900, 88172]

# 애니메이션을 위한 레이아웃 설정
margins_P = {"l": 25, "r": 25, "t": 50, "b": 50}
layout = {
    "title": "코로나 19 발생현황",
    "xaxis": {"title": "날짜", "showgrid": False, "range":x_range},
    "yaxis": {"title": "확진자수", "range":y_range},
    "margin": margins_P,
    "updatemenus": [{
        "type": "buttons",
        "showactive": False,
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
        }, {
            "label": "Pause",
            "method": "animate",
            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
        }]
    }]
}

# Figure 생성
fig = go.Figure(
    data=[
        {
            "type": "scatter",
            "mode": "markers",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "marker": {"color": "red"}
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "line": {"color": "blue", "dash": "dash"}
        }
    ],
    layout=layout,
    frames=frames
)

fig.show()
  
  
------
127p.

import plotly.express as px
!pip install palmerpenguins
from palmerpenguins import load_penguins


penguins = load_penguins()
penguins.head()

# x: bill_length_mm
# y: bill_depth_mm  
fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    size= "size",
    size_max=30
)

fig.update_traces(marker=dict(size=20), opacity = 0.7))

# 레이아웃 업데이트
# fig=px.scatter(penguins, x= "bill_length_mm", y= "bill_depth_mm", color="species", trendline="ols")
# dict() = {}
fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이와 깊이", font=dict(color="white", size=24)),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(
        font=dict(color="white", size=14),  # 범례 폰트 크기 조정
        title=dict(text="펭귄 종", font=dict(color="white", size=14))  # 범례 제목 조정
    )
)
fig.update_traces(marker=dict(size=12, opacity=0.7)) 
fig.show()



from sklearn.linear_model import LinearRegression

model = LinearRegression()
penguins=penguins.dropna()
x=penguins[["bill_length_mm"]]
y=penguins["bill_depth_mm"]

model.fit(x, y)
linear_fit=model.predict(x)
model.coef_
model.intercept_

fig.add_trace(
    go.Scatter(
        mode="lines",
        x=penguins["bill_length_mm"], y=linear_fit,
        name="선형회귀직선",
        line=dict(dash="dot", color="white")
    )
)
fig.show()


# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=False)
penguins_dummies.columns
penguins_dummies.iloc[:,-3:]
# x와 y 설정 범주형(문자열) 변수를 숫자형으로 바꾸니 modelfit  가능
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

model.coef_
model.intercept_
# 모델 학습
model = LinearRegression()
model.fit(x, y)


# 위 코드는 다음과 같은 회귀 직선임
# y = 0.2 * bill_length_mm + (-.193 * species_Chinstrap) + (-5.1 * species_Gentoo) + 10.56

# species    island  bill_length_mm  ...  body_mass_g     sex  year
# Adelie     Torgersen            39.5  ...       3800.0  female  2007
# Chinstrap  Torgersen            40.5  ...       3800.0  female  2007
# Gentoo     Torgersen            40.5  ...       3800.0  female  2007
# x1, x2, x3
# 39.5, 0, 0
# 40.5, 1, 0
# y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56
# adeli 에 비해 다른 종들의 부리 깊이가 -1.93, -5.1 만큼 작다
0.2 * 40.5 -1.93 * True -5.1* False + 10.56

# 입력값이 범주형인 변수 시각화

penguins_dummies['predicted_bill_depth_mm'] = model.predict(x)

# 시각화
fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    trendline="ols",
    trendline_color_override="black",
    labels={"bill_length_mm": "부리 길이 (mm)", "bill_depth_mm": "부리 깊이 (mm)"}
)

# 회귀선 추가
species_list = ["Adelie", "Chinstrap", "Gentoo"]
for species in species_list:
    filtered_data = penguins_dummies[penguins_dummies[f'species_{species}'] == 1]
    fig.add_scatter(
        x=filtered_data["bill_length_mm"],
        y=filtered_data["predicted_bill_depth_mm"],
        mode="lines",
        name=f"{species} 회귀선"
    )

# 레이아웃 업데이트
fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이와 깊이", font=dict(color="white", size=24)),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")),
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")),
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'
    ),
    legend=dict(
        font=dict(color="white", size=14),
        title=dict(text="펭귄 종", font=dict(color="white", size=14))
    )
)

fig.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x["bill_length_mm"], y, color='black', hue=penguins["species"])
sns.scatterplot(x["bill_length_mm"], penguins_dummies['predicted_bill_depth_mm'], s=1) 

plt.show()
plt.clf()

