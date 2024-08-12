import plotly.express as px
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    size_max=30,
    
)
fig.show()
fig.update_layout(
    title=dict(
        text="<span style='color:blue;font-weight:bold;'>팔머펭귄</span>",  # HTML 태그를 문자열로 처리
        x=0.5, 
        xanchor='center',
        yanchor='top'
    )
)
# <span> ... </span> # 글자의 서식을 나타내는 구문

