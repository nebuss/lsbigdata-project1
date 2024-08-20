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

# subplot 관련 패키지 불러오기

from plotly.subplots import make_subplots

fig_subplot= make_subplots(
    rows=3, cols=3,
    subplot_titles= ('Adeli', 'Gentoo', 'Chinstrap')
)

adelie_penguins = penguins.query('species == "Adelie"')

fig_subplot.add_trace({
    'type': 'scatter', 'mode' : 'markers',
    'x' : adelie_penguins['bill_length_mm'],
    'y': adelie_penguins["bill_depth_mm"],
    'name' : 'Adeli',
    },
    row=1, col=1
)

gentoo_penguins = penguins.query('species == "Gentoo"')

fig_subplot.add_trace({
    'type': 'scatter', 'mode' : 'markers',
    'x' : gentoo_penguins['bill_length_mm'],
    'y': gentoo_penguins["bill_depth_mm"],
    'name' : 'Gentoo',
    },
    row=1, col=2
)

chinstrap_penguins = penguins.query('species == "Chinstrap"')

fig_subplot.add_trace({
    'type': 'scatter', 'mode' : 'markers',
    'x' : chinstrap_penguins['bill_length_mm'],
    'y': chinstrap_penguins["bill_depth_mm"],
    'name' : 'Chinstrap',
    },
    row=1, col=3
)

fig_subplot.update_layout(title=dict(text= "펭귄 종별 부리 길이 vs. 깊이",  x = 0.5))



# 레이아웃 배치 조정
# Using the penguins dataset for plotting
# Create a scatter plot using plotly.subplots with a large total plot and three species-specific subplots below
# The large plot occupies 2 rows and the species-specific plots occupy 1 row

fig = make_subplots(
    rows=3, cols=3,
    specs=[[{'colspan': 3, 'rowspan': 2}, None, None],
           [None, None, None],
           [{'colspan': 1}, {'colspan': 1}, {'colspan': 1}]],
           # This row is needed to accommodate the rowspan of 2 for the total plot
    subplot_titles=["전체 데이터", "아델리", "친스트랩", "젠투"],
    row_heights=[0.35, 0.35, 0.3],  # Heights for the total plot and the species-specific plots
    shared_xaxes=True,
    horizontal_spacing=0.05,
    vertical_spacing=0.05)

# Colors for different species
colors = {
    "Adelie": "blue",
    "Chinstrap": "red",
    "Gentoo": "green"
}

import plotly.graph_objects as go
# Plot for all data with different colors for each species
for species, color in colors.items():
    subset = penguins[penguins['species'] == species]
    fig.add_trace(
        go.Scatter(
            x=subset['bill_length_mm'], 
            y=subset['bill_depth_mm'], 
            mode='markers',
            marker=dict(size=8, color=color),
            name=f'{species}'
        ),
        row=1, col=1
    )

# Subplots for each species in the third logical row
for i, species in enumerate(penguins['species'].unique(), 1):
    subset = penguins[penguins['species'] == species]
    fig.add_trace(
        go.Scatter(
            x=subset['bill_length_mm'], 
            y=subset['bill_depth_mm'], 
            mode='markers',
            marker=dict(size=7, line=dict(width=1), color=colors[species]),
            name=f'{species}'
        ),
        row=3, col=i
    )

# Update xaxis and yaxis properties for all subplots
fig.update_xaxes(title_text="부리 길이 (mm)")
fig.update_yaxes(title_text="부리 깊이 (mm)")

# Update layout and size, center title
fig.update_layout(height=900, width=1000, title_text="펭귄 종별 부리 치수", title_x=0.5)
fig