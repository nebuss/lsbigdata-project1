import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# 샘플 데이터프레임 생성
df = pd.DataFrame({
    'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 'Bananas'],
    'Amount': [4, 1, 2, 2, 4, 5],
    'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
})

# Dash 애플리케이션 초기화
app = dash.Dash(__name__)

# 애플리케이션 레이아웃 정의
app.layout = html.Div([
    dcc.Dropdown(
        id='city-dropdown',
        options=[
            {'label': 'SF', 'value': 'SF'},
            {'label': 'Montreal', 'value': 'Montreal'}
        ],
        value='SF'
    ),
    dcc.Graph(id='bar-graph')
])

# 콜백 함수 정의
@app.callback(
    Output('bar-graph', 'figure'),
    Input('city-dropdown', 'value')
)
def update_graph(selected_city):
    filtered_df = df[df['City'] == selected_city]
    fig = px.bar(filtered_df, x='Fruit', y='Amount', title=f'Fruit Amounts in {selected_city}')
    return fig

# 애플리케이션 실행
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
