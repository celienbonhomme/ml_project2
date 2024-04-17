import plotly.graph_objects as go
import pandas as pd

data = pd.read_csv('../data/data_wind_imputed.csv')

col = 'Precipitación total por hora (m)'
ts = {}
for col in data.drop(columns=['Fecha']).columns:
    ts[col] = go.Figure()
    ts[col].add_trace(go.Scatter(x=data['Fecha'], y=data[col], mode='lines', name=col))
    ts[col].update_layout(title=f'{col} over time',
                          xaxis_title='Date', 
                          yaxis_title=col,
                          margin=dict(l=20, r=20, t=40, b=20))
