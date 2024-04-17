from dash import html, dcc, Dash, Input, Output
import pandas as pd

from var import data, ts

app = Dash(__name__)

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='EDA', children=[
            dcc.Dropdown(
                id='selected-columns',
                options=data.drop(columns=['Fecha']).columns,
                value=['Precipitación total por hora (m)'],
                multi=True
            ),
            html.Div(id='time-series')
        ]),

        dcc.Tab(label='Introdution', children=[
            # TODO
        ]),
    ])
])

@app.callback(
    Output('time-series', 'children'),
    [Input('selected-columns', 'value')]
)
def update_time_series(selected_columns):
    return [dcc.Graph(figure=ts[col]) for col in selected_columns]

if __name__ == '__main__':
    app.run_server(debug=True)