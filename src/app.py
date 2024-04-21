from dash import html, dcc, Dash, Input, Output
import pandas as pd
from datetime import datetime as dt

from utils import DATA, get_ts, get_boxplot, get_histogram, get_scatterplot, get_scatterplot_output

app = Dash(__name__)

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Introdution', children=[
            html.Div([
                html.Img(
                    src="https://www.uninorte.edu.co/image/layout_set_logo?img_id=19855734&t=1706555249523",
                    style={'height': '25%', 'width': 'auto', 'float': 'right'}
                ),
                html.H2(
                    "Wind speed regression",
                    style={'text-align': 'center', 'font-family': 'Verdana, Geneva, sans-serif'}
                ),
                html.H3(
                    "Célien BONHOMME, Leonardo VAIA",
                    style={'text-align': 'center', 'font-family': 'Verdana, Geneva, sans-serif'}
                ),
            ], style={'align-items': 'center', 'margin-top': '30px', 'margin-bottom': '50px', 'margin-right': '40px', 'margin-left': '40px'}),

            html.Div([
                html.Img(
                    src="https://daxg39y63pxwu.cloudfront.net/images/blog/applications-of-machine-learning-in-energy-sector/Applications_of_Machine_Learning_in_Energy_Sector.webp",
                    style={'height': '250px', 'width': 'auto', 'margin-right': '20px'}
                ),
                html.P(
                    "The reliable prediction of wind speeds is crucial for optimizing the efficiency of wind energy production, a key component of the transition towards sustainable and renewable energy sources. Wind speed forecasting plays a pivotal role in the effective management of wind farms, enabling operators to make informed decisions to maximize energy output and grid stability.",
                    style={'display': 'inline-block', 'text-align': 'justify', 'font-family': 'Verdana, Geneva, sans-serif', 'font-size': '18px'}
                ),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-left': '20px', 'margin-right': '20px', 'margin-bottom': '40px'}),

            html.Div([
                html.Img(
                    src="https://ai-forum.com/wp-content/uploads/wind-energy-ai.jpg?fbclid=IwAR2uAwOX86kpwZJtm5Ws0KHDGwke1u6HksfyRH2G7V6A0skACAtGoGvel1E",
                    style={'height': '250px', 'width': 'auto', 'float': 'right', 'margin-left': '20px'}
                ),
                html.P(
                    "This project focuses on utilizing machine learning techniques to forecast wind speeds for the following day. By developing an accurate predictive model, we aim to address a critical challenge in the renewable energy sector: the variability of wind resources. With reliable wind speed forecasts, wind farm operators can strategically adjust turbine operations, enhancing overall energy production efficiency.", 
                    style={'display': 'inline-block', 'text-align': 'justify', 'font-family': 'Verdana, Geneva, sans-serif', 'font-size': '18px'}
                ),
            ], style={'display': 'flex', 'flex-direction': 'row-reverse', 'align-items': 'center', 'margin-left': '20px', 'margin-right': '20px', 'margin-bottom': '40px'}),

            html.Div([
                html.Img(
                    src="https://img.saurenergy.com/2022/06/artifical-intelligence-and-renewable-energy.jpg?fbclid=IwAR242z0QQkQioavKnfZrok5jt9II3yf1mBj3rgVUhS8q6LmsT8DDT9N1gVo",
                    style={'height': '250px', 'width': 'auto', 'margin-right': '20px'}
                ),
                html.P(
                    "The application of machine learning in wind speed prediction offers practical benefits, empowering the renewable energy industry to integrate wind power more effectively into the energy mix. The outcomes of this research will directly contribute to the advancement of green energy technologies, supporting the transition towards a cleaner and more sustainable energy landscape.", 
                    style={'display': 'inline-block', 'text-align': 'justify', 'font-family': 'Verdana, Geneva, sans-serif', 'font-size': '18px'}
                ),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-left': '20px', 'margin-right': '20px', 'margin-bottom': '40px'}),
        ]),

        dcc.Tab(label='EDA', children=[
            html.Div([
                html.Div([ # Dates + reset button
                    html.Div([ # Dates + labels
                        html.Div([ # Date min
                            html.Label('Start date: ', style={'margin-right': '10px'}),
                            dcc.Input(
                                id='day-start',
                                type='number',
                                placeholder='Enter start day',
                                value=1,
                                min=1,
                                max=31
                            ),
                            dcc.Input(
                                id='month-start',
                                type='number',
                                placeholder='Enter start month',
                                value=1,
                                min=1,
                                max=12
                            ),
                            dcc.Input(
                                id='year-start',
                                type='number',
                                placeholder='Enter start year',
                                value=2000,
                                min=2000,
                                max=2010
                            ),
                            dcc.Input(
                                id='hour-start',
                                type='number',
                                placeholder='Enter start hour',
                                value=12,
                                min=0,
                                max=23
                            ),
                        ]),
                        html.Div([ # Date max
                            html.Label('End date: ', style={'margin-right': '14px'}),
                            dcc.Input(
                                id='day-end',
                                type='number',
                                placeholder='Enter end day',
                                value=19,
                                min=1,
                                max=31
                            ),
                            dcc.Input(
                                id='month-end',
                                type='number',
                                placeholder='Enter end month',
                                value=4,
                                min=1,
                                max=12
                            ),
                            dcc.Input(
                                id='year-end',
                                type='number',
                                placeholder='Enter end year',
                                value=2010,
                                min=2000,
                                max=2010
                            ),
                            dcc.Input(
                                id='hour-end',
                                type='number',
                                placeholder='Enter end hour',
                                value=23,
                                min=0,
                                max=23
                            ),
                        ]),
                    ]),
                    html.Button('Reset Dates', id='reset-dates-button', style={'margin-left': '10px'}),
                ], style={'display': 'flex', 'margin-bottom': '20px'}),           
            ], style={'margin-top': '20px'}), 
            dcc.Dropdown(
                id='selected-columns',
                options=DATA.drop(columns=['Date', 'Hour']).columns.sort_values(),
                value=['Atmospheric pressure max (Pa)'],
                multi=True,
                placeholder='Select columns'
            ),
            html.Div(id='time-series'),
            html.Div([
                html.Div(id='boxplot', style={'width': '50%', 'display': 'inline-block'}),
                html.Div(id='histogram', style={'width': '50%', 'display': 'inline-block'}),
            ]),
            html.Div(id='bivariate-output'),
            html.Div([
                dcc.Dropdown(
                    id='bivariate-column1',
                    options=DATA.drop(columns=['Date']).columns.sort_values(),
                    value='Atmospheric pressure at station level (Pa)',
                    placeholder='Select first column'
                ),
                dcc.Dropdown(
                    id='bivariate-column2',
                    options=DATA.drop(columns=['Date']).columns.sort_values(),
                    value='Atmospheric pressure max (Pa)',
                    placeholder='Select second column'
                ),
                html.Div(id='bivariate')
            ]),
        ]),

        dcc.Tab(label='Results', children=[
            # TODO
        ]),
    ])
])

@app.callback(
    [Output('day-start', 'value'),
     Output('month-start', 'value'),
     Output('year-start', 'value'),
     Output('hour-start', 'value'),
     Output('day-end', 'value'),
     Output('month-end', 'value'),
     Output('year-end', 'value'),
     Output('hour-end', 'value')],
    [Input('reset-dates-button', 'n_clicks')]
)
def reset_dates(n_clicks):
    return 1, 1, 2000, 12, 19, 4, 2010, 23

@app.callback(
    [Output('time-series', 'children'),
     Output('boxplot', 'children'),
     Output('histogram', 'children'),
     Output('bivariate-output', 'children')],
    [Input('selected-columns', 'value'),
     Input('year-start', 'value'),
     Input('month-start', 'value'),
     Input('day-start', 'value'),
     Input('hour-start', 'value'),
     Input('year-end', 'value'),
     Input('month-end', 'value'),
     Input('day-end', 'value'),
     Input('hour-end', 'value')]
)
def update_time_series(selected_columns, year_start, month_start, day_start, hour_start, year_end, month_end, day_end, hour_end):
    start_date = dt(year_start, month_start, day_start, hour_start)
    end_date = dt(year_end, month_end, day_end, hour_end)
    
    time_series = [dcc.Graph(figure=get_ts(col, start_date, end_date)) for col in selected_columns]
    boxplot = [dcc.Graph(figure=get_boxplot(col, start_date, end_date)) for col in selected_columns]
    histogram = [dcc.Graph(figure=get_histogram(col, start_date, end_date)) for col in selected_columns]
    bivariate = [dcc.Graph(figure=get_scatterplot_output(col, start_date, end_date)) for col in selected_columns]

    return time_series, boxplot, histogram, bivariate

@app.callback(
    Output('bivariate', 'children'),
    [Input('bivariate-column1', 'value'),
     Input('bivariate-column2', 'value'),
     Input('year-start', 'value'),
     Input('month-start', 'value'),
     Input('day-start', 'value'),
     Input('hour-start', 'value'),
     Input('year-end', 'value'),
     Input('month-end', 'value'),
     Input('day-end', 'value'),
     Input('hour-end', 'value')]
)

def update_bivariate(bivariate_column1, bivariate_column2, year_start, month_start, day_start, hour_start, year_end, month_end, day_end, hour_end):
    start_date = dt(year_start, month_start, day_start, hour_start)
    end_date = dt(year_end, month_end, day_end, hour_end)
    col1 = bivariate_column1
    col2 = bivariate_column2
    scatterplot = [dcc.Graph(figure=get_scatterplot(col1, col2, start_date, end_date))]
    return scatterplot

if __name__ == '__main__':
    app.run_server(debug=True)