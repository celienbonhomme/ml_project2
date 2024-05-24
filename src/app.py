from dash import html, dcc, Dash, Input, Output
from datetime import datetime as dt
import pandas as pd
from utils_ui import get_ts, get_boxplot, get_histogram, get_scatterplot, get_scatterplot_output
from utils import model_evaluation_lr
import os
import pickle
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from time import time


import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('../data/data_preprocessed.csv')
X, y = data.drop(columns=['Wind speed (m/s)']), data['Wind speed (m/s)']


X_test = X[int(0.8*len(X)):]
y_test = y[int(0.8*len(y)):]

directory_path_models = Path("../pickle/models")
path_models = os.listdir(directory_path_models)
models = {path_model.split('.')[0].replace('_', ' '): {} for path_model in path_models}

results = pd.DataFrame(columns=['Model', 'Training time (s)', 'Predicting time (ms)', 'R2', 'RMSE'])
results_residuals = pd.DataFrame(columns=['Model', 'Mean', 'Std'])
nb_res = 0
for path_model in path_models:
    path_model = str(directory_path_models / path_model)
    path_cpu_time = path_model.replace('models', 'cpu_time').replace('.pkl', '_time.pkl')
    with open(path_model, 'rb') as f1:
        model = pickle.load(f1)
    with open(path_cpu_time, 'rb') as f2:
        training_time = pickle.load(f2)
    name = path_model.split('/')[-1].split('.')[0].replace('_', ' ')
    models[name]['model'] = model
    start = time()
    models[name]['predictions'] = model.predict(X_test.drop(columns=['Date']))
    end = time()
    predict_time = round((end - start)*1000)
    eval = model_evaluation_lr(y_test, models[name]['predictions'])
    results.loc[nb_res] = [name, training_time, predict_time, eval['r2'], eval['rmse']]
    residuals = y_test - models[name]['predictions']
    norm_res = norm.fit(residuals)
    results_residuals.loc[nb_res] = [name, round(norm_res[0], 2), round(norm_res[1], 2)]
    nb_res += 1

# Feature importance
feature_importance = models['Gradient Boosting']['model'].best_estimator_['model'].feature_importances_
features = X.drop(columns=['Date']).columns

sorted_indices = np.argsort(feature_importance)
sorted_feature_importance = feature_importance[sorted_indices]
sorted_features = features[sorted_indices]
fig = go.Figure(go.Bar(x=sorted_feature_importance, y=sorted_features, orientation='h'))
fig.update_layout(title='Feature importance (GB)', xaxis_title='Importance', yaxis_title='Feature', width=750, height=330)

results_md = results.sort_values(by='R2', ascending=False).to_markdown(index=False)
tab_residuals_md = results_residuals.sort_values(by='Std').to_markdown(index=False)


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
                    "This project focuses on using machine learning techniques to forecast wind speeds for the following day. By developing an accurate predictive model, we aim to address a critical challenge in the renewable energy sector: the variability of wind resources. With reliable wind speed forecasts, wind farm operators can strategically adjust turbine operations, enhancing overall energy production efficiency.", 
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
                html.H2('Exploratory Data Analysis'),
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
                options=data.drop(columns=['Date', 'Hour']).columns.sort_values(),
                value=['Wind speed (m/s)'],
                multi=True,
                placeholder='Select columns'
            ),
            html.Div(id='time-series'),
            html.Div([
                html.Div(id='boxplot', style={'width': '50%', 'display': 'inline-block'}),
                html.Div(id='histogram', style={'width': '50%', 'display': 'inline-block'}),
            ]),
            html.Div(id='bivariate-output'),
            html.H2('Bivariate Analysis'),
            html.Div([
                dcc.Dropdown(
                    id='bivariate-column1',
                    options=data.drop(columns=['Date']).columns.sort_values(),
                    placeholder='Select first column'
                ),
                dcc.Dropdown(
                    id='bivariate-column2',
                    options=data.drop(columns=['Date']).columns.sort_values(),
                    placeholder='Select second column'
                ),
                html.Div(id='bivariate')
            ]),
        ]),
        dcc.Tab(label='Results (without time considerations)', children=[
            html.Div([
                dcc.Markdown(results_md, style={'margin-right': '50px'}),
                dcc.Graph(figure=fig),
            ], style={'display': 'flex', 'margin-top': '20px', 'margin-bottom': '20px'}),
            dcc.Dropdown(
                id='selected-model',
                options=sorted(models),
                placeholder='Select model',
                value=['Gradient Boosting', 'Linear Regression'],
                multi=True,
                style={'margin-bottom': '20px'}
            ),
            html.Div([
                html.Div(id='testVSpred'),
                html.Div(id='time-series-prediction'),
                html.Div([
                    dcc.Markdown(tab_residuals_md, style= {'margin-right': '50px'}),
                    html.Div(id='residuals')
                ], style={'display': 'flex'})
            ]),
        ]),
    ])
], style={'margin-left': '20px', 'margin-right': '20px'})

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
    filtered_data = data[(pd.to_datetime(data['Date']) >= start_date) & (pd.to_datetime(data['Date']) <= end_date)]
    
    time_series = [dcc.Graph(figure=get_ts(col, filtered_data)) for col in selected_columns]
    boxplot = [dcc.Graph(figure=get_boxplot(col, filtered_data)) for col in selected_columns]
    histogram = [dcc.Graph(figure=get_histogram(col, filtered_data)) for col in selected_columns]
    bivariate = [dcc.Graph(figure=get_scatterplot_output(col, filtered_data)) for col in selected_columns]

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
    filtered_data = data[(pd.to_datetime(data['Date']) >= start_date) & (pd.to_datetime(data['Date']) <= end_date)]

    col1 = bivariate_column1
    col2 = bivariate_column2
    if col1 is None or col2 is None:
        return []
    scatterplot = [dcc.Graph(figure=get_scatterplot(col1, col2, filtered_data))]
    return scatterplot

@app.callback(
    [Output('testVSpred', 'children'),
     Output('time-series-prediction', 'children'),
     Output('residuals', 'children')],
    Input('selected-model', 'value'),
)

def update_results(selected_models):
    if selected_models is None:
        return [], [], []
    plot_testVS_pred = go.Figure()
    for selected_model in selected_models:
        plot_testVS_pred.add_trace(go.Scatter(x=y_test, y=models[selected_model]['predictions'], mode='markers', name=selected_model))
    plot_testVS_pred.update_layout(
        title='Actual vs Predicted',
        xaxis_title='Actual',
        yaxis_title='Predicted',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    plot_testVS_pred.add_trace(go.Scatter(x=[0, 10], y=[0, 10], mode='lines', line=dict(color='black', width=2), name='y=x'))

    plot_ts_predictions = go.Figure()
    plot_ts_predictions.add_trace(go.Scatter
        (x=X_test['Date'], y=y_test, mode='markers+lines', name='actual', marker=dict(color='black')))
    for selected_model in selected_models:
        plot_ts_predictions.add_trace(go.Scatter
            (x=X_test['Date'], y=models[selected_model]['predictions'], mode='markers+lines', name=selected_model))
    plot_ts_predictions.update_layout(
        title='Wind speed prediction over time',
        xaxis_title='Date',
        yaxis_title='Wind speed',
        margin=dict(l=20, r=20, t=40, b=20)
    )

    plot_residuals = go.Figure()
    for selected_model in selected_models:
        residuals = y_test - models[selected_model]['predictions']
        plot_residuals.add_trace(go.Histogram(x=residuals, name=selected_model, histnorm='probability', opacity=0.3))
    plot_residuals.update_layout(
        xaxis_title='Residual',
        yaxis_title='Probability',
        margin=dict(l=20, r=20, t=40, b=20),
        barmode='overlay',
        height=300,
        width=900,
    )
    plot_residuals.update_xaxes(range=[-2, 2])
    return dcc.Graph(figure=plot_testVS_pred), dcc.Graph(figure=plot_ts_predictions), dcc.Graph(figure=plot_residuals)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=9000)