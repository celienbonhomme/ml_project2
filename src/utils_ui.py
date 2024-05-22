import plotly.graph_objects as go

def get_ts(col, data):
    plot = go.Figure()
    plot.add_trace(go.Scatter(x=data['Date'], y=data[col], mode='lines', name=col))
    plot.update_layout(
        title=f'{col} over time',
        xaxis_title='Date', 
        yaxis_title=col,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return plot

def get_boxplot(col,  data):
    plot = go.Figure()
    plot.add_trace(go.Box(y=data[col], name=col))
    plot.update_layout(
        # title=f'{col} boxplot',
        # yaxis_title=col,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return plot

def get_histogram(col, data):
    plot = go.Figure()
    plot.add_trace(go.Histogram(x=data[col], name=col))
    plot.update_layout(
        # title=f'{col} histogram',
        xaxis_title=col,
        yaxis_title='Frequency',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return plot

def get_scatterplot_output(col, data):
    plot = go.Figure()
    plot.add_trace(go.Scatter(x=data['Wind speed (m/s)'], y=data[col], mode='markers', name=f'{col} VS Wind speed (m/s)'))
    plot.update_layout(
        title=f'{col} VS Wind speed (m/s)',
        xaxis_title='Wind speed',
        yaxis_title='Wind gust max',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return plot


def get_scatterplot(col_x, col_y, data):
    plot = go.Figure()
    plot.add_trace(go.Scatter(x=data[col_x], y=data[col_y], mode='markers', name=f'{col_x} vs {col_y}'))
    plot.update_layout(
        title=f'{col_x} VS {col_y}',
        xaxis_title=col_x,
        yaxis_title=col_y,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return plot
