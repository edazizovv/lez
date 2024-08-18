
import plotly.graph_objects as go
import numpy as np

t = np.linspace(-1, 1.2, 2000)
xx = (t**3) + (0.3 * np.random.randn(2000))
y0 = (t**6) + (0.3 * np.random.randn(2000))
y1 = (t**6) + (-0.3 * np.random.randn(2000))

x = np.concatenate((xx, xx))
y = np.concatenate((y0, y1))

colors = np.random.choice([0, 1], size=x.shape[0])
sizes = np.random.normal(loc=2, size=x.shape[0])
sizes[sizes<=0] = 1

fig = go.Figure()
fig.add_trace(go.Histogram2dContour(
    x=xx,
    y=y0,
    # colorscale='Blues',
    # reversescale=True,
    xaxis='x',
    yaxis='y',
    # contours=dict(coloring='lines')
    line=dict(color='green'),
    contours=dict(coloring='none'),
    # colorscale=None,
))
fig.add_trace(go.Histogram2dContour(
    x=xx,
    y=y1,
    # colorscale='Blues',
    # reversescale=True,
    xaxis='x',
    yaxis='y',
    # contours=dict(coloring='lines')
    line=dict(color='red'),
    contours=dict(coloring='none'),
    # colorscale=None,
))
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    xaxis='x',
    yaxis='y',
    mode='markers',
    marker=dict(
        color=colors,
        size=sizes
    )
    # marker=dict(
    #     color='rgba(0,0,0,0.3)',
    #     size=3
    # )
))


fig.show()
