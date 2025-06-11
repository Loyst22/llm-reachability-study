import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go

# === Simulated Data ===
x_labels = [f'X{i}' for i in range(10)]
y_labels = [f'Y{i}' for i in range(8)]
z_labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
accuracy_data = np.random.rand(len(z_labels), len(y_labels), len(x_labels))  # shape: (Z, Y, X)

# Sort Z labels
def sorted_z_labels(z):
    try:
        return sorted(z, key=lambda v: float(v.strip('Z')))
    except:
        return sorted(z)

z_labels = sorted_z_labels(z_labels)

# === Dash App ===
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("3D Accuracy Heatmap Viewer with Multi-Point Z Trends"),

    html.Div([
        html.Label("Z value:"),
        dcc.Slider(
            id='z-slider',
            min=0,
            max=len(z_labels) - 1,
            step=1,
            value=0,
            marks={i: z_labels[i] for i in range(len(z_labels))},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Button("Play", id="play-button", n_clicks=0),
        dcc.Interval(id='interval', interval=1000, disabled=True),
    ], style={'width': '80%', 'margin': '20px auto'}),

    dcc.Graph(id='heatmap', clear_on_unhover=True),

    html.Div([
        html.Div(id='selected-coord-text', style={'margin': '10px 0', 'fontWeight': 'bold'}),
        html.Button("Clear Selection", id="clear-selection", n_clicks=0)
    ], style={'margin': '10px'}),

    dcc.Graph(id='lineplot'),

    # Store for clicked coordinates
    dcc.Store(id='clicked-points', data=[])
])

# === Update Heatmap ===
@app.callback(
    Output('heatmap', 'figure'),
    Input('z-slider', 'value')
)
def update_heatmap(z_index):
    z_label = z_labels[z_index]
    z_slice = accuracy_data[z_index]

    fig = go.Figure(data=go.Heatmap(
        z=z_slice,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        colorbar=dict(title='Accuracy')
    ))
    fig.update_layout(title=f"Accuracy Heatmap at Z = {z_label}")
    return fig

# === Play/Pause Logic ===
@app.callback(
    Output('interval', 'disabled'),
    Output('play-button', 'children'),
    Input('play-button', 'n_clicks'),
    State('interval', 'disabled')
)
def toggle_play(n_clicks, is_disabled):
    return (False, "Pause") if n_clicks % 2 == 1 else (True, "Play")

# === Animate Slider ===
@app.callback(
    Output('z-slider', 'value'),
    Input('interval', 'n_intervals'),
    State('z-slider', 'value')
)
def animate_slider(n, current_val):
    return (current_val + 1) % len(z_labels)

# === Track Clicked Points ===
@app.callback(
    Output('clicked-points', 'data'),
    Input('heatmap', 'clickData'),
    Input('clear-selection', 'n_clicks'),
    State('clicked-points', 'data'),
    prevent_initial_call=True
)
def update_clicked_points(click_data, clear_clicks, stored_points):
    trigger = ctx.triggered_id

    if trigger == "clear-selection":
        return []

    if not click_data:
        return stored_points

    x_val = click_data['points'][0]['x']
    y_val = click_data['points'][0]['y']
    new_point = {'x': x_val, 'y': y_val}
    
    if new_point not in stored_points:
        stored_points.append(new_point)
    return stored_points

# === Plot All Selected Coordinates ===
@app.callback(
    Output('lineplot', 'figure'),
    Output('selected-coord-text', 'children'),
    Input('clicked-points', 'data')
)
def plot_lines(clicked_points):
    if not clicked_points:
        return go.Figure(), "Click cells to compare accuracy over Z for multiple (X, Y) points."

    fig = go.Figure()
    for pt in clicked_points:
        x_val = pt['x']
        y_val = pt['y']
        x_idx = x_labels.index(x_val)
        y_idx = y_labels.index(y_val)
        acc_curve = [accuracy_data[z, y_idx, x_idx] for z in range(len(z_labels))]
        fig.add_trace(go.Scatter(
            x=z_labels,
            y=acc_curve,
            mode='lines+markers',
            name=f'({x_val}, {y_val})'
        ))
    fig.update_layout(title='Accuracy Across Z',
                      xaxis_title='Z',
                      yaxis_title='Accuracy')

    label_text = "Selected coordinates: " + ", ".join([f"({pt['x']}, {pt['y']})" for pt in clicked_points])
    return fig, label_text

# Run it
if __name__ == '__main__':
    app.run_server(debug=True)

