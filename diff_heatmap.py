import sys
import os
import pandas as pd
import numpy as np
import dash
from dash import State, dcc, html, Input, Output, ctx
import plotly.graph_objs as go
import plotly.express as px
import socket
import webbrowser

# -------------------------
# DASH UTILITY
# -------------------------
def find_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

# -------------------------
# METRICS CONFIG (copy from your script)
# -------------------------

METRICS_CONFIG = {
    'Base Accuracy': {
        'column': 'Base_Accuracy',
        'colorscale': [
            [0.0, '#0000ff'],   # df2 < df1 (negative)
            [0.5, '#000000'],   # 0 difference → neutral
            [1.0, '#ff0000'],   # df2 > df1 (positive)
        ],
        'zmin': -1,  # adjust for diffs
        'zmax': 1
    },
    'RFWR Rate': {
        'column': 'RFWR_Rate',
        'colorscale': [
            [0.0, '#0000ff'],   # df2 < df1 (negative)
            [0.5, '#000000'],   # 0 difference → neutral
            [1.0, '#ff0000'],   # df2 > df1 (positive)
        ],
        'zmin': -1,
        'zmax': 1
    },
    'Adjusted Accuracy': {
        'column': 'Adjusted_Accuracy',
        'colorscale': [
            [0.0, '#0000ff'],   # df2 < df1 (negative)
            [0.5, '#000000'],   # 0 difference → neutral
            [1.0, '#ff0000'],   # df2 > df1 (positive)
        ],
        'zmin': -1,
        'zmax': 1
    },
    'Accuracy Drop': {
        'column': 'Accuracy_Drop',
        'colorscale': [
            [0.0, '#0000ff'],   # df2 < df1 (negative)
            [0.5, '#000000'],   # 0 difference → neutral
            [1.0, '#ff0000'],   # df2 > df1 (positive)
        ],
        'zmin': -1,
        'zmax': 1
    },
}

# -------------------------
# LOAD AND CLEAN CSV
# -------------------------
def load_csv(base_path):
    df = pd.read_csv(f"{base_path}/statistics.csv", sep=r'\s+|,', engine='python')
    df.rename(columns=lambda x: x.strip("_"), inplace=True)
    df["Depth"] = df["Depth"].astype(int)
    df["Context"] = df["Context"].astype(int)
    df["Comments"] = df["Comments"].astype(int)
    return df

def get_experiment_name(path):
    # Normalize path separators
    path = os.path.normpath(path)
    parts = path.split(os.sep)
    # We want the folder just after "exp_out"
    if "exp_out" in parts:
        idx = parts.index("exp_out")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # fallback: last directory
    return os.path.basename(os.path.dirname(path))

# -------------------------
# COMPUTE DIFF
# -------------------------
def compute_diff(df1, df2):
    join_keys = ["Depth", "Context", "Comments"] #, "Model"]
    merged = pd.merge(df1, df2, on=join_keys, suffixes=("_run1", "_run2"))

    for metric_name, config in METRICS_CONFIG.items():
        col = config["column"]
        if f"{col}_run1" in merged and f"{col}_run2" in merged:
            merged[col] = merged[f"{col}_run2"] - merged[f"{col}_run1"]

    # Keep only original column names
    merged = merged[join_keys + [config["column"] for config in METRICS_CONFIG.values() if config["column"] in merged.columns]]
    return merged


# -------------------------
# GROUPED DEPTHS UTIL
# -------------------------
def get_grouped_depths(depths):
    return sorted(list(set(abs(d) for d in depths)))

# -------------------------
# CREATE METRIC ARRAYS
# -------------------------
def create_metric_arrays(df, x_labels, y_labels, z_labels, grouped_x_labels, metric_column):
    metric_data = np.full((len(z_labels), len(y_labels), len(x_labels)), np.nan)
    
    for _, row in df.iterrows():
        if metric_column in row and not pd.isna(row[metric_column]):
            z_idx = z_labels.index(row["Comments"])
            y_idx = y_labels.index(row["Context"])
            x_idx = x_labels.index(row["Depth"])
            metric_data[z_idx, y_idx, x_idx] = row[metric_column]
    
    grouped_metric_data = np.full((len(z_labels), len(y_labels), len(grouped_x_labels)), np.nan)
    for z_idx in range(len(z_labels)):
        for y_idx in range(len(y_labels)):
            for grouped_x_idx, abs_depth in enumerate(grouped_x_labels):
                matching_depths = [d for d in x_labels if abs(d) == abs_depth]
                values = []
                for depth in matching_depths:
                    orig_x_idx = x_labels.index(depth)
                    val = metric_data[z_idx, y_idx, orig_x_idx]
                    if not np.isnan(val):
                        values.append(val)
                if values:
                    grouped_metric_data[z_idx, y_idx, grouped_x_idx] = np.mean(values)
    return metric_data, grouped_metric_data

# -------------------------
# DASH APP
# -------------------------
def run_dash(df, xp1_name, xp2_name):
    x_labels = sorted(df["Depth"].unique())
    y_labels = sorted(df["Context"].unique())
    z_labels = sorted(df["Comments"].unique(), key=int)
    grouped_x_labels = get_grouped_depths(x_labels)

    METRIC_DATA = {}
    for metric_name, config in METRICS_CONFIG.items():
        if config['column'] in df.columns:
            original, grouped = create_metric_arrays(
                df, x_labels, y_labels, z_labels, grouped_x_labels, config['column']
            )
            METRIC_DATA[metric_name] = {'original': original, 'grouped': grouped}

    grouped_x_labels_str = [str(x) for x in grouped_x_labels]
    z_labels_str = [str(z) for z in z_labels]

    # Minimal Dash layout example
    app = dash.Dash(__name__)
    app.layout = html.Div([

        # Sidebar (controls)
        html.Div([
            html.H2("⚙️ Controls", style={'textAlign': 'center', 'marginBottom': '20px'}),

            # Metric selection
            html.Div([
                html.Label("Select Metric:", style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='metric-select',
                    options=[{'label': k.replace('_', ' ').title(), 'value': k} for k in METRIC_DATA.keys()],
                    value=list(METRIC_DATA.keys())[0] if METRIC_DATA else None,
                    labelStyle={'display': 'block', 'margin': '5px 0'}
                ),
            ], style={'marginBottom': '20px'}),

            # Color Sensitivity Slider
            html.Div([
                html.Label("Color Sensitivity:", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='color-sensitivity-slider',
                    min=0.01,           # minimum sensitivity
                    max=1.0,            # maximum sensitivity
                    step=0.01,          # small steps for fine control
                    value=0.45,          # default sensitivity
                    marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode='drag'
                ),
                html.Div(id='color-sensitivity-value', style={'marginTop': '5px', 'textAlign': 'center'}),

                # Auto-cycle toggle
                dcc.Checklist(
                    id='auto-sensitivity-toggle',
                    options=[{'label': '▶ Auto-cycle Sensitivity', 'value': 'auto'}],
                    value=[],
                    style={'marginTop': '10px'}
                ),
                # html.Button("▶ Play", id="auto-sensitivity-toggle", n_clicks=0, style={
                #     'marginTop': '10px',
                #     'padding': '5px 12px',
                #     'borderRadius': '6px',
                #     'border': 'none',
                #     'backgroundColor': '#3498db',
                #     'color': 'white',
                #     'cursor': 'pointer'
                # }),
                dcc.Interval(
                    id='color-sensitivity-interval',
                    interval=200,  # 200 ms updates
                    disabled=True
                ),
                dcc.Store(id='color-sensitivity-direction', data='forward')

            ], style={'marginBottom': '20px'}),

        ], style={
            'width': '15%',
            'padding': '20px',
            'borderRight': '1px solid #ddd',
            'backgroundColor': '#f9f9f9',
            'height': '100vh',
            'position': 'fixed',
            'left': 0,
            'top': 0,
            'overflowY': 'auto'
        }),


        # Main content
        html.Div([
            html.H1("Metric Differences Heatmap", style={
                'textAlign': 'center',
                'marginBottom': '30px',
                'fontWeight': 'bold',
                'color': '#2c3e50'
            }),

            dcc.Graph(id='heatmap')

        ], style={
            'marginLeft': '17%',
            'padding': '20px'
        }),

    ])

    # Toggle interval
    @app.callback(
        Output('color-sensitivity-interval', 'disabled'),
        Input('auto-sensitivity-toggle', 'value')
    )
    def toggle_interval(auto_values):
        # Enable interval if checkbox is ticked
        return not ('auto' in auto_values)


    # Auto cycle slider
    @app.callback(
        Output('color-sensitivity-slider', 'value'),
        Output('color-sensitivity-direction', 'data'),
        Input('color-sensitivity-interval', 'n_intervals'),
        State('color-sensitivity-slider', 'value'),
        State('color-sensitivity-direction', 'data')
    )
    def auto_cycle_slider(n, current_value, direction):
        step = 0.05

        if direction == 'forward':
            new_value = current_value + step
            if new_value >= 1.0:
                new_value = 1.0
                direction = 'backward'
        else:  # backward
            new_value = current_value - step
            if new_value <= 0.01:
                new_value = 0.01
                direction = 'forward'

        return new_value, direction


    # Merge heatmap + display update
    @app.callback(
        [Output('heatmap', 'figure'),
         Output('color-sensitivity-value', 'children')],
        [Input('metric-select', 'value'),
         Input('color-sensitivity-slider', 'value')]
    )
    def update_heatmap_and_display(metric_name, sensitivity):
        metric_array = METRIC_DATA[metric_name]['original']

        zmin = -sensitivity
        zmax = sensitivity

        if metric_array.shape[0] == 0:  # No comments levels
            fig = go.Figure()
            fig.update_layout(title=f"No data available for {metric_name}")
            return fig, f"Sensitivity: {sensitivity:.2f}"

        # Use first available comments level
        z = metric_array[0, :, :]

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=[str(x) for x in x_labels],
            y=[str(y) for y in y_labels],
            colorscale=METRICS_CONFIG[metric_name]['colorscale'],
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=f"Difference in {metric_name}")
        ))
        fig.update_layout(
            title=f"Difference in {metric_name} ({exp2_name} - {exp1_name}) "
                  f"(Comments={z_labels_str[0]})"
        )

        return fig, f"Sensitivity: {sensitivity:.2f}"



    port = find_free_port()
    app.run(debug=True, port=port, use_reloader=False)
    webbrowser.open_new(f"http://127.0.0.1:{port}/")

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python diff_viewer.py run1/statistics.csv run2/statistics.csv")
        sys.exit(1)

    path1, path2 = sys.argv[1], sys.argv[2]
    if not os.path.exists(path1) or not os.path.exists(path2):
        raise FileNotFoundError("One or both CSV files do not exist")
    exp1_name = get_experiment_name(path1)
    exp2_name = get_experiment_name(path2)

    df1 = load_csv(path1)
    df2 = load_csv(path2)
    df_diff = compute_diff(df1, df2)

    run_dash(df_diff, exp1_name, exp2_name)