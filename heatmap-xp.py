import errno
import random
import socket
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go
import plotly.express as px
import sys
import os

base_path = sys.argv[1]
# Load data from CSV
df = pd.read_csv(f"{base_path}/statistics.csv", sep=r'\s+|,', engine='python')

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

xp_name = get_experiment_name(base_path)
    
def read_stats(input_dirs):
    # Read the first file normally to get the header
    first_path = input_dirs[0]
    df_first = pd.read_csv(f"{first_path}/statistics.csv", sep=r'\s+|,', engine='python')
    dataframes = [df_first]
    columns = df_first.columns

    for other_path in input_dirs[1:]:
        # Skip first row (header), and manually set the columns
        df = pd.read_csv(f"{other_path}/statistics.csv", skiprows=1, header=None, sep=r'\s+|,', engine='python')
        df.columns = columns
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# Global variable containing the stats
df = read_stats(sys.argv[1:])

# Rename columns to remove leading underscores
df = df.rename(columns=lambda x: x.strip("_"))

# Ensure correct types
df["Depth"] = df["Depth"].astype(int)
df["Context"] = df["Context"].astype(int)
df["Comments"] = df["Comments"].astype(int)

# Original axis labels
x_labels = sorted(df["Depth"].unique())
y_labels = sorted(df["Context"].unique())
z_labels = sorted(df["Comments"].unique(), key=int)

# Get unique models from the DataFrame
unique_models = sorted(df["Model"].unique())

# Create grouped depth labels (absolute values)
def get_grouped_depths(depths):
    """Get unique absolute depth values, sorted"""
    return sorted(list(set(abs(d) for d in depths)))

grouped_x_labels = get_grouped_depths(x_labels)

# Define metrics configuration - ADD YOUR METRICS HERE
METRICS_CONFIG = {
    'accuracy': {
        'column': 'Base_Accuracy',
        'colorscale': [
            [0.0, '#000000'],      # Gray for 0 (missing data)
            [0.3, '#990000'],      # Dark Red for 0.3 (worse than random performance - bad)
            [0.5, '#ff0000'],      # Red for 0.5 (random performance - bad)
            [0.7, '#ff6600'],      # Orange for 0.7
            [0.9, '#fde725'],      # Yellow for 0.9
            [1.0, '#00ff00'],      # Green for 1.0
        ],
        'colorbar': dict(
            title='Accuracy',
            tickvals=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
            ticktext=['Awful', 'Really bad', 'Random', 'Poor', 'Ok', 'Good']
        ),
        'zmin': 0,
        'zmax': 1
    },
    'rwr_rate': {
        'column': 'RFWR_Rate',  # Assuming you have this column
        'colorscale': [
            [0.0, '#0000ff'],      # Blue for 0.0 
            [0.1, '#fde725'],      # Yellow for 0.1
            [0.3, '#ff6600'],      # Orange for 0.3
            [0.5, '#ff0000'],      # Red for 0.5
            [0.7, '#990000'],      # Dark Red for 0.7
            [1.0, '#000000'],      # Black for 1.0
        ],
        'colorbar': dict(
            title='% RWR',
            tickvals=[0, 0.1, 0.3, 0.5, 0.7, 1.0],
            ticktext=['None', 'Some', 'Quite a few', 'Lots', 'Almost all', 'All']
        ),
        'zmin': 0,
        'zmax': 1
    },
    'adjusted accuracy': {
        'column': 'Adjusted_Accuracy',  # Add if you have this column
        'colorscale': [
            [0.0, '#000000'],      # Gray for 0 (missing data)
            [0.3, '#990000'],      # Dark Red for 0.3 (worse than random performance - bad)
            [0.5, '#ff0000'],      # Red for 0.5 (random performance - bad)
            [0.7, '#ff6600'],      # Orange for 0.7
            [0.9, '#fde725'],      # Yellow for 0.9
            [1.0, '#00ff00'],      # Green for 1.0
        ],
        'colorbar': dict(
             title='Accuracy',
            tickvals=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
            ticktext=['Awful', 'Really bad', 'Random', 'Poor', 'Ok', 'Good']
        ),
        'zmin': 0,
        'zmax': 1
    },
    'drop': {
        'column': 'Accuracy_Drop',  # Assuming you have this column
        'colorscale': [
            [0.0, '#0000ff'],      # Blue for 0.0 
            [0.1, '#fde725'],      # Yellow for 0.1
            [0.3, '#ff6600'],      # Orange for 0.3
            [0.5, '#ff0000'],      # Red for 0.5
            [0.7, '#990000'],      # Dark Red for 0.7
            [1.0, '#000000'],      # Black for 1.0
        ],
        'colorbar': dict(
            title='% RWR',
            tickvals=[0, 0.1, 0.3, 0.5, 0.7, 1.0],
            ticktext=['None', 'Some', 'Quite a few', 'Lots', 'Almost all', 'All']
        ),
        'zmin': 0,
        'zmax': 1
    },
}

# Create 3D accuracy arrays for both original and grouped data
def create_metric_arrays(metric_column):
    # Original metric array
    metric_data = np.full((len(z_labels), len(y_labels), len(x_labels)), np.nan)
    
    # Fill original metric values
    for _, row in df.iterrows():
        if metric_column in row and not pd.isna(row[metric_column]):
            z_idx = z_labels.index(row["Comments"])
            y_idx = y_labels.index(row["Context"])
            x_idx = x_labels.index(row["Depth"])
            metric_data[z_idx, y_idx, x_idx] = row[metric_column]
    
    # Grouped metric array
    grouped_metric_data = np.full((len(z_labels), len(y_labels), len(grouped_x_labels)), np.nan)
    
    # Fill grouped metric values by averaging positive/negative pairs
    for z_idx in range(len(z_labels)):
        for y_idx in range(len(y_labels)):
            for grouped_x_idx, abs_depth in enumerate(grouped_x_labels):
                # Find all depths with this absolute value
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

# Create data arrays for all metrics
METRIC_DATA = {}
for metric_name, config in METRICS_CONFIG.items():
    if config['column'] in df.columns:
        original, grouped = create_metric_arrays(config['column'])
        METRIC_DATA[metric_name] = {
            'original': original,
            'grouped': grouped
        }

# Convert labels to strings for display
z_labels = [str(z) for z in z_labels]
grouped_x_labels_str = [str(x) for x in grouped_x_labels]

# Dash App with Sidebar Layout
app = dash.Dash(__name__)
app.layout = html.Div([
    # Sidebar (controls)
    html.Div([
        html.H2("⚙️ Controls", style={'textAlign': 'center', 'marginBottom': '20px'}),

        # Metric selection
        html.Div([
            html.Label("Select Metric:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='metric-selector',
                options=[
                    {'label': metric_name.replace('_', ' ').title(), 'value': metric_name}
                    for metric_name in METRIC_DATA.keys()
                ],
                value=list(METRIC_DATA.keys())[0] if METRIC_DATA else 'accuracy',
                labelStyle={'display': 'block', 'margin': '5px 0'}
            ),
        ], style={'marginBottom': '20px'}),

        # Comments slider
        html.Div([
            html.Label("Comments (Z):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='z-slider',
                min=0,
                max=len(z_labels) - 1,
                step=1,
                value=-1,
                marks={i: str(z_labels[i]) for i in range(len(z_labels))},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Button("▶ Play", id="play-button", n_clicks=0, style={
                'marginTop': '10px',
                'padding': '5px 12px',
                'borderRadius': '6px',
                'border': 'none',
                'backgroundColor': '#3498db',
                'color': 'white',
                'cursor': 'pointer'
            }),
            dcc.Interval(id='interval', interval=1000, disabled=True),
        ], style={'marginBottom': '20px'}),

        # Model selection
        html.Div([
            html.Label("Select Model:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='model-selector',
                options=[{'label': m, 'value': m} for m in unique_models],
                value=unique_models[0],
                clearable=False
            ),
        ], style={'marginBottom': '20px'}),

        # Axis spacing toggles
        html.Div([
            dcc.Checklist(
                id='equal-spacing-toggle',
                options=[{'label': 'Equal spacing for axis ticks', 'value': 'equal'}],
                value=['equal']
            ),
            dcc.Checklist(
                id='group-depths-toggle',
                options=[{'label': 'Group positive/negative depths', 'value': 'group'}],
                value=[]
            ),
        ], style={'marginBottom': '20px'}),

        # Slice selections
        html.Div([
            html.Label("Slice by Depth (X):", style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='x-slice',
                options=[{'label': str(x), 'value': x} for x in sorted(set(x_labels), key=int)],
                value=[],
                labelStyle={'display': 'block'}
            ),
            html.Label("Slice by Context (Y):", style={'fontWeight': 'bold', 'marginTop': '15px'}),
            dcc.Checklist(
                id='y-slice',
                options=[{'label': str(y), 'value': y} for y in sorted(y_labels, key=int)],
                value=[],
                labelStyle={'display': 'block'}
            ),
        ]),

    ], style={
        'width': '20%',
        'padding': '20px',
        'borderRight': '1px solid #ddd',
        'backgroundColor': '#f9f9f9',
        'overflowY': 'auto',
        'height': '100vh',
        'position': 'fixed',
        'left': 0,
        'top': 0
    }),

    # Main content (graphs)
    html.Div([
        html.H1("3D Metric Heatmap Viewer", style={
            'textAlign': 'center',
            'marginBottom': '30px',
            'fontWeight': 'bold',
            'color': '#2c3e50'
        }),

        dcc.Graph(id='heatmap', style={'marginBottom': '30px'}),
        dcc.Graph(id='slice-heatmap')
    ], style={
        'marginLeft': '22%',
        'padding': '20px'
    }),

    dcc.Store(id='slice-info', data={})
])

@app.callback(
    Output('x-slice', 'options'),
    Input('group-depths-toggle', 'value')
)
def update_x_slice_options(group_depths):
    use_grouped = 'group' in group_depths
    if use_grouped:
        return [{'label': x, 'value': x} for x in grouped_x_labels_str]
    else:
        return [{'label': x, 'value': x} for x in x_labels]

@app.callback(
    Output('heatmap', 'figure'),
    Input('z-slider', 'value'),
    Input('slice-info', 'data'),
    Input('equal-spacing-toggle', 'value'),
    Input('group-depths-toggle', 'value'),
    Input('metric-selector', 'value'),
    Input('model-selector', 'value')
)
def update_heatmap(z_index, slice_info, equal_spacing, group_depths, selected_metric, selected_model):
    # Filter the DataFrame by selected model
    filtered_df = df[df["Model"] == selected_model]

    # Recompute axis labels for the filtered model
    x_labels = sorted(filtered_df["Depth"].unique())
    y_labels = sorted(filtered_df["Context"].unique())
    z_labels = sorted(filtered_df["Comments"].unique())
    grouped_x_labels = sorted(list(set(abs(d) for d in x_labels)))
    grouped_x_labels_str = [str(x) for x in grouped_x_labels]

    # Recompute metric arrays for the filtered model
    def create_metric_arrays(metric_column):
        metric_data = np.full((len(z_labels), len(y_labels), len(x_labels)), np.nan)
        for _, row in filtered_df.iterrows():
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

    # Get metric configuration
    metric_config = METRICS_CONFIG.get(selected_metric, METRICS_CONFIG['accuracy'])
    metric_data, grouped_metric_data = create_metric_arrays(metric_config['column'])

    # # Get metric data
    # if selected_metric not in METRIC_DATA:
    #     # Fallback to first available metric if selected one doesn't exist
    #     selected_metric = list(METRIC_DATA.keys())[0]

    # metric_data = METRIC_DATA[selected_metric]

    # Determine which data and labels to use
    use_grouped = 'group' in group_depths
    use_equal_spacing = 'equal' in equal_spacing
    if use_grouped:
        z_slice = grouped_metric_data[z_index]
        current_x_labels = grouped_x_labels_str
    else:
        z_slice = metric_data[z_index]
        current_x_labels = x_labels

    # Create coordinate axes
    x = list(range(len(current_x_labels))) if use_equal_spacing else [int(label) for label in current_x_labels]
    y = list(range(len(y_labels))) if use_equal_spacing else y_labels

    fig = go.Figure(data=go.Heatmap(
        z=z_slice,
        x=x,
        y=y,
        colorscale=metric_config['colorscale'],
        zmin=metric_config['zmin'],
        zmax=metric_config['zmax'],
        colorbar=metric_config['colorbar']
    ))

    title_suffix = " (Grouped ±Depths)" if use_grouped else ""
    metric_display_name = selected_metric.replace('_', ' ').title()
    fig.update_layout(
        title=f"{metric_display_name} Heatmap at Comments = {z_labels[z_index]}{title_suffix} | Model: {selected_model} | Exp: {xp_name}",
        xaxis_title='Depth (Absolute)' if use_grouped else 'Depth',
        yaxis_title='Context',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(current_x_labels))) if use_equal_spacing else [int(label) for label in current_x_labels],
            ticktext=current_x_labels
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(y_labels))) if use_equal_spacing else y_labels,
            ticktext=y_labels
        )
    )

    # Add slicing indicator
    if slice_info:
        if slice_info.get('type') == 'x':
            x_vals = slice_info['value']
            if not isinstance(x_vals, list):
                x_vals = [x_vals]  # ensure it's a list
            for x_val in x_vals:
                try:
                    if use_grouped:
                        x_pos = (
                            grouped_x_labels_str.index(str(x_val))
                            if use_equal_spacing else int(x_val)
                        )
                    else:
                        x_pos = (
                            x_labels.index(int(x_val))
                            if use_equal_spacing else int(x_val)
                        )

                    fig.add_shape(
                        type="line",
                        x0=x_pos, x1=x_pos,
                        y0=min(y), y1=max(y),
                        line=dict(color="red", width=2)
                    )
                except (ValueError, IndexError):
                    pass  # Skip if not found
        elif slice_info.get('type') == 'y':
            y_vals = slice_info['value']
            if not isinstance(y_vals, list):
                y_vals = [y_vals]  # ensure it's a list

            for y_val in y_vals:
                # Determine y position for the line
                y_pos = y_labels.index(y_val) if use_equal_spacing else y_val

                fig.add_shape(
                    type="line",
                    y0=y_pos, y1=y_pos,
                    x0=min(x), x1=max(x),
                    line=dict(color="red", width=2)
                )

    return fig

@app.callback(
    Output('interval', 'disabled'),
    Output('play-button', 'children'),
    Input('play-button', 'n_clicks'),
    State('interval', 'disabled')
)
def toggle_play(n_clicks, is_disabled):
    return (False, "Pause") if n_clicks % 2 == 1 else (True, "Play")

@app.callback(
    Output('z-slider', 'value'),
    Input('interval', 'n_intervals'),
    State('z-slider', 'value')
)
def animate_slider(n, current_val):
    return (current_val + 1) % len(z_labels)

@app.callback(
    Output('slice-info', 'data'),
    Input('x-slice', 'value'),
    Input('y-slice', 'value'),
    prevent_initial_call=True
)
def update_slice_info(x_slice_val, y_slice_val):
    trigger = ctx.triggered_id

    if trigger == 'x-slice' and x_slice_val:
        return {'type': 'x', 'value': x_slice_val}
    if trigger == 'y-slice' and y_slice_val:
        return {'type': 'y', 'value': y_slice_val}

    return {}

@app.callback(
    Output('slice-heatmap', 'figure'),
    Input('z-slider', 'value'),
    Input('slice-info', 'data'),
    Input('equal-spacing-toggle', 'value'),
    Input('group-depths-toggle', 'value'),
    Input('metric-selector', 'value'),
    Input('model-selector', 'value')
)
def update_slice_heatmap(z_index, slice_info, equal_spacing, group_depths, selected_metric, selected_model):
    if not slice_info:
        return go.Figure()

    # Filter by model
    filtered_df = df[df["Model"] == selected_model]

    # Recompute axis labels (same as update_heatmap)
    x_labels = sorted(filtered_df["Depth"].unique())
    y_labels = sorted(filtered_df["Context"].unique())
    z_labels = sorted(filtered_df["Comments"].unique())
    grouped_x_labels = sorted(list(set(abs(d) for d in x_labels)))
    grouped_x_labels_str = [str(x) for x in grouped_x_labels]


    # Rebuild metric arrays (same helper as update_heatmap)
    def create_metric_arrays(metric_column):
        metric_data = np.full((len(z_labels), len(y_labels), len(x_labels)), np.nan)
        for _, row in filtered_df.iterrows():
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

    # Metric config
    metric_config = METRICS_CONFIG.get(selected_metric, METRICS_CONFIG['accuracy'])
    metric_data, grouped_metric_data = create_metric_arrays(metric_config['column'])

    # Pick grouped or original view for depths finding
    use_grouped = 'group' in group_depths
    use_equal_spacing = 'equal' in equal_spacing
    
    if use_grouped:
        z_slice = grouped_metric_data[z_index]
        current_x_labels = grouped_x_labels_str
    else:
        z_slice = metric_data[z_index]
        current_x_labels = x_labels

    metric_display_name = selected_metric.replace('_', ' ').title()
    
    # --- Slice by Depth (x) ---
    if slice_info['type'] == 'x':
        selected_depths = slice_info['value']
        selected_depths.sort()     
        # Create coordinate axes
        x = list(range(len(selected_depths)))
        y = list(range(len(y_labels))) if use_equal_spacing else y_labels
        
        if use_grouped:
            unique_depths = sorted(set(abs(int(d)) for d in selected_depths))
            # grouped labels are strings, so cast to str
            x_indices = [current_x_labels.index(str(d)) for d in unique_depths]
        else:
            # original labels are ints, so cast to int
            x_indices = [current_x_labels.index(int(d)) for d in selected_depths]
        
        """ Debugging """
        # print()
        # print("y_labels :", y_labels)
        # print("current_x_labels :", current_x_labels)
        # print("selected depths :", selected_depths)
        # print("x indices :", x_indices)
        
        # Slice for the selected comment amount only
        # z_slice has shape (num_contexts, num_depths)
        z_slice_for_heatmap = z_slice[:, x_indices] # shape: (num_contexts, len(selected_depths))
        fig = go.Figure(data=go.Heatmap(
            z=z_slice_for_heatmap,
            x=x,
            y=y,
            colorscale=metric_config['colorscale'],
            zmin=metric_config['zmin'],
            zmax=metric_config['zmax'],
            colorbar=metric_config['colorbar']
        ))
        fig.update_layout(
            title=f'Comments={z_labels[z_index]} vs Context at Depth={selected_depths} ({metric_display_name})',
            xaxis_title='Depth (Absolute)' if use_grouped else 'Depth',
            yaxis_title='Context',
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(selected_depths))),
                ticktext=selected_depths if not use_grouped else unique_depths
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(y_labels))) if use_equal_spacing else y_labels,
                ticktext=y_labels
            )
        )
        return fig

    # --- Slice by Context (y) ---
    elif slice_info['type'] == 'y':
        selected_contexts = slice_info['value']  # list
        selected_contexts.sort()
        # Create coordinate axes
        x = list(range(len(current_x_labels)))
        y = list(range(len(selected_contexts))) if use_equal_spacing else selected_contexts
        
        y_indices = [y_labels.index(c) for c in selected_contexts]
        # Slice for the selected comment amount only
        z_slice_for_heatmap = z_slice[y_indices, :]  # select multiple contexts

        """ Debugging """
        # print()
        # print("y_labels :", y_labels)
        # print("current_x_labels :", current_x_labels)
        # print("selected contexts :", selected_contexts)
        # print("y indices :", y_indices)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_slice_for_heatmap,
            x=x,
            y=y,
            colorscale=metric_config['colorscale'],
            zmin=metric_config['zmin'],
            zmax=metric_config['zmax'],
            colorbar=metric_config['colorbar']
        ))
        fig.update_layout(
            title=f'Comments={z_labels[z_index]} vs Depth at Context={selected_contexts} ({metric_display_name})',
            xaxis_title='Depth (Absolute)' if use_grouped else 'Depth',
            yaxis_title='Context',
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(current_x_labels))),
                ticktext=current_x_labels
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(y_labels))) if use_equal_spacing else selected_contexts,
                ticktext=selected_contexts
            )
        )
        return fig

    return go.Figure()

def find_free_port(start_port=8050, max_port=8100):
    ports = list(range(start_port, max_port))
    random.shuffle(ports)
    for port in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                continue
            else:
                raise
    raise RuntimeError("No free port found in range.")

if __name__ == '__main__':
    # --- Load CSV files ---
        
    # --- Start Dash app ---
    port = find_free_port()
    app.run(debug=True, port=port)
    import webbrowser
    url = f'http://127.0.0.1:{port}/'
    webbrowser.open_new(url)