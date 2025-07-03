import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go
import plotly.express as px
import sys

base_path = sys.argv[1]
# Load data from CSV
df = pd.read_csv(f"{base_path}/statistics.csv", sep=r'\s+|,', engine='python')

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

df = read_stats(sys.argv[1:])

# Rename columns to remove leading underscores
df = df.rename(columns=lambda x: x.strip("_"))

# Ensure correct types
df["Depth"] = df["Depth"].astype(int)
df["Context"] = df["Context"].astype(int)
df["Comments"] = df["Comments"].astype(int)

# Check if N_tokens column exists, if not create it or use a placeholder
if "N_tokens" not in df.columns:
    print("Warning: N_tokens column not found. Creating placeholder values.")
    df["N_tokens"] = df["Context"] * (20 + df["Comments"] * 40)  # Placeholder: assume N_tokens correlates with Context

# Original axis labels
x_labels = sorted(df["Depth"].unique())
y_labels = sorted(df["Context"].unique())
z_labels = sorted(df["Comments"].unique())
n_tokens_labels = sorted(df["N_tokens"].unique())

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
    # Original metric array (3D: Comments x Context x Depth)
    metric_data_3d = np.full((len(z_labels), len(y_labels), len(x_labels)), np.nan)
    
    # Fill original metric values
    for _, row in df.iterrows():
        if metric_column in row and not pd.isna(row[metric_column]):
            z_idx = z_labels.index(row["Comments"])
            y_idx = y_labels.index(row["Context"])
            x_idx = x_labels.index(row["Depth"])
            metric_data_3d[z_idx, y_idx, x_idx] = row[metric_column]
    
    # Grouped metric array (3D: Comments x Context x |Depth|)
    grouped_metric_data_3d = np.full((len(z_labels), len(y_labels), len(grouped_x_labels)), np.nan)
    
    # Fill grouped metric values by averaging positive/negative pairs
    for z_idx in range(len(z_labels)):
        for y_idx in range(len(y_labels)):
            for grouped_x_idx, abs_depth in enumerate(grouped_x_labels):
                # Find all depths with this absolute value
                matching_depths = [d for d in x_labels if abs(d) == abs_depth]
                values = []
                
                for depth in matching_depths:
                    orig_x_idx = x_labels.index(depth)
                    val = metric_data_3d[z_idx, y_idx, orig_x_idx]
                    if not np.isnan(val):
                        values.append(val)
                
                if values:
                    grouped_metric_data_3d[z_idx, y_idx, grouped_x_idx] = np.mean(values)
    
    # Create 2D arrays (N_tokens x Depth) by averaging over Comments and Context
    metric_data_2d = np.full((len(n_tokens_labels), len(x_labels)), np.nan)
    grouped_metric_data_2d = np.full((len(n_tokens_labels), len(grouped_x_labels)), np.nan)
    
    # Fill 2D metric values
    for _, row in df.iterrows():
        if metric_column in row and not pd.isna(row[metric_column]):
            n_tokens_idx = n_tokens_labels.index(row["N_tokens"])
            x_idx = x_labels.index(row["Depth"])
            
            # Store individual values first, then average later if needed
            if np.isnan(metric_data_2d[n_tokens_idx, x_idx]):
                metric_data_2d[n_tokens_idx, x_idx] = row[metric_column]
            else:
                # Average with existing value
                metric_data_2d[n_tokens_idx, x_idx] = (metric_data_2d[n_tokens_idx, x_idx] + row[metric_column]) / 2
    
    # Fill grouped 2D metric values
    for n_tokens_idx in range(len(n_tokens_labels)):
        for grouped_x_idx, abs_depth in enumerate(grouped_x_labels):
            # Find all depths with this absolute value
            matching_depths = [d for d in x_labels if abs(d) == abs_depth]
            values = []
            
            for depth in matching_depths:
                orig_x_idx = x_labels.index(depth)
                val = metric_data_2d[n_tokens_idx, orig_x_idx]
                if not np.isnan(val):
                    values.append(val)
            
            if values:
                grouped_metric_data_2d[n_tokens_idx, grouped_x_idx] = np.mean(values)
    
    return {
        '3d_original': metric_data_3d,
        '3d_grouped': grouped_metric_data_3d,
        '2d_original': metric_data_2d,
        '2d_grouped': grouped_metric_data_2d
    }

# Create data arrays for all metrics
METRIC_DATA = {}
for metric_name, config in METRICS_CONFIG.items():
    if config['column'] in df.columns:
        METRIC_DATA[metric_name] = create_metric_arrays(config['column'])

# Convert labels to strings for display
z_labels = [str(z) for z in z_labels]
grouped_x_labels_str = [str(x) for x in grouped_x_labels]
n_tokens_labels_str = [str(x) for x in n_tokens_labels]

# Dash App
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("3D/2D Metric Heatmap Viewer with Z Trends & Slicing"),

    # Metric selection radio buttons
    html.Div([
        html.Label("Select Metric:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
        dcc.RadioItems(
            id='metric-selector',
            options=[
                {'label': metric_name.replace('_', ' ').title(), 'value': metric_name}
                for metric_name in METRIC_DATA.keys()
            ],
            value=list(METRIC_DATA.keys())[0] if METRIC_DATA else 'accuracy',
            labelStyle={'display': 'inline-block', 'marginRight': '20px'},
            style={'marginBottom': '20px'}
        ),
    ], style={'width': '80%', 'margin': '20px auto', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),

    # View mode toggle
    html.Div([
        dcc.Checklist(
            id='view-mode-toggle',
            options=[{'label': '2D View (Depth vs N_tokens only)', 'value': '2d'}],
            value=[],  # Default to 3D view
            style={'marginBottom': '20px', 'fontWeight': 'bold'}
        ),
    ], style={'width': '80%', 'margin': 'auto'}),

    # Z-slider (only visible in 3D mode)
    html.Div(id='z-slider-container', children=[
        html.Label("Z value (Comments):"),
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

    html.Div([
        dcc.Checklist(
            id='equal-spacing-toggle',
            options=[{'label': 'Equal spacing for axis ticks', 'value': 'equal'}],
            value=['equal'],  # Default to equal spacing
            style={'marginBottom': '20px'}
        ),
        dcc.Checklist(
            id='group-depths-toggle',
            options=[{'label': 'Group positive/negative depths (average values)', 'value': 'group'}],
            value=[],  # Default to not grouped
            style={'marginBottom': '20px'}
        ),
    ], style={'width': '80%', 'margin': 'auto'}),

    # Slicing controls (only visible in 3D mode)
    html.Div(id='slicing-controls', children=[
        html.Label("Slice by Depth (X):"),
        dcc.Dropdown(id='x-slice', placeholder="Select Depth"),

        html.Label("Slice by Context (Y):"),
        dcc.Dropdown(id='y-slice', options=[{'label': y, 'value': y} for y in y_labels], placeholder="Select Context"),
    ], style={'width': '50%', 'margin': '0 auto'}),

    dcc.Graph(id='heatmap'),

    dcc.Graph(id='slice-heatmap'),

    dcc.Store(id='slice-info', data={})
])

@app.callback(
    [Output('z-slider-container', 'style'),
     Output('slicing-controls', 'style')],
    Input('view-mode-toggle', 'value')
)
def toggle_controls_visibility(view_mode):
    is_2d = '2d' in view_mode
    hidden_style = {'display': 'none'}
    visible_style_slider = {'width': '80%', 'margin': '20px auto'}
    visible_style_slice = {'width': '50%', 'margin': '0 auto'}
    
    return (hidden_style if is_2d else visible_style_slider,
            hidden_style if is_2d else visible_style_slice)

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
    [Input('z-slider', 'value'),
     Input('slice-info', 'data'),
     Input('equal-spacing-toggle', 'value'),
     Input('group-depths-toggle', 'value'),
     Input('metric-selector', 'value'),
     Input('view-mode-toggle', 'value')]
)
def update_heatmap(z_index, slice_info, equal_spacing, group_depths, selected_metric, view_mode):
    # Get metric configuration
    metric_config = METRICS_CONFIG.get(selected_metric, METRICS_CONFIG['accuracy'])
    
    # Get metric data
    if selected_metric not in METRIC_DATA:
        selected_metric = list(METRIC_DATA.keys())[0]
    
    metric_data = METRIC_DATA[selected_metric]
    
    # Determine view mode
    is_2d = '2d' in view_mode
    use_grouped = 'group' in group_depths
    use_equal_spacing = 'equal' in equal_spacing
    
    if is_2d:
        # 2D View: N_tokens vs Depth
        if use_grouped:
            z_slice = metric_data['2d_grouped']
            current_x_labels = grouped_x_labels_str
            x_title = 'Depth (Absolute)'
        else:
            z_slice = metric_data['2d_original']
            current_x_labels = x_labels
            x_title = 'Depth'
        
        # Create coordinate axes
        x = list(range(len(current_x_labels))) if use_equal_spacing else [int(label) for label in current_x_labels]
        y = list(range(len(n_tokens_labels_str))) if use_equal_spacing else [int(label) for label in n_tokens_labels_str]

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
            title=f"2D {metric_display_name} Heatmap: N_tokens vs Depth{title_suffix}",
            xaxis_title=x_title,
            yaxis_title='N_tokens',
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(current_x_labels))) if use_equal_spacing else [int(label) for label in current_x_labels],
                ticktext=current_x_labels
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(n_tokens_labels_str))) if use_equal_spacing else [int(label) for label in n_tokens_labels_str],
                ticktext=n_tokens_labels_str
            )
        )
        
    else:
        # 3D View: Original behavior
        z_label = z_labels[z_index]
        
        if use_grouped:
            z_slice = metric_data['3d_grouped'][z_index]
            current_x_labels = grouped_x_labels_str
        else:
            z_slice = metric_data['3d_original'][z_index]
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
            title=f"{metric_display_name} Heatmap at Comments = {z_label}{title_suffix}",
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

        # Add slicing indicator (only for 3D mode)
        if slice_info:
            if slice_info.get('type') == 'x':
                x_val = slice_info['value']
                try:
                    if use_grouped:
                        x_pos = grouped_x_labels_str.index(str(x_val)) if use_equal_spacing else int(x_val)
                    else:
                        x_pos = x_labels.index(int(x_val)) if use_equal_spacing else int(x_val)
                    fig.add_shape(type="line", x0=x_pos, x1=x_pos,
                                  y0=min(y), y1=max(y),
                                  line=dict(color="red", width=2))
                except (ValueError, IndexError):
                    pass  # Skip if value not found
            elif slice_info.get('type') == 'y':
                y_val = slice_info['value']
                y_pos = y_labels.index(y_val) if use_equal_spacing else y_val
                fig.add_shape(type="line", y0=y_pos, y1=y_pos,
                              x0=min(x), x1=max(x),
                              line=dict(color="red", width=2))

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
    [Input('slice-info', 'data'),
     Input('group-depths-toggle', 'value'),
     Input('metric-selector', 'value'),
     Input('view-mode-toggle', 'value')]
)
def update_slice_heatmap(slice_info, group_depths, selected_metric, view_mode):
    # Hide slice heatmap in 2D mode
    if '2d' in view_mode:
        return go.Figure()
        
    if not slice_info:
        return go.Figure()

    # Get metric configuration
    metric_config = METRICS_CONFIG.get(selected_metric, METRICS_CONFIG['accuracy'])
    
    # Get metric data
    if selected_metric not in METRIC_DATA:
        selected_metric = list(METRIC_DATA.keys())[0]
    
    metric_data = METRIC_DATA[selected_metric]

    use_grouped = 'group' in group_depths
    current_metric_data = metric_data['3d_grouped'] if use_grouped else metric_data['3d_original']
    current_x_labels = grouped_x_labels_str if use_grouped else x_labels

    if slice_info['type'] == 'x':
        try:
            x_idx = current_x_labels.index(str(slice_info['value']) if use_grouped else slice_info['value'])
        except ValueError:
            return go.Figure()
        
        slice_data = current_metric_data[:, :, x_idx]  # shape (Z, Y)
        
        fig = go.Figure(data=go.Heatmap(
            z=slice_data,
            x=y_labels,
            y=z_labels,
            colorscale=metric_config['colorscale'],
            zmin=metric_config['zmin'],
            zmax=metric_config['zmax'],
            colorbar=metric_config['colorbar']
        ))
        depth_label = f'|Depth| = {slice_info["value"]}' if use_grouped else f'Depth = {slice_info["value"]}'
        metric_display_name = selected_metric.replace('_', ' ').title()
        fig.update_layout(
            title=f'Comments vs Context Slice at {depth_label} ({metric_display_name})',
            xaxis_title='Context', yaxis_title='Comments'
        )
        return fig

    elif slice_info['type'] == 'y':
        y_idx = y_labels.index(slice_info['value'])
        slice_data = current_metric_data[:, y_idx, :]  # shape (Z, X)
        
        fig = go.Figure(data=go.Heatmap(
            z=slice_data,
            x=current_x_labels,
            y=z_labels,
            colorscale=metric_config['colorscale'],
            zmin=metric_config['zmin'],
            zmax=metric_config['zmax'],
            colorbar=metric_config['colorbar']
        ))
        x_title = 'Depth (Absolute)' if use_grouped else 'Depth'
        metric_display_name = selected_metric.replace('_', ' ').title()
        fig.update_layout(
            title=f'Comments vs Depth Slice at Context = {slice_info["value"]} ({metric_display_name})',
            xaxis_title=x_title, yaxis_title='Comments'
        )
        return fig

    return go.Figure()

if __name__ == '__main__':
    app.run(debug=True)
    import webbrowser
    url = 'http://127.0.0.1:8050/'
    webbrowser.open_new(url)