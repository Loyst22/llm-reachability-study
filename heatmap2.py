import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go
import plotly.express as px
import base64
import io

# Initialize with default data
def load_default_data():
    """Load the default CSV data"""
    try:
        df = pd.read_csv("accuracy_data.csv", sep=r'\s+|,', engine='python')
        return df
    except FileNotFoundError:
        # Create sample data if default file doesn't exist
        print("Default file not found, creating sample data")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    data = []
    for depth in [-10, -5, 5, 10]:
        for context in [100, 200, 300]:
            for comments in [0, 1, 2]:
                accuracy = np.random.uniform(0.5, 0.95)
                data.append({
                    'Depth': depth,
                    'Context': context,
                    'Comments': comments,
                    'Accuracy': accuracy
                })
    return pd.DataFrame(data)

def process_dataframe(df):
    """Process the dataframe and return all necessary data structures"""
    # Rename columns to remove leading underscores
    df = df.rename(columns=lambda x: x.strip("_"))
    
    # Ensure correct types
    df["Depth"] = df["Depth"].astype(int)
    df["Context"] = df["Context"].astype(int)
    df["Comments"] = df["Comments"].astype(int)
    
    # Original axis labels
    x_labels = sorted(df["Depth"].unique())
    y_labels = sorted(df["Context"].unique())
    z_labels = sorted(df["Comments"].unique())
    
    # Create grouped depth labels (absolute values)
    def get_grouped_depths(depths):
        """Get unique absolute depth values, sorted"""
        return sorted(list(set(abs(d) for d in depths)))
    
    grouped_x_labels = get_grouped_depths(x_labels)
    
    # Create 3D accuracy arrays for both original and grouped data
    def create_accuracy_arrays():
        # Original accuracy array
        accuracy_data = np.full((len(z_labels), len(y_labels), len(x_labels)), np.nan)
        
        # Fill original accuracy values
        for _, row in df.iterrows():
            z_idx = z_labels.index(row["Comments"])
            y_idx = y_labels.index(row["Context"])
            x_idx = x_labels.index(row["Depth"])
            accuracy_data[z_idx, y_idx, x_idx] = row["Accuracy"]
        
        # Grouped accuracy array
        grouped_accuracy_data = np.full((len(z_labels), len(y_labels), len(grouped_x_labels)), np.nan)
        
        # Fill grouped accuracy values by averaging positive/negative pairs
        for z_idx in range(len(z_labels)):
            for y_idx in range(len(y_labels)):
                for grouped_x_idx, abs_depth in enumerate(grouped_x_labels):
                    # Find all depths with this absolute value
                    matching_depths = [d for d in x_labels if abs(d) == abs_depth]
                    accuracies = []
                    
                    for depth in matching_depths:
                        orig_x_idx = x_labels.index(depth)
                        acc_val = accuracy_data[z_idx, y_idx, orig_x_idx]
                        if not np.isnan(acc_val):
                            accuracies.append(acc_val)
                    
                    if accuracies:
                        grouped_accuracy_data[z_idx, y_idx, grouped_x_idx] = np.mean(accuracies)
        
        # Replace missing values with 0
        accuracy_data = np.nan_to_num(accuracy_data, nan=0.0)
        grouped_accuracy_data = np.nan_to_num(grouped_accuracy_data, nan=0.0)
        
        return accuracy_data, grouped_accuracy_data
    
    accuracy_data, grouped_accuracy_data = create_accuracy_arrays()
    
    # Convert labels to strings for display
    z_labels_str = [str(z) for z in z_labels]
    grouped_x_labels_str = [str(x) for x in grouped_x_labels]
    
    return {
        'df': df,
        'x_labels': x_labels,
        'y_labels': y_labels,
        'z_labels': z_labels,
        'z_labels_str': z_labels_str,
        'grouped_x_labels': grouped_x_labels,
        'grouped_x_labels_str': grouped_x_labels_str,
        'accuracy_data': accuracy_data,
        'grouped_accuracy_data': grouped_accuracy_data
    }

# Load initial data
initial_df = load_default_data()
initial_data = process_dataframe(initial_df)

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("3D Accuracy Heatmap Viewer with Z Trends & Slicing"),
    
    # File Upload Section
    html.Div([
        html.H4("Upload Data File"),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'backgroundColor': '#f9f9f9'
            },
            multiple=False
        ),
        html.Div(id='upload-status', style={'margin': '10px', 'color': 'green'}),
        html.Button("Reset to Default Data", id="reset-data-button", n_clicks=0, 
                   style={'margin': '10px', 'backgroundColor': '#ff6b6b', 'color': 'white', 'border': 'none', 'padding': '10px', 'borderRadius': '5px'})
    ], style={'border': '1px solid #ddd', 'padding': '20px', 'margin': '20px', 'borderRadius': '10px'}),

    html.Div([
        html.Label("Z value (Comments):"),
        dcc.Slider(
            id='z-slider',
            min=0,
            max=len(initial_data['z_labels_str']) - 1,
            step=1,
            value=0,
            marks={i: initial_data['z_labels_str'][i] for i in range(len(initial_data['z_labels_str']))},
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
            options=[{'label': 'Group positive/negative depths (average accuracy)', 'value': 'group'}],
            value=[],  # Default to not grouped
            style={'marginBottom': '20px'}
        ),
    ], style={'width': '80%', 'margin': 'auto'}),

    html.Div([
        html.Label("Slice by Depth (X):"),
        dcc.Dropdown(id='x-slice', placeholder="Select Depth"),

        html.Label("Slice by Context (Y):"),
        dcc.Dropdown(id='y-slice', options=[{'label': y, 'value': y} for y in initial_data['y_labels']], placeholder="Select Context"),
    ], style={'width': '50%', 'margin': '0 auto'}),

    dcc.Graph(id='heatmap', clear_on_unhover=True),

    html.Div([
        html.Div(id='selected-coord-text', style={'margin': '10px 0', 'fontWeight': 'bold'}),
        html.Button("Clear Selection", id="clear-selection", n_clicks=0)
    ], style={'margin': '10px'}),

    dcc.Graph(id='lineplot'),
    dcc.Graph(id='slice-heatmap'),

    # Storage components
    dcc.Store(id='clicked-points', data=[]),
    dcc.Store(id='slice-info', data={}),
    dcc.Store(id='data-store', data=initial_data)  # Store processed data
])

def parse_uploaded_file(contents, filename):
    """Parse uploaded file and return dataframe"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename.lower():
            # Try multiple separators for CSV files
            try:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=r'\s+|,', engine='python')
            except:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, f"Unsupported file type: {filename}"
        
        # Validate that required columns exist
        required_columns = ['Depth', 'Context', 'Comments', 'Accuracy']
        df_columns = [col.strip('_') for col in df.columns]
        
        missing_columns = [col for col in required_columns if col not in df_columns]
        if missing_columns:
            return None, f"Missing required columns: {missing_columns}. Required: {required_columns}"
        
        return df, f"Successfully loaded {filename} with {len(df)} rows"
        
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

@app.callback(
    Output('data-store', 'data'),
    Output('upload-status', 'children'),
    Output('upload-status', 'style'),
    Input('upload-data', 'contents'),
    Input('reset-data-button', 'n_clicks'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_data_store(contents, reset_clicks, filename):
    trigger = ctx.triggered_id
    
    if trigger == 'reset-data-button':
        initial_df = load_default_data()
        data = process_dataframe(initial_df)
        return data, "Reset to default data", {'margin': '10px', 'color': 'green'}
    
    if contents is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    df, message = parse_uploaded_file(contents, filename)
    
    if df is not None:
        try:
            data = process_dataframe(df)
            return data, message, {'margin': '10px', 'color': 'green'}
        except Exception as e:
            return dash.no_update, f"Error processing data: {str(e)}", {'margin': '10px', 'color': 'red'}
    else:
        return dash.no_update, message, {'margin': '10px', 'color': 'red'}

@app.callback(
    Output('z-slider', 'max'),
    Output('z-slider', 'marks'),
    Output('z-slider', 'value'),
    Input('data-store', 'data'),
    Input('interval', 'n_intervals'),
    State('z-slider', 'value')
)
def update_z_slider(data, n_intervals, current_val):
    trigger = ctx.triggered_id
    max_val = len(data['z_labels_str']) - 1
    marks = {i: data['z_labels_str'][i] for i in range(len(data['z_labels_str']))}
    
    if trigger == 'data-store':
        # Reset to 0 when new data is loaded
        return max_val, marks, 0
    elif trigger == 'interval':
        # Animate the slider
        new_val = (current_val + 1) % len(data['z_labels_str'])
        return max_val, marks, new_val
    else:
        # Default case
        return max_val, marks, current_val if current_val is not None else 0

@app.callback(
    Output('y-slice', 'options'),
    Input('data-store', 'data')
)
def update_y_slice_options(data):
    return [{'label': y, 'value': y} for y in data['y_labels']]

@app.callback(
    Output('x-slice', 'options'),
    Input('group-depths-toggle', 'value'),
    Input('data-store', 'data')
)
def update_x_slice_options(group_depths, data):
    use_grouped = 'group' in group_depths
    if use_grouped:
        return [{'label': x, 'value': x} for x in data['grouped_x_labels_str']]
    else:
        return [{'label': x, 'value': x} for x in data['x_labels']]

@app.callback(
    Output('heatmap', 'figure'),
    Input('z-slider', 'value'),
    Input('slice-info', 'data'),
    Input('equal-spacing-toggle', 'value'),
    Input('group-depths-toggle', 'value'),
    Input('data-store', 'data')
)
def update_heatmap(z_index, slice_info, equal_spacing, group_depths, data):
    z_label = data['z_labels_str'][z_index]
    
    # Determine which data and labels to use
    use_grouped = 'group' in group_depths
    use_equal_spacing = 'equal' in equal_spacing
    
    if use_grouped:
        z_slice = data['grouped_accuracy_data'][z_index]
        current_x_labels = data['grouped_x_labels_str']
    else:
        z_slice = data['accuracy_data'][z_index]
        current_x_labels = data['x_labels']

    # Create coordinate axes
    x = list(range(len(current_x_labels))) if use_equal_spacing else [int(label) for label in current_x_labels]
    y = list(range(len(data['y_labels']))) if use_equal_spacing else data['y_labels']

    # Create custom colorscale: gray for 0 (missing), blue-green-yellow-orange-red for 0.5-1.0
    custom_colorscale = [
        [0.0, '#808080'],      # Gray for 0 (missing data)
        [0.001, '#808080'],    # Keep gray until just above 0
        [0.3, '#990000'],      # Dark Red for 0.5 (worse than random performance - bad)
        [0.5, '#ff0000'],      # Red for 0.5 (random performance - bad)
        [0.7, '#ff6600'],      # Orange for 0.6.5
        [0.9, '#fde725'],      # Yellow for 0.8
        [1.0, '#00ff00'],      # Green for 0.8
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_slice,
        x=x,
        y=y,
        colorscale=custom_colorscale,
        zmin=0,
        zmax=1,
        colorbar=dict(
            title='Accuracy',
            tickvals=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
            ticktext=['Missing', 'Really bad', 'Random', 'Poor', 'Ok', 'Good']
        )
    ))

    title_suffix = " (Grouped ±Depths)" if use_grouped else ""
    fig.update_layout(
        title=f"Accuracy Heatmap at Comments = {z_label}{title_suffix}",
        xaxis_title='Depth (Absolute)' if use_grouped else 'Depth',
        yaxis_title='Context',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(current_x_labels))) if use_equal_spacing else [int(label) for label in current_x_labels],
            ticktext=current_x_labels
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(data['y_labels']))) if use_equal_spacing else data['y_labels'],
            ticktext=data['y_labels']
        )
    )

    # Add slicing indicator
    if slice_info:
        if slice_info.get('type') == 'x':
            x_val = slice_info['value']
            try:
                if use_grouped:
                    x_pos = data['grouped_x_labels_str'].index(str(x_val)) if use_equal_spacing else int(x_val)
                else:
                    x_pos = data['x_labels'].index(int(x_val)) if use_equal_spacing else int(x_val)
                fig.add_shape(type="line", x0=x_pos, x1=x_pos,
                              y0=min(y), y1=max(y),
                              line=dict(color="red", width=2))
            except (ValueError, IndexError):
                pass  # Skip if value not found
        elif slice_info.get('type') == 'y':
            y_val = slice_info['value']
            y_pos = data['y_labels'].index(y_val) if use_equal_spacing else y_val
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
    Output('clicked-points', 'data'),
    Output('slice-info', 'data'),
    Input('heatmap', 'clickData'),
    Input('x-slice', 'value'),
    Input('y-slice', 'value'),
    Input('clear-selection', 'n_clicks'),
    State('clicked-points', 'data'),
    prevent_initial_call=True
)
def update_clicked_points(click_data, x_slice_val, y_slice_val, clear_clicks, stored_points):
    trigger = ctx.triggered_id
    if trigger == "clear-selection":
        return [], {}

    if trigger == 'x-slice' and x_slice_val:
        return stored_points, {'type': 'x', 'value': x_slice_val}
    if trigger == 'y-slice' and y_slice_val:
        return stored_points, {'type': 'y', 'value': y_slice_val}

    if not click_data:
        return stored_points, {}

    x_val = click_data['points'][0]['x']
    y_val = click_data['points'][0]['y']
    new_point = {'x': x_val, 'y': y_val}

    if new_point not in stored_points:
        stored_points.append(new_point)
    return stored_points, {}

@app.callback(
    Output('lineplot', 'figure'),
    Output('selected-coord-text', 'children'),
    Input('clicked-points', 'data'),
    Input('group-depths-toggle', 'value'),
    Input('data-store', 'data')
)
def plot_lines(clicked_points, group_depths, data):
    if not clicked_points:
        return go.Figure(), "Click cells to compare accuracy over Comments for multiple (Depth, Context) points."

    use_grouped = 'group' in group_depths
    current_accuracy_data = data['grouped_accuracy_data'] if use_grouped else data['accuracy_data']
    current_x_labels = data['grouped_x_labels_str'] if use_grouped else data['x_labels']

    fig = go.Figure()
    color_palette = px.colors.qualitative.Plotly
    for i, pt in enumerate(clicked_points):
        try:
            x_idx = current_x_labels.index(str(pt['x']) if use_grouped else pt['x'])
            y_idx = data['y_labels'].index(pt['y'])
        except ValueError:
            continue
        acc_curve = [current_accuracy_data[z, y_idx, x_idx] for z in range(len(data['z_labels_str']))]
        fig.add_trace(go.Scatter(
            x=data['z_labels_str'],
            y=acc_curve,
            mode='lines+markers',
            name=f"({pt['x']}, {pt['y']})",
            marker=dict(color=color_palette[i % len(color_palette)]),
            line=dict(color=color_palette[i % len(color_palette)]),
            hovertemplate=f'Point: ({pt["x"]}, {pt["y"]})<br>Comments: %{{x}}<br>Accuracy: %{{y:.3f}}<extra></extra>'
        ))
    
    title_suffix = " (Grouped Data)" if use_grouped else ""
    fig.update_layout(
        title=f'Accuracy Across Comments for Selected Points{title_suffix}',
        xaxis_title='Comments',
        yaxis_title='Accuracy'
    )
    label_text = "Selected coordinates: " + ", ".join([f"({pt['x']}, {pt['y']})" for pt in clicked_points])
    return fig, label_text

@app.callback(
    Output('slice-heatmap', 'figure'),
    Input('slice-info', 'data'),
    Input('group-depths-toggle', 'value'),
    Input('data-store', 'data')
)
def update_slice_heatmap(slice_info, group_depths, data):
    if not slice_info:
        return go.Figure()

    use_grouped = 'group' in group_depths
    current_accuracy_data = data['grouped_accuracy_data'] if use_grouped else data['accuracy_data']
    current_x_labels = data['grouped_x_labels_str'] if use_grouped else data['x_labels']

    if slice_info['type'] == 'x':
        try:
            x_idx = current_x_labels.index(str(slice_info['value']) if use_grouped else slice_info['value'])
        except ValueError:
            return go.Figure()
        
        slice_data = current_accuracy_data[:, :, x_idx]  # shape (Z, Y)
        
        # Custom colorscale for slice heatmap: blue-green-yellow-orange-red
        custom_colorscale = [
            [0.0, '#808080'],      # Gray for 0 (missing data)
            [0.001, '#808080'],    # Keep gray until just above 0
            [0.3, '#990000'],      # Dark Red for 0.5 (worse than random performance - bad)
            [0.5, '#ff0000'],      # Red for 0.5 (random performance - bad)
            [0.7, '#ff6600'],      # Orange for 0.6.5
            [0.9, '#fde725'],      # Yellow for 0.8
            [1.0, '#00ff00'],      # Green for 0.8
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=slice_data,
            x=data['y_labels'],
            y=data['z_labels_str'],
            colorscale=custom_colorscale,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title='Accuracy',
                tickvals=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                ticktext=['Missing', 'Really bad', 'Random', 'Poor', 'Ok', 'Good']
            )
        ))
        depth_label = f'|Depth| = {slice_info["value"]}' if use_grouped else f'Depth = {slice_info["value"]}'
        fig.update_layout(
            title=f'Comments vs Context Slice at {depth_label}',
            xaxis_title='Context', yaxis_title='Comments'
        )
        return fig

    elif slice_info['type'] == 'y':
        y_idx = data['y_labels'].index(slice_info['value'])
        slice_data = current_accuracy_data[:, y_idx, :]  # shape (Z, X)
        
        # Custom colorscale for slice heatmap: blue-green-yellow-orange-red
        custom_colorscale = [
            [0.0, '#808080'],      # Gray for 0 (missing data)
            [0.001, '#808080'],    # Keep gray until just above 0
            [0.3, '#990000'],      # Dark Red for 0.5 (worse than random performance - bad)
            [0.5, '#ff0000'],      # Red for 0.5 (random performance - bad)
            [0.7, '#ff6600'],      # Orange for 0.6.5
            [0.9, '#fde725'],      # Yellow for 0.8
            [1.0, '#00ff00'],      # Green for 0.8
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=slice_data,
            x=current_x_labels,
            y=data['z_labels_str'],
            colorscale=custom_colorscale,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title='Accuracy',
                tickvals=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                ticktext=['Missing', 'Really bad', 'Random', 'Poor', 'Ok', 'Good']
            )
        ))
        x_title = 'Depth (Absolute)' if use_grouped else 'Depth'
        fig.update_layout(
            title=f'Comments vs Depth Slice at Context = {slice_info["value"]}',
            xaxis_title=x_title, yaxis_title='Comments'
        )
        return fig

    return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)