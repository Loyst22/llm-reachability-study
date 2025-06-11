import pandas as pd
import streamlit as st
import plotly.express as px

# Load a specific hard-coded CSV file
@st.cache
def load_data():
    return pd.read_csv("./exp_out/output_all.csv")  # Replace with your CSV file path

answer_col = "_answer type"
# Main Streamlit App

context_mapping = {
    "small" : 30,
    "smallish" : 50,
    "medium": 75,
    "medium-plus": 100,
    "medium-plus-plus": 125,
    "medium-large": 150,
    "largish" : 200,
    "largish-plus": 250,
    "very-large": 300,
    "very-very-large": 350,
    "huge": 400,
}

def map_context_size(row):
    # Sort the keys by length in descending order to match the longest words first
    for word in sorted(context_mapping.keys(), key=len, reverse=True):
        if word in row:
            return context_mapping[word]
    return None  # In case no match is found, return None or another default value

def make_clickable(path):
    return f'<a href="file://{path}" target="_blank">{path}</a>'

def line_plot_for_col(df, selected_num_col):
    # Plot the percentage as a line plot (one line per answer category)
    group = df.groupby([selected_num_col, answer_col]).size().unstack(fill_value=0)
            
    # Calculate the percentage of each answer category within each numerical value
    group_percentage = group.divide(group.sum(axis=1), axis=0) * 100
            
    # Reset index for plotting (so we can plot 'selected_num_col' on the x-axis)
    group_percentage = group_percentage.reset_index()
    fig = px.line(group_percentage, x=selected_num_col, y=group_percentage.columns.drop(selected_num_col),
                title=f"Percentage of {answer_col} Categories by {selected_num_col} Values",
                labels={selected_num_col: selected_num_col, 'value': 'Percentage'})
    
    # Display the plot
    st.plotly_chart(fig)


def depth_vs_context_plot(df, selected_answer_type, answer_col):
    """
    Creates a plot of depth vs context size for a selected answer type.
    Percentage is calculated relative to all rows.
    """
    # Filter for the selected answer type
    filtered_df = df[df[answer_col] == selected_answer_type]
    
    # Group by Depth and context_size, counting occurrences
    total_counts = df.groupby(['Depth', 'context_size']).size().reset_index(name='total_count')
    selected_counts = filtered_df.groupby(['Depth', 'context_size']).size().reset_index(name='selected_count')
    
    # Merge total and selected counts
    merged = pd.merge(total_counts, selected_counts, on=['Depth', 'context_size'], how='left')
    merged['selected_count'] = merged['selected_count'].fillna(0)  # Fill missing selected counts with 0
    
    # Calculate percentages (selected_count / total_count * 100)
    merged['percentage'] = (merged['selected_count'] / merged['total_count']) * 100
    
    # Create the plot
    fig = px.line(
        merged,
        x='Depth',
        y='percentage',
        color='context_size',
        # color_continuous_scale='Viridis',  # Use a sequential color scale
        # doesnt work for lines
        title=f"Evolution of Depth by Context Size for {selected_answer_type}",
        labels={'percentage': 'Percentage (%)', 'Depth': 'Depth', 'context_size': 'Context Size'}
    )
    fig.update_yaxes(range=[0, 100])
    # Display the plot
    st.plotly_chart(fig)


def depth_vs_context_plot2(df, selected_answer_type, answer_col):
    """
    Creates a plot of depth vs context size for a selected answer type.
    Percentage is calculated relative to all rows. Uses discrete color mapping for lines.
    """
    # Filter for the selected answer type
    filtered_df = df[df[answer_col] == selected_answer_type]
    
    # Group by Depth and context_size, counting occurrences
    total_counts = df.groupby(['Depth', 'context_size']).size().reset_index(name='total_count')
    selected_counts = filtered_df.groupby(['Depth', 'context_size']).size().reset_index(name='selected_count')
    
    # Merge total and selected counts
    merged = pd.merge(total_counts, selected_counts, on=['Depth', 'context_size'], how='left')
    merged['selected_count'] = merged['selected_count'].fillna(0)  # Fill missing selected counts with 0
    
    # Calculate percentages (selected_count / total_count * 100)
    merged['percentage'] = (merged['selected_count'] / merged['total_count']) * 100
    
    # Ensure Depth is sorted
    merged = merged.sort_values('Depth')
    
    # Create a line plot with discrete color mapping
    fig = px.line(
        merged,
        x='Depth',
        y='percentage',
        color='context_size',  # Lines for each context size
        title=f"Evolution of Depth by Context Size for {selected_answer_type}",
        labels={'percentage': 'Percentage (%)', 'Depth': 'Depth', 'context_size': 'Context Size'},
        color_discrete_sequence=px.colors.sequential.Viridis[:len(merged['context_size'].unique())]  # Discrete color palette
    )
    
    # Force the y-axis to range from 0 to 100
    fig.update_yaxes(range=[0, 100])
    
    # Display the plot
    st.plotly_chart(fig)

def depth_vs_context_plot_with_swap(df, selected_answer_type, answer_col):
    """
    Creates a plot of depth vs context size (or vice versa) for a selected answer type.
    Allows swapping the x-axis and the line dimension dynamically.
    """
    # Filter for the selected answer type
    filtered_df = df[df[answer_col] == selected_answer_type]
    
    # Group by Depth and context_size, counting occurrences
    total_counts = df.groupby(['Depth', 'context_size']).size().reset_index(name='total_count')
    selected_counts = filtered_df.groupby(['Depth', 'context_size']).size().reset_index(name='selected_count')
    
    # Merge total and selected counts
    merged = pd.merge(total_counts, selected_counts, on=['Depth', 'context_size'], how='left')
    merged['selected_count'] = merged['selected_count'].fillna(0)  # Fill missing selected counts with 0
    
    # Calculate percentages (selected_count / total_count * 100)
    merged['percentage'] = (merged['selected_count'] / merged['total_count']) * 100
    
    # Allow the user to choose which dimension is on the x-axis
    axis_option = st.radio(
        "Choose the x-axis dimension:",
        options=["Depth", "context_size"],
        index=0,  # Default to "Depth"
        horizontal=True
    )
    
    # Set x-axis and line grouping dynamically
    x_axis = axis_option
    line_group = "context_size" if x_axis == "Depth" else "Depth"
    
    # Ensure the selected x-axis is sorted
    merged = merged.sort_values(x_axis)
    
    # Create a line plot with discrete color mapping
    fig = px.line(
        merged,
        x=x_axis,
        y='percentage',
        color=line_group,  # Lines for the other dimension
        title=f"Evolution of {x_axis} by {line_group} for {selected_answer_type}",
        labels={'percentage': 'Percentage (%)', x_axis: x_axis, line_group: line_group},
        color_discrete_sequence=px.colors.sequential.Viridis[:len(merged[line_group].unique())]  # Discrete color palette
    )
    
    # Force the y-axis to range from 0 to 100
    fig.update_yaxes(range=[0, 100])
    
    # Display the plot
    st.plotly_chart(fig)


def arbitrary_column_line_plot(df, answer_col, selected_answer_type):
    """
    Create a line plot for two arbitrary columns, selected by the user.
    Handles both numerical and categorical data for the grouping column.
    """
    st.write("### Plot Two Arbitrary Columns")
    
    # Select columns for x-axis and grouping
    available_columns = list(df.columns)
    x_axis_col = st.selectbox("Select column for x-axis", available_columns, index=0)
    group_col = st.selectbox("Select column for line grouping", available_columns, index=1)

    # Filter data for the selected answer type
    filtered_df = df[df[answer_col] == selected_answer_type]
    
    # Check if the group column is categorical or numerical
    if pd.api.types.is_numeric_dtype(df[group_col]):
        group_type = "numerical"
        group_col_values = filtered_df[group_col].unique()
    else:
        group_type = "categorical"
        group_col_values = filtered_df[group_col].astype(str).unique()

    # Group data by x-axis and the group column
    grouped = (
        filtered_df.groupby([x_axis_col, group_col])
        .size()
        .reset_index(name='selected_count')
    )
    
    # Add total counts for normalization
    total_counts = (
        df.groupby([x_axis_col, group_col])
        .size()
        .reset_index(name='total_count')
    )
    merged = pd.merge(total_counts, grouped, on=[x_axis_col, group_col], how="left")
    merged['selected_count'] = merged['selected_count'].fillna(0)

    # Calculate percentages
    merged['percentage'] = (merged['selected_count'] / merged['total_count']) * 100

    # Sort by x-axis
    merged = merged.sort_values(x_axis_col)

    # Plot the line graph
    if group_type == "categorical":
        fig = px.line(
            merged,
            x=x_axis_col,
            y='percentage',
            color=group_col,
            title=f"Percentage of {selected_answer_type} by {x_axis_col} and {group_col}",
            labels={'percentage': 'Percentage (%)', x_axis_col: x_axis_col, group_col: group_col},
            color_discrete_sequence=px.colors.qualitative.Set1  # Nice discrete palette for categories
        )
    else:
        fig = px.line(
            merged,
            x=x_axis_col,
            y='percentage',
            color=group_col,
            title=f"Percentage of {selected_answer_type} by {x_axis_col} and {group_col}",
            labels={'percentage': 'Percentage (%)', x_axis_col: x_axis_col, group_col: group_col},
            color_continuous_scale=px.colors.sequential.Viridis
        )

    # Force the y-axis to range from 0 to 100
    fig.update_yaxes(range=[0, 100])

    # Display the plot
    st.plotly_chart(fig)

def main():
    st.title("Data Explorer: Compare Subsets")

    # Load the data
    df_init = load_data()  # Assuming load_data is a function you have to load your DataFrame

    # df_init['result_file_link'] = df_init['result_file_path'].apply(make_clickable)


    # Show the original data
    st.write("### Original Data ...")
    st.dataframe(df_init.head())

    # Default columns to remove based on name patterns
    default_columns_to_remove = [col for col in df_init.columns if 'Amount' in col or 'Percent' in col or 'path' in col or "number" in col]

    # Allow user to select columns to add back
    columns_to_add_back = st.multiselect(
        "Select columns to add back", 
        options=default_columns_to_remove, 
        default=[]  # By default, they are removed
    )

    # Remove selected columns from the default removed columns
    columns_to_remove = set(default_columns_to_remove) - set(columns_to_add_back)

    # Create a new DataFrame by removing the selected columns
    df_cleaned = df_init.drop(columns=columns_to_remove)

    # Allow user to select additional columns to remove from the cleaned DataFrame
    columns_to_remove_extra = st.multiselect(
        "Select additional columns to remove", 
        options=[col for col in df_cleaned.columns if col not in default_columns_to_remove],  # Only show columns that are still in df_cleaned
        default=[]  # By default, no extra columns are removed
    )

    # Add the extra columns to remove
    print("Columns before removal:", df_init.columns)
    df_cleaned = df_init.drop(columns=columns_to_remove)
    df_cleaned = df_cleaned.drop(columns=columns_to_remove_extra)
    df_cleaned['context_size'] = df_cleaned['_Context size'].apply(map_context_size)
    df_cleaned = df_cleaned.drop(columns=['_Context size'])
    print("Columns after removal:", df_cleaned.columns)
    # Show the updated DataFrame
    st.write("### Data After Removing/Adding Columns")
    st.dataframe(df_cleaned)  # This should now reflect the correct DataFrame

    # Now we work on subset filtering and other parts of the app
    # Choose property to filter subsets
    st.write("### Select Subset Filters")
    subset_filters = {}
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object' or df_cleaned[col].nunique() < 20:
            values = st.multiselect(f"Filter by {col}", options=df_cleaned[col].unique())
            if values:
                subset_filters[col] = values

    # Apply filters to create subset
    filtered_df = df_cleaned.copy()
    for col, values in subset_filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(values)]

    st.write("### Filtered Subset")
    st.dataframe(filtered_df)

    # Compare subsets
    st.write("### Compare Subsets")
    subset_group = st.radio("Choose property for grouping", df_cleaned.columns)
    if subset_group:
        fig = px.histogram(filtered_df, x=subset_group, color=answer_col, barmode="group", title=f"Distribution of {subset_group}")
        st.plotly_chart(fig)

    # Numerical analysis
    st.write("### Numerical Property Analysis")
    num_cols = [col for col in df_cleaned.columns if pd.api.types.is_numeric_dtype(df_cleaned[col])]
    print(num_cols)
    st.write("### Depth")
    line_plot_for_col(filtered_df, "Depth")
    st.write("### Context size")
    line_plot_for_col(filtered_df, "context_size")
    if num_cols:
        selected_num_col = st.selectbox("Choose numerical column", num_cols)
        print(selected_num_col)
        if selected_num_col:
              # Group by the numerical column and the answer_col, counting occurrences
            group = filtered_df.groupby([selected_num_col, answer_col]).size().unstack(fill_value=0)
            
            # Calculate the percentage of each answer category within each numerical value
            group_percentage = group.divide(group.sum(axis=1), axis=0) * 100
            
            # Reset index for plotting (so we can plot 'selected_num_col' on the x-axis)
            group_percentage = group_percentage.reset_index()

            ctx = [col for col in df_cleaned.columns if "context" in col]
            depth = [col for col in df_cleaned.columns if "Depth" in col]
           
            # Plot the percentage as a line plot (one line per answer category)
            #fig = px.line(group_percentage, x=selected_num_col, y=group_percentage.columns.drop(selected_num_col),
            #            title=f"Percentage of {answer_col} Categories by {selected_num_col} Values",
            #            labels={selected_num_col: selected_num_col, 'value': 'Percentage'})
            
            # Display the plot
            #st.plotly_chart(fig)

     # Numerical analysis: Depth vs Context Size
    st.write("### Depth vs Context Size")
    
    # Select the answer type
    answer_types = df_cleaned[answer_col].unique()
    selected_answer_type = st.selectbox("Select answer type to analyze", answer_types)
    
    # Generate and display the plot
    depth_vs_context_plot_with_swap(filtered_df, selected_answer_type, answer_col)

    arbitrary_column_line_plot(filtered_df, answer_col, selected_answer_type)


# Run the app
if __name__ == "__main__":
    main()
