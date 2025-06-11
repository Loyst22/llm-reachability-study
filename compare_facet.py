import pandas as pd
import streamlit as st
import plotly.express as px

# Load your dataset
@st.cache
def load_data():
    return pd.read_csv("./exp_out/output_all.csv")  # Replace with your CSV file path

answer_col = "_answer type"

def map_context_size(row):
    # Sort the keys by length in descending order to match the longest words first
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
    for word in sorted(context_mapping.keys(), key=len, reverse=True):
        if word in row:
            return context_mapping[word]
    return None  # In case no match is found

def facet_line_plot(df, selected_cols):
    # Extract the selected columns
    x_axis_col, group_col, facet_col = selected_cols

    # Create the plot
    fig = px.line(
        df,
        x=x_axis_col,
        y="value",  # This should be the column you're plotting percentages in, e.g., "value"
        color=group_col,
        facet_col=facet_col,
        facet_col_wrap=4,  # Optional: Controls layout of the facets
        title=f"Analysis of {group_col} by {x_axis_col} and Facet by {facet_col}",
        labels={x_axis_col: x_axis_col, "value": "Percentage"},
    )
    st.plotly_chart(fig)

def main():
    # Function to determine the default facet column
    def get_default_facet(categorical_cols, numerical_cols):
        # Rule 1: If there's only one categorical, it's the default facet
        if len(categorical_cols) == 1:
            return categorical_cols[0]
        # Rule 2: If there are multiple categorical columns, choose the one with the least unique values
        elif len(categorical_cols) > 1:
            return min(categorical_cols, key=lambda col: df_init[col].nunique())
        # Rule 3: If only numerical columns, choose the one with the least unique values
        elif len(categorical_cols) == 0 and len(numerical_cols) > 0:
            return min(numerical_cols, key=lambda col: df_init[col].nunique())
        # Fallback: If no default is found, return None or prompt the user to choose
        return None

    st.title("Data Explorer with Faceting")

    # Load the data
    df_init = load_data()

    # Show the original data
    st.write("### Original Data")
    st.dataframe(df_init.head())

    # Allow user to select columns for analysis
    numerical_cols = [col for col in df_init.columns if pd.api.types.is_numeric_dtype(df_init[col])]
    categorical_cols = [col for col in df_init.columns if pd.api.types.is_object_dtype(df_init[col])]

    default_facet = get_default_facet(categorical_cols, numerical_cols)


    # Let user select dimensions: two numerical and one categorical
    selected_numerical_cols = st.multiselect("Select numerical columns", options=numerical_cols, default=numerical_cols[:2])
    selected_categorical_cols = st.multiselect("Select categorical columns", options=categorical_cols, default=categorical_cols[:1])

    # Exit early if no selections
    if not selected_numerical_cols or not selected_categorical_cols:
        st.warning("Please select at least one numerical column and one categorical column to proceed.")
        return
    if answer_col not in df_init.columns:
        st.error(f"Column '{answer_col}' does not exist in the dataframe!")
        return  # Exit early if column is missing


    # Ensure user selects exactly three columns
    selected_cols = selected_numerical_cols + selected_categorical_cols
    if len(selected_cols) < 3:
        st.warning("Please select at least three columns (e.g., two numerical and one categorical).")
        return

    # Dynamically determine grouping columns
    grouping_columns = selected_numerical_cols + selected_categorical_cols + [answer_col]
    # Select the facet column based on user input and ensure it's in the dataframe
    facet_col = st.radio("Select the column for faceting", selected_cols, index=selected_cols.index(default_facet))

    # Filter the dataframe and compute the group percentages
    filtered_df = df_init.copy()
    group = filtered_df.groupby([facet_col, answer_col]).size().unstack(fill_value=0)

    group_percentage = group.divide(group.sum(axis=1), axis=0) * 100

    # Flatten multi-index columns
    group_percentage.columns = group_percentage.columns.to_flat_index()

    # Reset index and show the result
    group_percentage = group_percentage.reset_index()

    # Debugging: Show final group_percentage
    st.write("### Final Group Percentage Data", group_percentage)

    # Check if there is a mismatch in the column names
    st.write("### Columns in group_percentage", group_percentage.columns)
    facet_line_plot(group_percentage, [selected_numerical_cols[0], selected_categorical_cols[0], facet_col])



# Run the app
if __name__ == "__main__":
    main()
