import pandas as pd
import numpy as np
from itertools import product
import sys

def load_and_clean_csv(file_path):
    """Load CSV and clean the data"""
    df = pd.read_csv(file_path)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Clean string columns
    string_cols = ['_answer type', '_Experiment', '_Model', 'answer category', 'answer subcategory']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()
    
    return df

def compute_accuracy_summary(df):
    """
    Compute accuracy for all combinations of Depth, _Comments, and _Context
    Accuracy = percentage of responses where _answer type is 'RIGHT'
    """
    
    # Create binary accuracy column
    df['is_right'] = (df['_answer type'] == 'RIGHT').astype(int)
    
    # Group by the three dimensions
    summary = df.groupby(['Depth', '_Comments', '_Context', '_Model']).agg({
        'is_right': ['count', 'sum', 'mean'],
        # '_Model': 'first',  # Get model name (assuming same for each group)
        '_Experiment': 'first'  # Get experiment name
    }).round(4)
    
    # Flatten column names
    summary.columns = ['Total_Responses', 'Right_Responses', 'Accuracy', 'Model', 'Experiment']
    
    # Reset index to make grouping columns regular columns
    summary = summary.reset_index()
    
    # Calculate percentage
    summary['Accuracy_Percent'] = (summary['Accuracy'] * 100).round(2)
    
    return summary

def compute_rfwr_within_right_summary(df):
    """
    For each (Depth, _Comments, _Context) group:
    - Restrict to _answer type == 'RIGHT'
    - Compute percentage of those that are 'RIGHT_FOR_WRONG_REASON'
    
    Returns a DataFrame with the same structure as compute_accuracy_summary.
    """
    # Filter to only RIGHT answers
    right_df = df[df['_answer type'] == 'RIGHT'].copy()

    # Add binary column: 1 if category is RIGHT_FOR_WRONG_REASON
    right_df['is_rfwr'] = (right_df['answer category'] == 'RIGHT_FOR_WRONG_REASON').astype(int)

    # Group by relevant dimensions
    summary = right_df.groupby(['Depth', '_Comments', '_Context', '_Model']).agg({
        'is_rfwr': ['count', 'sum', 'mean'],
        # '_Model': 'first',
        '_Experiment': 'first'
    }).round(4)
    
    # Rename columns
    summary.columns = ['Total_Responses', 'Right_Responses', 'Accuracy', 'Model', 'Experiment']
    summary = summary.reset_index()
    
    # Percentage column
    summary['Accuracy_Percent'] = (summary['Accuracy'] * 100).round(2)
    
    return summary

def compute_comprehensive_accuracy_summary(df):
    """
    Compute comprehensive accuracy metrics for all combinations of Depth, *Comments, and *Context
    
    Returns:
    - Base accuracy: percentage of responses where _answer type is 'RIGHT'
    - RFWR rate: percentage of RIGHT responses that are 'RIGHT_FOR_WRONG_REASON'
    - Adjusted accuracy: base accuracy excluding RIGHT_FOR_WRONG_REASON cases
    """
    # Create binary columns for analysis
    df['is_right'] = (df['_answer type'] == 'RIGHT').astype(int)
    df['is_rfwr'] = (df['answer category'] == 'RIGHT_FOR_WRONG_REASON').astype(int)
    df['is_right_good_reason'] = ((df['_answer type'] == 'RIGHT') & 
                                  (df['answer category'] != 'RIGHT_FOR_WRONG_REASON')).astype(int)
    
    # Group by the three dimensions and compute all metrics
    summary = df.groupby(['Depth', '_Comments', '_Context', '_Model']).agg({
        'is_right': ['count', 'sum', 'mean'],
        'is_rfwr': 'sum',
        'is_right_good_reason': 'sum',
        # '_Model': 'first',
        '_Experiment': 'first'
    }).round(4)
    
    # Flatten and rename columns
    summary.columns = ['Total_Cases', 'Right_Cases', 'Base_Accuracy', 
                      'RFWR_Cases', 'Right_Good_Reason_Cases', 'Experiment']
    
    # Reset index to make grouping columns regular columns
    summary = summary.reset_index()
    
    # Calculate additional metrics
    summary['Base_Accuracy_Percent'] = (summary['Base_Accuracy'] * 100).round(2)
    
    # RFWR rate among right answers (handle division by zero)
    summary['RFWR_Rate'] = (summary['RFWR_Cases'] / summary['Right_Cases']).fillna(0).round(4)
    summary['RFWR_Rate_Percent'] = (summary['RFWR_Rate'] * 100).round(2)
    
    # Adjusted accuracy (excluding RFWR cases)
    summary['Adjusted_Accuracy'] = (summary['Right_Good_Reason_Cases'] / summary['Total_Cases']).round(4)
    summary['Adjusted_Accuracy_Percent'] = (summary['Adjusted_Accuracy'] * 100).round(2)
    
    summary['Accuracy_Drop'] = (summary['Base_Accuracy'] - summary['Adjusted_Accuracy']).round(4)
    summary['Accuracy_Drop_Percent'] = (summary['Base_Accuracy_Percent'] - summary['Adjusted_Accuracy_Percent']).round(2)

    # Reorder columns for better readability
    column_order = [
        'Depth', '_Comments', '_Context', '_Model', 'Experiment',
        'Total_Cases', 'Right_Cases', 'Base_Accuracy', 'Base_Accuracy_Percent',
        'RFWR_Cases', 'RFWR_Rate', 'RFWR_Rate_Percent',
        'Right_Good_Reason_Cases', 'Adjusted_Accuracy', 'Adjusted_Accuracy_Percent','Accuracy_Drop','Accuracy_Drop_Percent'
    ]
    
    return summary[column_order]

def print_summary_statistics(summary_df):
    """Print overall statistics"""
    print("=== OVERALL STATISTICS ===")
    print(f"Total combinations: {len(summary_df)}")
    print(f"Overall accuracy: {summary_df['Base_Accuracy_Percent'].mean():.2f}%")
    print(f"Best accuracy: {summary_df['Base_Accuracy_Percent'].max():.2f}%")
    print(f"Worst accuracy: {summary_df['Base_Accuracy_Percent'].min():.2f}%")
    print(f"Standard deviation: {summary_df['Base_Accuracy_Percent'].std():.2f}%")
    print()

def analyze_csv_accuracy(file_path):
    """Main function to analyze CSV and print results"""
    
    # Load data
    print("Loading CSV data...")
    df = load_and_clean_csv(file_path)
    print(f"Loaded {len(df)} rows")
    print()
    
    # Compute summary
    print("Computing accuracy summary...")
    summary = compute_comprehensive_accuracy_summary(df)
    
    # Print overall statistics
    print_summary_statistics(summary)
    
    # Print detailed summary table
    print("=== DETAILED ACCURACY SUMMARY ===")
    print("(Depth, Comments, Context) -> Accuracy")
    print("-" * 60)
    
    # Sort by accuracy descending
    summary_sorted = summary.sort_values('Accuracy_Percent', ascending=False)
    
    for _, row in summary_sorted.iterrows():
        print(f"Depth={row['Depth']:2}, Comments={row['_Comments']:3}, Context={row['_Context']:3} -> "
              f"{row['Right_Responses']:3}/{row['Total_Responses']:3} = {row['Accuracy_Percent']:6.2f}%")
    
    print()
    
    return summary

def analyze_csv_accuracy_full(file_path):
    """Main function to analyze CSV and print results"""
    # Load data
    print("Loading CSV data...")
    df = load_and_clean_csv(file_path)
    print(f"Loaded {len(df)} rows")
    print()
    
    # Compute summary
    print("Computing comprehensive accuracy summary...")
    summary = compute_comprehensive_accuracy_summary(df)
    
    # Print overall statistics
    print_summary_statistics(summary)
    
    # Print detailed summary table
    print("=== DETAILED ACCURACY SUMMARY ===")
    print("(Depth, Comments, Context) -> Base Accuracy | RFWR Rate | Adjusted Accuracy")
    print("-" * 85)
    
    # Sort by base accuracy descending
    summary_sorted = summary.sort_values('Base_Accuracy_Percent', ascending=False)
    
    for _, row in summary_sorted.iterrows():
        print(f"Depth={row['Depth']:2}, Comments={row['_Comments']:3}, Context={row['_Context']:3} -> "
              f"{row['Right_Cases']:3}/{row['Total_Cases']:3} = {row['Base_Accuracy_Percent']:6.2f}% | "
              f"RFWR: {row['RFWR_Cases']:2}/{row['Right_Cases']:3} = {row['RFWR_Rate_Percent']:5.1f}% | "
              f"Adjusted: {row['Right_Good_Reason_Cases']:3}/{row['Total_Cases']:3} = {row['Adjusted_Accuracy_Percent']:6.2f}%")
    
    print()
    
    # Print additional analysis
    print("=== PERFORMANCE COMPARISON ===")
    print(f"Overall Base Accuracy:     {summary['Base_Accuracy_Percent'].mean():.2f}%")
    print(f"Overall RFWR Rate:         {summary['RFWR_Rate_Percent'].mean():.2f}%")
    print(f"Overall Adjusted Accuracy: {summary['Adjusted_Accuracy_Percent'].mean():.2f}%")
    print(f"Accuracy Drop due to RFWR: {summary['Base_Accuracy_Percent'].mean() - summary['Adjusted_Accuracy_Percent'].mean():.2f} percentage points")
    print()
    
    # Print best and worst performers
    best_base = summary_sorted.iloc[0]
    worst_base = summary_sorted.iloc[-1]
    
    print("=== BEST/WORST CONFIGURATIONS ===")
    print(f"Best Base Accuracy:  Depth={best_base['Depth']}, Comments={best_base['_Comments']}, Context={best_base['_Context']} ({best_base['Base_Accuracy_Percent']:.2f}%)")
    print(f"Worst Base Accuracy: Depth={worst_base['Depth']}, Comments={worst_base['_Comments']}, Context={worst_base['_Context']} ({worst_base['Base_Accuracy_Percent']:.2f}%)")
    
    # Sort by adjusted accuracy for comparison
    summary_adj_sorted = summary.sort_values('Adjusted_Accuracy_Percent', ascending=False)
    best_adj = summary_adj_sorted.iloc[0]
    worst_adj = summary_adj_sorted.iloc[-1]
    
    print(f"Best Adjusted Accuracy:  Depth={best_adj['Depth']}, Comments={best_adj['_Comments']}, Context={best_adj['_Context']} ({best_adj['Adjusted_Accuracy_Percent']:.2f}%)")
    print(f"Worst Adjusted Accuracy: Depth={worst_adj['Depth']}, Comments={worst_adj['_Comments']}, Context={worst_adj['_Context']} ({worst_adj['Adjusted_Accuracy_Percent']:.2f}%)")
    
    # Find highest RFWR rate
    highest_rfwr = summary.loc[summary['RFWR_Rate_Percent'].idxmax()]
    print(f"Highest RFWR Rate: Depth={highest_rfwr['Depth']}, Comments={highest_rfwr['_Comments']}, Context={highest_rfwr['_Context']} ({highest_rfwr['RFWR_Rate_Percent']:.1f}%)")
    print()
    
    return summary

def save_results(summary_df, output_prefix="accuracy_analysis"):
    """Save results to CSV files"""
    
    # Save detailed summary
    summary_df.to_csv(f"{output_prefix}.csv", index=False)
    print(f"Saved detailed summary to {output_prefix}.csv")

# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    base_path = sys.argv[1]
    print(base_path)
    csv_file_path = f"{base_path}/aggregated.csv"
    output_file = f"{base_path}/statistics"
    #csv_file_path = "exp_out/output_temp0.csv"
    #csv_file_path = "mid.csv"
    
    try:
        summary = analyze_csv_accuracy_full(csv_file_path)
        
        # Optionally save results
        save_results(summary, output_file)
        
    except FileNotFoundError:
        print(f"File {csv_file_path} not found. Please update the file path.")
    except Exception as e:
        print(f"Error: {e}")

