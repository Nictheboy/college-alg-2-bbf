import pandas as pd
import numpy as np

def analyze_results_per_dataset(df_dataset, dataset_name_for_print):
    """Helper function to analyze and print results for a single dataset collection."""
    print(f"\n--- Statistics for Dataset: {dataset_name_for_print} (avg \u00B1 std per algorithm) ---")

    value_cols_map = {
        "build_time_ms": "Build Time (ms)",
        "query_time_ms": "Query Time (ms)",
        "accuracy_percent": "Accuracy (%)",
        "avg_dist_ratio": "Avg. Dist. Ratio",
        "space_bytes": "Space (Bytes)"
    }
    
    unique_algorithms = df_dataset['algorithm_name'].unique()
    summary_table_data_for_dataset = []

    for alg_name in unique_algorithms:
        alg_df = df_dataset[df_dataset['algorithm_name'] == alg_name].copy()
        print(f"\n  Algorithm: {alg_name} (Dataset: {dataset_name_for_print})")
        
        alg_summary_row = {"Algorithm": alg_name}

        for col_key, col_print_name in value_cols_map.items():
            valid_data = alg_df[col_key].dropna() 
            
            mean_val_str = "N/A"
            std_val_str = "N/A"
            combined_str = "N/A"

            if not valid_data.empty:
                mean_val = valid_data.mean()
                std_val = valid_data.std(ddof=0) 
                
                if col_key == "space_bytes":
                    if mean_val > 1024 * 1024 * 1024: # GB
                        mean_val_disp, std_val_disp, unit = mean_val / (1024*1024*1024), std_val / (1024*1024*1024), "GB"
                    elif mean_val > 1024 * 1024: # MB
                        mean_val_disp, std_val_disp, unit = mean_val / (1024*1024), std_val / (1024*1024), "MB"
                    elif mean_val > 1024: # KB
                        mean_val_disp, std_val_disp, unit = mean_val / 1024, std_val / 1024, "KB"
                    else: # Bytes
                        mean_val_disp, std_val_disp, unit = mean_val, std_val, "B"
                    combined_str = f"{mean_val_disp:.2f} \u00B1 {std_val_disp:.2f} {unit}"
                else:
                    combined_str = f"{mean_val:.4f} \u00B1 {std_val:.4f}"
                
                alg_summary_row[col_print_name] = combined_str
            else:
                alg_summary_row[col_print_name] = "No Data"

            print(f"    {col_print_name}: {combined_str}")
            
            if col_key in ["accuracy_percent", "avg_dist_ratio"]:
                num_nan_original = alg_df[col_key].isnull().sum()
                if num_nan_original > 0:
                    print(f"      ({num_nan_original} out of {len(alg_df)} files had NaN for this metric, excluded from avg/std)")
        
        summary_table_data_for_dataset.append(alg_summary_row)
    
    return summary_table_data_for_dataset

def analyze_results_main(results_file="results.txt"):
    try:
        df = pd.read_csv(results_file)
    except FileNotFoundError:
        print(f"Error: Results file '{results_file}' not found.")
        print("Please run the C++ program first to generate it.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Results file '{results_file}' is empty.")
        return

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    value_cols = ["build_time_ms", "query_time_ms", "accuracy_percent", "avg_dist_ratio", "space_bytes"]
    for col in value_cols:
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Expected column '{col}' not found in results file.")

    # Handle ERROR_OPENING_FILE entries
    error_mask = df['algorithm_name'] == 'ERROR_OPENING_FILE'
    error_df = df[error_mask]
    if not error_df.empty:
        print("\n--- File Opening Errors ---")
        for dataset_name, group in error_df.groupby('dataset_name'):
            print(f"  Dataset: {dataset_name}, Files with errors: {len(group)} (IDs: {sorted(list(group['file_id'].unique()))})")
    
    # Filter out error rows for numerical analysis
    df_valid = df[~error_mask].copy() # Use .copy() to avoid SettingWithCopyWarning
    if df_valid.empty:
        print("\nNo valid data rows found after filtering out errors. Cannot proceed with analysis.")
        return

    all_datasets_summary_tables = []
    unique_dataset_names = df_valid['dataset_name'].unique()

    for ds_name in unique_dataset_names:
        current_dataset_df = df_valid[df_valid['dataset_name'] == ds_name].copy()
        if current_dataset_df.empty:
            print(f"\nNo valid data for dataset: {ds_name}")
            continue
        summary_data = analyze_results_per_dataset(current_dataset_df, ds_name)
        if summary_data:
            summary_df_for_ds = pd.DataFrame(summary_data)
            # Add dataset name for multi-dataset summary table
            summary_df_for_ds.insert(0, 'Dataset', ds_name)
            all_datasets_summary_tables.append(summary_df_for_ds)

    print("\n\n--- Overall Summary Table (avg \u00B1 std per Algorithm per Dataset) ---")
    if all_datasets_summary_tables:
        overall_summary_df = pd.concat(all_datasets_summary_tables, ignore_index=True)
        value_cols_map_names = ["Build Time (ms)", "Query Time (ms)", "Accuracy (%)", "Avg. Dist. Ratio", "Space (Bytes)"]
        cols_ordered = ["Dataset", "Algorithm"] + value_cols_map_names
        cols_present_in_df = [col for col in cols_ordered if col in overall_summary_df.columns]
        print(overall_summary_df[cols_present_in_df].to_string(index=False))
    else:
        print("No data to display in overall summary table.")

if __name__ == "__main__":
    analyze_results_main()
