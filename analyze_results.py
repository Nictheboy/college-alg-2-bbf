import pandas as pd
import numpy as np

def analyze_results_simplified(results_file="results.txt"):
    try:
        df = pd.read_csv(results_file)
    except FileNotFoundError:
        print(f"Error: Results file '{results_file}' not found.")
        print("Please run the C++ program first to generate it.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Results file '{results_file}' is empty.")
        return

    # Handle potential NaN strings from C++ double output
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['avg_dist_ratio'] = pd.to_numeric(df['avg_dist_ratio'], errors='coerce')


    print("--- Statistics (avg ± std per algorithm) ---")

    value_cols_map = {
        "build_time_ms": "Build Time (ms)",
        "query_time_ms": "Query Time (ms)",
        "accuracy_percent": "Accuracy (%)",
        "avg_dist_ratio": "Avg. Dist. Ratio", # Added average distance ratio
        "space_bytes": "Space (Bytes)"
    }
    
    unique_algorithms = df['algorithm_name'].unique()
    summary_table_data = []

    for alg_name in unique_algorithms:
        alg_df = df[df['algorithm_name'] == alg_name].copy()
        print(f"\nAlgorithm: {alg_name}")
        
        alg_summary_row = {"Algorithm": alg_name}

        for col_key, col_print_name in value_cols_map.items():
            # For avg_dist_ratio, NaNs from C++ (if any) are already handled by pd.to_numeric
            valid_data = alg_df[col_key].dropna() 
            
            mean_val_str = "N/A" # Default for print
            std_val_str = "N/A"  # Default for print
            combined_str = "N/A" # Default for table and print

            if not valid_data.empty:
                mean_val = valid_data.mean()
                std_val = valid_data.std(ddof=0) 
                
                if col_key == "space_bytes":
                    # ... (space formatting logic same as before) ...
                    if mean_val > 1024 * 1024: # MB
                        mean_val_disp, std_val_disp, unit = mean_val / (1024*1024), std_val / (1024*1024), "MB"
                    elif mean_val > 1024: # KB
                        mean_val_disp, std_val_disp, unit = mean_val / 1024, std_val / 1024, "KB"
                    else: # Bytes
                        mean_val_disp, std_val_disp, unit = mean_val, std_val, "B"
                    combined_str = f"{mean_val_disp:.2f} ± {std_val_disp:.2f} {unit}"
                else: # For time, accuracy, and avg_dist_ratio
                    combined_str = f"{mean_val:.4f} ± {std_val:.4f}"
                
                alg_summary_row[col_print_name] = combined_str
            else:
                alg_summary_row[col_print_name] = "No Data" # Or np.nan for internal, "No Data" for table

            print(f"  {col_print_name}: {combined_str}")
            
            # Additional info for metrics that might have NaNs from C++
            if col_key in ["accuracy_percent", "avg_dist_ratio"]:
                num_nan_original = alg_df[col_key].isnull().sum() # Count NaNs before dropna()
                if num_nan_original > 0:
                     # This message reflects NaNs present in the input file for this column for this alg
                    print(f"    ({num_nan_original} out of {len(alg_df)} datasets had NaN for this metric, excluded from avg/std)")
        
        summary_table_data.append(alg_summary_row)

    print("\n\n--- Summary Table (avg ± std) ---")
    if summary_table_data:
        summary_df = pd.DataFrame(summary_table_data)
        cols_ordered = ["Algorithm"] + [name for name in value_cols_map.values()]
        cols_present_in_df = [col for col in cols_ordered if col in summary_df.columns]
        print(summary_df[cols_present_in_df].to_string(index=False))
    else:
        print("No data to display in summary table.")

if __name__ == "__main__":
    analyze_results_simplified()
