import pandas as pd
import os
from tkinter import filedialog

#---------------------------------------- Dataset preview ---------------------------------------#                 
"""
1. since the size of the dataset is more then 12 GB, we can preview it without open it
"""

# *** Open, previwew, cut & export, etc... *** #
def open_csv(chunk: int = None,
             dataset_dim: bool= False,
             target_desc: str = None,
             all_names: bool = False,
             split_by_feature_type: bool = True,
             export: bool = False) -> pd.DataFrame:
    print(f"{'-'*90}")
    print(f"Function ---> open_csv\n\nSummary:")
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

    if not file_path:
        raise SystemExit("\n---> open_csv Error 1: CSV file not loaded, try again!")
    else:
        file_name = os.path.basename(file_path)
        print(f"\nCSV file: {file_name} loaded successfully!")
 
    directory = os.path.dirname(file_path)
    print(f"File path: {file_path}")
    print(f"Directory: {directory}")

    # count row & columns.
    if dataset_dim:
        with open(file_path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for line in f) - 1
        
        col_names = pd.read_csv(file_path, nrows=0).columns
        col_count = len(col_names)
        print(f"\n---> The dataset has {row_count} rows & {col_count} columns.")

        if target_desc:
            print(f"\n---> Analyzing target column: '{target_desc}'")
            df_target = pd.read_csv(file_path, usecols=[target_desc])
            
            if target_desc not in df_target.columns:
                raise ValueError(f"\n---> Error: Column '{target_desc}' not found in the dataset.")
            
            class_counts = df_target[target_desc].value_counts().sort_index()
            num_classes = class_counts.shape[0]
            print(f"---> Number of unique classes: {num_classes}")
            print(f"---> Samples per class:")
            
            for cls, count in class_counts.items():
                print(f"     Class {cls}: {count} samples")

            # # print(f"{target_desc} column has {df_target[target_desc].isnull().sum()} of nulls values or empty spaces.")

            # print(f"---> Analyzing nulls vals or empty spaces in the dataset.")
            # df = pd.read_csv(file_path)
            # print(f"The dataset has a total of {df.isnull().sum()} nulls values or empty spaces.")
            
        if all_names:
            print(f"\n---> Dataset features(all):\n{col_names.to_list()}")
        else:
            print(f"\n---> Dataset features:\n{col_names}")
        raise SystemExit(f"\ndataset_dim visualization finished, exiting... ")
    
    # cut dataset.
    if chunk is not None:
        print(f"\n---> Processing first {chunk} rows using chunksize...\n")

        chunk_iter = pd.read_csv(file_path, chunksize=chunk)
        first_chunk = next(chunk_iter)
        df = first_chunk

        # export.
        if export:
            base_name = os.path.splitext(file_name)[0]
            output_filename = f"{base_name}_{chunk}_rows.csv"
            export_path = os.path.join(directory, output_filename)
            df.to_csv(export_path, index=False)
            print(f"\nProcessed CSV saved at: {export_path}")

        return df
    
    if split_by_feature_type:
        df_cleaned, df_technical, df_lag = separate_feature_type(file_path= file_path,
                                                           save=export)
        return df_cleaned
    
    else:
        if df is not None:
            return df
        
def separate_feature_type(file_path: str, save: bool = True) -> tuple:
    print(f"{'-'*90}")
    print("Function ---> separate_feature_type\n\nSummary:")

    # Load dataset
    df = pd.read_csv(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    directory = os.path.dirname(file_path)

    print(f"\nTotal columns before processing: {df.shape[1]}")

    # Ensure the target column exists
    if '飆股' not in df.columns:
        raise ValueError("\nTarget column '飆股' not found in dataset.")

    target_col = '飆股'

    # Identify technical analysis columns (summarized indicators, not sliding windows)
    technical_cols = [col for col in df.columns if (
        ('天' in col or '日' in col) and
        any(keyword in col for keyword in ['波動度', '報酬率', '乖離率']) and
        '前' not in col
    )]

    df_technical = df[technical_cols + [target_col]].copy()
    print(f"Found {len(technical_cols)} technical analysis features.")

    # Identify lag features (sliding windows like '前1天', '前2天', etc.)
    lag_cols = [col for col in df.columns if '前' in col and any(str(n) in col for n in range(1, 21))]
    print(f"\nFound {len(lag_cols)} raw time-series (lag) features.")

    # From each group of lag features, keep only the '前1天' version
    base_feature_map = {}
    for col in lag_cols:
        if '前1天' in col:
            base = col.replace('前1天', '')
            base_feature_map[base] = col

    selected_lag_cols = list(base_feature_map.values())
    df_lag = df[selected_lag_cols + [target_col]].copy()
    print(f"Selected {len(selected_lag_cols)} '前1天' time-series features.")

    # Create cleaned dataset (excluding both technical and lag features)
    cols_to_remove = technical_cols + lag_cols
    remaining_cols = [col for col in df.columns if col not in cols_to_remove and col != target_col]
    df_cleaned = df[remaining_cols + [target_col]].copy()
    print(f"\nColumns remaining in cleaned dataset: {df_cleaned.shape[1]}")

    # Save all resulting datasets if requested
    if save:
        tech_path = os.path.join(directory, f"{base_name}_analisis_tecnico.csv")
        lag_path = os.path.join(directory, f"{base_name}_time_series.csv")
        clean_path = os.path.join(directory, f"{base_name}_cleaned.csv")

        df_technical.to_csv(tech_path, index=False)
        df_lag.to_csv(lag_path, index=False)
        df_cleaned.to_csv(clean_path, index=False)

        print(f"\n--> Technical analysis dataset saved at: {tech_path}")
        print(f"--> Time-series dataset saved at: {lag_path}")
        print(f"--> Cleaned dataset saved at: {clean_path}")

    print(f"{'-'*90}")
    return df_cleaned, df_technical, df_lag

if __name__ == '__main__':
    open_csv(chunk=None,
             dataset_dim=False,
             target_desc="飆股",
             all_names=False,
             split_by_feature_type=True,
             export=True)
