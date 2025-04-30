import pandas as pd
import os
from tkinter import filedialog
import argparse

#---------------------------------------- Dataset preview ---------------------------------------#                 
"""
1. since the size of the dataset is more then 12 GB, we can preview it without open it
"""

# *** Open, previwew, cut & export, etc... *** #
def open_csv(chunk: int = None,
             dataset_dim: bool= False,
             target_desc: str = None,
             all_names: bool = False,
             create_balanced: bool = False,
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

    if create_balanced:
        balanced_dataset(file_path=file_path,
                         target_col='飆股',
                         save=export)

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
        return

def balanced_dataset(file_path: str,
                     target_col: str,
                     save: bool = True,
                     chunksize: int = 1000) -> pd.DataFrame:
    """
    Creates a balanced dataset by:
    - Taking all samples of the minority class.
    - Selecting an equal number of majority class samples with the fewest nulls.

    Args:
        file_path (str): Path to the original CSV.
        target_col (str): Name of the target column with class labels.
        save (bool): Whether to export the balanced dataset to CSV.
        chunksize (int): Number of rows to read per chunk.

    Returns:
        pd.DataFrame: The balanced dataset.
    """
    print("-"*90)
    print("Function ---> balanced_dataset\n")
    print(f"Reading in chunks of {chunksize} rows...")
    
    # First pass: count samples per class
    class_counts = {}
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        for cls, cnt in chunk[target_col].value_counts().to_dict().items():
            class_counts[cls] = class_counts.get(cls, 0) + cnt
    
    if len(class_counts) < 2:
        raise ValueError(f"Only one class found in '{target_col}'.")
    
    # Determine minority class and its size
    cls_counts_series = pd.Series(class_counts)
    minority_class = cls_counts_series.idxmin()
    minority_n = cls_counts_series.min()
    print(f"Minority class: {minority_class} with {minority_n} samples")
    
    # Collect minority samples and maintain top majority samples
    minority_list = []
    majority_best = pd.DataFrame()
    
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        # Minority rows
        min_chunk = chunk[chunk[target_col] == minority_class]
        if not min_chunk.empty:
            minority_list.append(min_chunk)
        # Majority rows
        maj_chunk = chunk[chunk[target_col] != minority_class].copy()
        if not maj_chunk.empty:
            maj_chunk['_null_count'] = maj_chunk.isnull().sum(axis=1)
            combined = pd.concat([majority_best, maj_chunk])
            # Keep rows with fewest nulls
            majority_best = combined.nsmallest(minority_n, '_null_count').drop(columns=['_null_count'])

        print(f"---> chunk number {chunk} finished!")
    # Concatenate all minority and sampled majority
    minority_df = pd.concat(minority_list)
    if len(minority_df) > minority_n:
        minority_df = minority_df.sample(n=minority_n, random_state=42)
    
    balanced = pd.concat([minority_df, majority_best]).reset_index(drop=True)
    print(f"Balanced dataset size: {balanced.shape[0]} rows ({minority_n} per class)")
    
    # Export if requested
    if save:
        base = os.path.splitext(os.path.basename(file_path))[0]
        directory = os.path.dirname(file_path)
        out_path = os.path.join(directory, f"{base}_balanced.csv")
        balanced.to_csv(out_path, index=False)
        print(f"Balanced dataset saved at: {out_path}")
    
    print("-"*90)
    return balanced
        
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

def main(args):
    open_csv(
        chunk=args.chunk,
        dataset_dim=args.dataset_dim,
        target_desc=args.target_desc,
        all_names=args.all_names,
        create_balanced=args.create_balanced,
        split_by_feature_type=args.split_by_feature_type,
        export=args.export
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Load and preprocess a large CSV with various options."
    )
    parser.add_argument(
        "--chunk", type=int, default=None,
        help="Number of rows to read as a chunk (for preview)."
    )
    parser.add_argument(
        "--dataset_dim", type=bool, default=True,
        help="If set, only print dataset dimensions and exit."
    )
    parser.add_argument(
        "--target_desc", type=str, default="飆股",
        help="Name of the target/label column to analyze (e.g. '飆股')."
    )
    parser.add_argument(
        "--all_names",type=bool, default=False,
        help="If set, print all column names instead of a summary."
    )
    parser.add_argument(
        "--create_balanced", type=bool, default=False,
        help="If set, build and optionally save a balanced subset."
    )
    parser.add_argument(
        "--split_by_feature_type",type=bool, default=False,
        help="If set, split features by type (technical, time series, cleaned)."
    )
    parser.add_argument(
        "--export",type=bool, default=False,
        help="If set, export any generated datasets to CSV."
    )

    args = parser.parse_args()
    main(args)


