from tkinter import Tk, filedialog
from typing import Optional, Union
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
import pandas as pd
import numpy as np
import joblib
import os

def open_dataset(purpose: str | None,
                 rows_to_show: Optional[int] = 10) -> pd.DataFrame:
    root = Tk()
    root.withdraw()
    
    try:
        df_path = filedialog.askopenfilename(
            title=f"Select the CSV file for {purpose}",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not df_path:
            raise FileNotFoundError(f"\nNo CSV file was selected!, exiting...")
        else:
            df_name = os.path.basename(df_path)
            print(f"\n{df_name} dataset loaded successfully!")
    except FileExistsError:
        print(f"\nError type:FileExistsError, Error: Could not find the csv file")
        return
    
    root.destroy()
    df = pd.read_csv(df_path)

    if "ID" in df.columns:
        df.set_index("ID", inplace=True)
        print("\nID column now is the dataset index!")

    print(f"\n===== first {rows_to_show} rows of the dataset =====\n\n{df.head(rows_to_show)}") 
    print(f"\ndataset dimensions: {df.shape[0]} rows | {df.shape[1]} columns")

    return df

def split_data(df: pd.DataFrame,
                validation_size : float,
                test_size: float,
                target_col: Optional[str] = None,
                ):
    if target_col is not None and target_col not in df.columns:
        raise KeyError(f"Target column: {target_col} is not in the dataset!")
    
    if target_col is None:
        target_col = df.columns[-1] # if not target col the default is the last col of df.
        print(f"\nNo target column specified. Using '{target_col}' as default target.")

    if test_size + validation_size >= 1.0:
        raise ValueError("The sum of test_size and validation_size must be less than 1.0")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    val_ratio = validation_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
    )

    print(f"\nX_train: {len(X_train)} | y_train: {len(y_train)}")
    print(f"X_val: {len(X_val)} | y_val: {len(y_val)}, (this is {val_ratio} of validation ratio!)")
    print(f"X_test: {len(X_test)} | y_test: {len(y_test)}")

    # Distribution of the clases
    for name, arr in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        labels, counts = np.unique(arr, return_counts=True)
        print(f"\nDistribution in {name}:")
        for label, count in zip(labels, counts):
            print(f"Label {label}: {count} Samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_checkpoint(path: str):
    ckpt = joblib.load(path)
    model = ckpt["model"]
    hyperparams = ckpt["hyperparams"]
    print(f"\nLoaded checkpoint from {path}\nwith hyperparams: {hyperparams}\n")
    
    return model, hyperparams

