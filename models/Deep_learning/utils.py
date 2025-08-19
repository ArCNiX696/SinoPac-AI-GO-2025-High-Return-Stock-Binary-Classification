import torch
from tkinter import Tk, filedialog
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import os

def open_dataset(purpose: str | None,
                 df_path: str| None = None,
                 rows_to_show: Optional[int] = 10) -> pd.DataFrame:
    root = Tk()
    root.withdraw()
    
    try:
        if df_path is None:
            df_path = filedialog.askopenfilename(
                title=f"Select the CSV file for {purpose}",
                filetypes=[("CSV Files", "*.csv")]
            )
        if not df_path:
            raise FileNotFoundError(f"\nNo CSV file was selected!, exiting...")
        else:
            df_name = os.path.basename(df_path)
            print(f"\n{df_name} dataset loaded successfully!")
            # print(df_path)
            # exit()
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
               target_col: Optional[str] = None
               ):
    if target_col is not None and target_col not in df.columns:
        raise KeyError(f"Target column: {target_col} is not in the dataset!")
    
    if target_col is None:
        target_col = df.columns[-1] # if not target col the default is the last col of df.
        print(f"\nNo target column specified. Using '{target_col}' as default target.")

    if test_size + validation_size >= 1.0:
        raise ValueError("The sum of test_size and validation_size must be less than 1.0")

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

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

def inspect_loader(loader, name):
    for xb, yb in loader:
        print(f"\n=== {name} Loader ===")
        print(f"Batch X shape: {xb.shape} | Batch y shape: {yb.shape}")
        print(f"First 5 y values: {yb[:5].cpu().numpy()}")

        # stadistics
        print(f"X min:  {xb.min().item():.4f}")
        print(f"X max:  {xb.max().item():.4f}")
        print(f"X mean: {xb.mean().item():.4f}")

        # first 5 rows of the batch
        print("\nFirst 5 rows of X batch:\n", xb[:5].cpu().numpy())
        break  # Inspect only the first bacht

def load_data(purpose: str,
              validation_size : float,
              test_size: float,
              batch_size: int,
              df_path: str| None = None,
              target_col: Optional[str] = None,
              test_split: bool = False,
              test_scaler_path: Optional[str] = None,
              scale: bool = True,
              scaler_method: str = "minmax",
              scaler_dir: Optional[str] = None,
              model_name: Optional[str] = None):
    # --- Open and split the input dataset. ---
    df = open_dataset(purpose, df_path=df_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df,
                                                                validation_size,
                                                                test_size,
                                                                target_col)
    if test_split:
        print(f"\n{'=' * 80}\nScale test split option activated!")

        if test_scaler_path is None:
            test_scaler_path = filedialog.askopenfilename(
                title="Select the scaler to transform the test split",
                filetypes=[("Scaler files", "*.pkl *.joblib")]
            )
            if not test_scaler_path:
                print("\nNo Scaler file selected!, exiting ...")
                return
        else:
            if not os.path.exists(test_scaler_path):
                print(f"\nScaler file not found in: {test_scaler_path}")
                return

        # --- in case the file is corrupted ---
        try:
            test_scaler = joblib.load(test_scaler_path)
            print("Test scaler loaded successfully!")
        except Exception as e:
            print(f"\nError loading scaler: {e}")
            return
        
        # --- load the test scaler ---
        test_scaler = joblib.load(test_scaler_path)
        print("Test scaler loaded successfully!")

        # --- Scale, reshape and load the test split only ---
        X_test = test_scaler.transform(X_test)
        print(f"\nScalating test data...")

        X_test  = torch.tensor(X_test,  dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1) 
        
        print(f"\n{'=' * 60}")
        print(f"\nTest data shape:")
        print(f"\nX_test:\n{X_test.shape}  | y_test: {y_test.shape}")
        print(f"\n{'=' * 60}")

        test_ds = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
        inspect_loader(test_loader, "Test")
        
        return test_loader

    # --- Normalize the input data and save the scaler ---
    if scale:
        if scaler_method == "minmax":
            print(f"\nNormalizing input data...")
            scaler = MinMaxScaler()
            scaler_name = f"{model_name}_MinMaxScaler.pkl"
        elif scaler_method == "standard":
            print(f"\nStandardizing input data...")
            scaler = StandardScaler()
            scaler_name = f"{model_name}_StandardScaler.pkl"
        else:
            raise ValueError("scaler_method must be 'standard' or 'minmax'")

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # --- Save the scaler ---
        if scaler_dir and model_name:
            scaler_dir = os.path.dirname(scaler_dir)
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, scaler_name)
            joblib.dump(scaler, scaler_path)
            print(f"{scaler_name} file save in --> {scaler_dir}")
    
    # --- reshape X and y splits. (float for nn.BCEwithLogitsLoss) binary classification. ---
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) 
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1) 
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1) 
    
    print(f"\n{'=' * 60}")
    print(f"\nInput data shape:")
    print(f"\nX_train:{X_train.shape} | y_train: {y_train.shape}")
    print(f"\nX_val:\n{X_val.shape} | y_val: {y_val.shape}")
    print(f"\nX_test:\n{X_test.shape}  | y_test: {y_test.shape}")
    print(f"\n{'=' * 60}")

    # --- create the dataset loaders and inspect them. ---
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    inspect_loader(train_loader, "Train")
    inspect_loader(val_loader, "Validation")
    inspect_loader(test_loader, "Test")

    return train_loader, val_loader, test_loader

def load_checkpoint(path: str):
    ckpt = joblib.load(path)
    model = ckpt["model"]
    hyperparams = ckpt["hyperparams"]
    print(f"\nLoaded checkpoint from {path}\nwith hyperparams: {hyperparams}\n")
    
    return model, hyperparams

if __name__ == "__main__":
    df = open_dataset(purpose="in utils test")
    load_data(df,
              validation_size=0.2,
              test_size=0.1,
              scaler_method="minmax",
              scaler_dir="./stats/Deep_learning/mlp_classifier/",
              batch_size=32,
              model_name="mlp_classifier")

