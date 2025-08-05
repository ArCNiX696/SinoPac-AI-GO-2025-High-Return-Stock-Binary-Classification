from tkinter import Tk, filedialog
from typing import Optional, Union
import pandas as pd
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
    print(f"\n===== first {rows_to_show} rows of the dataset =====\n\n{df.head(rows_to_show)}") 
    print(f"\ndataset dimensions: {df.shape[0]} rows | {df.shape[1]} columns")

    return df

def load_checkpoint(path: str):
    ckpt = joblib.load(path)
    model = ckpt["model"]
    hyperparams = ckpt["hyperparams"]
    print(f"\nLoaded checkpoint from {path}\nwith hyperparams: {hyperparams}\n")
    
    return model, hyperparams

