from typing import List, Dict, Optional, Union, Tuple, Any
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import json
import pandas as pd
import os
import argparse
import numpy as np
from arch import arch_model
from tkinter import filedialog

"""
NOTE: Since Yahoo Finance no longer allows free downloads,
   this script focuses only on preprocessing. The data will be obtained from Nasdaq.com.
"""

#---------------------------------------- Preprocessing code ---------------------------------------#
class Preprocessing:
    def __init__(self,
                 args: argparse.Namespace) -> None:
        self.args = args
        self.verbose = args.verbose

        #** Paths. **#
        self.open_path = args.open_path
        self.training_path = args.training_path
        self.prediction_path = args.prediction_path
        self.save = args.save
        self.jason_path = args.jason_path

        #** Windows and technical and others. **#
        self.windows = args.windows
        self.technical = args.technical
        self.Garch = args.GARCH
        self.forMoE = args.forMoE

        #** Scale dataset. **#
        self.normalization = args.normalization
        self.normalizer = MinMaxScaler()
        self.standardization = args.standardization
        self.scaler = StandardScaler()

#---------------------------------------- Open Dataset ---------------------------------------#                 
    def open_csv(self) -> pd.DataFrame: 
        """
        because we are using datasets from Nasdaq.com,
        this function is not only for opening the input dataset but for
        filtering $ usd simbols in the columns with price, and also
        for renaming the column "Close/Last" as "Adj Close".
        """
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        
        if not file_path:
            raise SystemExit(f"\n---> open_csv Error 1: csv file of not loaded, try again!")
        else:
            file = os.path.basename(file_path)
            print(f"\ncvs file: {file} loaded successfully!")
            
        # ** get the name of the file which assuming is the name of the stock, to rename the proccessed dataset.
        if file:
            self.filename = os.path.splitext(file)[0]
            print(f"\nfile name: {self.filename} | assuming is the name of the stock, to rename the proccessed dataset. ")
            
        df = pd.read_csv(file_path)
        df.rename(columns={"Close/Last": "Adj Close"}, inplace=True)

        price_columns = ["Adj Close", "Open", "High", "Low"]  # Lists of columns with "$"
        for col in price_columns:
            df[col] = df[col].astype(str).str.replace("$", "", regex=False).astype(float)

        if "Volume" in df.columns: # this is the only col as int64 type, to avoid future warnings and problems should be converted.
            df["Volume"] = df["Volume"].astype("float64")
            print('*' * 50)
            print(f"\n---> ✅ Column Volume was converted to float64 to avoid future problems and warnings.\n")
            print('*' * 50)

        # Sort the dataset by date in ascending order (oldest first, newest last)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])  
            df = df.sort_values(by="Date", ascending=True)  
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d") 

        if self.verbose == 1:
            print(f"\n{'=' * 90}")
            print(f'\n1).open_csv | input data visualization:\n\n{df.head(10)}') 
            
        return df

#---------------------------------------- Process Cols and Rows ---------------------------------------#
    def set_col_index(self,
                      df: pd.DataFrame,
                      column_name: str = 'Date'):
        
        df.set_index([column_name] , inplace=True)

        print('*' * 50)
        print(f"\n---> Column: {column_name} was set as the index of the dataset.\n")
        print('*' * 50)
        return df
    
    def move_col_to_end(self,
                        df: pd.DataFrame,
                        col_name: str = 'Adj Close'):
        
        cols = [col for col in df.columns if col != col_name] + [col_name]  
        print('*' * 50)
        print(f"\n---> Column: {col_name} was moved at the final column of the dataset.\n")
        print('*' * 50)
        return df[cols]
    
    def split_data(self,
                   df: pd.DataFrame,
                   months: int = 1,
                   c: int = 0):
        
        if df.empty:
            raise SystemExit(f"\n🛑---> split_data func. Error: the dataset is empty or None!")
        
        days = (months * 22) + c  # calculate only an approx. of operational days of the stock market.
        rows_from = max(len(df) - days, 0)
        df_train = df.iloc[:rows_from, :]
        df_test = df.iloc[rows_from:, :]

        if self.forMoE:
            n_train = len(df_train)
            subset_size = n_train // self.forMoE
            remainder = n_train % self.forMoE
            subsets = {}
            start_idx = 0

            for i in range(self.forMoE):

                if i == self.forMoE -1:
                    end_idx = start_idx + subset_size + remainder
                else:
                    end_idx = start_idx + subset_size
                
                subsets[f"Subset {i+1}"] = df_train.iloc[start_idx:end_idx, :]

                if self.verbose:
                    print(f"---> 📝 Subset {i+1}: rows from {start_idx} to {end_idx -1} | total of rows: {end_idx - start_idx}")
                
                start_idx = end_idx

            if self.verbose:
                print(f"\n{'='*100}")
                print(f"📊 Split Dataset Summary 📊")
                print(f"{'-'*100}")
                print(f"📌 Original dataset: {len(df)} rows")
                print(f"✅ Training set: {len(df_train)} rows divided by {self.forMoE} subsets")
                print(f"🧪 Test set: {len(df_test)} rows (From index {rows_from} to {len(df) - 1})")
                print(f"{'-'*100}\n")

            return subsets, df_test

        else:
            if c != 0:
                print(f"\n---> WARNING ⚠️: a compensation of {c} days was made in order to match the date with other datasets.")  
                print(f"\nif this action was not intentionally just set c = 0.")

            if self.verbose:
                print(f"\n{'='*100}")
                print(f"📊 Split Dataset Summary 📊")
                print(f"{'-'*100}")
                print(f"📌 Original dataset: {len(df)} rows")
                print(f"✅ Training set: {len(df_train)} rows (From index 0 to {rows_from - 1})")
                print(f"🧪 Test set: {len(df_test)} rows (From index {rows_from} to {len(df) - 1})")
                print(f"{'-'*100}")
                
                print(f"\n📝 First 10 rows of Training Set:\n{df_train.head(10)}")
                print(f"\n📝 Last 10 rows of Training Set:\n{df_train.tail(10)}")

                print(f"\n📝 First 10 rows of Test Set:\n{df_test.head(10)}")
                print(f"\n📝 Last 10 rows of Test Set:\n{df_test.tail(10)}")
                print(f"{'='*100}\n")

        return df_train, df_test
    
    def extract_dates(self,
                      df: pd.DataFrame,
                      mode: str = 'train') -> Dict[str, str]:
        if df.index.name != "Date":
            raise SystemExit("🛑--->extract_dates func. Error: 'Date' is not set as the index.")
        
        first_date = df.index[0]
        last_date = df.index[-1]

        first_date = pd.to_datetime(first_date).strftime("%Y-%m-%d")
        last_date = pd.to_datetime(last_date).strftime("%Y-%m-%d")
        date_range = {"start_date": first_date, "end_date": last_date}

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"📅 {mode} dataset Date Range Extracted:")
            print(f"🟢 Start Date: {first_date}")
            print(f"🔴 End Date: {last_date}")
            print(f"{'='*50}\n")

        return date_range
    
    def isolate_cols(self,
                    df: pd.DataFrame,
                    cols_to_isolate: Optional [Union[list[str], str]]):
        
        if isinstance(cols_to_isolate, list):
            cols_to_isolate = [cols_to_isolate]

        df_copy = df[[cols_to_isolate]].copy()

        if self.verbose:
            print(f"\n{'=' * 90}\n---> ✅ applying isolate_cols function:\n")
            print(f'\ndataset first 10 rows with the isolate cols :\n{df_copy.head(10)}\n')
            print(f'\ndataset last 10 rows with the isolate cols :\n{df_copy.tail(10)}\n')

        return df_copy 
    
    def drop_columns(self,
                     df: pd.DataFrame,
                     columns_name: str):
        
        if isinstance(columns_name, str):
            columns_name = [columns_name]
        
        for col in columns_name:
            df.drop(col,axis=1, inplace=True)
        
        if self.verbose:
            print('*' * 50)
            print(f"\n---> ✅ applying drop_columns function, column: {columns_name} was removed from the dataset!\n")
            print('*' * 50)
        
        return df

#---------------------------------------- Scale Dataset ---------------------------------------#    
    def normalize(self,
                  df: pd.DataFrame,
                  start_col: Optional[str] = None,
                  end_col: Optional[Union[str, List[str]]] = None,
                  dates: Optional[Dict[str, str]] = None,
                  mode: str = "train") -> pd.DataFrame:
        
        df = df.copy()

        excluded_cols = None
        # Case 1: Normalize from start_col to end_col
        if start_col is not None and isinstance(end_col, str):
            assert start_col in df.columns and end_col in df.columns, f'\n{start_col} and/or {end_col} not found in the DataFrame\n'
            cols_to_normalize = df.loc[:, start_col:end_col].columns

        # Case 2: Exclude columns from normalization
        elif end_col is not None:
            if isinstance(end_col, str):
                end_col = [end_col]
            missing_cols = [col for col in end_col if col not in df.columns]# Check that all columns in end_col are in the DataFrame
            assert not missing_cols, f'\nThe following columns are not in the DataFrame: {missing_cols}\n'
            excluded_cols = df[end_col].copy()
            cols_to_normalize = df.columns.drop(end_col)

        # Case 3: Normalize the entire DataFrame
        else:
            cols_to_normalize = df.columns

        # Normalize the selected columns
        df_normalized = self.normalizer.fit_transform(df[cols_to_normalize])
        df.loc[:, cols_to_normalize] = pd.DataFrame(df_normalized, 
                                            columns=cols_to_normalize, 
                                            index=df.index)

        # ** for time series preproccessing ** #
        if mode == "test" and self.windows:
            normalization_values = {
                'Adj Close': {
                    "min": float(self.normalizer.data_min_[-1]),  # Last normalized column (Adj Close)
                    "max": float(self.normalizer.data_max_[-1])   # Last normalized columna (Adj Close)
                }
            }

            if dates:
                start_date = dates.get("start_date", None)
                end_date = dates.get("end_date", None)
            else:
                raise SystemExit(f"\n🛑---> normalize func. Error: No dates imported, please check it!")

            if self.save:
                # Save the min and max values to a JSON file
                json_save_path = os.path.join(self.jason_path, f"{self.filename}_time_series_{start_date}_to_{end_date}_{mode}.json")
                with open(json_save_path, 'w') as file:
                    json.dump(normalization_values, file, indent=4)
                    self.message = f"\n---> ✅ 📝 jason file saved in: {json_save_path}"

        # ** for technical analysis preproccessing ** #
        elif mode == "test" and self.technical:
            normalization_values = {
            col: {
                "min": float(self.normalizer.data_min_[i]),
                "max": float(self.normalizer.data_max_[i])  
            }for i, col in enumerate(cols_to_normalize)#The for loop at the final.
            }

            if dates:
                start_date = dates.get("start_date", None)
                end_date = dates.get("end_date", None)
            else:
                raise SystemExit(f"\n🛑---> normalize func. Error: No dates imported, please check it!")

            if self.save:
                # Save the min and max values to a JSON file
                json_save_path = os.path.join(self.jason_path, f"{self.filename}_technical_{start_date}_to_{end_date}_{mode}.json")
                with open(json_save_path, 'w') as file:
                    json.dump(normalization_values, file, indent=4)
                    self.message = f"\n---> ✅ 📝 jason file saved in: {json_save_path}"

        # ** for GARCH preproccessing ** #
        elif mode == "test" and self.Garch:
            normalization_values = {
                'Adj Close': {
                    "min": float(self.normalizer.data_min_[-1]),  # Última columna normalizada (Adj Close)
                    "max": float(self.normalizer.data_max_[-1])   # Última columna normalizada (Adj Close)
                }
            }
            
            if dates:
                start_date = dates.get("start_date", None)
                end_date = dates.get("end_date", None)
            else:
                raise SystemExit(f"\n🛑---> normalize func. Error: No dates imported, please check it!")

            if self.save:
                # Save the min and max values to a JSON file
                json_save_path = os.path.join(self.jason_path, f"{self.filename}_garch_{start_date}_to_{end_date}_{mode}.json")
                with open(json_save_path, 'w') as file:
                    json.dump(normalization_values, file, indent=4)
                    self.message = f"\n---> ✅ 📝 jason file saved in: {json_save_path}"

        # ** for a normal preproccessing ** #
        else:
            if mode == "test":
                normalization_values = {
                    'Adj Close': {
                        "min": float(self.normalizer.data_min_[-1]),  # Última columna normalizada (Adj Close)
                        "max": float(self.normalizer.data_max_[-1])   # Última columna normalizada (Adj Close)
                    }
                }
            
                if dates:
                    start_date = dates.get("start_date", None)
                    end_date = dates.get("end_date", None)
                else:
                    raise SystemExit(f"\n🛑---> normalize func. Error: No dates imported, please check it!")

                if self.save:
                    # Save the min and max values to a JSON file
                    json_save_path = os.path.join(self.jason_path, f"{self.filename}_normal_{start_date}_to_{end_date}_{mode}.json")
                    with open(json_save_path, 'w') as file:
                        json.dump(normalization_values, file, indent=4)
                        self.message = f"\n---> ✅ 📝 jason file saved in: {json_save_path}"
            
        # Restore the excluded columns after normalization
        if excluded_cols is not None:
            df[end_col] = excluded_cols

        if self.verbose:
            print(f"\n{'=' * 90}")
            print(f'\n2).normalize | normaliced {mode} data visualization:\n') 
            print(f'\nThis is the Normalized for {mode} dataset (first 10 rows):\n{df.head(10)}\n')
            print(f'\nThis is the Normalized for {mode} dataset (last 10 rows):\n{df.tail(10)}\n')
        return df

#---------------------------------------- Time series ---------------------------------------#     
    def windows_preprocessing(self,
                              df: pd.DataFrame,
                              windows_size: int,
                              cols_to_windows: Optional[Union[list[str], str]],
                              idx: bool = True):
        
        if isinstance(cols_to_windows, str):
            cols_to_windows = [cols_to_windows]

        index_dates = []
        df_copy = df[cols_to_windows].copy()

        for col in cols_to_windows:
            for i in range(1, windows_size +1):
                df_copy[f'{col}(t-{i})'] = df_copy[col].shift(i)

        df_copy.dropna(inplace=True)#Avoid empty or Nan values in the dataset.
        df_copy = df_copy.iloc[:, ::-1]#Flip columns

        if idx:
            for i in range(len(df_copy)):
                date_of_pred = df.index[i + windows_size]#Calculate the date till the windows size ends.
                index_dates.append(f"{date_of_pred}")
        else:
            index_dates = list(range(len(df_copy)))
        
        df_copy.index = index_dates
        df_copy.index.name = "Date"

        # if self.Garch:
        #     #Drop garch_vol col to avoid giving info about actual Adj close to the model.
        #     self.drop_columns(df_copy, 'garch_vol')

        if self.verbose == 2:
            print(f"\n{'=' * 90}\n3).windows_preprocessing | dataset visualization:\n")
            print(f'\nDataset  first 10 rows with windows size :{windows_size}:\n{df_copy.head(10)}\n')
            print(f'\nDataset  last 10 rows with windows size :{windows_size}:\n{df_copy.tail(10)}\n')

        return df_copy

#---------------------------------------- technical analysis ---------------------------------------#  
    # ** logaritmic return. ** #
    def log_return(self,
               df: pd.DataFrame,
               cols:Optional[Union[list[str], str]],
               window_size: int):
        
        if isinstance(cols, str):
            cols = [cols]  
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.intersection(cols)
        
        for col in numeric_cols:
            df[f'LogR_{col}_t-{window_size}'] = np.log(df[col] / df[col].shift(window_size))
        
        df.dropna(inplace=True)
        
        if self.verbose == 4:
            print(f"\n{'=' * 90}\n---> ✅ applying log_return:\n")
            print(f'\nDataset  first 10 rows after applying "log_return" func :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after applying "log_return" func :\n{df.tail(10)}\n')
        
        return df
    
    # ** Relative Strength Index (RSI) ** #
    def RSI(self,
            df: pd.DataFrame,
            col: str = 'Adj Close',
            periods: int = 14):
       
        delta = df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        df[f'RSI_{col}_{periods}_periods'] = 100 - (100 / (1 + rs))
        df.dropna(inplace=True)
        
        if self.verbose == 5:
            print(f"\n{'=' * 90}\n---> ✅ applying RSI (Relative Strength Index):\n")
            print(f'\nDataset  first 10 rows after applying "RSI" func :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after applying "RSI" func :\n{df.tail(10)}\n')
        
        return df
    
    # ** Momentum ratio ** #
    def momentum_ratio(self,
                       df: pd.DataFrame,
                       col: str = 'Adj Close',
                       periods: int = 14):
        
        df[f'Momentum_Ratio_{col}_{periods}d'] = (df[col] / df[col].shift(periods)) * 100
        df.dropna(inplace=True)
        
        if self.verbose == 6:
            print(f"\n{'=' * 90}\n---> ✅ applying momentum_ratio:\n")
            print(f'\nDataset  first 10 rows after applying "momentum_ratio" func :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after applying "momentum_ratio" func :\n{df.tail(10)}\n')
        
        return df
    
    # ** True Range (TR) ** #
    def true_range(self,
                   df: pd.DataFrame):
       
        previous_close = df['Adj Close'].shift(1)
    
        df['True_Range'] = df.apply(
            lambda row: max(
                row['High'] - row['Low'],
                abs(row['High'] - previous_close[row.name]),  # Gap to raise.
                abs(row['Low'] - previous_close[row.name])    # Gap to down.
            ), axis=1
        )
        df.reset_index(drop=True, inplace=True)
    
        if self.verbose == 7:
            print(f"\n{'=' * 90}\n---> ✅ applying true_range:\n")
            print(f'\nDataset  first 10 rows after applying "true_range" func :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after applying "true_range" func :\n{df.tail(10)}\n')
        
        return df
    
    # **Calculates the Average True Range (ATR) using the True Range (TR) ** #
    def average_true_range(self,
                           df: pd.DataFrame,
                           periods: int = 14):
       
        df = self.true_range(df)
        
        df['ATR'] = df['True_Range'].rolling(window=periods).mean()
        df.dropna(inplace=True)
     
        for i in range(periods, len(df)):
            previous_atr = df.at[i-1, 'ATR']
            current_tr = df.at[i, 'True_Range']
            df.at[i, 'ATR'] = ((previous_atr * (periods - 1)) + current_tr) / periods
        df.reset_index(drop=True, inplace=True)

        if self.verbose == 7:
            print(f"\n{'=' * 90}\n---> ✅ applying average_true_range:\n")
            print(f'\nDataset  first 10 rows after applying "average_true_range" func :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after applying "average_true_range" func :\n{df.tail(10)}\n')
        
        return df
    
    # ** Parabolic SAR ** #
    def parabolic_sar(self,
                      df: pd.DataFrame,
                      af_start: float = 0.02,
                      af_step: float = 0.02,
                      af_max: float = 0.2):
        # Initial settings
        af = af_start
        ep = df['Low'][0]
        psar = df['High'][0]
        psar_list = [psar]
        uptrend = True

        for i in range(1, len(df)):
            prev_psar = psar

            # Calculate the current PSAR
            if uptrend:
                psar = prev_psar + af * (ep - prev_psar)
            else:
                psar = prev_psar - af * (prev_psar - ep)
            
            # Determine the trend and update EP, AF
            if uptrend:
                if df['Low'][i] < psar:
                    uptrend = False
                    psar = ep  # Set PSAR to EP when trend reverses
                    ep = df['Low'][i]
                    af = af_start
                else:
                    if df['High'][i] > ep:
                        ep = df['High'][i]
                        af = min(af + af_step, af_max)
            else:
                if df['High'][i] > psar:
                    uptrend = True
                    psar = ep
                    ep = df['High'][i]
                    af = af_start
                else:
                    if df['Low'][i] < ep:
                        ep = df['Low'][i]
                        af = min(af + af_step, af_max)

            psar_list.append(psar)
        
        df['PSAR'] = psar_list

        if self.verbose == 8:
            print(f"\n{'=' * 90}\n---> ✅ applying parabolic_sar:\n")
            print(f'\nDataset first 10 rows after applying "Parabolic SAR":\n{df.head(10)}\n')
            print(f'\nDataset last 10 rows after applying "Parabolic SAR":\n{df.tail(10)}\n')

        return df
    
    def commodity_channel_index(self,
                                df: pd.DataFrame,
                                periods: int = 20):
        
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Adj Close']) / 3
        df['SMA_TP'] = df['Typical_Price'].rolling(window=periods).mean()
        df['Mean_Deviation'] = df['Typical_Price'].rolling(window=periods).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=False)
        df['CCI'] = (df['Typical_Price'] - df['SMA_TP']) / (0.015 * df['Mean_Deviation'])
        df.drop(columns=['Typical_Price', 'SMA_TP', 'Mean_Deviation'], inplace=True)
        df.dropna(inplace=True)
        
        if self.verbose == 9:
            print(f"\n{'=' * 90}\n---> ✅ applying commodity_channel_index:\n")
            print(f'\nDataset first 10 rows after applying "CCI":\n{df.head(10)}\n')
            print(f'\nDataset last 10 rows after applying "CCI":\n{df.tail(10)}\n')

        return df
    
    # ** Simple Moving Average. ** #
    def SMA(self,
            df: pd.DataFrame,
            cols: str = 'Adj Close',
            periods: int = 20):
        
        if isinstance(cols, str):
            cols = [cols]

        for col in cols:
            sma_series = df[col].rolling(window=periods).mean()
            sma_col_name = f'SMA_{col}_{periods}_periods'
            df[sma_col_name] = sma_series
        
        sma_columns = [f'SMA_{col}_{periods}_periods' for col in cols]
        df.dropna(subset=sma_columns, inplace=True)

        if self.verbose == 10:
            print(f"\n{'=' * 90}\n---> ✅ applying SMA:\n")
            print(f'\nDataset  first 10 rows after applying "SMA" func :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after applying "SMA" func :\n{df.tail(10)}\n')
        
        return df
    
# ----------------------------------- for GARCH model -----------------------------------  #
    """
    NOTE:
    This function implements a GARCH(1,1) model to estimate the conditional volatility of a time series.
    The GARCH(1,1) model is defined as:
        σ_t² = ω + α * (ε_(t-1)²) + β * (σ_(t-1)²)
    where:
        - ω (omega) is the baseline variance,
        - α (alpha) measures the impact of the previous period's squared error,
        - β (beta) captures the persistence of past volatility.

    In this function:
    - The 'arch_model' is instantiated with vol='Garch', p=1, and q=1, specifying a GARCH(1,1) model.
    - The 'dist' parameter is set to 't' to model the error terms with a Student's t-distribution,
      which is useful for capturing heavy tails in financial data.
    - The 'fit' method is used to estimate the model parameters via maximum likelihood.
      Here, 'update_freq=5' controls how often progress is printed during optimization, and 'disp' is set to 'off' to suppress output.
    - Finally, the estimated conditional volatility (res.conditional_volatility)
      is added to the DataFrame as 'garch_vol'.
    """
    def GARCH(self,
              df: pd.DataFrame,
              cols: Optional[Union[list[str], list]],
              rescale: int = 100):
        
        if isinstance(cols, list):
            cols = [cols]
        
        # since we fit arch_model with log returns and these are in a very small scale we should rescaleted it.
        df_rescaled = df[cols] * rescale # rescale variable should set deppending of the recommended warnings.

        am = arch_model(df_rescaled, vol='Garch', p=1, q=1, dist='t', rescale=False)
        res = am.fit(update_freq=5, disp='off')
        
        df['garch_vol'] = res.conditional_volatility
        self.drop_columns(df, 'LogR_Adj Close_t-1') #Drop the log return col.

        if self.verbose == 11:
            print(f"\n{'=' * 90}\n---> ✅ applying GARCH:\n")
            print(f'\ndataset  first 10 rows after apply GARCH :\n{df.head(10)}\n')
            print(f'\ndataset  last 10 rows after apply GARCH :\n{df.tail(10)}\n')

        return df


#---------------------------------------- Visualization ---------------------------------------#  
    def analize_df(self,
                   df: pd.DataFrame,
                   mode: str = "train") -> pd.DataFrame:
        
        if not df.empty and self.verbose:
            print(f"\n{'=' * 90}")
            print(f'\nanalize_df | deep visualization of the preproccessed {mode} dataset:\n') 
            df.info()
            print(f'\nNumber of Null values per column:\n{df.isnull().sum()}\n')
            print(f'\nDescription:\n{df.describe()}\n') #Describe all the columns even those that are not numerical
            print(f'\n{mode} dataset after preprocessing first 10 rows:\n{df.head(20)}\n\nlast 10 rows:\n{df.tail(50)}\n')
         
        return df      

#---------------------------------------- main ---------------------------------------#
    def main(self):
        df = self.open_csv()

        # ** for time series preproccessing ** #
        if self.windows:
            print('-' * 50)
            print(f"\n---> 📝 dataset preproccessing for ---> time series models.\n")
            print('-' * 50)
            df = self.set_col_index(df)
            df = self.windows_preprocessing(df,
                                            14,'Adj Close',
                                            idx=True)
            if self.forMoE:
                #create a folder to save the train subsets in case it does not exist
                subset_folder = os.path.join(self.training_path, f"{self.filename}_MoE_subsets_TimeSeries_{self.forMoE}_subs")
                if not os.path.exists(subset_folder):
                    print(f"\n🛑---> folder for MoE subsets does not exist! | ✅ creating the folder...\n")
                    os.makedirs(subset_folder)
                
                train_subsets, df_test = self.split_data(df)

                # # ** test dataset ** #
                df_test_dates = self.extract_dates(df_test, mode="test")
                
                if self.normalization:
                    df_test = self.normalize(df_test, dates=df_test_dates, mode="test")
                    df_test = self.analize_df(df_test, mode="test") 

                    if self.save: 
                        # ** save test dataset ** #
                        test_start_date, test_end_date = df_test_dates["start_date"], df_test_dates["end_date"]
                        test_csv_path = os.path.join(self.prediction_path, f"{self.filename}_time_series_{test_start_date}_to_{test_end_date}_test.csv")
                        df_test.to_csv(test_csv_path, index=True)
                        print(f"\n---> ✅ 📊 test dataset saved in: {test_csv_path}")
                        print(self.message)
                else:
                    df_test = self.analize_df(df_test, mode="test")
                 
                # ** train subsets ** #
                for key, train_subset in train_subsets.items():
                    subset_dates = self.extract_dates(train_subset)
                  
                    if self.normalization:
                        print(f"\n---> {key} normalization:")
                        train_subset = self.normalize(train_subset)
                        print(f"\n---> {key} description:")
                        train_subset = self.analize_df(train_subset)

                        if self.save: 
                            # ** save train subsets ** #
                            train_sub_start_date, train_sub_end_date = subset_dates["start_date"], subset_dates["end_date"]
                            subset_csv_path = os.path.join(subset_folder, f"{self.filename}_time_series_{train_sub_start_date}_to_{train_sub_end_date}_train_{key}.csv")
                            train_subset.to_csv(subset_csv_path, index=True)
                            print(f"\n---> ✅ 🚀 train subset saved in: {subset_csv_path}")
                    
                    else:
                        print(f"\n---> {key} description:")
                        train_subset = self.analize_df(train_subset)

            else:
                df_train, df_test = self.split_data(df)
                df_train_dates = self.extract_dates(df_train)
                df_test_dates = self.extract_dates(df_test, mode="test")
                
                # assuming that i only wanna save the dataset if it is normalized.
                if self.normalization:
                    # ** train dataset ** #
                    df_train = self.normalize(df_train)
                    df_train = self.analize_df(df_train)
                    # # ** test dataset ** #
                    df_test = self.normalize(df_test, dates=df_test_dates, mode="test")
                    df_test = self.analize_df(df_test, mode="test") 
                    
                    if self.save: 
                        # ** save train dataset ** #
                        train_start_date, train_end_date = df_train_dates["start_date"], df_train_dates["end_date"]
                        train_csv_path = os.path.join(self.training_path, f"{self.filename}_time_series_{train_start_date}_to_{train_end_date}_train.csv")
                        df_train.to_csv(train_csv_path, index=True)
                        print(f"\n---> ✅ 🚀 train dataset saved in: {train_csv_path}")
                        # ** save test dataset ** #
                        test_start_date, test_end_date = df_test_dates["start_date"], df_test_dates["end_date"]
                        test_csv_path = os.path.join(self.prediction_path, f"{self.filename}_time_series_{test_start_date}_to_{test_end_date}_test.csv")
                        df_test.to_csv(test_csv_path, index=True)
                        print(f"\n---> ✅ 📊 test dataset saved in: {test_csv_path}")
                        print(self.message)
                        
                else:
                    df_train = self.analize_df(df_train)
                    df_test = self.analize_df(df_test, mode="test")
        
        # ** for technical analysis preproccessing ** #
        elif self.technical:
            print('-' * 50)
            print(f"\n---> 📝 dataset preproccessing for ---> technical analysis models.\n")
            print('-' * 50)
            cols = [col for col in df.columns]
            df = self.log_return(df, cols=cols, window_size=4)
            df = self.RSI(df)
            df = self.momentum_ratio(df)
            df = self.average_true_range(df) # true_range func is applied inside.
            df = self.parabolic_sar(df)
            df = self.commodity_channel_index(df)
            df = self.SMA(df)
            df = self.set_col_index(df)
            df = self.move_col_to_end(df)

            if self.forMoE:
                #create a folder to save the train subsets in case it does not exist
                subset_folder = os.path.join(self.training_path, f"{self.filename}_MoE_subsets_Technical_{self.forMoE}_subs")
                if not os.path.exists(subset_folder):
                    print(f"\n🛑---> folder for MoE subsets does not exist! | ✅ creating the folder...\n")
                    os.makedirs(subset_folder)
                
                train_subsets, df_test = self.split_data(df)

                # # ** test dataset ** #
                df_test_dates = self.extract_dates(df_test, mode="test")
                
                if self.normalization:
                    df_test = self.normalize(df_test, dates=df_test_dates, mode="test")
                    df_test = self.analize_df(df_test, mode="test") 

                    if self.save: 
                        # ** save test dataset ** #
                        test_start_date, test_end_date = df_test_dates["start_date"], df_test_dates["end_date"]
                        test_csv_path = os.path.join(self.prediction_path, f"{self.filename}_technical_{test_start_date}_to_{test_end_date}_test.csv")
                        df_test.to_csv(test_csv_path, index=True)
                        print(f"\n---> ✅ 📊 test dataset saved in: {test_csv_path}")
                        print(self.message)
                else:
                    df_test = self.analize_df(df_test, mode="test")
                 
                # ** train subsets ** #
                for key, train_subset in train_subsets.items():
                    subset_dates = self.extract_dates(train_subset)
                  
                    if self.normalization:
                        print(f"\n---> {key} normalization:")
                        train_subset = self.normalize(train_subset)
                        print(f"\n---> {key} description:")
                        train_subset = self.analize_df(train_subset)

                        if self.save: 
                            # ** save train subsets ** #
                            train_sub_start_date, train_sub_end_date = subset_dates["start_date"], subset_dates["end_date"]
                            subset_csv_path = os.path.join(subset_folder, f"{self.filename}_technical_{train_sub_start_date}_to_{train_sub_end_date}_train_{key}.csv")
                            train_subset.to_csv(subset_csv_path, index=True)
                            print(f"\n---> ✅ 🚀 train subset saved in: {subset_csv_path}")
                    
                    else:
                        print(f"\n---> {key} description:")
                        train_subset = self.analize_df(train_subset)

            else:
                df_train, df_test = self.split_data(df)
                df_train_dates = self.extract_dates(df_train)
                df_test_dates = self.extract_dates(df_test, mode="test")

                # assuming that i only wanna save the dataset if it is normalized.
                if self.normalization:
                    # ** train dataset ** #
                    df_train = self.normalize(df_train)
                    df_train = self.analize_df(df_train)
                    # # ** test dataset ** #
                    df_test = self.normalize(df_test, dates=df_test_dates, mode="test")
                    df_test = self.analize_df(df_test, mode="test")
                    
                    if self.save: 
                        # ** save train dataset ** #
                        train_start_date, train_end_date = df_train_dates["start_date"], df_train_dates["end_date"]
                        train_csv_path = os.path.join(self.training_path, f"{self.filename}_technical_{train_start_date}_to_{train_end_date}_train.csv")
                        df_train.to_csv(train_csv_path, index=True)
                        print(f"\n---> ✅ 🚀 train dataset saved in: {train_csv_path}")
                        # ** save test dataset ** #
                        test_start_date, test_end_date = df_test_dates["start_date"], df_test_dates["end_date"]
                        test_csv_path = os.path.join(self.prediction_path, f"{self.filename}_technical_{test_start_date}_to_{test_end_date}_test.csv")
                        df_test.to_csv(test_csv_path, index=True)
                        print(f"\n---> ✅ 📊 test dataset saved in: {test_csv_path}")
                        print(self.message)
                        
                else:
                    df_train = self.analize_df(df_train)
                    df_test = self.analize_df(df_test, mode="test")
        
        # ** for GARCH preproccessing ** #
        elif self.Garch:
                print('-' * 50)
                print(f"\n---> 📝 dataset preproccessing for ---> GARCH models.\n")
                print('-' * 50)
                df = self.set_col_index(df)
                df = self.isolate_cols(df, 'Adj Close')
                df = self.log_return(df, cols='Adj Close', window_size=1)
                df = self.GARCH(df, 'LogR_Adj Close_t-1')
                df = self.windows_preprocessing(df, 7, ['Adj Close', 'garch_vol'], idx=True )
                df_train, df_test = self.split_data(df)
                df_train_dates = self.extract_dates(df_train)
                df_test_dates = self.extract_dates(df_test, mode="test")
                
                # assuming that i only wanna save the dataset if it is normalized.
                if self.normalization:
                    # ** train dataset ** #
                    df_train = self.normalize(df_train)
                    df_train = self.analize_df(df_train)
                    # # ** test dataset ** #
                    df_test = self.normalize(df_test, dates=df_test_dates, mode="test")
                    df_test = self.analize_df(df_test, mode="test")
                    
                    if self.save: 
                        # ** save train dataset ** #
                        train_start_date, train_end_date = df_train_dates["start_date"], df_train_dates["end_date"]
                        train_csv_path = os.path.join(self.training_path, f"{self.filename}_garch_{train_start_date}_to_{train_end_date}_train.csv")
                        df_train.to_csv(train_csv_path, index=True)
                        print(f"\n---> ✅ 🚀 train dataset saved in: {train_csv_path}")
                        # ** save test dataset ** #
                        test_start_date, test_end_date = df_test_dates["start_date"], df_test_dates["end_date"]
                        test_csv_path = os.path.join(self.prediction_path, f"{self.filename}_garch_{test_start_date}_to_{test_end_date}_test.csv")
                        df_test.to_csv(test_csv_path, index=True)
                        print(f"\n---> ✅ 📊 test dataset saved in: {test_csv_path}")
                        print(self.message)
                        
                else:
                    df_train = self.analize_df(df_train)
                    df_test = self.analize_df(df_test, mode="test")

        # ** for a normal preproccessing ** #
        else:
            print('-' * 50)
            print(f"\n---> 📝 normal preproccessing ---> a simple split and normalization of the dataset.\n")
            print('-' * 50)
            df = self.set_col_index(df)
            df = self.move_col_to_end(df)
            df_train, df_test = self.split_data(df)
            df_train_dates = self.extract_dates(df_train)
            df_test_dates = self.extract_dates(df_test, mode="test")

            # assuming that i only wanna save the dataset if it is normalized.
            if self.normalization:
                # ** train dataset ** #
                df_train = self.normalize(df_train)
                df_train = self.analize_df(df_train)
                # # ** test dataset ** #
                df_test = self.normalize(df_test, dates=df_test_dates, mode="test")
                df_test = self.analize_df(df_test, mode="test")
                
                if self.save: 
                    # ** save train dataset ** #
                    train_start_date, train_end_date = df_train_dates["start_date"], df_train_dates["end_date"]
                    train_csv_path = os.path.join(self.training_path, f"{self.filename}_normal_{train_start_date}_to_{train_end_date}_train.csv")
                    df_train.to_csv(train_csv_path, index=True)
                    print(f"\n---> ✅ 🚀 train dataset saved in: {train_csv_path}")
                    # ** save test dataset ** #
                    test_start_date, test_end_date = df_test_dates["start_date"], df_test_dates["end_date"]
                    test_csv_path = os.path.join(self.prediction_path, f"{self.filename}_normal_{test_start_date}_to_{test_end_date}_test.csv")
                    df_test.to_csv(test_csv_path, index=True)
                    print(f"\n---> ✅ 📊 test dataset saved in: {test_csv_path}")
                    print(self.message)
                    
            else:
                df_train = self.analize_df(df_train)
                df_test = self.analize_df(df_test, mode="test")
            
if __name__ =='__main__':
    
    # ** Hyperparameters **#
    def args():
        parser = argparse.ArgumentParser()
        
        #** Verbose. **#
        parser.add_argument('--verbose', type=int, default=2,
                            help="""Visualize functions results depending on the number,
                            True means that the verbose in that function will be visualize if verbose is not None.

                            verbose = None ---> Nothing
                            verbose = 1 ---> open_csv | visualize the opened dataset.
                            verbose = True ---> normalize | normalization visualization.
                            verbose = True ---> extract_dates | see the extracted dates to name the out datasets.
                            verbose = True ---> analize_df | deep visualization of the preproccessed dataset.

                            ** time series **
                            (NOTE: some functions could be applied in anothe kind of preproccessing as well.)
                            
                            verbose = 2 ---> windows_preprocessing | dataset visualization.
                            
                            ** technical analysis **
                            (NOTE: some functions could be applied in anothe kind of preproccessing as well.)
                            
                            verbose = 4 ---> log_return | visualize the resulting dataset after applying 'log_return' func.
                            verbose = 5 ---> RSI | visualize the resulting dataset after applying 'RSI' func.
                            verbose = 6 ---> momentum_ratio | visualize the resulting dataset after applying 'momentum_ratio' func.
                            verbose = 7 ---> true_range & average_true_range | visualize the resulting dataset after applying 'true_range & average_true_range' funcs.
                            verbose = 8 ---> parabolic_sar | visualize the resulting dataset after applying 'parabolic_sar' funcs.
                            verbose = 9 ---> commodity_channel_index | visualize the resulting dataset after applying 'commodity_channel_index' funcs.
                            verbose = 10 ---> SMA | visualize the resulting dataset after applying 'SMA' func.
                            
                            ** GARCH **
                            (NOTE: some functions could be applied in anothe kind of preproccessing as well.)
                            
                            verbose = True ---> isolate_cols | dataset visualization.
                            verbose = True ---> drop_columns | dataset visualization.
                            verbose = 11 ---> GARCH | visualize the resulting dataset after applying 'GARCH' func.
                            """)

        #** Paths. **#
        parser.add_argument('--open_path', type=str, 
        default= './Datasets/', 
        help='Open dataset path')

        parser.add_argument('--training_path', type=str, 
        default= './input datasets/Training/', 
        help='Save dataset path')

        parser.add_argument('--prediction_path', type=str, 
        default= './input datasets/Prediction/', 
        help='Save dataset path')

        parser.add_argument('--jason_path', type=str, 
        default= './input datasets/Prediction/', 
        help='Save json file path')

        #** Windows and technical and others. **#
        # NOTE : if all these are False, then a normal preprocessing is gonna be perform.
        parser.add_argument('--windows', type=bool, default=False, help='Transform a feature in a time series data')
        parser.add_argument('--technical', type=bool, default=False, help='Perform technical preprocessing the dataset')
        parser.add_argument('--GARCH', type=bool, default=True, help='Perform GARCH preprocessing the dataset')
        parser.add_argument('--forMoE', type=int, default=None, help='split the train dataset in subsets to train the MoE experts')
        
        #** Scale dataset. **#
        parser.add_argument('--normalization', type=bool, default=True, help='Normalize the dataset or not')
        parser.add_argument('--standardization', type=bool, default=False, help='Standardize the dataset or not')
        
        parser.add_argument('--save', type=bool, default=True, help='Save or not the dataset after preprocessing')
        args = parser.parse_args([])
        return args 
    
    pre_args = args() 
    pre = Preprocessing(pre_args)
    pre.main()
