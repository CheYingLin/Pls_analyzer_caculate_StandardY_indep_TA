import numpy as np
import pandas as pd
import os
from typing import Tuple, List, Dict, Any, Optional

class Pre_processing:
    """導入外參數設定"""
    def __init__(self,):
        pass

    def preprocess_data(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """數據預處理：移除NaN並檢查數據充足性"""
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
        X_valid = X[mask]
        Y_valid = Y[mask]
               
        return X_valid, Y_valid
    
    def import_merged_file(self, file_path):
        ParquetFolder  = 'data_in_parquetFolder'
        # 分離路徑與檔名
        folder, fname = os.path.split(file_path)   # folder = 路徑, fname = 'result.csv'
        # 再分離檔名與副檔名
        name, ext = os.path.splitext(fname) 
        parquet_file = os.path.join(ParquetFolder, name + ".parquet")
        # 如果 parquet 已存在，就直接讀取
        if os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
        else:
            # 根據檔案類型讀取
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            else:
                df = pd.read_excel(file_path)
            # 存成 Parquet
            df.to_parquet(parquet_file, index=False)
        df['Time'] = pd.to_datetime(df['Time'])    
        # 清理欄名空格
        df.columns = df.columns.str.strip()
        return df 
    
    def import_excel_file(self, full_path,sheet_name):
        if full_path.endswith('.csv'):
            df_timeRef = pd.read_csv(full_path, encoding='utf-8-sig')
        else:
            df_timeRef = pd.read_excel(full_path, sheet_name=sheet_name)#,index_col=0
        df_timeRef["Time"] = (df_timeRef["Time"].astype(str).str.replace(" PM", "", regex=False).str.replace(" AM", "", regex=False))
        df_timeRef["Time"] = df_timeRef["Time"].astype(str).str.strip()
        df_timeRef["Time"] = pd.to_datetime(df_timeRef["Time"],format="mixed",errors="coerce")
        return df_timeRef
