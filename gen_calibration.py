import numpy as np
import pandas as pd
import os
from datetime import datetime

def run_output_calibration_Excel(X_ori,comp_cols, intercepts, coefs,selected_algorithm,selected_factor):
    # mw_range_nums = [i + 1 for i in selected_indices]
    if coefs.shape[0] <= 36 :
        spec_cols = [f"MW{i+1}" for i in range(X_ori.shape[1])]
    else:
        spec_cols = ['TA'] + [f"MW{i+1}" for i in range(X_ori.shape[1])]
    # 確定前綴
    prefix = 'MW' 
    labels = ['intercept', 'TA'] + [f"{prefix}{i}" for i in range(1, 37)]

    # 創建輸出表格
    df_out = pd.DataFrame(0.0, index=labels, columns = comp_cols)

    # 使用 iloc 方法填充係數
    for j, comp in enumerate(comp_cols):
        # 1. 填充截距 (第0行)
        df_out.iloc[0, j] = intercepts[j]
        
        # 2. 創建係數映射字典
        coef_map = dict(zip(spec_cols, coefs[:, j]))
        
        # 3. 填充 TA (第1行)
        if 'TA' in coef_map:
            df_out.iloc[1, j] = coef_map['TA']

        # 4. 填充 MW/MWTN 通道 (第2-37行)
        for i in range(1, 37):
            row_idx = i + 1  # 通道 i 對應 DataFrame 行索引 i+1
            
            # 構建查找鍵
            if prefix == 'MWTN-':
                orig_key = f"MWTN-{i}"
            else:
                orig_key = f"MW{i}"
            
            # 如果該通道有係數，則填充
            if orig_key in coef_map:
                df_out.iloc[row_idx, j] = coef_map[orig_key]
        
    # 調整列名格式
    if len(comp_cols) == 1:
        df_out.columns = ['值']

    # 11. 創建時間戳和輸出資料夾
    FolderOut = 'calibration_Excel_data_out'#固定輸出資料夾位置
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    timestamp_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_folder = os.path.join(FolderOut, f"Calibration_Export_{timestamp}")
    
    os.makedirs(export_folder, exist_ok=True)
    # 12. 定義檔案路徑
    calibration_filename = f"Calibration_{selected_algorithm}_F{selected_factor}_{timestamp_file}.xlsx"
    # temp_correction_filename = f"TempCorrection_{selected_algorithm}_F{selected_factor}_{timestamp}.xlsx"
    
    calibration_path = os.path.join(export_folder, calibration_filename)
    # temp_correction_path = os.path.join(export_folder, temp_correction_filename)
    # 13. 儲存兩個檔案
    
    df_out.to_excel(calibration_path)
    # temp_correction_df.to_excel(temp_correction_path, index=False, header=True)
   
    return 0   