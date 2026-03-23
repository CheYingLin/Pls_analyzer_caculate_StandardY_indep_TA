import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import Tuple, List, Dict, Any, Callable, Optional

class SVR_multi_model_backtest:
    """導入外參數設定"""
    def __init__(self,):
        pass

    def run_output_calibration_Excel(self,X_ori,comp_cols, intercepts, coefs,selected_algorithm,selected_factor):
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
        calibration_filename = f"{selected_algorithm}_Calibration_F{selected_factor}_{timestamp_file}.xlsx"
        # temp_correction_filename = f"TempCorrection_{selected_algorithm}_F{selected_factor}_{timestamp}.xlsx"
        
        calibration_path = os.path.join(export_folder, calibration_filename)
        # temp_correction_path = os.path.join(export_folder, temp_correction_filename)
        # 13. 儲存兩個檔案
        
        df_out.to_excel(calibration_path)
        # temp_correction_df.to_excel(temp_correction_path, index=False, header=True)
    
        return 0 

    def run_svr_multi_model_backtest(self,  X_ori: np.ndarray, X: np.ndarray, multi_algorithm_results_SVR,backtest_method_task_list,comp_cols,param_setIn, YesorNo=False):
        

        # 儲存多算法回測結果    
        predictions_dict = {}
        for model_name in backtest_method_task_list:
            algorithm_name = model_name.split(r'算法')[1].split(r'_(')[0]
            """數據預處理：移除NaN並檢查數據充足性"""
            X_ori = X_ori[algorithm_name]
            X = X[algorithm_name]
            if X_ori.shape[0]== X.shape[1]:
                mask = (np.isnan(X_ori).any(axis=1) | np.isnan(X).any(axis=1))
                X_ori[mask] = 0
                X[mask] = 0   
                X_pred = X
            else:
                mask = ~(np.isnan(X_ori).any(axis=1) )
                X_ori = X_ori[mask]  
                X_pred = X

            model_data = multi_algorithm_results_SVR[model_name]
            results = multi_algorithm_results_SVR[model_name]
            svr_result = results['svr_model']
            cv_result = results['svr_cv']
            
            # 4.3 取得係數
            intercepts_std = np.zeros(len(comp_cols))
            B_orig_i_table = np.zeros([X.shape[1],len(comp_cols)])    
            param_set_list = []
            Y_pred_std = []
            for idx, comp in enumerate(comp_cols):
                if  param_setIn:
                    param_set = param_setIn[idx]
                else:
                    param_set = cv_result['best_parameters_set'][comp] 
                param_set_list = np.hstack([param_set_list,format(param_set['Epsilon'], ".1e"),format(param_set['Cost'], ".1e")])
                # 獲取PLS和CV結果
                if param_set['Epsilon'] in cv_result['param_epsilon_results']:  
                    svr_param_epsilon_result = svr_result['param_epsilon_results'].get(param_set['Epsilon'])
                    svr_param_Cost_result = svr_param_epsilon_result[param_set['Cost']]
                    
                else:
                    svr_param_epsilon_result = svr_result['param_epsilon_results'].get(str(param_set['Epsilon']))
                    svr_param_Cost_result = svr_param_epsilon_result[str(param_set['Cost'])]
                # 4.4 截距 - 從訓練結果中獲取
                intercepts_std[idx] = svr_param_Cost_result['model_intercepts'][idx]
                B_orig_i_table[:,idx] =  svr_param_Cost_result['model_coefs'][:,idx]    


            # 4.5 執行預測
            Y_pred_std_tmp = (X_pred).dot(B_orig_i_table) + intercepts_std
            # Y_pred_std.append(Y_pred_std_tmp)
            Y_pred_std = Y_pred_std_tmp
                 #========輸出檢量線====================
        if YesorNo:
            self.run_output_calibration_Excel(X_ori,comp_cols,intercepts_std,B_orig_i_table, model_name, param_set_list)#輸出標準化Y的檢量線
            
            print('完成輸出SVR檢量線')
        else:
            print('無輸出SVR檢量線')

        # 4.6 存儲結果
        unique_key = f"{model_name}_P{param_set_list}"
        predictions_dict[unique_key] = {
            'predictions': Y_pred_std ,
            'comp_names': comp_cols,
            'model_name': model_name,
            'param_set_list': param_set_list
        }  

        return predictions_dict          
