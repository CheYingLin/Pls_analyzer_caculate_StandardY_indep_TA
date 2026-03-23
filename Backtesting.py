import numpy as np
import pandas as pd
import os
import time
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Callable, Optional
from gen_calibration import run_output_calibration_Excel
matplotlib.rc('font', family='serif', serif=['ABC', 'MingLiU']) 
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def run_multi_model_backtest(X: np.ndarray ,factor: int, model_data, comp_cols: List[str], 
                             pls_model, stats, unique_key, model_name):
    predictions_dict = {}
    X_pred = X    
        # 4.3 計算係數
    coefs = pls_model.coef_ if pls_model.coef_.shape[0] != len(comp_cols) else pls_model.coef_.T

    # 4.4 計算截距 - 從訓練結果中獲取
    # model_result = multi_algorithm_results[model_name]
    model_result = model_data
    X_valid = model_result['pls']['X_valid']
    Y_valid = model_result['pls']['Y_valid']

    # 計算訓練數據的均值
    X_mean = X_valid.mean(axis=0)
    Y_mean = Y_valid.mean(axis=0)

    # 計算截距
    intercepts = Y_mean - X_mean.dot(coefs)

    # 4.5 執行預測
    Y_pred = X_pred.dot(coefs) + intercepts

    # 4.6 存儲結果
    predictions_dict[unique_key] = {
        'predictions': Y_pred,
        'comp_names': comp_cols,
        'stats': stats,
        'model_name': model_name,
        'factor': factor
    }
    return{
            f"{model_name}_F{factor}": {
                    'predictions': Y_pred,
                    'comp_names': comp_cols,
                    'stats': {},
                    'model_name': model_name,
                    'factor': factor
                }
            }# 轉換為新格式並調用多模型繪圖函數

# def run_multi_model_backtest_NEW(X: np.ndarray ,factor: int, model_data, comp_cols: List[str], 
#                              pls_model, scalerY, stats, unique_key, model_name , unselect):
#     predictions_dict = {}
#     coefs_list = []
#     X_pred = X   
#     mask = np.ones(X.shape[1], dtype=bool)
#     mask[unselect] = False 
#     coefs_table = np.zeros([X.shape[1],len(comp_cols)]) 
#     intercepts = np.zeros(len(comp_cols)) 
#         # 4.3 計算係數
#     for i in range(len(comp_cols)):
#         coefs_list.append( pls_model[0][i].coef_ if pls_model.shape[0] != len(comp_cols) else pls_model[0][i].coef_.T)
#     coefs = np.vstack(coefs_list).T  
#     coefs_table[mask] = coefs
#     # 4.4 計算截距 - 從訓練結果中獲取
#     # model_result = multi_algorithm_results[model_name]
#     model_result = model_data
#     X_valid = model_result['pls']['X_valid']
#     Y_valid = model_result['pls']['Y_valid']

#     # 計算訓練數據的均值
#     X_mean = X_valid.mean(axis=0)
#     Y_mean = Y_valid.mean(axis=0)

#     # 計算截距
#     intercepts = Y_mean - X_mean.dot(coefs)

#     # 4.5 執行預測
#     Y_pred = X_pred.dot(coefs_table) + intercepts

#     # 4.6 存儲結果
#     predictions_dict[unique_key] = {
#         'predictions': Y_pred,
#         'comp_names': comp_cols,
#         'stats': stats,
#         'model_name': model_name,
#         'factor': factor
#     }
#     return{
#             f"{model_name}_F{factor}": {
#                     'predictions': Y_pred,
#                     'comp_names': comp_cols,
#                     'stats': {},
#                     'model_name': model_name,
#                     'factor': factor
#                 }
#             }# 轉換為新格式並調用多模型繪圖函數

def run_multi_model_backtest_NEWII(predictions_dict,predictions_dict_no_scaleY,
                                X_ori,X: np.ndarray ,factor: List, factor_no_scaleY: int, model_data, comp_cols: List[str], 
                                model_name, unselect, filename ,YesorNo=False):
    unique_key = f"{model_name}_F{factor}"
    unique_key_no_scaleY =  f"{model_name}_F{factor_no_scaleY:}"
    coefs_list = []
    coefs_list_no_scaleY = []
    scalerY = []
    """數據預處理：移除NaN並檢查數據充足性"""
    if X_ori.shape[0]== X.shape[0]:
        mask = (np.isnan(X_ori).any(axis=1) | np.isnan(X).any(axis=1))
        X_ori[mask] = 0
        X[mask] = 0   
        X_pred = X
    else:
        mask = ~(np.isnan(X_ori).any(axis=1) )
        X_ori = X_ori[mask]  
        X_pred = X
    # mask = np.ones(X.shape[1], dtype=bool)
    # mask[unselect] = False 
    # 獲取model results
    pls_results = model_data.get('pls', {})
    factor_result = pls_results.get('factor_results', {})

    coefs_table = np.zeros([X.shape[1],len(comp_cols)]) 
    coefs_table_no_scaleY = np.zeros([X.shape[1],len(comp_cols)]) 
    # intercepts = np.zeros(len(comp_cols)) 
        # 4.3 計算係數
    for i in range(len(comp_cols)):
        # mask = np.ones(X.shape[1], dtype=bool)
        # mask[unselect[i]] = False 
        model_info = factor_result[factor[i]]
        model_info_no_scaleY = factor_result[factor_no_scaleY]
        pls_model = model_info.get('model')
        pls_model_no_scaleY = model_info_no_scaleY.get('moedl_no_scaleY')
        scalerY.append(model_info.get('Y_scalers')[i])
        stats = model_info.get('stats', {})

        coefs_list.append( pls_model[0][i].coef_ if pls_model.shape[0] != len(comp_cols) else pls_model[0][i].coef_.T)
    coefs_list_no_scaleY= pls_model_no_scaleY[0][0].coef_.T if pls_model_no_scaleY.shape[0] != len(comp_cols) else pls_model_no_scaleY[0][0].coef_.T
    # coefs = np.vstack(coefs_list).T  
    
    # 4.4 計算截距 - 從訓練結果中獲取
    # model_result = multi_algorithm_results[model_name]
    model_result = model_data
    X_valid = model_result['pls']['X_valid']
    Y_valid = model_result['pls']['Y_valid']

    # 計算訓練數據的均值
    # X_mean = np.zeros_like()
    intercepts_std = np.zeros(len(comp_cols)) 
    intercepts_no_scaleY= np.zeros(len(comp_cols))
    for i in range(len(comp_cols)):
        mask = np.ones(X_ori.shape[1], dtype=bool)
        mask_no_scaleY = np.ones(X_ori.shape[1], dtype=bool)
        if not unselect or unselect[i].size==0:
                pass
        else:
            mask[unselect[i]] = False
            if not any(arr.size == 0 for arr in unselect): 
                mask_no_scaleY[unselect[0]] = False
        X_mean = X_valid[:,mask].mean(axis=0)
        X_mean_no_scaleY = X_valid[:,mask_no_scaleY].mean(axis=0)
        Y_mean = Y_valid.mean(axis=0)
        coefs_table[mask,i] = np.array(coefs_list[i]).flatten()
        coefs_table_no_scaleY [mask_no_scaleY,i] = np.array(coefs_list_no_scaleY)[:,i].flatten()
        # 計算截距
        intercepts_std[i] = - X_mean.dot(coefs_list[i].ravel())
        intercepts_no_scaleY[i] = Y_mean[i] - X_mean_no_scaleY .dot(np.array(coefs_list_no_scaleY)[:,i].ravel())
    
    
    # 4.5 執行預測
    # Y_pred = X_pred.dot(coefs_table) + intercepts
    Y_pred_std = X_pred.dot(coefs_table) + intercepts_std
    Y_pred_no_scaleY = X_pred.dot(coefs_table_no_scaleY) + intercepts_no_scaleY

    Y_pred_new = np.zeros_like(Y_pred_std)

    for i, sc in enumerate(scalerY):
        Y_pred_new[:, i] = sc.inverse_transform(
            Y_pred_std[:, i].reshape(-1, 1)
        ).ravel()
    # #========輸出檢量線====================
    # if YesorNo:
    #     run_output_calibration_Excel(X_ori,comp_cols,intercepts_std,coefs_table,f'{model_name}_標準化Y',factor)#輸出標準化Y的檢量線
    #     run_output_calibration_Excel(X_ori,comp_cols,intercepts_no_scaleY,coefs_table_no_scaleY,f'{model_name}_無標準化Y',factor_no_scaleY)#輸出無標準化Y的檢量線
    #     print('完成輸出檢量線')
    # else:
    #     print('無輸出檢量線')
    #========手動驗證逆標準化Y的係數========
    # 訓練用 X（完整）
    X_train_full = X_valid

    # 每個 y 對應的子空間平均
    X_mean_list = []
    for i in range(len(comp_cols)):
        mask = np.ones(X_train_full.shape[1], dtype=bool)
        if not unselect or unselect[i].size==0:
            pass
        else:
            mask[unselect[i]] = False
        X_mean_list.append(X_train_full[:, mask].mean(axis=0))

    intercepts_std = np.zeros(len(comp_cols)) 
    Y_2 = np.zeros_like(Y_pred_std)
    B_orig_i_table = np.zeros([X.shape[1],len(comp_cols)])
    for i in range(len(comp_cols)):
        mask = np.ones(X_ori.shape[1], dtype=bool)
        if not unselect or unselect[i].size==0:
            pass
        else:
            mask[unselect[i]] = False
        model_info = factor_result[factor[i]]
        pls_model = model_info.get('model')
        B_scaled_i = pls_model[0][i].coef_.ravel()
        sigma_Y_i  = scalerY[i].scale_[0]
        mu_Y_i     = scalerY[i].mean_[0]
        X_mean_i   = X_mean_list[i]
        # X_mean_i = pls_model[0][i]._x_mean
        B_orig_i = B_scaled_i * sigma_Y_i
        b_orig_i = mu_Y_i - X_mean_i @ B_orig_i
        B_orig_i_table[mask,i] = B_orig_i
        intercepts_std[i] = b_orig_i
        Y_2[:, i] = X_pred @ B_orig_i_table[:,i] + b_orig_i
        
    print(f'{np.allclose(Y_pred_new, Y_2, atol=1e-10)}驗證')
    #========輸出檢量線====================
    if YesorNo:
        run_output_calibration_Excel(X_ori,comp_cols,intercepts_std,B_orig_i_table,f'{model_name}_標準化Y',filename,factor)#輸出標準化Y的檢量線
        run_output_calibration_Excel(X_ori,comp_cols,intercepts_no_scaleY,coefs_table_no_scaleY,f'{model_name}_無標準化Y',filename,factor_no_scaleY)#輸出無標準化Y的檢量線
        print('完成輸出檢量線')
    else:
        print('無輸出檢量線')
    # 4.6 存儲結果
    predictions_dict[unique_key] = {
        'predictions': Y_pred_new,
        'comp_names': comp_cols,
        'stats': stats,
        'model_name': model_name,
        'factor': factor
    }
    predictions_dict_no_scaleY[unique_key_no_scaleY] = {
        'predictions': Y_pred_no_scaleY,
        'comp_names': comp_cols,
        'stats': stats,
        'model_name': model_name,
        'factor': factor
    }
    # return{
    #         f"{model_name}_F{factor}": {
    #                 'predictions':  Y_pred_new,
    #                 'comp_names': comp_cols,
    #                 'stats': {},
    #                 'model_name': model_name,
    #                 'factor': factor
    #             }
    #         },{# 轉換為新格式並調用多模型繪圖函數
    #         f"{model_name}_F{factor_no_scaleY}": {
    #                 'predictions':  Y_pred_no_scaleY,
    #                 'comp_names': comp_cols,
    #                 'stats': {},
    #                 'model_name': model_name,
    #                 'factor': factor
    #             }
    #         }
    return predictions_dict, predictions_dict_no_scaleY

def import_merged_file(file_path):
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

def run_selected_specturm(X_tmp_rows, unselect, training_has_ta):
    mw_row_avg = []
    ta_row_avg = []
    ta_col = "temperature"
    if training_has_ta:
        if ta_col in X_tmp_rows.columns:
            ta_values = X_tmp_rows[ta_col].dropna()
            ta_row_avg.append(ta_values if not ta_values.isna().all() else np.nan)
        else:
            ta_row_avg.append(np.nan)
    for i in range(1, 37):
        # if i in unselect:
        #     continue  # 跳過這個 i
        # MW數據
        # mw_col = f"MW{i}"#MW_absorb_normalized,_normalized
        mw_col = f"{i}-MW_NON"
        if mw_col in X_tmp_rows.columns:
            mw_values = X_tmp_rows[mw_col].dropna()
            mw_row_avg.append(mw_values if not mw_values.isna().all() else np.nan) 
        else:
            mw_row_avg.append(np.nan)

    if len(ta_row_avg) == 0:
        mw_mat = mw_row_avg.copy()
    else:
        mw_mat = np.vstack((ta_row_avg, mw_row_avg))       
    return np.array(mw_mat).T 

def read_folder_tables(folder_path, sheet_number=0):
    output = {}

    files = sorted(os.listdir(folder_path))

    for file in files:
        if not file.lower().endswith(('.xls', '.xlsx')):
            continue

        full_path = os.path.join(folder_path, file)
        name = os.path.splitext(file)[0]

        df = pd.read_excel(full_path, sheet_name=sheet_number,index_col=0)
        output[name] = df

    return output    

# #===============main=====================
if __name__ == '__main__':

    file_path = r"C:\Users\Jason.lin\Desktop\workfile\建模資料庫\20260225_九峰山\Original_backtest_file_九峰山_OQC.csv"
    #=================================
    df = import_merged_file(file_path)

    df_timeRef = pd.read_excel( r"C:\Users\Jason.lin\Desktop\workfile\建模資料庫\20260225_九峰山\concentration_list_九峰山SAOQC.xlsx",
                            sheet_name="工作表2")
    # df_calibration = pd.read_excel(r"C:\Users\Jason.lin\Desktop\workfile\建模資料庫\20260225_九峰山\九峰山Calibration_算法X_標準化Y_F[5, 3]_20260225_152155.xlsx",
    #                         sheet_name="Sheet1",index_col=0)
    # # 清除空白
    # df_calibration.index = df_calibration.index.astype(str).str.strip()
    #new
    df_calibration_file = "multi_cailbration_backtest_Folder"
    df_calibration_all = read_folder_tables(df_calibration_file)

    #luna MW_data電鍍
    # ========處理時間格式清理空格==========
    df_timeRef["Time"] = (df_timeRef["Time"].astype(str).str.replace(" PM", "", regex=False).str.replace(" AM", "", regex=False))
    df_timeRef["Time"] = df_timeRef["Time"].astype(str).str.strip()
    df_timeRef["Time"] = pd.to_datetime(df_timeRef["Time"],format="mixed",errors="coerce")

    # ========== 3.Parameters Setting ================
    train_ratio = 1 #1 for all,0.8 for 80%
    Concentration_slec = np.array([1,2]) #全部幾種濃度液體
    Concentration_pred = np.array([1,2]) #要預測哪幾種濃度液體
    Training_has_ta = True # True,False
    # time_window = [timedelta(minutes=5), timedelta(minutes=0)]
    time_window = [5,0]#([before,after] minutes) #捷捷薇選[6,-4]or[6,-5]
    # ========= Constant Parameters===============
    Concentration_slec_colnum = 2
    # Concentration_slec_num = 3 #要看幾種濃度液體
    timedata = df.iloc[:,1:2]
    timedata.head()
    # y是目標值
    # print(df.columns)
    # comp_cols = df_timeRef.columns.tolist()[Concentration_slec_colnum:(Concentration_slec_colnum+Concentration_slec_num)]
    comp_cols_all_name = df_timeRef.columns[Concentration_slec_colnum + (Concentration_slec-1)].tolist()
    comp_preds = df_timeRef.columns[Concentration_slec_colnum + (Concentration_pred-1)].tolist()
    comp_indices = [comp_cols_all_name.index(name) for name in comp_preds if name in comp_cols_all_name]
    # print(comp_cols_all_name)
    # Y.head()
    #================= 獲取 X=======================
    channel_unselect = []
    X_pred= run_selected_specturm(df, channel_unselect,Training_has_ta)

    for page in range(len(comp_preds)):
        # fig = plt.figure(figsize=(14, 4))
        fig, axs = plt.subplots(2, 1,figsize=(14, 4),gridspec_kw={'height_ratios': [3, 1]})
        axs = axs.flatten()

        ax = axs[0]
        # 準備makers
        markers = ['s','^','D','v','P','X','*']
        # =====獲取檢量線資訊====
        for idx, name in enumerate(df_calibration_all):
            df_calibration = df_calibration_all[name] 
            intercepts = df_calibration.loc["intercept"].values #"B_0" , "intercept"
            # 4. 提取係數矩陣 (第 1 行之後，保留所有行)
            coef_df = df_calibration.iloc[1:]  # 去掉 intercept 行
            
            # 5. 提取所有行索引作為特徵名稱
            spec_cols = coef_df.index.tolist()  # 例如 ['TA', 'MWTN-1', 'MWTN-2', ..., 'MWTN-36']
            
            # 6. 提取完整係數矩陣（包含零值行）
            coefs = coef_df.values 
            Y_pred = X_pred.dot(coefs) + intercepts 



            time_data = df['Time'].values
            colors = plt.cm.tab10(np.linspace(0, 1, 10))  # tab10 色盤
            marker = markers[idx % len(markers)] # 改變maker
            
            # 準備顏色
            
            # 選擇顏色
            color = colors[idx % len(colors)]
            ax.plot(
                    time_data, 
                    # Y_pred[:, page]+bias[page],
                    Y_pred[:, page],
                    linestyle=' ',
                    color=color,
                    marker= marker,
                    markerfacecolor='none',
                    markersize=5,
                    label = f"predict_{name}",
                    alpha=0.6
                )
            
        ref_time = df_timeRef['Time'].values
        ref_values = df_timeRef[comp_preds[page]].values
        # 繪製參考數據為紅色星形標記
        ax.plot(
            ref_time,
            ref_values,
            linestyle=' ',
            color='red',
            marker='.',
            markersize=12,
            markerfacecolor='red',
            label='Reference Data',
            alpha=0.9,
            zorder=100
        )

        # 設置圖表標題和標籤
        ax.set_title(
            f"{comp_preds[page]} 模型回測對比",
            fontsize=11,
            fontweight='bold'
        )
        # ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Predicted Value', fontsize=9)
        # ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        y_min =-10 ; y_max = 15 
        # ax.set_ylim(y_min, y_max)  
        
        # 添加圖例（放在子圖外側右方或下方）
        ax.legend(
            loc='lower left',
            bbox_to_anchor=(0.01, 0.5),
            fontsize=8,
            framealpha=0.9
        )

        #畫溫度
        df_TA = df['temperature'].values
        ax = axs[1]
        ax.plot(
                time_data, 
                # Y_pred[:, page]+bias[page],
                df_TA,
                linestyle=' ',
                color='blue',
                marker='.',
                markersize=3,
                label = '',
                alpha=0.5
            )
        ax.set_title(
            f"Temperture",
            fontsize=11,
            fontweight='bold'
        )
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('(T)', fontsize=9)
        # ax.tick_params(axis='x', rotation=45, labelsize=8)
        # ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        y_min =-10 ; y_max = 15
        
        # 調整佈局以防止重疊
        plt.tight_layout()
    plt.show()    

    print('Done!!!!!!!!!!!!!!!!!!!!')
