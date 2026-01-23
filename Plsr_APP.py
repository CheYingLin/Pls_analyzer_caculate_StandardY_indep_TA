#import modules
import pandas as pd
import numpy as np
import os
import time
from datetime import timedelta
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from typing import Tuple, List, Dict, Any, Optional
from Cross_validation import run_cross_validation_analysis
from Backtesting import run_multi_model_backtest_NEWII
from Plotgrop import run_plot_group_new,run_plot_group_scatter_new ,run_plot_display_multi_algorithm_results,run_plot_display_indepY_algorithm_results
from Plotgrop import run_create_prediction_comparison_chart, run_plot_backtest_results,run_plot_backtest_results_with_score
# 導入新的分析模組
from Principal_Component_Analysis import principal_component_analysis
from PLSR_Analysis.plsr_analysis_APP import PLSR_Analysis
# import seaborn as sns

tic = time.time()
'''
def _preprocess_data( X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """數據預處理：移除NaN並檢查數據充足性"""
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X_valid = X[mask]
    Y_valid = Y[mask]
    
    n_samples = X_valid.shape[0]
    if n_samples < 2:
        raise ValueError(f"篩選後資料點不足 ({n_samples}), 至少需 2 筆")
    
    return X_valid, Y_valid, n_samples

def _determine_max_factor( n_sample: int, n_features: int, max_factor: int = 16) -> int:
    """確定最大Factor數量（與交叉驗證邏輯一致）"""
    if n_sample < 16:
        max_factor = max(1, n_sample - 2)
    
    max_factor = min(max_factor-2, n_features)
    
    if max_factor < 1:
        raise ValueError(f"無法進行分析：數據點數 ({n_sample}) 或特徵數 ({n_features}) 不足")
    
    return max_factor

def _fit_single_factor_final( X: np.ndarray, ch_unselect, Y: np.ndarray, n_component: int) -> Tuple[PLSRegression, np.ndarray]:
    Y_train = np.zeros_like(Y)
    scalers_y = []
    for i in range(Y.shape[1]):
        sc = StandardScaler()
        Y_train[:, i] = sc.fit_transform(
            Y[:, i].reshape(-1, 1)
        ).ravel()
        scalers_y.append(sc)

    # train + predict
    Y_pred_All = []
    Y_val_All = []
    pls_model_All = []
    for i in range(Y.shape[1]):
        mask = np.ones(X.shape[1], dtype=bool)
        if not ch_unselect:
            pass
        else:
            mask[ch_unselect[i]] = False 
        pls = PLSRegression(n_components = n_component, scale=False)
        pls.fit(X[:,mask], Y_train[:, i])
        y_pred_std = pls.predict(X[:,mask]).ravel()
        y_pred = scalers_y[i].inverse_transform(
                                y_pred_std.reshape(-1, 1)).ravel()
        Y_pred_All.append(y_pred)
        Y_val_All.append(Y[:, i]) 
        pls_model_All.append(pls)
    # 合併累積數據
    y_pred_list = np.vstack(Y_pred_All)      
    y_val_list = np.vstack(Y_val_All)   
    cv_pls_model_list= np.vstack(pls_model_All)
    return cv_pls_model_list.T, y_pred_list.T, scalers_y 
        
def _fit_single_factor_no_scaleY_final( X: np.ndarray, ch_unselect, Y: np.ndarray, n_component: int) -> Tuple[PLSRegression, np.ndarray]:    
    scalers_y = np.full((Y.shape[1], 1), None)
    # train + predict
    Y_pred_All = []
    Y_val_All = []
    pls_model_All = []
    for i in range(Y.shape[1]):
        mask = np.ones(X.shape[1], dtype=bool)
        if not ch_unselect:
            pass
        else:
            mask[ch_unselect[0]] = False  #此處0保留都不可篩選，通道應該是不隨不同濃度液體篩選
        pls = PLSRegression(n_components = n_component, scale=False)
        pls.fit(X[:,mask], Y)
        y_pred = pls.predict(X[:,mask])
        Y_pred_All.append(y_pred[:, i])
        Y_val_All.append(Y[:, i]) 
    pls_model_All.append(pls) 
    # 合併累積數據
    y_pred_list = np.vstack(Y_pred_All)      
    y_val_list = np.vstack(Y_val_All)   
    cv_pls_model_list= np.vstack(pls_model_All) 
    return cv_pls_model_list.T, y_pred_list.T, scalers_y    

def _fit_single_factor( X: np.ndarray, Y: np.ndarray, n_component: int) -> Tuple[PLSRegression, np.ndarray]:
    """單個Factor的PLS建模"""
    pls = PLSRegression(n_components=n_component, scale=False)
    pls.fit(X, Y)
    Y_pred = pls.predict(X)
    return pls, Y_pred

def _calculate_regression_stats( Y_true: np.ndarray, Y_pred: np.ndarray, 
                                  comp_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """計算回歸統計信息"""
    stats = {}
    
    # 計算各成分的個別統計
    for idx, comp in enumerate(comp_cols):
        y_true = Y_true[:, idx]
        y_pred = Y_pred[:, idx]
        
        r2 = r2_score(y_true, y_pred)
        coeffs = np.polyfit(y_true, y_pred, 1)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        stats[comp] = {
            'r2': r2,
            'slope': coeffs[0],
            'intercept': coeffs[1],
            'rmse': rmse
        }
    
    # 計算 explained variance（比照 cross_validation 算法）
    # 1. 標準化所有成分的 y_prediction 和 y_reference
    # scaler_true = StandardScaler()
    # scaler_pred = StandardScaler()
    
    # Y_true_scaled = scaler_true.fit_transform(Y_true)
    # Y_pred_scaled = scaler_pred.fit_transform(Y_pred)
    # NEW Scale
    scaler_all = StandardScaler()

    Y_true_scaled = scaler_all.fit_transform(Y_true)  # ← 學尺度
    Y_pred_scaled = scaler_all.transform(Y_pred)      # ← 用同一尺度

    
    # 2. 計算所有成分的 R² 並取平均
    component_r2_values = []
    for idx in range(Y_true.shape[1]):
        r2 = r2_score(Y_true_scaled[:, idx], Y_pred_scaled[:, idx])
        component_r2_values.append(r2)
    
    explained_variance = np.mean(component_r2_values)
    
    # 3. 將 explained_variance 添加到每個成分的統計中
    for idx, comp in enumerate(comp_cols):
        stats[comp]['explained_variance'] = explained_variance
        stats[comp]['explained_variance_per_y'] = component_r2_values[idx]
    
    return stats

def run_pls_factor_scan(X: np.ndarray, ch_unselect ,Y: np.ndarray, 
                        comp_cols: List[str], 
                        max_factor: int = 16, 
                        progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """
    執行PLS Factor掃描分析
    """
    X_valid, Y_valid, n_samples = _preprocess_data(X, Y)
    n_features = X_valid.shape[1]
    max_factor = _determine_max_factor(n_samples, n_features, max_factor)

    # for i in range(len(comp_cols)):
    #     max_factor = min(max_factor,36-len(ch_unselect[i]))
    if np.array([len(x) for x in ch_unselect]).size > 0:
        max_factor = min(max_factor,min(36-np.array([len(x) for x in ch_unselect])))

    factor_results = {}
    # Factor掃描
    for factor in range(1, max_factor + 1):
        if progress_callback:
            progress_callback(factor, max_factor)
        
        pls, Y_pred,Y_scalers = _fit_single_factor_final(X_valid,ch_unselect, Y_valid, factor)
        stats = _calculate_regression_stats(Y_valid, Y_pred, comp_cols)

        pls_no_scaleY, Y_pred_no_scaleY,Y_scalers_no_scaleY = _fit_single_factor_no_scaleY_final(X_valid,ch_unselect, Y_valid, factor)
        stats_no_scaleY = _calculate_regression_stats(Y_valid, Y_pred_no_scaleY, comp_cols)
            
        factor_results[factor] = {
            'model': pls, #模型訓練出來的 參數 ex.y^​=X⋅coef_+intercept 可得 coef_
            'Y_pred': Y_pred,
            'Y_scalers':Y_scalers, 
            'stats': stats,
            'moedl_no_scaleY':pls_no_scaleY,
            'Y_pred_no_scaleY': Y_pred_no_scaleY,
            'Y_scalers_no_scaleY':Y_scalers_no_scaleY, 
            'stats_no_scaleY': stats_no_scaleY

        }

        result = {
                'factor_results': factor_results,
                'X_valid': X_valid,
                'Y_valid': Y_valid,
                'max_factor': max_factor,
                'comp_cols': comp_cols,
                'n_samples': n_samples
            }
        
        
    return result
'''
def run_selected_specturm(X_tmp_rows, unselect,training_has_ta):
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

def run_Y_standart_Ind(Y_In):
    scalers_y = {}
    Y_train_std = {}
    # Y_test_std = {}

    for i, y_train in enumerate(Y_In.T, start=1):

        scaler = StandardScaler()
        Y_train_std[f'y{i}'] = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        # Y_test_std[f'y{i}'] = scaler.transform(y_test.reshape(-1, 1)).ravel()
        scalers_y[f'y{i}'] = scaler
    return Y_train_std    

def run_selected(X_tmp, time_window, df_timeRef,training_has_ta):
    mw_mat = []
    time_before = pd.Timedelta(minutes=time_window[0])
    time_after = pd.Timedelta(minutes=time_window[1])
    # 找到時間窗口範圍內的所有數據點（非對稱）
    for t in df_timeRef['Time']:
        time_mask = ( pd.to_datetime(X_tmp['Time']) >= t - time_before) & ( pd.to_datetime(X_tmp['Time']) <= t + time_after)
        mw_row_avg = []
        ta_row_avg = []
        matched_rows = X_tmp[time_mask]
        ta_col = "temperature"
        if training_has_ta:
            if ta_col in matched_rows.columns:
                ta_values = matched_rows[ta_col].dropna()
                ta_row_avg.append(ta_values.mean() if not ta_values.empty else np.nan)
            else:
                ta_row_avg.append(np.nan)
        for i in range(1, 37):
            
            #     continue  # 跳過這個 i
            # MW數據
            # mw_col = f"MW{i}"#MW_absorb_normalized,_normalized
            mw_col = f"{i}-MW_NON"
            if mw_col in matched_rows.columns:
                mw_values = matched_rows[mw_col].dropna()
                mw_row_avg.append(mw_values.mean() if not mw_values.empty else np.nan)
            else:
                mw_row_avg.append(np.nan)
        mw_mat.append(np.hstack((ta_row_avg,mw_row_avg)))
    return np.array(mw_mat)  

#===============main======================================
#===========import dataset path================================
pca = principal_component_analysis()
plsr = PLSR_Analysis()
# file_path = r"C:\Users\Jason.lin\Desktop\workfile\20260107_luna微蝕data\luna微蝕data\luna微蝕data\Original_backtest_file.xlsx"#luna MW_data
file_path = r"C:\Users\Jason.lin\Desktop\workfile\20260116Luna_電鍍銅\Luna_電鍍銅all_data_for_Jason\all_data_for_Jason\25024Original_backtest_file_20251221_2CSV.csv"#luna MW_data電鍍
#=================================
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

# df = pd.read_excel(
#     r"C:\Users\Jason.lin\Desktop\workfile\20260107_luna微蝕data\luna微蝕data\luna微蝕data\LUNA_25026_MW_20251221.xlsx"
# )
# df = pd.read_excel("data_out\CombinationData_20251226.xlsx")
# df = pd.read_excel("data_out\data_SC1_25042for_AI.xlsx")
# df = pd.read_excel("data_out\data_SC1_25043for_AI.xlsx")
# df = pd.read_excel("data_out\data_SC1_25049for_AI.xlsx")
# df = pd.read_excel("data_out\CombinationData2_20260102_All42_43_49Machine.xlsx") 
# df = pd.read_excel("data_out\Original_backtest_file_normalizeCol.xlsx") # 43 and 49
# df = pd.read_excel("data_out\CombinationData_20251231.xlsx")# 43 and 49
df['Time'] = pd.to_datetime(df['Time'])
#==========================concentration Table path==========================
# df_timeRef = pd.read_excel("金像電本廠-化驗值(含標準點).xlsx")
# df_timeRef = pd.read_excel("data_out\concentration_list_SC-1藥水 DOE-20251224-含變溫(包含氨水重測)_pls.xlsx")
# df_timeRef = pd.read_excel( r"C:\Users\Jason.lin\Desktop\workfile\20260107_luna微蝕data\luna微蝕data\luna微蝕data\concentration_list_亞智luna25026_sa.xlsx",
#                             sheet_name="工作表1")#luna concentration
# df_timeRef = pd.read_excel( r"C:\Users\Jason.lin\Desktop\workfile\20260107_luna微蝕data\luna微蝕data\luna微蝕data\concentration_list_亞智luna25026_sa_backtest.xlsx",
#                             sheet_name="工作表3")#luna concentration
df_timeRef = pd.read_excel( r"C:\Users\Jason.lin\Desktop\workfile\20260116Luna_電鍍銅\Luna_電鍍銅all_data_for_Jason\all_data_for_Jason\concentration_list_Luna電鍍_2setupSA.xlsx",
                            sheet_name="工作表4")
#luna MW_data電鍍
# ==================
df_timeRef["Time"] = (df_timeRef["Time"].astype(str).str.replace(" PM", "", regex=False).str.replace(" AM", "", regex=False))
df_timeRef["Time"] = df_timeRef["Time"].astype(str).str.strip()
df_timeRef["Time"] = pd.to_datetime(df_timeRef["Time"],format="mixed",errors="coerce")
#==============Paremeter Setting==============
train_ratio = 0.8 #1 for all
Training_has_ta = False # True,False

split_ratio_idx = int(len(df_timeRef) * train_ratio)
df_timeRef_train = df_timeRef.iloc[0:split_ratio_idx,:]
print(split_ratio_idx,len(df_timeRef)-split_ratio_idx,len(df_timeRef))
#observing dataset
# df.head()
Concentration_slec_colnum = 2
Concentration_slec = np.array([1,2,3]) #要看幾種濃度液體
# Concentration_slec_num = 3 #要看幾種濃度液體
timedata = df.iloc[:,1:2]
# time_window = [timedelta(minutes=5), timedelta(minutes=0)]
time_window = [5, 0]#([before,after] minutes)
timedata.head()
#X是所有可能的影響變因
#取得所有的列的0,1,2,3,4欄位
# X_df = df.iloc[:,7+72:7+36+72]
# X_df.head()
# y是目標值
# print(df.columns)
# 清理欄名空格
df.columns = df.columns.str.strip()
# comp_cols = df_timeRef.columns.tolist()[Concentration_slec_colnum:(Concentration_slec_colnum+Concentration_slec_num)]
comp_cols = df_timeRef.columns[Concentration_slec_colnum + (Concentration_slec-1)].tolist()
# print(comp_cols)
comp_indices = [comp_cols.index(name) for name in comp_cols if name in comp_cols]

# Y.head()
#==============plot part I Rawdatas################
# prefix = 'MW'#-MW_NON
# run_plot_group_new(prefix,timedata, df_timeRef['NH4OH'], run_selected(df, time_window, df_timeRef))
# run_plot_group_scatter_new(prefix,timedata, df_timeRef[comp_cols[2]], run_selected(df, time_window, df_timeRef))
#==================================################
# X_tmp = df.iloc[:,7+72:7+36+72].values
# X_tmp = df.iloc[:,7+72:7+36+72]
channel_unselect = []
# channel_unselect = [
#         np.array([5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,30,31,32,34,35,36])
#                     ]
# channel_unselect = [
#         np.array([1,2,3,5,8,12,16,19,22,23,24]),
#         np.array([1,2,3,5,8,12,16,19,22,23,24]),
#         np.array([1,2,3,5,8,12,16,19,22,23,24])
#                     ]
channel_unselect = [
        np.array([1,2,3,5,7,8,12,14,19,20,21,22,23,24,26,27,28,29,31,32]),
        np.array([1,2,5,7,8,12,19,20,21,22,23,24,26,27,28,29,31,32]),
        np.array([1,2,3,5,8,12,16,19,22,23,24])
                    ]
# channel_unselect = [
#         np.array([5,6,9,10,11,12,22,23,24,30]),
#         np.array([5,6,9,10,11,12,22,23,24,30]),
#         np.array([5,6,9,10,11,12,22,23,24,30])
#                     ]
# channel_unselect = [
#         np.array([5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,30,31,32,34,35,36]),
#         np.array([5,6,7,8,9,10,20,21,22,23,24,30,31,32,33,34,35,36]),
#         np.array([5,6,7,8,9])
#                     ]
# channel_unselect = [
#      np.array([6,7,8,9,10,11,12,13,21,22,23,24,31,32,33,34,35,36]),
#      np.array([6,7,8,9,10,11,12,13,21,22,23,24,31,32,33,34,35,36]),
#      np.array([6,7,8,9,10,11,12,13,21,22,23,24,31,32,33,34,35,36])
#                     ]# luna
# channel_unselect = [
#     np.array([1,2,3,10,11,12,25,26,33,34]),
#     np.array([1,2,3,10,11,12,25,26,33,34])
#         ]# SC1
# channel_unselect = [
#      np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,19,20,21,22,23,24,29,30,31,32,35,36]),
#      np.array([1,2,3,10,11,12,18,19,20,21,22,23,24,25,26,27,31,32,33,34,35,36])
#                     ]# SC1
if not Training_has_ta:
    channel_unselect = [x - 1 for x in channel_unselect]  # 保證從0開始數  
                    
X = run_selected(df, time_window, df_timeRef_train,Training_has_ta)
X_PCA = pca.run_PCA_analyzer(X,Training_has_ta)
# Y = df.iloc[:, [i+Concentration_slec_colnum for i in comp_indices]].values
Y_temp = df_timeRef_train.iloc[:, [i+Concentration_slec_colnum for i in comp_indices]].values
toc = time.time()
word = f"preprocessing花費時間：{toc - tic:.3f} 秒" 
print(f"\x1b[32m{word}\x1b[0m")
# I
# # 1️⃣ 壓縮 y 的分布
pt = PowerTransformer(method="yeo-johnson")
Y_scaler = StandardScaler()
# Y_t = pt.fit_transform(Y_temp)
# # 2️⃣ 再做標準化（仍然建議）
# Y = Y_scaler.fit_transform(Y_t)

#  II
# Y = Y_scaler.fit_transform(Y_temp) #使用時記得去改plotgrop裡的(line157,line239)

# III
Y = Y_temp
# 執行PLS分析
# VI
# Y = run_Y_standart_Ind(Y_temp)

pls_result = plsr.run_pls_factor_scan(
                    X, channel_unselect, Y, comp_cols, max_factor = 16)
# 執行交叉驗證
cv_result = run_cross_validation_analysis(
    X, channel_unselect, Y, comp_cols, max_factor = 16
)

# 儲存結果（包含時間窗口設定）
# 儲存多算法結果
multi_algorithm_results = {}
multi_algorithm_results["算法X"] = {
    'pls': pls_result,
    'cv': cv_result,
    }

#==============plot part II PLS predict################

# run_plot_display_multi_algorithm_results(multi_algorithm_results)
# run_create_prediction_comparison_chart( multi_algorithm_results,Y_scaler)
run_plot_display_indepY_algorithm_results(multi_algorithm_results)
#==============Backtesting Part===========
model_name = "算法X"
factor_no_scaleY = multi_algorithm_results[model_name]['cv']['best_factor']
# factor = [2,2] # select by user choice SC1
# factor = [5,3,6] # select by user choice Luna
factor = [5,7,1] # select by user choice Luna電鍍
Output_calibrationYesorNo = False #False & True
#==============plot part III PLS predict################
run_create_prediction_comparison_chart( multi_algorithm_results,Y_scaler, factor)

#========varible importantance in projection
vip_scores_idx = pca.vip(multi_algorithm_results["算法X"]['pls']['factor_results'],factor,channel_unselect ,Training_has_ta)

# unique_key = f"{model_name}_F{factor}"

model_data = multi_algorithm_results[model_name]
pls_results = model_data.get('pls', {})
factor_results = pls_results.get('factor_results', {})

# model_info = factor_results[factor]
# pls_model = model_info.get('model')
# Y_scaler = model_info.get('Y_scalers')
# stats = model_info.get('stats', {})
# 4.2 準備預測數據
#================backtest data prepare=======================
# 判斷是否使用參考通道
# X_pred = X.copy() 
# df_backtest = pd.read_excel("data_out\data_25049for_AI.xlsx")
# X_pred = run_selected_specturm(df_backtest, channel_unselect)
#===============================================
X_pred_allSample = run_selected_specturm(df, channel_unselect,Training_has_ta)
df_backtest_allSample = df.copy()
X_pred_blindSample = run_selected(df_backtest_allSample, time_window, df_timeRef,Training_has_ta)

df_timeRef_backtest = df_timeRef.copy()
#=====================for luna test================

# df_timeRef_backtest = pd.read_excel( r"C:\Users\Jason.lin\Desktop\workfile\20260107_luna微蝕data\luna微蝕data\luna微蝕data\concentration_list_亞智luna25026_sa_backtest.xlsx",
#                                    sheet_name="工作表1")
# df_timeRef_backtest["Time"] = pd.to_datetime(df_timeRef_backtest["Time"],format="mixed",errors="coerce")
#=======================
# df_backtest = pd.read_excel(
#     r"C:\Users\Jason.lin\Desktop\workfile\20260107_luna微蝕data\luna微蝕data\luna微蝕data\LUNA_25026_MW_20251221.xlsx"
# )
# X_pred = run_selected(df_backtest, time_window, df_timeRef_backtest)
#================backtest data prepare=======================
backtest_result_allSample, backtest_result_allSample_no_scaleY= run_multi_model_backtest_NEWII(X,X_pred_allSample,factor,factor_no_scaleY, model_data, comp_cols,
                                            factor_results,model_name, channel_unselect,Output_calibrationYesorNo)

backtest_result_blindtest, backtest_result_blindtest_no_scaleY  = run_multi_model_backtest_NEWII(X,X_pred_blindSample,factor,factor_no_scaleY, model_data, comp_cols,
                                            factor_results,model_name, channel_unselect)
#==============plot part III Backtesting################

run_plot_backtest_results(backtest_result_allSample, df_backtest_allSample , backtest_result_allSample[f"算法X_F{factor}"]['comp_names'], df_timeRef_backtest ,Y_scaler,pt)
run_plot_backtest_results_with_score(backtest_result_blindtest, df_timeRef_backtest , backtest_result_blindtest[f"算法X_F{factor}"]['comp_names'], df_timeRef_backtest ,Y_scaler,pt,split_ratio_idx)

run_plot_backtest_results(backtest_result_allSample_no_scaleY, df_backtest_allSample , backtest_result_allSample_no_scaleY[f"算法X_F{factor_no_scaleY}"]['comp_names'], df_timeRef_backtest ,Y_scaler,pt)
run_plot_backtest_results_with_score(backtest_result_blindtest_no_scaleY, df_timeRef_backtest , backtest_result_blindtest_no_scaleY[f"算法X_F{factor_no_scaleY}"]['comp_names'], df_timeRef_backtest ,Y_scaler,pt,split_ratio_idx)

print('Done!!!!!!!!!!!!!!!!!!!!')








