#import modules
import pandas as pd
import numpy as np
import scipy.signal as signal
import h5py
import pickle
import os
import time
import matplotlib.pyplot as plt

from datetime import timedelta
from scipy.signal import medfilt
from scipy.signal import find_peaks
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from typing import Tuple, List, Dict, Any, Optional
from Factor_h5_file.H5dy import H5dy
from Backtesting import run_multi_model_backtest_NEWII
from Plotgrop import run_plot_group,run_plot_group_new,run_plot_group_newII,run_plot_group_scatter_new
from Plotgrop import run_plot_display_multi_algorithm_results,run_plot_display_indepY_algorithm_results, run_create_prediction_comparison_chart 
from Plotgrop import run_plot_backtest_results,run_plot_backtest_results_with_score
from Plot_Observation import run_plot_MW_zoom_out,run_plot_MW_vs_concentration,run_plot_MWLine_VS_MWwindow_scatter_new,run_plot_MW_zoom_out_process
# from plot_fun_group.plot_group import run_svr_prediction_comparison_chart,run_svr_display_CV_results

# 導入新的分析模組
from Principal_Component_Analysis import principal_component_analysis
from PLSR_Analysis.pls_with_cross_validation import PLS_with_cross_validation
from SVR_Analysis.svr_with_cross_validation import SVR_with_cross_validation
from SVR_Analysis.svr_multi_model_backtest import SVR_multi_model_backtest
from common.Pre_processing import Pre_processing
# 導入新的畫圖模組
from plot_fun_group.plot_group import SVR_plot_group
# import seaborn as sns


def run_selected_specturm(X_tmp_rows, unselect, training_has_ta,task_list):
    task_list_out = {}
    for task_idx, task_name in enumerate(task_list):
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
            mw_col = f"{i}-{task_name}"
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

def run_selected(X_tmp, time_window, df_timeRef,training_has_ta,task_list):
    task_list_out = {}
    for task_idx, task_name in enumerate(task_list):
        mw_mat = []
        time_before = pd.Timedelta(minutes=time_window[0])
        time_after = pd.Timedelta(minutes=time_window[1])
        # 找到時間窗口範圍內的所有數據點（非對稱）
        X_tmp_time = X_tmp['Time']
        for t in df_timeRef['Time']:
            time_mask = ( X_tmp_time >= t - time_before) & ( X_tmp_time <= t + time_after)
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
                mw_col = f"{i}-{task_name}"
                if mw_col in matched_rows.columns:
                    mw_values = matched_rows[mw_col].dropna()
                    mw_row_avg.append(mw_values.mean() if not mw_values.empty else np.nan)
                else:
                    mw_row_avg.append(np.nan)
            mw_mat.append(np.hstack((ta_row_avg,mw_row_avg)))
        task_list_out[task_name] = np.array(mw_mat)   
    return  task_list_out  

def run_selected_fast(X_tmp, time_window, df_timeRef,training_has_ta,task_list):
    task_list_out = {}
    for task_idx, task_name in enumerate(task_list):
        mw_mat = []
        ta_out_row_avg = []
        time_before = pd.Timedelta(minutes=time_window[0]).to_numpy()
        time_after = pd.Timedelta(minutes=time_window[1]).to_numpy()
        X_tmp_time = X_tmp['Time'].values
        df_ref_time = df_timeRef['Time'].values
        # 找到時間窗口範圍內的所有數據點（非對稱）
        for t in df_ref_time:
            start = t - time_before
            end   = t + time_after
            left  = np.searchsorted(X_tmp_time, start)
            right = np.searchsorted(X_tmp_time, end)
            mw_row_avg = []
            ta_row_avg = []
            matched_rows =  X_tmp.iloc[left:right]
            ta_col = "temperature"
            if training_has_ta:
                if ta_col in matched_rows.columns:
                    ta_values = matched_rows[ta_col].dropna()
                    ta_row_avg.append(ta_values.mean() if not ta_values.empty else np.nan)
                else:
                    ta_row_avg.append(np.nan)
            if ta_col in matched_rows.columns:
                ta_values = matched_rows[ta_col].dropna()
                ta_out_row_avg.append(ta_values.mean() if not ta_values.empty else np.nan)
            else:
                ta_out_row_avg.append(np.nan)        
            for i in range(1, 37):
                # continue  # 跳過這個 i
                # MW數據
                # mw_col = f"MW{i}"#MW_absorb_normalized,_normalized
                mw_col = f"{i}-{task_name}"
                if mw_col in matched_rows.columns:
                    mw_values = matched_rows[mw_col].dropna()
                    mw_row_avg.append(mw_values.mean() if not mw_values.empty else np.nan)
                else:
                    mw_row_avg.append(np.nan)
            mw_mat.append(np.hstack((ta_row_avg,mw_row_avg)))
        task_list_out[task_name] = np.array(mw_mat)   
    return  task_list_out, ta_out_row_avg

def run_selected_fast_ALL_n_blind(X_tmp, time_window, df_timeRef,split_ratio_idx,training_has_ta,task_list):
    '''
    進行一次篩選吐出三個資料:
    資料一:用滴定表篩選出時間的sample，再根據切割比率分割出部分sample來進行訓練
    資料二:用(全)滴定表篩選出時間的sample。如果沒進行切割資料一會等於資料二
    資料三:目的是篩選出預處理方法的MW資料，如MW_MON..來進行全數據Sample回測
    '''
    task_list_out_train = {}
    task_list_out_slec_sample = {}
    task_list_out_slec_task = {}
    for task_idx, task_name in enumerate(task_list):
        mw_mat = []
        mw_mat_all = []
        ta_out_row_avg = []
        time_before = pd.Timedelta(minutes=time_window[0]).to_numpy()
        time_after = pd.Timedelta(minutes=time_window[1]).to_numpy()
        X_tmp_time = X_tmp['Time'].values
        df_ref_time = df_timeRef['Time'].values
        # 找到時間窗口範圍內的所有數據點（非對稱）
        for t in df_ref_time:
            start = t - time_before
            end   = t + time_after
            left  = np.searchsorted(X_tmp_time, start)
            right = np.searchsorted(X_tmp_time, end)
            mw_row_avg = []
            mw_row_all_avg = []
            ta_row_avg = []
            ta_row_all_avg = []
            matched_rows =  X_tmp.iloc[left:right]
            ta_col = "temperature"
            if training_has_ta:
                if ta_col in matched_rows.columns:
                    ta_values = matched_rows[ta_col].dropna()
                    ta_row_avg.append(ta_values.mean() if not ta_values.empty else np.nan)
                else:
                    ta_row_avg.append(np.nan)
                if ta_col in X_tmp.columns:
                    ta_values = X_tmp[ta_col].dropna()
                    ta_row_all_avg.append(ta_values if not ta_values.isna().all() else np.nan)
                else:
                    ta_row_all_avg.append(np.nan)    
            if ta_col in matched_rows.columns:
                ta_values = matched_rows[ta_col].dropna()
                ta_out_row_avg.append(ta_values.mean() if not ta_values.empty else np.nan)
            else:
                ta_out_row_avg.append(np.nan)        
            for i in range(1, 37):
                # continue  # 跳過這個 i
                # MW數據
                # mw_col = f"MW{i}"#MW_absorb_normalized,_normalized
                mw_col = f"{i}-{task_name}"
                if mw_col in matched_rows.columns:
                    mw_values = matched_rows[mw_col].dropna()
                    mw_row_avg.append(mw_values.mean() if not mw_values.empty else np.nan)
                else:
                    mw_row_avg.append(np.nan)
                # 產生選擇預處理的所有 Xsample    
                if mw_col in X_tmp.columns:
                    mw_values = X_tmp[mw_col].dropna()
                    mw_row_all_avg.append(mw_values if not mw_values.isna().all() else np.nan) 
                else:
                    mw_row_all_avg.append(np.nan)    

            mw_mat.append(np.hstack((ta_row_avg,mw_row_avg)))
            # 整理出預處理的資料是否又加入溫度
            if len(ta_row_all_avg) == 0:
                mw_mat_all = mw_row_all_avg.copy()
            else:
                mw_mat_all = np.vstack((ta_row_all_avg, mw_row_all_avg))   

        task_list_out_train[task_name] = np.array(mw_mat)[0:split_ratio_idx,:]
        task_list_out_slec_sample [task_name] =np.array(mw_mat)
        task_list_out_slec_task[task_name] =np.array(mw_mat_all).T
           
    return  task_list_out_train, task_list_out_slec_sample ,task_list_out_slec_task, ta_out_row_avg

tic = time.time()
#===============main======================================
pcocess = Pre_processing()
pca = principal_component_analysis()
plsr_cv = PLS_with_cross_validation()
svr_cv= SVR_with_cross_validation()
svr_plot = SVR_plot_group()
svr_backtest = SVR_multi_model_backtest()
h5dy = H5dy()
#===========1.mport dataset path================================
file = r"C:\Users\Jason.lin\Desktop\workfile\建模資料庫"
# file_path = r"C:\Users\Jason.lin\Desktop\workfile\建模資料庫\20260225_九峰山\Original_backtest_file_九峰山.csv"
# file_path = r"C:\Users\Jason.lin\Desktop\workfile\20260224_佳美\for_Jason\Original_backtest_file佳美_combined_data_all.csv"
# file_path = r"C:\Users\Jason.lin\Desktop\workfile\20260209欣興超出化藥水\Original_backtest_file_欣興250209AP3006.csv"
# file_path = r"C:\Users\Jason.lin\Desktop\workfile\20260116Luna_電鍍銅\Luna_電鍍銅all_data_for_Jason\all_data_for_Jason\25024Original_backtest_file_20251221_2CSV.csv"#luna MW_data
# file_path = r"C:\Users\Jason.lin\Desktop\workfile\20260127_超幹\超幹data\Original_backtest_file.csv"#luna MW_data電鍍
# file_path = r"C:\Users\Jason.lin\Desktop\workfile\20260129_捷捷微\捷捷微forJason\捷捷微forJason\Original_backtest_file_20260112093132_20260116081236.csv"
# filename = "20260224_佳美"
#---------------------------------------------------------------
# filename = r"20260225_九峰山\Original_backtest_file_九峰山.csv"
filename = r"20260309_臨港基塔BOE\Original_backtest_file_臨港基塔BOE.csv"
# filename = r"20260311_超淦科技\Original_backtest_file_20260311_超淦科技.csv"
# filename = r"20260316_日本凸板\Original_backtest_file_20260316_日本凸板.csv"
# filename = r"20260318_SiliconBox\Original_backtest_file_2026018_SiliconBox.csv"
file_path = os.path.join(file, filename)
#=================================
df = pcocess.import_merged_file(file_path)
#  -----------雜訊太多嘗試減少取樣點---------
# zoom_out_ratio = 0.25
# linespace =np.linspace(0, len(df), 
#                             num=int(len(df)*zoom_out_ratio), 
#                             endpoint=False, 
#                             retstep=False, 
#                             dtype=int)
# df = df.iloc[linespace]
#-----------------------------------------
# df = pd.read_excel(
#     r"C:\Users\Jason.lin\Desktop\workfile\20260107_luna微蝕data\luna微蝕data\luna微蝕data\LUNA_25026_MW_20251221.xlsx"
# )
#observing dataset
# df.head()
#===========2.concentration Table path==========================
# df_timeRef = pd.read_excel("金像電本廠-化驗值(含標準點).xlsx")
# df_timeRef = pd.read_excel("data_out\concentration_list_SC-1藥水 DOE-20251224-含變溫(包含氨水重測)_pls.xlsx")
# df_timeRef = pd.read_excel( r"C:\Users\Jason.lin\Desktop\workfile\20260107_luna微蝕data\luna微蝕data\luna微蝕data\concentration_list_亞智luna25026_sa.xlsx",
#                             sheet_name="工作表1")#luna concentration
# df_timeRef = pd.read_excel( r"C:\Users\Jason.lin\Desktop\workfile\20260116Luna_電鍍銅\Luna_電鍍銅all_data_for_Jason\all_data_for_Jason\concentration_list_Luna電鍍_2setupSA.xlsx",
#                             sheet_name="工作表4")#luna concentration
# df_timeRef = pd.read_excel( r"C:\Users\Jason.lin\Desktop\workfile\20260127_超幹\超幹data\concentration_list_超幹SA.xlsx",
#                             sheet_name="工作表1")
# df_timeRef = pd.read_excel( r"C:\Users\Jason.lin\Desktop\workfile\20260129_捷捷微\捷捷微forJason\捷捷微forJason\concentration_list_HORIBA.xlsx",
#                             sheet_name="工作表1")
# df_timeRef = pd.read_excel( r"C:\Users\Jason.lin\Desktop\workfile\20260209欣興超出化藥水\concentration_list_欣興250209AP3006SA.xlsx",
#                             sheet_name="工作表3")
# df_timeRef = pcocess.import_excel_file( r"C:\Users\Jason.lin\Desktop\workfile\20260224_佳美\for_Jason\concentration_list_佳美SA.xlsx",
#                             sheet_name="工作表1")
# df_timeRef = pd.read_excel( r"C:\Users\Jason.lin\Desktop\workfile\建模資料庫\20260225_九峰山\concentration_list_九峰山SA.xlsx",
#                             sheet_name="工作表1")
# df_timeRef = pcocess.import_excel_file( r"C:\Users\Jason.lin\Desktop\workfile\20260224_佳美\for_Jason\concentration_list_file_佳美SA_MORE.csv",
#                             sheet_name="工作表1")
#==============================================================
# timeRef_filename = r"20260225_九峰山\concentration_list_九峰山SA.xlsx"
timeRef_filename = r"20260309_臨港基塔BOE\concentration_list_臨港基塔BOE_SA.xlsx"
# timeRef_filename = r"20260311_超淦科技\concentration_list_超淦SA.xlsx"
# timeRef_filename = r"20260316_日本凸板\concentration_list__CZ8401_AP3006_more2.csv"
# timeRef_filename = r"20260316_日本凸板\Concentration_list_CZ8401_AP3006.xlsx"
# timeRef_filename = r"20260318_SiliconBox\HORIBA_merged_data_all.xlsx"
#--------------------------------------------------------------
timeRef_file_path = os.path.join(file, timeRef_filename)
df_timeRef = pcocess.import_excel_file( timeRef_file_path,
                            sheet_name="工作表2")
#==============3.Paremeter Setting==============
#  #-----------滴定點太多使用downsample不調整滴定點數據長度---------
# zoom_out_ratio = 0.25
# yy = df_timeRef.iloc[0:10000,:]
# linespace =np.linspace(0, len(yy), 
#                             num=int(len(yy)*zoom_out_ratio), 
#                             endpoint=False, 
#                             retstep=False, 
#                             dtype=int)
# df_timeRef = df_timeRef.iloc[linespace]
#   #---------------濾波滴定點保留波峰波谷值-------------------------
# y = np.array(df_timeRef.iloc[:,2:3])
# y_smooth = medfilt(y.ravel(), kernel_size=3)
# peaks, _ = find_peaks(y_smooth, distance=10, prominence=0.02)
# valleys, _ = find_peaks(-y_smooth, distance=10, prominence=0.02)

# idx = np.sort(np.concatenate([peaks, valleys]))
# df_timeRef = df_timeRef.iloc[idx]
# ========== 3.Parameters Setting ================
train_ratio = 1 #1 for all,0.8 for 80%
Concentration_num =  2 #全部幾種濃度液體
Concentration_pred = np.array([1,2]) #要預測哪幾種濃度液體
Training_has_ta = False # True,False
# time_window = [timedelta(minutes=5), timedelta(minutes=0)]
time_window = [5,0]#([before,after] minutes) #捷捷薇選[6,-4]or[6,-5] #SiliconBox用[-4,6][-2,3]
# ========= Constant Parameters===============
Concentration_slec_colnum = 2
# Concentration_slec_num = 3 #要看幾種濃度液體
timedata = df.iloc[:,1:2]
timedata.head()
# y是目標值
# print(df.columns)
# comp_cols = df_timeRef.columns.tolist()[Concentration_slec_colnum:(Concentration_slec_colnum+Concentration_slec_num)]
comp_cols_all_name = df_timeRef.columns[Concentration_slec_colnum :(Concentration_slec_colnum + Concentration_num)].tolist()
comp_preds = df_timeRef.columns[Concentration_slec_colnum + (Concentration_pred-1)].tolist()
comp_indices = [comp_cols_all_name.index(name) for name in comp_preds if name in comp_cols_all_name]
# print(comp_cols_all_name)
# Y.head()
split_ratio_idx = int(len(df_timeRef) * train_ratio)
df_timeRef_train = df_timeRef.iloc[0:split_ratio_idx,:]
# Y_temp = df_timeRef_train.iloc[:, [i+Concentration_slec_colnum for i in comp_indices]].values
Y_temp = df_timeRef_train[comp_preds].values
print(split_ratio_idx,len(df_timeRef)-split_ratio_idx,len(df_timeRef))
#==================== Plot Ori rawdatas===================
prefix = 'MW'#-MW_NON
# run_plot_MW_vs_concentration(prefix,comp_cols_all_name[0],df_timeRef.iloc[8430:8630,:],df.iloc[3810:3910,:]) #滴定點太多，滴定時間跟MW有延遲現象
#                                                                                                    可用此圖加滴定Excel時間欄位來判斷時間window的選擇
#   ================= 1.XX縮點處理(加工處理)調整滴定點數據長度(只觀察前"1萬點到3萬點")====================
# df_timeRef_part = df_timeRef_train.iloc[0:3000,:]
# zoom_out_ratio = 0.25 
# linespace =np.linspace(0, len(df_timeRef_part), 
#                             num=int(len(df_timeRef_part)*zoom_out_ratio), 
#                             endpoint=False, 
#                             retstep=False, 
#                             dtype=int)
# df_timeRef_part = df_timeRef_part.iloc[linespace]
# #  -------------------1.XX縮點處理後畫圖---------------------------------
# has_zoom_df = run_selected(df, time_window, df_timeRef_part,Training_has_ta,["MW_NON"])
# run_plot_MW_zoom_out(prefix,comp_cols_all_name[0],df_timeRef_part,has_zoom_df["MW_NON"]) #比對滴定點與MW波的時間曲線
# run_plot_MW_zoom_out_process(prefix,comp_cols_all_name[0],df_timeRef_part,has_zoom_df["MW_NON"]) # 濾波滴定點保留波峰波谷值
#   ==============plot part I Rawdatas################

# run_plot_group(prefix,timedata,time_window, df_timeRef,df,Training_has_ta)# 滴定濃度穩定'時間內'的通道光譜強度
# run_plot_group_newII(prefix,timedata,df_timeRef[comp_cols_all_name[0]], run_selected(df, time_window, df_timeRef,Training_has_ta,["MW_NON"])["MW_NON"])#各滴定通道與強度圖(加工圖)
#==================================
# sel_df = run_selected(df, time_window, df_timeRef.iloc[8430:8530],Training_has_ta,["MW_NON"])
# ##timeRef = df_timeRef['Horiba-NH4OH'][0:1000]
# timeRef = df_timeRef[comp_cols_all_name[0]][3810:3910]
#---------------------------------------------------------
# sel_df = run_selected(df, time_window, df_timeRef,Training_has_ta,["MW_NON"])
# timeRef = df_timeRef[comp_cols_all_name[0]]
# run_plot_group_new(prefix,timedata, timeRef, sel_df["MW_NON"]) # 各通道強度MW與濃度趨勢圖
# run_plot_group_scatter_new(prefix,timedata, timeRef, sel_df["MW_NON"]) # 各通道MW強度與濃度趨勢圖(點圖)
# run_plot_MWLine_VS_MWwindow_scatter_new(prefix,df_timeRef, comp_cols_all_name,Training_has_ta,df, sel_df["MW_NON"]) # 各通道原始MW vs 取平均的MW =>看MW有沒有取到跳點
#================= 4.CH selection==========
channel_unselect = []
# channel_unselect = [
#         np.array([]),
#         np.array([1,2,3,4,5,6,7,8,15,16,17,18,19,29,30,31,32,33,34,35,36]),
#         np.array([1,2,3,4,5,6,7,8,30,31,32,33,34,35,36])
#                     ]#日本凸版20260316
# channel_unselect = [
#         np.array([]),
#         np.array([1,2,3,4,5,6,7,8,29,30,31,32,33,34,35,36]),
#         np.array([1,2,3,4,5,6,7,8,29,30,31,32,33,34,35,36])
#                     ]#日本凸版more20260316
# channel_unselect = [
#         np.array([4,5,6,9,10,20,21,22,23,24,25,26,28])
#                     ]
# channel_unselect = [
#         np.array([7,8,9,10,12,13,14,15,16,17,18,20,21,22,23,24,29,30,33])
#                     ]##superFXXX_observation可見光ch7-12 
# channel_unselect = [
#         np.array([8,10,12,13,14,15,16,17,18,20,21,22,23,24,29,30,33])
#                     ]#superFXXX_Computer
# channel_unselect = [
#         np.array([8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
#                     ]#superFXXX_EYES
# channel_unselect = [
#         np.array([9,10,11,12,13,14,15,17,18,19,20,21,22,25,26,27,28,29,30,31,32,33,34,35,36]),
#         np.array([9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
#                     ]#捷捷薇_EYES
# channel_unselect = [
#         np.array([9,10,20,21,22,25,26,27,28,29,30,31,32,33,34,35]),
#         np.array([9,10,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35])
#                     ]#捷捷薇_Computer
# channel_unselect = [
#      np.array([6,7,8,9,10,11,12,13,21,22,23,24,31,32,33,34,35,36]),
#      np.array([6,7,8,9,10,11,12,13,21,22,23,24,31,32,33,34,35,36]),
#      np.array([6,7,8,9,10,11,12,13,21,22,23,24,31,32,33,34,35,36])
#                     ]# luna
# channel_unselect = [
#      np.array([1,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]),
#      np.array([1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,31,33,34,35,36]),
#      np.array([1,2,3,4,5,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
#                     ]#興欣eye
# channel_unselect = [
#      np.array([8,9,10,13,14,15,16,17,20,27,28,30,31,32,33,34,35,36]),
#      np.array([5,6,7,8,11,13,14,16,17,20,25,27,28,35,36]),
#      np.array([5,6,7,8,11,13,14,16,17,20,25,27,28,35,36])
#                     ]#興欣computer,工作表3
# channel_unselect = [
#      np.array([1,8,9,10,13,14,15,16,17,20,21,22,23,24,26,27,28,30,31,32,33,34,35,36])
#                     ]#看濃度1興欣computer,工作表3
# channel_unselect = [
#      np.array([1,2,3,4,5,6,7,8,11,13,14,16,17,20,25,26,27,28,35,36]),
#      np.array([1,2,3,4,5,6,7,8,11,13,14,16,17,20,25,26,27,28,35,36])
#                     ]#看濃度2,3,興欣computer,工作表3
# channel_unselect = [
#         np.array([5,6,7,8,11,13,14,16,17,20,25,27,28,35,36]),
#         np.array([5,6,7,8,11,13,14,16,17,20,25,27,28,35,36])
#                     ]#看濃度2,3,興欣computer,工作表3
# channel_unselect = [
#         np.array([9,14,18,27,28,29,31,32,33,36]),
#         np.array([1,2,3,5,6,8,13,14,18,19,20,25,26,27,28])
                    # ]#,佳美,工作表1
# channel_unselect = [
#         np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,27,28,29,31,32,33,36]),
#         np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,19,20])
#                     ]#,佳美2,工作表1
# channel_unselect = [
#         np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,27,28,29,31,32,33,36]),
#         np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,19,20,25,26,27,28])
#                     ]#,佳美2,工作表1
# channel_unselect = [
#         np.array([1,2,3,4,5,6,7,8,9,10,11,12,24,28,29,31,32,33,34,35,36]),
#         np.array([1,2,3,4,5,6,7,8,9,10,11,12,24,28,29,31,32,33,34,35,36])
#                     ]# 九峰山,工作表1
# channel_unselect = [
#         np.array([1,2,3,4,5,6,7,8,9,10,11,12,17,18,25,30,32,33,34,35,36]),
#         np.array([1,2,3,4,5,6,7,8,9,10,11,12,17,18,25,30,32,33,34,35,36])
#                     ]# 九峰山,工作表1
channel_unselect = [
        np.array([1,2,3,4,5,6,7,8,9,10,11,12]),
        np.array([1,2,3,4,5,6,7,8,9,10,11,12])
                    ]# 九峰山,工作表1,臨港基塔,工作表2
# channel_unselect = [
#         np.array([9,13,14,15,16,17,18,19,20,21,22,23,24,32,33,34,35,36]),
#         np.array([9,13,14,15,16,17,18,19,20,21,22,23,24,32,33,34,35,36])
#                     ]
# channel_unselect = [
#         np.array([1,2,3,7,8,11,12,13,14,15,16,18,9,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
#                     ]
if not Training_has_ta:
    channel_unselect = [x - 1 for x in channel_unselect]  # 保證從0開始數  
#================== 5.X Y input rawdatas processing====================  
print('開始數據篩選!!!!!!!!!!!!!!!!!!!!')    
pre_processing_method_task_list =["MW_NON"] #,"MW_normalized_maxCol","MW_absorb"
              
# #X_list,_ = run_selected_fast(df, time_window, df_timeRef_train,Training_has_ta,pre_processing_method_task_list)
X_list, X_list_blind , X_list_task,selected_df_TA = run_selected_fast_ALL_n_blind(df, time_window, df_timeRef_train, split_ratio_idx,Training_has_ta,pre_processing_method_task_list)

# ================= XX點數太多縮點處理(加工處理)用下面的====================
# # X_list ,selected_df_TA = run_selected_fast(df, time_window,df_timeRef_part,Training_has_ta,pre_processing_method_task_list)
# X_list, X_list_blind , X_list_task, selected_df_TA = run_selected_fast_ALL_n_blind(df, time_window, df_timeRef_part, split_ratio_idx,Training_has_ta,pre_processing_method_task_list)
# Y_temp = df_timeRef_part[comp_preds].values
# split_ratio_idx = int(len(df_timeRef_part) * train_ratio)
# print(split_ratio_idx,len(df_timeRef_part)-split_ratio_idx,len(df_timeRef_part))
# ======================================================================
# Y = df.iloc[:, [i+Concentration_slec_colnum for i in comp_indices]].values # 滴定濃度跟rawdatas放再一起 
toc = time.time()
word = f"preprocessing花費時間：{toc - tic:.3f} 秒" 
print(f"\x1b[32m{word}\x1b[0m")
# --------I
# # 1️⃣ 壓縮 y 的分布
pt = PowerTransformer(method="yeo-johnson")
Y_scaler = StandardScaler()
# Y_t = pt.fit_transform(Y_temp)
# # 2️⃣ 再做標準化（仍然建議）
# Y = Y_scaler.fit_transform(Y_t)
# ---------II
# Y = Y_scaler.fit_transform(Y_temp) #使用時記得去改plotgrop裡的(line157,line239)
# ---------III
Y = Y_temp

# ----------VI
# Y = run_Y_standart_Ind(Y_temp)
# 執行PLS分析
# 儲存多算法結果
multi_algorithm_results = {}
multi_algorithm_results_SVR = {}
for algorithm_name in pre_processing_method_task_list[0:3]: # 0:3等於取全算法 , 先暫時利用0:1 或1:2取單一的算法回測
    X = X_list[algorithm_name]
    # X = signal.medfilt(X_list[algorithm_name], kernel_size=(3, 1))#使用中值率波
    X_PCA = pca.run_PCA_analyzer(X,algorithm_name,Training_has_ta)#光譜通道數量很大時 如:100~，再使用X_PCA當輸入X
    X_In, Y_In = pcocess.preprocess_data(X,Y)
    # ============== 6.1PLSR model ===================
    tic = time.time()
    
    multi_algorithm_results = plsr_cv.run_PLS_with_cross_validation(X_In, channel_unselect, Y_In, comp_preds,multi_algorithm_results, algorithm_name,max_factor = 16)
       
    toc = time.time()
    word = f"運算花費時間：{toc - tic:.3f} 秒" 
    print(f"\x1b[32m{word}\x1b[0m")

    # ----- 7.varible importantance in projection------
    # vip_scores_idx = pca.vip(multi_algorithm_results,
    #                          algorithm_name,channel_unselect ,Training_has_ta)
    # ========== Save data as H5 =====================
    # with h5py.File("Factor_h5_file\data.h5", "w") as f:
    #     h5dy.save_dict_to_h5(f, multi_algorithm_results)
    # with h5py.File("Factor_h5_file\data.h5", "w") as f:
    #     for k, v in multi_algorithm_results.items():
    #         f.create_dataset(k, data=v)

    # ============== 6.2 SVR model ===================
    tic = time.time()
    multi_algorithm_results_SVR = svr_cv.run_SVR_with_cross_validation(X_In, channel_unselect, Y_In, comp_preds,multi_algorithm_results_SVR, algorithm_name)
    toc = time.time()
    word = f"SVR運算花費時間：{toc - tic:.3f} 秒" 
    print(f"\x1b[32m{word}\x1b[0m")

print('模型運算Done!!!!!!!!!!!!!!!!!!!!') 
plt.close('all') 
#==============8.plot part II PLS predict =================
Ob_indices = [0] #選擇要看的預處理方法表裡的方法   [0,1,2]
# Ob_list = [f"算法{pre_processing_method_task_list[i]}" for i in Ob_indices] 
Ob_list = [list(multi_algorithm_results.keys())[i] for i in Ob_indices]
Ob_multi_algorithm_results = {k: v for k, v in multi_algorithm_results.items() if k in Ob_list}
Ob_multi_algorithm_results_SVR = {k: v for k, v in multi_algorithm_results_SVR.items() if k in Ob_list}
# for name in Ob_list
#----------------------PLSR ------------------------------------
run_plot_display_multi_algorithm_results(Ob_multi_algorithm_results)
run_create_prediction_comparison_chart(Ob_multi_algorithm_results,Y_scaler)
run_plot_display_indepY_algorithm_results(Ob_multi_algorithm_results)

#==============plot part III PLS predict ===============================
## model_name =f"算法{algorithm_name}"
## factor = multi_algorithm_results[model_name]['cv']['best_factor_StdY']
# #----------------------------------------------------------------------
factor = [] #手動設定 不須關閉可[ ]
run_create_prediction_comparison_chart( Ob_multi_algorithm_results,Y_scaler, factor)
#----------------------SVR ------------------------------------
svr_plot.run_svr_display_CV_results(multi_algorithm_results_SVR)
svr_plot.run_svr_prediction_comparison_chart(multi_algorithm_results_SVR)

plt.close('all') 
print('factor觀察Done!!!!!!!!!!!!!!!!!!!!') 
Output_calibrationYesorNo = False #False & True
# unique_key = f"{model_name}_F{factor}"
# model_info = factor_results[factor]
# pls_model = model_info.get('model')
# Y_scaler = model_info.get('Y_scalers')
# stats = model_info.get('stats', {})
#============== 9.Backtesting Part ===========
# 儲存多算法回測結果
backtest_result_allSample= {}
backtest_result_allSample_no_scaleY= {}
backtest_result_blindtest= {}
backtest_result_blindtest_no_scaleY= {}

To_backtest_indices = [0] #選擇處理方法表裡的方法進行回測與 輸出減量線   [0,1,2]
# #backtest_method_task_list = [pre_processing_method_task_list[i] for i in To_backtest_indices] 
backtest_method_task_list = [list(multi_algorithm_results.keys())[i] for i in To_backtest_indices]
backtest_method_svr_task_list = [list(multi_algorithm_results_SVR.keys())[i] for i in To_backtest_indices]
for model_name in backtest_method_task_list:
    algorithm_name = model_name.split(r'算法')[1].split(r'_(')[0]
    #------------PLSR 手動更改 fators -------------------------------
    factor_no_scaleY = multi_algorithm_results[model_name]['cv']['best_factor']
    factor = multi_algorithm_results[model_name]['cv']['best_factor_StdY']
    # factor = [4] # 手動設定StdY 改變要吐出的檢量線
    # factor_no_scaleY = 4 # 手動設定 改變要吐出的檢量線
    #------------SVR 手動更改 params -------------------------------
    # param_set = cv_result['best_parameters_set']
    model_data = multi_algorithm_results[model_name]
    # pls_results = model_data.get('pls', {})
    # factor_results = pls_results.get('factor_results', {})
    #===============10.準備預測數據=============
    # #X_pred_allSample = run_selected_specturm(df, channel_unselect,Training_has_ta,[algorithm_name])
    df_backtest_allSample = df.copy()
    tic = time.time()
    # X_pred_blindSample, selected_df_TA = run_selected_fast(df_backtest_allSample, time_window, df_timeRef,Training_has_ta,[algorithm_name])
    # X_pred_blindSample = X_list_blind[algorithm_name] 
    # ================= XX點數太多縮點處理(加工處理)用下面的====================
    # X_pred_blindSample = run_selected_fast(df_backtest_allSample, time_window, df_timeRef_part, Training_has_ta,[algorithm_name])
    # toc = time.time()
    # word = f"select花費時間：{toc - tic:.3f} 秒" 
    # print(f"\x1b[32m{word}\x1b[0m")
    df_timeRef_backtest = df_timeRef.copy()
    # #df_timeRef_backtest = df_timeRef_train.copy()
    # ================= XX點數太多縮點處理(加工處理)用下面的====================
    # df_timeRef_backtest = df_timeRef_part.copy()
    #================11.backtest data prepare=======================
    backtest_result_allSample, backtest_result_allSample_no_scaleY= run_multi_model_backtest_NEWII(backtest_result_allSample, backtest_result_allSample_no_scaleY,
                                                                                                   X,(X_list_task[algorithm_name]),factor,factor_no_scaleY, model_data, comp_preds,
                                                model_name, channel_unselect,filename,Output_calibrationYesorNo)

    backtest_result_blindtest, backtest_result_blindtest_no_scaleY  = run_multi_model_backtest_NEWII(backtest_result_blindtest, backtest_result_blindtest_no_scaleY,
                                                                                                     X,X_list_blind[algorithm_name]  ,factor,factor_no_scaleY, model_data, comp_preds,
                                                model_name, channel_unselect,filename)
# ===============執行SVR回測分析======================================
#暫時手動更改功能
Output_SVR_calibrationYesorNo = False #False & True
param_sets = []
# param_sets.append({'Epsilon': 0.001, 'Cost': 0.464})# E:[0.001, 0.003, 0.006, 0.008, 0.01 ]
# param_sets.append({'Epsilon': 0.001, 'Cost': 21.544}) # C:[0.1 ,0.215,0.464,1.0,2.154,4.642,10.0,21.544,46.416,100.0]
# param_set['Epsilon'] = 0.005
# param_set['Cost'] = 17.783
svr_backtest_result_allSample =svr_backtest.run_svr_multi_model_backtest(X_list,X_list_task, multi_algorithm_results_SVR,backtest_method_svr_task_list,comp_preds, param_sets,Output_SVR_calibrationYesorNo)     
svr_backtest_result_blindtest =svr_backtest.run_svr_multi_model_backtest(X_list,X_list_blind, multi_algorithm_results_SVR,backtest_method_svr_task_list,comp_preds, param_sets)    
#==============plot part III Backtesting################
# 選擇要看的預處理方法表裡的方法  [0,1,2] 
Ob_backtest_results_indices = [0]   
# #準備資料
df_TA = df['temperature'].values
# #標準化 Y
Ob_backtest_results_list = [list(backtest_result_allSample.keys())[i] for i in Ob_backtest_results_indices]
Ob_multi_allSample_results = {k: v for k, v in backtest_result_allSample.items() if k in Ob_backtest_results_list}
Ob_multi_blindtest_results = {k: v for k, v in backtest_result_blindtest.items() if k in Ob_backtest_results_list}
# #svr 標準化 Y
svr_Ob_backtest_results_list = [list(svr_backtest_result_allSample.keys())[i] for i in Ob_backtest_results_indices]
svr_Ob_multi_allSample_results = {k: v for k, v in svr_backtest_result_allSample.items() if k in svr_Ob_backtest_results_list }
svr_Ob_multi_blindtest_results = {k: v for k, v in svr_backtest_result_blindtest.items() if k in svr_Ob_backtest_results_list }
# #沒有標準化 Y
Ob_backtest_results_list_no_scaleY = [list(backtest_result_allSample_no_scaleY.keys())[i] for i in Ob_backtest_results_indices]
Ob_multi_allSample_results_no_scaleY = {k: v for k, v in backtest_result_allSample_no_scaleY.items() if k in Ob_backtest_results_list_no_scaleY}
Ob_multi_blindtest_results_no_scaleY = {k: v for k, v in backtest_result_blindtest_no_scaleY.items() if k in Ob_backtest_results_list_no_scaleY}

run_plot_backtest_results(Ob_multi_allSample_results, df_backtest_allSample , backtest_result_allSample[f"{model_name}_F{factor}"]['comp_names'], df_timeRef_backtest , df_TA ,Y_scaler,pt)
run_plot_backtest_results_with_score(Ob_multi_blindtest_results, df_timeRef_train , backtest_result_blindtest[f"{model_name}_F{factor}"]['comp_names'], df_timeRef_train ,selected_df_TA,Y_scaler,pt,split_ratio_idx)

# run_plot_backtest_results(Ob_multi_allSample_results_no_scaleY, df_backtest_allSample , backtest_result_allSample_no_scaleY[f"{model_name}_F{factor_no_scaleY}"]['comp_names'], df_timeRef_backtest, df_TA ,Y_scaler,pt)
# run_plot_backtest_results_with_score(Ob_multi_blindtest_results_no_scaleY, df_timeRef_backtest , backtest_result_blindtest_no_scaleY[f"{model_name}_F{factor_no_scaleY}"]['comp_names'], df_timeRef_backtest, selected_df_TA ,Y_scaler,pt,split_ratio_idx)

svr_plot.run_plot_backtest_results(svr_Ob_multi_allSample_results,df,comp_cols_all_name,df_timeRef, df_TA )
svr_plot.run_plot_backtest_results_with_score(svr_Ob_multi_blindtest_results,df_timeRef_train,comp_cols_all_name, df_timeRef_train, selected_df_TA ,split_ratio_idx)
plt.close('all') 


print('Done!!!!!!!!!!!!!!!!!!!!')
 








