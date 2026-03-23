import numpy as np
import pandas as pd
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
import textwrap
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Callable, Optional
from sklearn.linear_model import LinearRegression
# 導入新的分析模組
from common.Pre_processing import Pre_processing
matplotlib.rc('font', family='serif', serif=['ABC', 'MingLiU']) 
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def read_folder_tables(folder_path, sheet_number=0):
    output = {}
    output_method_name = []
    files = sorted(os.listdir(folder_path))

    for file in files:
        if not file.lower().endswith(('.xls', '.xlsx')):
            continue

        full_path = os.path.join(folder_path, file)
        name = os.path.splitext(file)[0]
        method= name.split(r'算法')[1].split(r'_(')
        # method1 = name.split(r'算法')[1]
        # method =  re.split('_無標準化|_標準化',method1)

        df = pd.read_excel(full_path, sheet_name=sheet_number,index_col=0)
        output[name] = df
        output_method_name.append(method[0])

    return output,output_method_name 

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
        task_list_out[task_name] = np.array(mw_mat).T        
    return task_list_out

def run_selected_fast(X_tmp, time_window, df_timeRef,training_has_ta,task_list):
    task_list_out = {}
    for task_idx, task_name in enumerate(task_list):
        mw_mat = []
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
    return  task_list_out


def msc(X):
    """
    Perform Multiplicative Scatter Correction (MSC) on the given spectral data.
    
    Parameters:
    X (ndarray): The input spectral data matrix, where rows are samples and columns are wavelengths.
    
    Returns:
    ndarray: The MSC corrected spectral data.
    """
    # Calculate the mean spectrum
    X_ref = np.mean(X, axis=0)
    
    # Initialize the corrected spectra matrix
    X_msc = np.zeros_like(X)
    
    # Perform MSC on each spectrum
    for i in range(X.shape[0]):
        # Reshape data for LinearRegression
        X_ref_reshaped = X_ref.reshape(-1, 1)
        X_i_reshaped = X[i].reshape(-1, 1)
        
        # Fit linear regression model
        model = LinearRegression().fit(X_ref_reshaped, X_i_reshaped)
        a = model.coef_[0][0]
        b = model.intercept_[0]
        
        # Apply correction
        X_msc[i] = (X[i] - b) / a

    return X_msc    

# #===============main=====================
if __name__ == '__main__':
    pcocess = Pre_processing()

    #===========1.mport dataset path================================
    file = r"C:\Users\Jason.lin\Desktop\workfile\建模資料庫"
    #---------------------------------------------------------------
    # filename = r"20260127_超幹\Original_backtest_file.csv"
    # filename = r"20260309_臨港基塔BOE\Original_backtest_file_臨港基塔BOE.csv"
    # filename = r"20260311_超淦科技\Original_backtest_file_20260311_超淦科技.csv"
    # filename = r"20260225_九峰山\Original_backtest_file_九峰山_OQC.csv"
    # filename = r"20260225_九峰山\Original_backtest_file_九峰山.csv"
    # filename = r"20260316_日本凸板\Original_backtest_file_20260316_日本凸板.csv"
    # filename = r"20260318_SiliconBox\Original_backtest_file_2026018_SiliconBox.csv"
    # file_path = os.path.join(file, filename)
    #---------------------------------------------------------------
    file_path = r"C:\Users\Jason.lin\Desktop\workfile\20260224_佳美\for_Jason\Original_backtest_file佳美_combined_data_all.csv"
    df = pcocess.import_merged_file(file_path)
    #===========2.concentration Table path==========================
    # timeRef_filename = r"20260127_超幹\concentration_list_超幹SA.xlsx"
    # timeRef_filename = r"20260309_臨港基塔BOE\concentration_list_臨港基塔BOE_SA.xlsx"
    # timeRef_filename = r"20260311_超淦科技\concentration_list_超淦SA.xlsx"
    # timeRef_filename = r"20260225_九峰山\concentration_list_九峰山SAOQC.xlsx"
    # timeRef_filename = r"20260225_九峰山\concentration_list_九峰山SA.xlsx"
    # timeRef_filename = r"20260316_日本凸板\concentration_list__CZ8401_AP3006_more2.csv"
    # timeRef_filename = r"20260318_SiliconBox\HORIBA_merged_data_all.xlsx"
    # timeRef_file_path = os.path.join(file, timeRef_filename)
    #---------------------------------------------------------------
    df_timeRef = pcocess.import_excel_file( r"C:\Users\Jason.lin\Desktop\workfile\20260224_佳美\for_Jason\concentration_list_佳美SA.xlsx",
                            sheet_name="工作表4")
    # df_timeRef = pcocess.import_excel_file( timeRef_file_path,
    #                             sheet_name="工作表3")
    #===========3.import calibration path to load Excel================================
    df_calibration_file = "multi_cailbration_backtest_Folder"
    df_calibration_all ,method_task_list = read_folder_tables(df_calibration_file)

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
    #================= 獲取 X 與 時間=======================
    channel_unselect = []
    unique_names_method_task_list = list(set(method_task_list))
    # 準備資料
    time_data = df['Time'].values
    X_pred= run_selected_specturm(df, channel_unselect,Training_has_ta,unique_names_method_task_list)
    # X_pred['MW_NON'] = msc(X_pred['MW_NON'])
    #================= 結束獲取 X=======================
    # 開始畫圖
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
            method_key = name.split(r'算法')[1].split(r'_(')[0] 
            # method_key1 = name.split(r'算法')[1]
            # method_key = re.split('_無標準化|_標準化',method_key1)[0]
            intercepts = df_calibration.loc["intercept"].values #"B_0" , "intercept"
            # 4. 提取係數矩陣 (第 1 行之後，保留所有行)
            coef_df = df_calibration.iloc[1:]  # 去掉 intercept 行
            
            # 5. 提取所有行索引作為特徵名稱
            spec_cols = coef_df.index.tolist()  # 例如 ['TA', 'MWTN-1', 'MWTN-2', ..., 'MWTN-36']
            
            # 6. 提取完整係數矩陣（包含零值行）
            coefs = coef_df.values 
            Y_pred = X_pred[method_key].dot(coefs) + intercepts 

            
            # 準備顏色
            colors = plt.cm.tab10(np.linspace(0, 1, 10))  # tab10 色盤
            marker = markers[idx % len(markers)] # 改變maker
            # 選擇顏色
            color = colors[idx % len(colors)]
            nums = 3 #  20個點畫出一個
            ax.plot(
                    time_data[::nums], 
                    # Y_pred[:, page]+bias[page],
                    Y_pred[::nums, page],
                    linestyle='-',
                    color=color,
                    marker= marker,
                    markerfacecolor=color,
                    markersize=2,
                    label = f"predict_{name}",
                    alpha=0.3
                )
        #加入滴定數據    
        ref_time = df_timeRef['Time'].values
        ref_values = df_timeRef[comp_preds[page]].values
        # 繪製參考數據為紅色星形標記
        ax.plot(
            ref_time,
            ref_values,
            linestyle=' ',
            color='red',
            marker='.',
            markersize=3,
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
            loc='upper left',
            bbox_to_anchor=(0.005, 0.99),
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

    #================點數少的製作表格===================
    # 準備X資料
    time_window = [0.5,0.5] #找預測點的前後一分鐘平均 or[0.5,0.5]
    X_list = run_selected_fast(df, time_window, df_timeRef,Training_has_ta,unique_names_method_task_list)
    
    #這裡用6種顏色做循環
    color_list = ["#89C6C6","#BC95E2","#BE7A7A","#91AF73","#9C5B9C","#AAAA69"]
    #column文字標籤
    column_labels=['Time']
    for page in range(len(comp_preds)):
        fig, ax = plt.subplots(figsize=(10, 5))

        # =====獲取檢量線資訊====
        row_labels = []
        data_All = df_timeRef['Time'].dt.strftime('%Y-%m-%d \n %H:%M:%S').values
        ref_values = df_timeRef[comp_preds[page]].values
        color=[color_list[-1]]
        for idx, name in enumerate(df_calibration_all):
            #選擇顏色
            colors = color_list[idx % len(color_list)]
            name_parts= name.split('_')[:]
            # name_temp = '_'.join(name_parts)
            if len(name_parts)<=8:
                name_temp = '_'.join([name_parts[i] for i in [0,1,2,3,4,6]])
            else:
                name_temp = '_'.join([name_parts[i] for i in [0,1,2,3,5,6,8]])
            column_labels.append(f"{name_temp}理論值")
            column_labels.append(f"{name_temp}預測值")

            df_calibration = df_calibration_all[name]
            method_key = name.split(r'算法')[1].split(r'_(')[0] 
            # method_key1 = name.split(r'算法')[1]
            # method_key = re.split('_無標準化|_標準化',method_key1)[0]
            intercepts = df_calibration.loc["intercept"].values #"B_0" , "intercept"
            # 4. 提取係數矩陣 (第 1 行之後，保留所有行)
            coef_df = df_calibration.iloc[1:]  # 去掉 intercept 行
            
            # 5. 提取所有行索引作為特徵名稱
            spec_cols = coef_df.index.tolist()  # 例如 ['TA', 'MWTN-1', 'MWTN-2', ..., 'MWTN-36']
            
            # 6. 提取完整係數矩陣（包含零值行）
            coefs = coef_df.values 
            Y_pred = X_list[method_key].dot(coefs) + intercepts
            data_All = np.column_stack((data_All, np.round(ref_values, 3)))
            data_All = np.column_stack((data_All, np.round(Y_pred[:, page], 3)))
            # color = np.hstack( (color,[colors,colors]))
            color.append([colors,colors])
        #關閉座標軸線
        ax.axis('off')
        column_labels = [
                        "\n".join(textwrap.wrap(label, 9))
                        for label in column_labels
                        ]
        # ax.table(cellText = data_All,colLabels=column_labels,loc="center")
        num = 6
        color_temp = [np.hstack(color) ] * (num)
        # color = np.hstack(color_temp)
        # color = np.hstack(color)
        table = ax.table(cellText=data_All[0:num], colLabels=column_labels,
                         colColours=color_temp[0],cellColours=color_temp, loc="center", cellLoc='center')
        table.auto_set_font_size(False) # 開啟會原生字太小
        table.set_fontsize(10)
        #設置表格長跟寬
        for cell in table.get_celld().values():
            cell.set_width(0.08)
            cell.set_height(0.15)
            

        ax.axis('tight')   
        # 調整佈局
        table.scale(1.3, 0.85) # 調整表格大小 
        # 設置圖表標題和標籤
        ax.set_title(
            f"{comp_preds[page]} 模型回測數值({num}筆)對比",
            fontsize=16,
            fontweight='bold'
        )    
        plt.tight_layout()    
    plt.show()    
    print('Done!!!!!!!!!!!!!!!!!!!!')
