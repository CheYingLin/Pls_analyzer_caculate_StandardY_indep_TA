import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import h5py
import os
import math
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from Factor_h5_file.H5dy import H5dy

matplotlib.rc('font', family='serif', serif=['ABC', 'MingLiU']) 
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SVR_plot_group:
    """導入外部畫圖函式"""
    def __init__(self,):
        pass

    def regression_score(self,y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return r2 ,rmse, mae, mape

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

    def run_svr_display_CV_results(self,multi_results):
        for algorithm_name, results in multi_results.items():               
            # 在該算法分頁中創建第四層Tab (Factor vs EV / 預測對比)
            svr_result = results['svr_model']
            cv_result = results['svr_cv']
            """創建 Param vs EV 趨勢圖"""
            comp_cols = svr_result ['comp_cols']

            # 準備數據
            #X軸資訊 
            param_grid = svr_result['param_grid']
            costs = param_grid['Cost']
            epsilons = param_grid['Epsilon']

            # 創建子圖
            n_comp = len(comp_cols)
            cols = min(n_comp, 2)
            rows = (n_comp + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows),dpi = 80)
            if n_comp == 1:
                axes = [axes]
            else:
                axes = np.array(axes).flatten()

            for idx, comp in enumerate(comp_cols):
                ax = axes[idx]
                # PLS EV數據 - 取第一個成分的EV
                svr_ev_data = []
                for epsilon in epsilons:
                    cost_ev_data = []
                    for cost in costs:
                        first_comp = comp_cols[idx]
                        epsilon_result = svr_result['param_epsilon_results'][str(epsilon)][str(cost)].get('stats')
                        cost_ev_data = np.hstack((cost_ev_data,epsilon_result[first_comp]['explained_variance_per_y']))
                    svr_ev_data.append(cost_ev_data)
                svr_ev_data = np.array(svr_ev_data)        
                # CV EV數據 - 使用total_explained_variance
                cv_ev_data = []
                for epsilon in epsilons:
                    cost_ev_data = []
                    for cost in costs:
                        epsilon_result = cv_result['param_epsilon_results'][str(epsilon)]
                        # cv_ev_data.append(epsilon_result[str(cost)].get('explained_variance_per_y')[idx])
                        cost_ev_data = np.hstack((cost_ev_data, epsilon_result[str(cost)].get('explained_variance_per_y')[idx]))
                    cv_ev_data.append(cost_ev_data)
                cv_ev_data = np.array(cv_ev_data) 

                svr_df = pd.DataFrame(svr_ev_data, index= np.round(epsilons, 2), columns=np.round(costs, 2)) 
                cv_df = pd.DataFrame(cv_ev_data, index= np.round(epsilons, 2), columns=np.round(costs, 2)) 

                sns.heatmap(
                            svr_df ,
                            annot = cv_df,
                            vmax=1.2, #設定bar最大值
                            fmt=".3f",
                            center = 1,
                            cmap="coolwarm_r",
                            linewidths = 1,
                            cbar_kws={"label":"SVR EV"},
                            annot_kws={"size":10},
                            ax=ax
                        ) 
                # param_set = cv_result['best_parameters_set'][comp] 
                best_idx = np.unravel_index(np.argmax(cv_ev_data), cv_ev_data.shape)
                
                ax.text(
                            best_idx[1]+ 0.5,#col
                            best_idx[0]+ 0.1,#row
                            "★",
                            ha="center",
                            va="center",
                            color="green",
                            fontsize=18
                        )
                ax.set_xlabel('Cost', fontsize=12)
                ax.set_ylabel('Epsilon', fontsize=12)
                ax.set_title(f'Explained Variance ({comp})({algorithm_name})', fontsize=14, fontweight='bold')
                # ax.legend(fontsize=10)
                # ax.set_label('Scale')
                ax.grid(True, alpha=0.3)
                # 隱藏多餘的子圖
                for ax in axes[n_comp:]:
                    ax.set_visible(False)
                
            plt.tight_layout()
        plt.show()     

    def run_svr_prediction_comparison_chart(self,multi_results, *args):
        for algorithm_name, results in multi_results.items():               
            # 在該算法分頁中創建第四層Tab (Factor vs EV / 預測對比)
            svr_result = results['svr_model']
            cv_result = results['svr_cv']
            comp_cols = svr_result ['comp_cols']

            # 創建子圖
            n_comp = len(comp_cols)
            cols = min(n_comp, 2)
            rows = (n_comp + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows),dpi = 80)
            if n_comp == 1:
                axes = [axes]
            else:
                axes = np.array(axes).flatten()  

            for idx, comp in enumerate(comp_cols):
                param_set = cv_result['best_parameters_set'][comp] 
                #暫時手動更改功能
                # param_set['Epsilon'] = 0.001
                # param_set['Cost'] = 100.0



                # 獲取PLS和CV結果
                if param_set['Epsilon'] in cv_result['param_epsilon_results']:  
                    svr_param_epsilon_result = svr_result['param_epsilon_results'].get(param_set['Epsilon'])
                    svr_param_Cost_result = svr_param_epsilon_result[param_set['Cost']]
                    cv_epsilon_result = cv_result['param_epsilon_results'].get(param_set['Epsilon'])
                    cv_Cost_result = cv_epsilon_result[param_set['Cost']]
                else:
                    svr_param_epsilon_result = svr_result['param_epsilon_results'].get(str(param_set['Epsilon']))
                    svr_param_Cost_result = svr_param_epsilon_result[str(param_set['Cost'])]
                    cv_epsilon_result = cv_result['param_epsilon_results'].get(str(param_set['Epsilon']))  
                    cv_Cost_result = cv_epsilon_result[str(param_set['Cost'])]

                Y_true = svr_result['Y_valid']
                svr_Y_pred = svr_param_Cost_result['Y_pred']
                cv_Y_true = cv_Cost_result['all_y_true_original']
                cv_Y_pred = cv_Cost_result['all_y_pred_original'] 
                #獲取計算y=x線的資料
                pls_XY_line = svr_param_Cost_result['stats'].get(comp)
                pls_XY_line_slpoe = pls_XY_line['slope']
                pls_XY_line_sintecept = pls_XY_line['intercept']


                ax = axes[idx]
                y_true = Y_true[:, idx]
                svr_y_pred = svr_Y_pred[:, idx]
                cv_y_true = cv_Y_true[:,idx]
                cv_y_pred = cv_Y_pred[:, idx]

                # 繪製散點圖
                
                ax.scatter(y_true, svr_y_pred, alpha=0.6, label='PLS', color='blue')
                ax.scatter(cv_y_true, cv_y_pred, alpha=0.6, label='CV', color='red', marker='s')   


                # 計算y=x線的範圍
                ax_xlim = ax.get_xlim()
                ax_ylim = ax.get_ylim()
                plot_min = min(ax_xlim[0], ax_ylim[0])
                plot_max = max(ax_xlim[1], ax_ylim[1])

                # 繪製y=x線
                ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k', 
                        linestyle='-', linewidth=1.2, alpha=0.8, label='y=x')
                ax.plot([plot_min, plot_max], pls_XY_line_slpoe*np.array([plot_min, plot_max])+pls_XY_line_sintecept, 'r', 
                        linestyle=':', linewidth=1.2, alpha=0.8, label=f'y={pls_XY_line_slpoe:1.1f}x+{pls_XY_line_sintecept:1.1f}')
                
                ax.set_xlim(plot_min, plot_max)
                ax.set_ylim(plot_min, plot_max)



                # 計算統計
                        
                svr_r2 ,svr_rmse, svr_mae, _ = self.regression_score(y_true, svr_y_pred)
                cv_r2 ,cv_rmse, cv_mae, _ = self.regression_score(cv_Y_true[:,idx], cv_y_pred)


                # II
                lines = [
                        (f'SVR  : R²={svr_r2:6.3f}, rmse={svr_rmse:6.3f}, mae={svr_mae:6.3f}', 'black'),
                        (f'CV   : R²={cv_r2:6.3f}, rmse={cv_rmse:6.3f}, mae={cv_mae:6.3f}', 'tab:red')
                        ]
                y0 = 0.96      # 起始高度（axes 座標）
                dy = 0.03      # 行距
                # 先畫一個透明框（只負責背景）
                ax.text(
                    dy, y0,
                    ' ' * 55 + '\n' * 2,
                    transform=ax.transAxes,
                    va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.8)
                )
                for i, (line, color) in enumerate(lines):
                    ax.text(
                        dy, y0 - i * dy, line,
                        transform=ax.transAxes,
                        va='top', ha='left',
                        fontfamily='monospace',
                        color=color
                    )

                
                ax.set_title(f'{comp} param[ε,C] :[{param_set['Epsilon']:.3f},{param_set['Cost']:.3f}]')
                ax.set_xlabel('Reference Y')
                ax.set_ylabel('Predicted Y')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 隱藏多餘的子圖
            for ax in axes[n_comp:]:
                ax.set_visible(False)
            
            plt.tight_layout()
        
        #暫時吐出檢量線去側
            # intercepts_std = svr_param_Cost_result['model_intercepts']
            # B_orig_i_table =  svr_param_Cost_result['model_coefs']
        
            # self.run_output_calibration_Excel(np.ones([3,36]),comp_cols,intercepts_std,B_orig_i_table,algorithm_name,[f'{param_set['Epsilon']:.3f}',f'{param_set['Cost']:.3f}'])#輸出標準化Y的檢量線
        plt.show()

    def run_plot_backtest_results(self,predictions_dict, df_time, comp_cols, df_timeRef,df_TA,selected_component = None):

        """繪製多模型對比回測結果圖表（單圖模式）
        
        Args:
            predictions_dict: 預測結果字典
            comp_cols: 成分名稱列表
            selected_component: 選擇要顯示的成分名稱，如果為 None 則顯示第一個成分
        """
        # time_data = df_timeRef['Time'].values
        time_data = df_time['Time'].values

        for page in range(len(comp_cols)):
        # 確定要顯示的成分
            if selected_component is None:
                selected_component = comp_cols[page] if comp_cols else None
            else:    
                selected_component = comp_cols[page] if comp_cols else None

            # fig = plt.figure(figsize=(14, 4))
            # ax = plt.subplot(1, 1, 1)
            fig, axs = plt.subplots(2, 1,figsize=(14, 4),gridspec_kw={'height_ratios': [3, 1]})
            axs = axs.flatten()

            ax = axs[0]
            # 準備顏色
            colors = plt.cm.tab10(np.linspace(0, 1, 10))  # tab10 色盤

            # 為每個模型繪製預測線
            for model_idx, (unique_key, pred_data) in enumerate(predictions_dict.items()):
                Y_pred = pred_data['predictions']
                # Y_pred = Y_scaler.inverse_transform(pred_data['predictions'])
                # Y_pred = pt.inverse_transform(pred_data['predictions'])
                model_name = pred_data['model_name']
                param_set = pred_data['param_set_list']
                stats = pred_data.get('stats', {})
                
                # 選擇顏色
                color = colors[model_idx % len(colors)]

                # 獲取該成分的 R² 統計數據（如果有）
                r2_score = None
                if stats and selected_component in stats:
                    comp_stats = stats[selected_component]
                    if isinstance(comp_stats, dict) and 'r2' in comp_stats:
                        r2_score = comp_stats['r2']
                
                # 構建圖例標籤
                if r2_score is not None:
                    label = f"{unique_key} (R²={r2_score:.3f})"
                else:
                    label = unique_key

                # 繪製散點（使用 plot，但 linestyle='none' 移除連線）
                ax.plot(
                    time_data, 
                    # Y_pred[:, page]+bias[page],
                    Y_pred[:, page],
                    linestyle='-',
                    color=color,
                    marker='.',
                    markersize=2,
                    label=label,
                    alpha=0.5
                )  

            # 繪製參考數據（實際值）- 最後繪製，顯示在最上層
        
            if df_timeRef is not None:
                try:
                    # 獲取參考數據的時間和成分值
                    if 'Time' in df_timeRef.columns and selected_component in df_timeRef.columns:
                        ref_time = df_timeRef['Time'].values
                        ref_values = df_timeRef[selected_component].values
                        
                        # 繪製參考數據為紅色星形標記
                        ax.plot(
                            ref_time,
                            ref_values,
                            linestyle='-',
                            color='red',
                            marker='.',
                            markersize=3,
                            markerfacecolor='red',
                            label='Reference Data',
                            alpha=0.8,
                            zorder=100
                        )
                except Exception as e:
                    print(f"無法繪製參考數據: {e}")
                y_min = min(ref_values) * 0.75
                y_max = max(ref_values) * 1.25
            # 設置圖表標題和標籤
            ax.set_title(
                f"{selected_component} 多模型回測對比",
                fontsize=11,
                fontweight='bold'
            )
            ax.set_xlabel('Time', fontsize=9)
            ax.set_ylabel('Predicted Value', fontsize=9)
            ax.tick_params(axis='x', rotation=30, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            # y_min =-10 ; y_max = 15 
            ax.set_ylim(y_min, y_max)  
            
            # 添加圖例（放在子圖外側右方或下方）
            ax.legend(
                loc='lower left',
                bbox_to_anchor=(1.02, 1),
                fontsize=8,
                framealpha=0.9
            )
            
            #畫溫度
            # df_TA = df['temperature'].values
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
            # y_min =-10 ; y_max = 25
            # 調整佈局以防止重疊
            plt.tight_layout()
        plt.show() 

    def run_plot_backtest_results_with_score(self, predictions_dict, df_time, comp_cols, df_timeRef,selected_df_TA,split_idx,selected_component=None):
        """繪製多模型對比回測結果圖表（單圖模式）
        
        Args:
            predictions_dict: 預測結果字典
            comp_cols: 成分名稱列表
            selected_component: 選擇要顯示的成分名稱，如果為 None 則顯示第一個成分
        """
        # time_data = df_timeRef['Time'].values
        time_data = df_time['Time'].values  


        for page in range(len(comp_cols)):
            # 確定要顯示的成分
            if selected_component is None:
                selected_component = comp_cols[page] if comp_cols else None
            else:    
                selected_component = comp_cols[page] if comp_cols else None

            # fig = plt.figure(figsize=(14, 4))
            # # fig, axs = plt.subplots(2, 1,figsize=(14, 4),gridspec_kw={'height_ratios': [1, 3]})
            # ax = plt.subplot(1, 1, 1)
            fig, axs = plt.subplots(2, 1,figsize=(14, 4),gridspec_kw={'height_ratios': [3, 1]})
            axs = axs.flatten()

            ax = axs[0]
            # 準備顏色
            colors = plt.cm.tab10(np.linspace(0, 1, 10))  # tab10 色盤
            # ax = axs[1]
            # 收集分數資訊
            score_text_list = []
            # 為每個模型繪製預測線
            for model_idx, (unique_key, pred_data) in enumerate(predictions_dict.items()):
                
                Y_pred = pred_data['predictions'] 
                model_name = pred_data['model_name']
                param_set = pred_data['param_set_list']
                stats = pred_data.get('stats', {})  
                # 選擇顏色
                color = colors[model_idx % len(colors)]
                
                # 獲取該成分的 R² 統計數據（如果有）
                r2_score = None
                if stats and selected_component in stats:
                    comp_stats = stats[selected_component]
                    if isinstance(comp_stats, dict) and 'r2' in comp_stats:
                        r2_score = comp_stats['r2']
                
                # 構建圖例標籤
                if r2_score is not None:
                    label = f"{unique_key} (R²={r2_score:.3f})"
                else:
                    label = unique_key


                # 繪製散點（使用 plot，但 linestyle='none' 移除連線）
                ax.plot(
                    time_data, 
                    # Y_pred[:, page]+bias[page],
                    Y_pred[:, page],
                    linestyle='-',
                    color=color,
                    marker='.',
                    markersize=5,
                    label=label,
                    alpha=0.8
                )

                # score text data prepare
                if 'Time' in df_timeRef.columns and selected_component in df_timeRef.columns:
                    ref_values = df_timeRef[selected_component].values
                _,all_rmse, all_mae, all_mape = self.regression_score(ref_values,Y_pred[:, page])
                if ref_values[split_idx:].size == 0:
                    blind_rmse, blind_mae, blind_mape = all_rmse, all_mae, all_mape
                else:
                    _,blind_rmse, blind_mae, blind_mape = self.regression_score(ref_values[split_idx:],Y_pred[split_idx:, page])

                
                score_text_list.append(
                    f'{unique_key:<20} All point:        RMSE: {all_rmse:6.3f}, MAE:{all_mae:6.3f}, MAPE:{all_mape:6.3f}%'
                )
                score_text_list.append(
                    f'{unique_key:<20} Blind test point: RMSE: {blind_rmse:6.3f}, MAE:{blind_mae:6.3f}, MAPE:{blind_mape:6.3f}%'
                )
                score_text_list.append(
                    f'----------------------------------------------------------------------------------------------'
                )

                textstr = "\n".join(score_text_list)
            ax.text(
                    0.01, 1.15,
                    textstr,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                    )
            # 繪製參考數據（實際值）- 最後繪製，顯示在最上層
            if df_timeRef is not None:
                try:
                    # 獲取參考數據的時間和成分值
                    if 'Time' in df_timeRef.columns and selected_component in df_timeRef.columns:
                        ref_time = df_timeRef['Time'].values
                        ref_values = df_timeRef[selected_component].values
                        
                        # 繪製參考數據為紅色星形標記
                        ax.plot(
                            ref_time,
                            ref_values,
                            linestyle='-',
                            color='red',
                            marker='.',
                            markersize=5,
                            markerfacecolor='none',
                            label='Reference Data',
                            alpha=0.5,
                            zorder=100
                        )
                except Exception as e:
                    print(f"無法繪製參考數據: {e}")
                # 繪製回測切割線 split_idx-1 避免沒從零開始取
                ax.plot([ref_time[split_idx-1],ref_time[split_idx-1]],[max(Y_pred[:, page]), min(Y_pred[:, page])],linestyle='--',color='k')        
            # #text data prepare    
            # 每個樣本的 residual magnitude
            residual = ref_values-Y_pred[:, page]
            residual_mag = np.linalg.norm(residual.reshape(-1,1), axis=1)
            outlier_idx = np.where(residual_mag > np.percentile(residual_mag, max(min(99,int(abs(100-all_rmse))),95)))[0]
            for n, i in enumerate(outlier_idx, start=0):
                ax.text( ref_time[i] ,Y_pred[:, page][i]+0.03,# y 座標# 往上偏一點             
                        f'x{ref_time[i]}',                 # 你要顯示的文字
                        fontsize=6,
                        color='g',
                        ha='left',
                        va='bottom',
                        alpha=0.7
                    )
                
            # 設置圖表標題和標籤
            ax.set_title(
                f"{selected_component}多模型回測對比{round(1-(split_idx/len(ref_time)),1)*100}%,total:{len(ref_time)}",
                fontsize=11,
                fontweight='bold'
            )
            ax.set_xlabel('Time', fontsize=9)
            ax.set_ylabel('Predicted Value', fontsize=9)
            ax.tick_params(axis='x', rotation=30, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            y_min =-10 ; y_max = 15 
            # ax.set_ylim(y_min, y_max)  
            
            # 添加圖例（放在子圖外側右方或下方）
            ax.legend(
                loc='lower left',
                bbox_to_anchor=(0.85, 1),
                fontsize=8,
                framealpha=0.9
            )
            
            
            # 設置圖表分數說明欄        
            # textstr = (
            #             f'{unique_key}All point:        RMSE: {all_rmse:6.3f}, MAE:{all_mae:6.3f}, MAPE:{all_mape:6.3f}%\n'
            #             f'{unique_key}Blind test point: RMSE: {blind_rmse:6.3f}, MAE:{blind_mae:6.3f}, MAPE:{blind_mape:6.3f}%'
            #              )
            # ax = axs[0]
            # ax.text(
            #     0.02, 1.55,
            #     textstr,
            #     transform=ax.transAxes,
            #     fontsize=10,
            #     verticalalignment='top',
            #     horizontalalignment='left',
            #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            # )
            # 調整佈局以防止重疊
            #畫溫度
            # df_TA = df['temperature'].values
            ax = axs[1]
            ax.plot(
                    time_data, 
                    # Y_pred[:, page]+bias[page],
                    selected_df_TA,
                    linestyle='-',
                    color='blue',
                    marker='.',
                    markersize=5,
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
            plt.tight_layout() 
            
        plt.show()