import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

matplotlib.rc('font', family='serif', serif=['ABC', 'MingLiU']) 
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def regression_score(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return rmse, mae, mape

def run_plot_group( prefix,timedata, merged_df):
    fig, axs = plt.subplots(6, 6, figsize=(17.5, 9.5),dpi = 100)
    for i in range(36):
        # col = f"{prefix}{i+1}"
        col = f"{i+1}{prefix}"
        if col in merged_df.columns:
            ax = axs[i // 6, i % 6]
            y_data = merged_df[col]
            x_data = timedata["time"] if "time" in timedata.columns else range(len(y_data))
            ax.plot(x_data, y_data)
            ax.set_title(col)
            ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def run_plot_group_new( prefix,timedata, comp_cols,merged_df): 
    import matplotlib.pyplot as plt

    plots_per_page = 12
    rows, cols = 3, 4

    figs = []  # 收集 figure

    for page in range(3):
        fig, axs = plt.subplots(rows, cols, figsize=(17.5, 9.5),dpi = 80)
        axs = axs.flatten()

        for i in range(plots_per_page):
            idx = page * plots_per_page + i
            if idx >= 36:
                break

            col = f"{prefix}{idx+1}"
            ax = axs[i]

            # if col in merged_df.columns:
            y = merged_df[:,idx]
            x = timedata["time"] if "time" in timedata.columns else range(len(y))#data time
            # x = comp_cols

            ax.plot(x, y)
            ax.set_title(col)
            ax.tick_params(axis='x', rotation=45)

        fig.tight_layout()
        figs.append(fig)

    # 🔥 一次顯示全部
    plt.show()

def run_plot_group_scatter_new( prefix,timedata, comp_cols,merged_df): 
    import matplotlib.pyplot as plt

    plots_per_page = 12
    rows, cols = 3, 4

    figs = []  # 收集 figure

    for page in range(3):
        fig, axs = plt.subplots(rows, cols, figsize=(17.5, 9.5),dpi = 80)
        axs = axs.flatten()

        for i in range(plots_per_page):
            idx = page * plots_per_page + i
            if idx >= 36:
                break

            col = f"{prefix}{idx+1}"
            ax = axs[i]

            # if col in merged_df.columns:
            y = merged_df[:,idx]
            # x = timedata["time"] if "time" in timedata.columns else range(len(y))#data time
            x = comp_cols

            ax.scatter(x, y)
            ax.set_title(col)
            ax.set_xlabel('concentation', fontsize=12)
            ax.tick_params(axis='x', rotation=45)

        fig.tight_layout()
        figs.append(fig)

    # 🔥 一次顯示全部
    plt.show()
   

def run_plot_display_multi_algorithm_results(multi_results):
    for algorithm_name, results in multi_results.items():               
        # 在該算法分頁中創建第四層Tab (Factor vs EV / 預測對比)
        pls_result = results['pls']
        cv_result = results['cv']
        """創建 Factor vs EV 趨勢圖"""
        comp_cols = pls_result['comp_cols']
        max_factor = pls_result['max_factor']

        # 準備數據
        factors = list(range(1, max_factor + 1))

        # PLS EV數據 - 取第一個成分的EV
        pls_ev_data = []
        for factor in factors:
            if factor in pls_result['factor_results']:
                first_comp = comp_cols[0]
                ev = pls_result['factor_results'][factor]['stats'][first_comp]['explained_variance']
                pls_ev_data.append(ev)
            else:
                pls_ev_data.append(np.nan)  

        # CV EV數據 - 使用total_explained_variance
        cv_ev_data = []
        for factor in factors:
            if factor in cv_result['factor_results']:
                ev = cv_result['factor_results'][factor].get('total_explained_variance', 0)
                cv_ev_data.append(ev)
            else:
                cv_ev_data.append(np.nan)      

        # 繪製圖表
        # matplotlib.rc('font', family='serif', serif=['ABC', 'MingLiU']) 
        # matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        fig, ax = plt.subplots(figsize=(10, 5))
        # 繪製兩條線
        ax.plot(factors, pls_ev_data, 'o-', label='PLS EV', linewidth=2, markersize=8, color='blue')
        ax.plot(factors, cv_ev_data, 's--', label='CV EV', linewidth=2, markersize=8, color='red')
        
        # 標記最佳Factor            
        best_factor = cv_result['best_factor']
        ax.axvline(x=best_factor, color='green', linestyle=':', alpha=0.7, linewidth=2, 
                   label=f'Best Factor: {best_factor}')
        
        # 在最佳Factor點添加星號標記
        if best_factor <= len(pls_ev_data):
            ax.plot(best_factor, pls_ev_data[best_factor-1], 'g*', markersize=15)
        if best_factor <= len(cv_ev_data):
            ax.plot(best_factor, cv_ev_data[best_factor-1], 'g*', markersize=15)
        
        ax.set_xlabel('Factor', fontsize=12)
        ax.set_ylabel('Explained Variance', fontsize=12)
        ax.set_title(f'Factor vs Explained Variance ({algorithm_name})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(factors)
        
        # 設置Y軸範圍
        all_ev_values = [v for v in pls_ev_data + cv_ev_data if not np.isnan(v)]
        if all_ev_values:
            y_min = min(all_ev_values) * 0.95
            y_max = max(all_ev_values) * 1.05
            ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.show()

def run_plot_display_indepY_algorithm_results(multi_results):
    for algorithm_name, results in multi_results.items():               
        # 在該算法分頁中創建第四層Tab (Factor vs EV / 預測對比)
        pls_result = results['pls']
        cv_result = results['cv']
        """創建 Factor vs EV 趨勢圖"""
        comp_cols = pls_result['comp_cols']
        max_factor = pls_result['max_factor']
        # 準備數據
        factors = list(range(1, max_factor + 1))

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
            pls_ev_data = []
            for factor in factors:
                if factor in pls_result['factor_results']:
                    first_comp = comp_cols[idx]
                    ev = pls_result['factor_results'][factor]['stats'][first_comp]['explained_variance_per_y']
                    pls_ev_data.append(ev)
                else:
                    pls_ev_data.append(np.nan)  

            # CV EV數據 - 使用total_explained_variance
            cv_ev_data = []
            cv_rms_data = []
            for factor in factors:
                if factor in cv_result['factor_results']:
                    ev = cv_result['factor_results'][factor].get('explained_variance_per_y')
                    rms = cv_result['factor_results'][factor].get('rmse_means')
                    if ev is None:
                        value = np.nan
                    else:
                        value = ev[idx]
                        rm_value = rms[idx] 
                    cv_ev_data.append(value)
                    cv_rms_data.append(rm_value)

                else:
                    cv_ev_data.append(np.nan) 
                    cv_rms_data.append(np.nan)
            # 繪製兩條線
            ax.plot(factors, pls_ev_data, 'o-', label='PLS EV=>R²', linewidth=2, markersize=8, color='blue')
            ax.plot(factors, cv_ev_data, 's--', label='CV EV', linewidth=2, markersize=8, color='red')
            ax.plot(factors, cv_rms_data, marker='d', label='CV RM', linestyle=':', linewidth=2, markersize=8, color='green')            

            ax.set_xlabel('Factor', fontsize=12)
            ax.set_ylabel('Explained Variance', fontsize=12)
            ax.set_title(f'Factor vs Explained Variance  ({comp})', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(factors)   

            # 設置Y軸範圍
            all_ev_values = [v for v in pls_ev_data + cv_ev_data if not np.isnan(v)]
            if all_ev_values:
                y_min = min(all_ev_values) * 0.95
                y_max = max(all_ev_values) * 1.05
                ax.set_ylim(y_min, y_max)     
        plt.tight_layout()
        plt.show()

def run_create_prediction_comparison_chart(multi_results,Y_scaler, *args):
    for algorithm_name, results in multi_results.items():               
        # 在該算法分頁中創建第四層Tab (Factor vs EV / 預測對比)
        pls_result = results['pls']
        cv_result = results['cv']
        comp_cols = pls_result['comp_cols']

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
            if len(args) > 0:
                factor = args[0][idx]
            else:     
                factor = cv_result['best_factor']
            
        
            # 獲取PLS和CV結果
            pls_factor_result = pls_result['factor_results'].get(factor)
            cv_factor_result = cv_result['factor_results'].get(factor)

            
            
            Y_true = pls_result['Y_valid']
            pls_Y_pred = pls_factor_result['Y_pred']
            pls_Y_pred_no_scaleY = pls_factor_result['Y_pred_no_scaleY']
            cv_Y_true = cv_factor_result['all_y_true_original']
            cv_Y_pred = cv_factor_result['all_y_pred_original']
            cv_Y_pred_no_scaleY = cv_factor_result['all_y_pred_no_scaleY_original']
            # Y_true = Y_scaler.inverse_transform(pls_result['Y_valid'])
            # pls_Y_pred = Y_scaler.inverse_transform(pls_factor_result['Y_pred'])
            # cv_Y_pred = Y_scaler.inverse_transform(cv_factor_result['all_y_pred_original'])
            #獲取計算y=x線的資料
            pls_XY_line = pls_factor_result['stats'].get(comp)
            pls_XY_line_slpoe = pls_XY_line['slope']
            pls_XY_line_sintecept = pls_XY_line['intercept']
            # for idx, comp in enumerate(comp_cols):
            ax = axes[idx]
            y_true = Y_true[:, idx]
            pls_y_pred = pls_Y_pred[:, idx]
            pls_y_pred_no_scaleY = pls_Y_pred_no_scaleY[:, idx]
            cv_y_true = cv_Y_true[:,idx]
            cv_y_pred = cv_Y_pred[:, idx]
            cv_y_pred_no_scaleY = cv_Y_pred_no_scaleY[:, idx]
            
            
            
            # 繪製散點圖
            
            ax.scatter(y_true, pls_y_pred, alpha=0.6, label='PLS', color='blue')
            ax.scatter(cv_y_true, cv_y_pred, alpha=0.6, label='CV', color='red', marker='s')
            ax.scatter(cv_y_true, cv_y_pred_no_scaleY, alpha=0.6, label='CV_noStd', color='green', marker='^')
            
            # 計算y=x線的範圍
            ax_xlim = ax.get_xlim()
            ax_ylim = ax.get_ylim()
            plot_min = min(ax_xlim[0], ax_ylim[0])
            plot_max = max(ax_xlim[1], ax_ylim[1])
            
            # 繪製y=x線
            ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k', 
                    linestyle='-', linewidth=1.2, alpha=0.8, label='y=x')
            ax.plot([plot_min, plot_max], pls_XY_line_slpoe*np.array([plot_min, plot_max])+pls_XY_line_sintecept, 'r', 
                    linestyle=':', linewidth=1.2, alpha=0.8, label='y=x')
            
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
            
            # 計算統計
            pls_r2 = r2_score(y_true, pls_y_pred)
            pls_r2_no_scaleY = r2_score(y_true, pls_y_pred_no_scaleY)
            cv_r2 = r2_score(cv_Y_true[:,idx], cv_y_pred)
            cv_r2_no_scaleY = r2_score(cv_Y_true[:,idx], cv_y_pred_no_scaleY)
            pls_rms = np.sqrt(mean_squared_error(y_true, pls_y_pred))
            pls_rms_no_scaleY = np.sqrt(mean_squared_error(y_true, pls_y_pred_no_scaleY))
            cv_rms = np.sqrt(mean_squared_error(cv_Y_true[:,idx], cv_y_pred))
            cv_rms_no_scaleY= np.sqrt(mean_squared_error(cv_Y_true[:,idx], cv_y_pred_no_scaleY))
            pls_mae = mean_absolute_error(y_true, pls_y_pred)
            pls_mae_no_scaleY = mean_absolute_error(y_true, pls_y_pred_no_scaleY)
            cv_mae = mean_absolute_error(cv_Y_true[:,idx], cv_y_pred)
            cv_mae_no_scaleY  = mean_absolute_error(cv_Y_true[:,idx], cv_y_pred_no_scaleY )
            bias = np.mean(cv_y_pred - cv_Y_true[:,idx], axis=0)
            print(bias)
            # 添加統計信息
            text = (
                    f'PLS      : R²={pls_r2:6.3f}, rmse={pls_rms:6.3f}, mae={pls_mae:6.3f}\n'
                    f'PLS_noStd: R²={pls_r2_no_scaleY:6.3f}, rmse={pls_rms_no_scaleY:6.3f}, mae={pls_mae_no_scaleY:6.3f}\n'
                    f'CV       : R²={cv_r2:6.3f}, rmse={cv_rms:6.3f}, mae={cv_mae:6.3f}\n'
                    f'CV_noStd : R²={cv_r2_no_scaleY:6.3f}, rmse={cv_rms_no_scaleY:6.3f}, mae={cv_mae_no_scaleY:6.3f}'
                )
            ax.text(0.05, 0.95, text, 
                    transform=ax.transAxes, va='top',ha='left',fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax.set_title(f'{comp} (Factor {factor})')
            ax.set_xlabel('參考值')
            ax.set_ylabel('預測值')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隱藏多餘的子圖
        for ax in axes[n_comp:]:
            ax.set_visible(False)
        
        plt.tight_layout()
    plt.show()

def run_plot_backtest_results(predictions_dict, df_time, comp_cols, df_timeRef,Y_scaler ,pt,selected_component=None):
    """繪製多模型對比回測結果圖表（單圖模式）
    
    Args:
        predictions_dict: 預測結果字典
        comp_cols: 成分名稱列表
        selected_component: 選擇要顯示的成分名稱，如果為 None 則顯示第一個成分
    """
    # time_data = df_timeRef['Time'].values
    time_data = df_time['Time'].values
    
    # # 確定要顯示的成分
    # if selected_component is None:
    #     selected_component = comp_cols[0] if comp_cols else None
                
    # # 獲取成分索引
    # comp_idx = comp_cols.index(selected_component)
    
    # 創建單一圖表
    # matplotlib.rc('font', family='serif', serif=['ABC', 'MingLiU']) 
    # matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    for page in range(len(comp_cols)):
        # 確定要顯示的成分
        if selected_component is None:
            selected_component = comp_cols[page] if comp_cols else None
        else:    
            selected_component = comp_cols[page] if comp_cols else None

        fig = plt.figure(figsize=(14, 4))
        ax = plt.subplot(1, 1, 1)
        # 準備顏色
        colors = plt.cm.tab10(np.linspace(0, 1, 10))  # tab10 色盤
        
        # 為每個模型繪製預測線
        for model_idx, (unique_key, pred_data) in enumerate(predictions_dict.items()):
            Y_pred = pred_data['predictions']
            # Y_pred = Y_scaler.inverse_transform(pred_data['predictions'])
            # Y_pred = pt.inverse_transform(pred_data['predictions'])
            model_name = pred_data['model_name']
            factor = pred_data['factor']
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
            
            bias = [0,0]
            # 繪製散點（使用 plot，但 linestyle='none' 移除連線）
            ax.plot(
                time_data, 
                # Y_pred[:, page]+bias[page],
                Y_pred[:, page],
                linestyle='none',
                color=color,
                marker='.',
                markersize=5,
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
                        linestyle='none',
                        color='red',
                        marker='o',
                        markersize=6,
                        markerfacecolor='none',
                        label='Reference Data',
                        alpha=0.9,
                        zorder=100
                    )
            except Exception as e:
                print(f"無法繪製參考數據: {e}")
        
        # 設置圖表標題和標籤
        ax.set_title(
            f"{selected_component} 多模型回測對比",
            fontsize=11,
            fontweight='bold'
        )
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Predicted Value', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        y_min =-10 ; y_max = 15 
        # ax.set_ylim(y_min, y_max)  
        
        # 添加圖例（放在子圖外側右方或下方）
        ax.legend(
            loc='lower left',
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
            framealpha=0.9
        )
        
        # 調整佈局以防止重疊
        plt.tight_layout()
    plt.show()

def run_plot_backtest_results_with_score(predictions_dict, df_time, comp_cols, df_timeRef,Y_scaler ,pt,split_idx,selected_component=None):
    """繪製多模型對比回測結果圖表（單圖模式）
    
    Args:
        predictions_dict: 預測結果字典
        comp_cols: 成分名稱列表
        selected_component: 選擇要顯示的成分名稱，如果為 None 則顯示第一個成分
    """
    # time_data = df_timeRef['Time'].values
    time_data = df_time['Time'].values
    
    # # 確定要顯示的成分
    # if selected_component is None:
    #     selected_component = comp_cols[0] if comp_cols else None
                
    # # 獲取成分索引
    # comp_idx = comp_cols.index(selected_component)
    
    # 創建單一圖表
    # matplotlib.rc('font', family='serif', serif=['ABC', 'MingLiU']) 
    # matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    for page in range(len(comp_cols)):
        # 確定要顯示的成分
        if selected_component is None:
            selected_component = comp_cols[page] if comp_cols else None
        else:    
            selected_component = comp_cols[page] if comp_cols else None

        fig = plt.figure(figsize=(14, 4))
        ax = plt.subplot(1, 1, 1)
        # 準備顏色
        colors = plt.cm.tab10(np.linspace(0, 1, 10))  # tab10 色盤
        
        # 為每個模型繪製預測線
        for model_idx, (unique_key, pred_data) in enumerate(predictions_dict.items()):
            Y_pred = pred_data['predictions']
            # Y_pred = Y_scaler.inverse_transform(pred_data['predictions'])
            # Y_pred = pt.inverse_transform(pred_data['predictions'])
            model_name = pred_data['model_name']
            factor = pred_data['factor']
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
            
            bias = [0,0]
            
            # 繪製散點（使用 plot，但 linestyle='none' 移除連線）
            ax.plot(
                time_data, 
                # Y_pred[:, page]+bias[page],
                Y_pred[:, page],
                linestyle='none',
                color=color,
                marker='.',
                markersize=5,
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
                        linestyle='none',
                        color='red',
                        marker='o',
                        markersize=6,
                        markerfacecolor='none',
                        label='Reference Data',
                        alpha=0.9,
                        zorder=100
                    )
            except Exception as e:
                print(f"無法繪製參考數據: {e}")
            # 繪製回測切割線
            ax.plot([ref_time[split_idx],ref_time[split_idx]],[max(Y_pred[:, page]), min(Y_pred[:, page])],linestyle='--',color='k')        
        #text data prepare
        all_rmse, all_mae, all_mape = regression_score(ref_values,Y_pred[:, page])
        if ref_values[split_idx:].size == 0:
            blind_rmse, blind_mae, blind_mape = all_rmse, all_mae, all_mape
        else:
            blind_rmse, blind_mae, blind_mape = regression_score(ref_values[split_idx:],Y_pred[split_idx:, page])
        # 每個樣本的 residual magnitude
        residual = ref_values-Y_pred[:, page]
        residual_mag = np.linalg.norm(residual.reshape(-1,1), axis=1)
        outlier_idx = np.where(residual_mag > np.percentile(residual_mag, max(int(abs(100-all_rmse)),95)))[0]
        for n, i in enumerate(outlier_idx, start=0):
            ax.text( ref_time[i] ,Y_pred[:, page][i]+0.03,# y 座標# 往上偏一點             
                    f'x{ref_time[i]}',                 # 你要顯示的文字
                    fontsize=6,
                    color='r',
                    ha='left',
                    va='bottom'
                )
        # 設置圖表分數說明欄        
        textstr = (
                    f'All point:        RMSE: {all_rmse:6.3f}, MAE:{all_mae:6.3f}, MAPE:{all_mape:6.3f}%\n'
                    f'Blind test point: RMSE: {blind_rmse:6.3f}, MAE:{blind_mae:6.3f}, MAPE:{blind_mape:6.3f}%'
                     )

        ax.text(
            0.02, 1.15,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        # 設置圖表標題和標籤
        ax.set_title(
            f"{selected_component}多模型回測對比{round(1-(split_idx/len(ref_time)),1)*100}%,total:{len(ref_time)}",
            fontsize=11,
            fontweight='bold'
        )
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Predicted Value', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        y_min =-10 ; y_max = 15 
        # ax.set_ylim(y_min, y_max)  
        
        # 添加圖例（放在子圖外側右方或下方）
        ax.legend(
            loc='lower left',
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
            framealpha=0.9
        )
        
        # 調整佈局以防止重疊
        plt.tight_layout()
    plt.show()    
    # # # 確保資料夾存在
    # Out_file_path = r"C:\Users\Jason.lin\Desktop\workfile\20260107_luna微蝕data\luna微蝕data\luna微蝕data"
    # filename = "output2.xlsx"
    # os.makedirs(Out_file_path, exist_ok=True)
    # # 完整檔案路徑
    # output_file = os.path.join(Out_file_path, filename)    
    # # 輸出 Excel
    # Y_pred_df = pd.DataFrame(Y_pred)
    # Y_pred_df.to_excel(output_file, index=False)