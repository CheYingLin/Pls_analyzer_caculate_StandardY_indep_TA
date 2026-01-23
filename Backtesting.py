import numpy as np
from typing import Tuple, List, Dict, Any, Callable, Optional
from gen_calibration import run_output_calibration_Excel

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

def run_multi_model_backtest_NEW(X: np.ndarray ,factor: int, model_data, comp_cols: List[str], 
                             pls_model, scalerY, stats, unique_key, model_name , unselect):
    predictions_dict = {}
    coefs_list = []
    X_pred = X   
    mask = np.ones(X.shape[1], dtype=bool)
    mask[unselect] = False 
    coefs_table = np.zeros([X.shape[1],len(comp_cols)]) 
    intercepts = np.zeros(len(comp_cols)) 
        # 4.3 計算係數
    for i in range(len(comp_cols)):
        coefs_list.append( pls_model[0][i].coef_ if pls_model.shape[0] != len(comp_cols) else pls_model[0][i].coef_.T)
    coefs = np.vstack(coefs_list).T  
    coefs_table[mask] = coefs
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
    Y_pred = X_pred.dot(coefs_table) + intercepts

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

def run_multi_model_backtest_NEWII(X_ori,X: np.ndarray ,factor: List, factor_no_scaleY: int, model_data, comp_cols: List[str], 
                             factor_result, model_name , unselect,YesorNo=False):
    unique_key = f"{model_name}_F{factor}"
    predictions_dict = {}
    predictions_dict_no_scaleY= {}
    coefs_list = []
    coefs_list_no_scaleY = []
    scalerY = []
    X_pred = X   
    # mask = np.ones(X.shape[1], dtype=bool)
    # mask[unselect] = False 
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
        if not unselect:
                pass
        else:
            mask[unselect[i]] = False
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
    X_train_full = X_ori

    # 每個 y 對應的子空間平均
    X_mean_list = []
    for i in range(len(comp_cols)):
        mask = np.ones(X_train_full.shape[1], dtype=bool)
        if not unselect:
            pass
        else:
            mask[unselect[i]] = False
        X_mean_list.append(X_train_full[:, mask].mean(axis=0))

    intercepts_std = np.zeros(len(comp_cols)) 
    Y_2 = np.zeros_like(Y_pred_std)
    B_orig_i_table = np.zeros([X.shape[1],len(comp_cols)])
    for i in range(len(comp_cols)):
        mask = np.ones(X_ori.shape[1], dtype=bool)
        if not unselect:
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
        run_output_calibration_Excel(X_ori,comp_cols,intercepts_std,B_orig_i_table,f'{model_name}_標準化Y',factor)#輸出標準化Y的檢量線
        run_output_calibration_Excel(X_ori,comp_cols,intercepts_no_scaleY,coefs_table_no_scaleY,f'{model_name}_無標準化Y',factor_no_scaleY)#輸出無標準化Y的檢量線
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
    predictions_dict_no_scaleY[unique_key] = {
        'predictions': Y_pred_no_scaleY,
        'comp_names': comp_cols,
        'stats': stats,
        'model_name': model_name,
        'factor': factor
    }
    return{
            f"{model_name}_F{factor}": {
                    'predictions':  Y_pred_new,
                    'comp_names': comp_cols,
                    'stats': {},
                    'model_name': model_name,
                    'factor': factor
                }
            },{# 轉換為新格式並調用多模型繪圖函數
            f"{model_name}_F{factor_no_scaleY}": {
                    'predictions':  Y_pred_no_scaleY,
                    'comp_names': comp_cols,
                    'stats': {},
                    'model_name': model_name,
                    'factor': factor
                }
            }