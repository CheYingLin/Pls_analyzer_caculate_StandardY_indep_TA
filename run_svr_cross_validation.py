import numpy as np
from typing import Tuple, List, Dict, Any, Callable, Optional
from sklearn.svm import SVR
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error

def determine_cv_strategy( n_samples: int) -> Tuple[Any, str, int]:
    """
    根據樣本數量自動確定最優CV策略
    
    決策邏輯:
    - n ≤ 20: Leave-One-Out CV (最大化數據利用)
    - n > 20: 20-fold CV (平衡效率和精度)
    
    Parameters:
    -----------
    n_samples : int
        樣本數量
    
    Returns:
    --------
    Tuple[Any, str, int]
        (cv_object, cv_type, n_folds)
    """
    if n_samples <= 9:
        cv_n = int(n_samples/2)
        return KFold(n_splits=cv_n, shuffle=True, random_state=42), "x-fold", cv_n
    else:
        return KFold(n_splits=5, shuffle=True, random_state=42), "5-fold", 5
    
def _calculate_total_explained_variance_standardized( y_true, y_pred):
    """
    計算標準化Y的總體解釋變異量
    
    此方法專門用於計算Total EV，確保跨成分公平比較
    輸入的y_true和y_pred必須是已經標準化後的數據
    
    Parameters:
    -----------
    y_true : np.ndarray
        標準化後的真實值 (n_samples, n_components)
    y_pred : np.ndarray
        標準化後的預測值 (n_samples, n_components)
    
    Returns:
    --------
    float
        總體解釋變異量 (0-1之間)
    """
    n_samples, n_components = y_true.shape
    
    # 總變異量（標準化後每個成分方差≈1）
    # ss_tot = n_samples * n_components #這裡不保證方差一定等於不建議用這個
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    
    # 殘差平方和
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # 解釋變異量
    explained_variance = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # --- 各 response 的 explained variance ---
    ev_per_y = []
    for j in range(y_true.shape[1]):
        ss_tot_j = np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2)
        ss_res_j = np.sum((y_true[:, j] - y_pred[:, j]) ** 2)
        ev_j = 1 - ss_res_j / ss_tot_j if ss_tot_j != 0 else 0
        ev_per_y.append(ev_j)
    
    return np.array(ev_per_y)    
    
def cross_validate_multi_param( X: np.ndarray, ch_unselect,  
                                   Y_original: np.ndarray,epsilon ,Cost,  
                                   comp_cols: List[str]) -> Dict[str, Any]:  
    # 自動確定CV策略
    cv_object, cv_type, n_folds = determine_cv_strategy(X.shape[0])  
    # 定義SVR模型
    svr = SVR(kernel='linear', epsilon=epsilon, C=Cost, gamma='scale') #'poly','linear','rbf'

    # 初始化累積數據容器（僅保存原始尺度）
    y_true_list = []
    y_pred_list = []
    y_pred_standardized_list = []# for EV運算
    y_true_standardized_list = []# for EV運算
    rmse_StdY_list = [] #算全部CV的RMSE

    # 執行累積交叉驗證循環
    for train_idx, val_idx in cv_object.split(X):
        # 數據分割
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y_original[train_idx], Y_original[val_idx]

        Y_train_std = np.zeros_like(Y_train)
        Y_val_std = np.zeros_like(Y_val)
        X_train_std = np.zeros_like(X_train)
        X_val_std = np.zeros_like(X_val)
        scalers_y = []
        # scalers_x = []
        scalers_x = StandardScaler()
        X_train_std = scalers_x.fit_transform(X_train)
        X_val_std = scalers_x.transform(X_val)
        
        for i in range(len(comp_cols)):
            sc_y = StandardScaler()
            Y_train_std[:, i] = sc_y.fit_transform(
                Y_train[:, i].reshape(-1, 1)
            ).ravel()
            Y_val_std[:, i] = sc_y.transform(
                Y_val[:, i].reshape(-1, 1)
            ).ravel()
            scalers_y.append(sc_y)

            
        # train + predict
        Y_pred_All = []
        Y_val_All = []
        # y_true_standardized_All = [] # for EV運算
        y_pred_standardized_All = [] # for EV運算
        rmse_StdY_folds= [] #算RMSE
        cv_pls_model_All = []
        
        for i in range(len(comp_cols)):
            mask = np.ones(X.shape[1], dtype=bool)
            if not ch_unselect or ch_unselect[i].size==0:
                pass
            else:
                mask[ch_unselect[i]] = False    
                
            svr.fit(X_train_std[:,mask], Y_train_std[:, i]) 
            y_pred_std = svr.predict(X_val_std[:,mask]).ravel() 
            y_pred = scalers_y[i].inverse_transform(
                y_pred_std.reshape(-1, 1)
            ).ravel()

            Y_pred_All.append(y_pred)
            Y_val_All.append(Y_val[:, i])
            y_pred_standardized_All.append(y_pred_std)
            rmse_StdY_folds.append(np.sqrt(mean_squared_error(Y_val_std[:,i], y_pred_std))) 

        y_pred_list_temp = np.vstack(Y_pred_All) 
        y_val_list_temp = np.vstack(Y_val_All)   
        y_pred_standardized_temp = np.vstack(y_pred_standardized_All) 
        y_true_standardized_temp  = Y_val_std   
        rmse_StdY_temp = np.vstack(rmse_StdY_folds)
        # cv_pls_model_list_temp= np.vstack(cv_pls_model_All) 
        # 累積結果
        y_pred_list.append(y_pred_list_temp.T)
        y_true_list.append(y_val_list_temp.T)
        y_pred_standardized_list.append(y_pred_standardized_temp.T)
        y_true_standardized_list.append(y_true_standardized_temp)
        rmse_StdY_list.append(rmse_StdY_temp.T)   

    # 合併累積數據
    y_true_original = np.vstack(y_true_list)
    y_pred_original = np.vstack(y_pred_list)
    y_pred_standardized_original = np.vstack(y_pred_standardized_list)
    y_true_standardized_original = np.vstack(y_true_standardized_list)
    rmse_StdY_original = np.vstack(rmse_StdY_list)  

    # 計算原始尺度指標（R²、RMSE）
    r2_original = []
    rmse_original = []
    rmse_std = []    

    for i in range(len(comp_cols)):
        try:
            # R²（原始尺度）
            r2 = r2_score(y_true_original[:, i], y_pred_original[:, i])
            r2_original.append(r2 if np.isfinite(r2) else 0.0)
            
            # RMSE（原始尺度）
            rmse = np.mean(rmse_StdY_original[:, i] )
            rmse_std.append(np.std(rmse_StdY_original[:, i], ddof=1))
            rmse_original.append(rmse if np.isfinite(rmse) else 0.0)
        except Exception as e:
            print(f"警告：成分 {comp_cols[i]} 計算失敗: {e}")
            r2_original.append(0.0)
            rmse_original.append(0.0)  

    # 額外計算：標準化尺度的Total EV（僅用於EV計算）
    y_true_standardized = y_pred_standardized_original
    y_pred_standardized = y_true_standardized_original
    # 計算總體解釋變異量（標準化尺度）
    explained_variance_per_y = _calculate_total_explained_variance_standardized(
        y_true_standardized, y_pred_standardized) 

    # 返回結果
    return {
        # 主要結果（原始尺度）- 用於所有業務指標
        'mean_cv_scores_original': r2_original,
        'rmse_means': rmse_original,
        'rmse_std': rmse_std,
        'all_y_true_original': y_true_original,
        'all_y_pred_original': y_pred_original,
        
        # EV結果（標準化尺度）- 僅用於Total EV計算
        'explained_variance_per_y': explained_variance_per_y,
        'all_y_true': y_true_standardized,  # 保留以兼容現有代碼
        'all_y_pred': y_pred_standardized,  # 保留以兼容現有代碼
        
        # 其他信息
        'cv_type': cv_type,
        'k_folds': n_folds,
        'mean_cv_scores': r2_original,  # 兼容舊版接口
        'std_cv_scores': [0.0] * len(comp_cols),
        'std_cv_scores_original': [0.0] * len(comp_cols),
        'rmse_stds': [0.0] * len(comp_cols)
        },explained_variance_per_y   

def run_svr_parameter_cross_validation_scan(X: np.ndarray, channel_unselect: List[int], Y: np.ndarray, comp_cols: List[str]):
    results = {}  # 儲存交叉驗證結果  
    # 數據預處理：移除包含NaN的行
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X_valid = X[mask]
    Y_valid = Y[mask]
    
    # 檢查數據點是否足夠
    n_samples = X_valid.shape[0]
    if n_samples < 3:
        raise ValueError(f"篩選後資料點不足 ({n_samples}), 至少需 3 筆進行交叉驗證")
    
    base = 0.001
    multipliers = [1, 2.5, 5, 10, 25, 50, 75, 100]

    epsilon_param = [base * m for m in multipliers]
    # epsilon_param=np.array([0.01,0.025,0.05,0.1,0.25,0.5,0.75,1])
    # epsilon_param=np.array( np.round(np.linspace(0.001, 0.05, 
    #                             num=5, 
    #                             endpoint=True, 
    #                             retstep=False, 
    #                             dtype=float),3))
    # Cost_param = np.array([0.01,0.025,0.05,0.1,0.25,0.5,0.75,1,2.5,5,7.5,10])
    Cost_param = np.array(np.round(np.logspace(-1, 2, num=5),3))
    # 存儲每個param的結果
    param_epsilon_results = {}
    best_parameters = {}
    # 推薦最佳Factorparam
    best_score = np.ones(len(comp_cols))*-50 #避免 R^2有負的值
    for idx_e, epsilon in enumerate(epsilon_param):
        param_Cost_results ={}
        for idx_c, Cost in enumerate(Cost_param):
            # 交叉驗證單個Cost
            result, score = cross_validate_multi_param(
                X_valid, channel_unselect, Y_valid,epsilon ,Cost, comp_cols )
            param_Cost_results[str(Cost)] = result 
            for i in range(len(comp_cols)):
                if score[i] > best_score[i]:#找到表现最好的参数
                    best_score[i] = score[i]
                    best_parameters[comp_cols[i]] = {'Epsilon':epsilon,'Cost':Cost}   
        param_epsilon_results[str(epsilon)] = param_Cost_results

    # 存儲結果
      
    results = {
            'param_epsilon_results':  param_epsilon_results,
            'best_parameters_set':best_parameters,
            'best_param_score_StdY(R2)':best_score,
            'n_samples': n_samples,
            'comp_cols': comp_cols,
            'algorithm_info': "算法svr"
        }
    return results