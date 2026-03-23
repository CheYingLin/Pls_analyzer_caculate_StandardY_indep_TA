import numpy as np
from typing import Tuple, List, Dict, Any, Callable, Optional
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class SVR_Analysis:
    """導入外參數設定"""
    def __init__(self,):
        pass
    
    def _fit_param_final(self, X: np.ndarray, ch_unselect, Y: np.ndarray, epsilon:int ,Cost:int) -> Tuple[SVR, np.ndarray]:
        scalers_y = []
        scalers_x = StandardScaler()
        X_train = scalers_x.fit_transform(X)
                
        Y_train = np.zeros_like(Y)
        for i in range(Y.shape[1]):
            sc_y = StandardScaler()
            Y_train[:, i] = sc_y.fit_transform(
                Y[:, i].reshape(-1, 1)
            ).ravel()
            scalers_y.append(sc_y)    

        # train + predict
        Y_pred_All = []
        Y_val_All = []
        svr_model_coefs = []
        svr_model_intercepts = []
        # 4.3 儲存計算係數
        # coef  = np.zeros(X.shape[1]) 
        svr_coefs_table = np.zeros([X.shape[1],Y.shape[1]]) 
        svr_intercepts_std = np.zeros(Y.shape[1]) 
        # 定義SVR模型
        svr = SVR(kernel='linear', epsilon=epsilon, C=Cost, gamma='scale') #'poly','linear','rbf'
        for i in range(Y.shape[1]):
            coef  = np.zeros(X.shape[1]) 
            mask = np.ones(X.shape[1], dtype=bool)
            if not ch_unselect or ch_unselect[i].size==0:
                pass
            else:
                mask[ch_unselect[i]] = False  
             
            svr.fit(X_train[:,mask], Y_train[:, i]) 
            y_pred_std = svr.predict(X_train[:,mask]).ravel()
            y_pred = scalers_y[i].inverse_transform(
                                    y_pred_std.reshape(-1, 1)).ravel()
            Y_pred_All.append(y_pred)
            Y_val_All.append(Y[:, i]) 
              
            coef[mask] = np.array(svr.coef_ if svr.shape_fit_[0] != Y.shape[0] else svr.coef_.T).ravel()
            bs = svr.intercept_
            sigma_x = scalers_x.scale_
            mu_x = scalers_x.mean_

            sigma_y = scalers_y[i].scale_[0]
            mu_y = scalers_y[i].mean_[0]
            # intercept = svr.intercept_
            # svr_coefs_table[mask,i] = np.array(svr[0][i].coef_ if svr.shape[0] != Y.shape[1] else svr[0][i].coef_.T)
            # 原始空間係數
            W = coef * (sigma_y / sigma_x)
            # svr_coefs_table[mask,i] = W
            svr_coefs_table[:,i] = W
            # 計算截距
            svr_intercepts_std[i] = mu_y + sigma_y * bs - np.sum(W * mu_x) # np.dot(W , mu_x) = np.sum(W * mu_x)
        # 合併累積數據
        y_pred_list = np.vstack(Y_pred_All)      
        y_val_list = np.vstack(Y_val_All) 
        
        return svr_coefs_table, svr_intercepts_std, y_pred_list.T, scalers_x,scalers_y         

    def _calculate_regression_stats( self,Y_true: np.ndarray, Y_pred: np.ndarray, 
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

    def run_svr_parameter_scan(self, X: np.ndarray, channel_unselect: List[int], Y: np.ndarray, comp_cols: List[str]):
        """
        執行SVR parameters掃描分析
        """
        X_valid, Y_valid = X,Y
        n_samples = X.shape[0]

        base = 0.001
        multipliers = [1, 2.5, 5, 10, 25, 50, 75, 100]

        epsilon_param = [base * m for m in multipliers]
        # epsilon_param=np.array([0.01,0.025,0.05,0.1,0.25,0.5,0.75,1])
        # epsilon_param=np.array( np.round(np.linspace(0.001, 0.01, 
        #                             num=5, 
        #                             endpoint=True, 
        #                             retstep=False, 
        #                             dtype=float),3))
        # Cost_param = np.array([0.01,0.025,0.05,0.1,0.25,0.5,0.75,1,2.5,5,7.5,10])
        Cost_param = np.array(np.round(np.logspace(-1, 2, num=5),3))
        param_grid = {
                    'Cost': Cost_param ,
                    'Epsilon': epsilon_param
                }
        # 存儲每個param的結果
        param_epsilon_results = {}

        for idx_e, epsilon in enumerate(epsilon_param):
            param_Cost_results = {}
            for idx_c, Cost in enumerate(Cost_param):
                svr_coefs_table, svr_intercepts_std, Y_pred,X_scalers,Y_scalers = self._fit_param_final(X_valid, channel_unselect, Y_valid, epsilon ,Cost)
                
                stats = self._calculate_regression_stats(Y_valid, Y_pred, comp_cols)

                param_Cost_results[str(Cost)] = {
                'model_coefs': svr_coefs_table, 
                'model_intercepts':svr_intercepts_std,
                'Y_pred': Y_pred,
                'X_scalers':X_scalers,
                'Y_scalers':Y_scalers, 
                'stats': stats
                }
            param_epsilon_results[str(epsilon)] = param_Cost_results 

        result = {
                'param_epsilon_results': param_epsilon_results,
                'X_valid': X_valid,
                'Y_valid': Y_valid,
                'epsilon_param': epsilon_param,
                'Cost_param': Cost_param,
                'param_grid' :param_grid,
                'comp_cols': comp_cols,
                'n_samples': n_samples
            }
            
            
        return result       