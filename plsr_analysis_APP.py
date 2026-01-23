import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

class PLSR_Analysis:
    """導入外參數設定"""
    def __init__(self,):
        pass

    def _preprocess_data(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """數據預處理：移除NaN並檢查數據充足性"""
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
        X_valid = X[mask]
        Y_valid = Y[mask]
        
        n_samples = X_valid.shape[0]
        if n_samples < 2:
            raise ValueError(f"篩選後資料點不足 ({n_samples}), 至少需 2 筆")
        
        return X_valid, Y_valid, n_samples
    
    def _determine_max_factor( self, n_sample: int, n_features: int, max_factor: int = 16) -> int:
        """確定最大Factor數量（與交叉驗證邏輯一致）"""
        if n_sample < 16:
            max_factor = max(1, n_sample - 2)
        
        max_factor = min(max_factor-2, n_features)
        
        if max_factor < 1:
            raise ValueError(f"無法進行分析：數據點數 ({n_sample}) 或特徵數 ({n_features}) 不足")
        
        return max_factor
    
    def _fit_single_factor_final( self, X: np.ndarray, ch_unselect, Y: np.ndarray, n_component: int) -> Tuple[PLSRegression, np.ndarray]:
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
        
    def _fit_single_factor_no_scaleY_final( self, X: np.ndarray, ch_unselect, Y: np.ndarray, n_component: int) -> Tuple[PLSRegression, np.ndarray]:    
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

    def _fit_single_factor( self, X: np.ndarray, Y: np.ndarray, n_component: int) -> Tuple[PLSRegression, np.ndarray]:
        """單個Factor的PLS建模"""
        pls = PLSRegression(n_components=n_component, scale=False)
        pls.fit(X, Y)
        Y_pred = pls.predict(X)
        return pls, Y_pred
    
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
    
    def run_pls_factor_scan(self, X: np.ndarray, ch_unselect ,Y: np.ndarray, 
                        comp_cols: List[str], 
                        max_factor: int = 16, 
                        progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        執行PLS Factor掃描分析
        """
        X_valid, Y_valid, n_samples = self._preprocess_data(X, Y)
        n_features = X_valid.shape[1]
        max_factor = self._determine_max_factor(n_samples, n_features, max_factor)

        # for i in range(len(comp_cols)):
        #     max_factor = min(max_factor,36-len(ch_unselect[i]))
        if np.array([len(x) for x in ch_unselect]).size > 0:
            max_factor = min(max_factor,min(36-np.array([len(x) for x in ch_unselect])))

        factor_results = {}
        # Factor掃描
        for factor in range(1, max_factor + 1):
            if progress_callback:
                progress_callback(factor, max_factor)
            
            pls, Y_pred,Y_scalers = self._fit_single_factor_final(X_valid,ch_unselect, Y_valid, factor)
            stats = self._calculate_regression_stats(Y_valid, Y_pred, comp_cols)

            pls_no_scaleY, Y_pred_no_scaleY,Y_scalers_no_scaleY = self._fit_single_factor_no_scaleY_final(X_valid,ch_unselect, Y_valid, factor)
            stats_no_scaleY = self._calculate_regression_stats(Y_valid, Y_pred_no_scaleY, comp_cols)
                
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