import numpy as np
from typing import Tuple, List, Dict, Any, Callable, Optional
from SVR_Analysis import run_svr_cross_validation
from SVR_Analysis.svr_analysis_APP import SVR_Analysis
from SVR_Analysis.run_svr_cross_validation import run_svr_parameter_cross_validation_scan


class SVR_with_cross_validation:
    """導入外參數設定"""
    def __init__(self,):
        self.svr = SVR_Analysis()

    def run_SVR_with_cross_validation(self, X: np.ndarray, channel_unselect, Y: np.ndarray ,comp_cols , multi_algorithm_results , algorithm_name):

        svr_model_result = self.svr.run_svr_parameter_scan(X, channel_unselect, Y, comp_cols)
        # 執行交叉驗證
        svr_model_cv_result = run_svr_parameter_cross_validation_scan(X, channel_unselect, Y, comp_cols)
        
         # 儲存多算法結果
        # multi_algorithm_results = {}
        multi_algorithm_results[f"算法{algorithm_name}_(SVR)"] = {
            'svr_model': svr_model_result,
            'svr_cv': svr_model_cv_result,
            }
        return  multi_algorithm_results