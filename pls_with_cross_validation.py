from PLSR_Analysis.plsr_analysis_APP import PLSR_Analysis
from PLSR_Analysis.Cross_validation import run_cross_validation_analysis

class PLS_with_cross_validation:
    """導入外參數設定"""
    def __init__(self,):
        self.plsr = PLSR_Analysis()
        self.pred_all_Y = False

    def run_PLS_with_cross_validation(self,X, channel_unselect, Y, comp_cols, multi_algorithm_results, algorithm_name, max_factor = 16): 
           
        pls_result = self.plsr.run_pls_factor_scan(
                                X, channel_unselect, Y, comp_cols, max_factor)

        # 執行交叉驗證
        cv_result = run_cross_validation_analysis(
                                X, channel_unselect, Y, comp_cols, self.pred_all_Y, max_factor)

        # 儲存結果（包含時間窗口設定）
        # 儲存多算法結果
        # multi_algorithm_results = {}
        multi_algorithm_results[f"算法{algorithm_name}_(PLSR)"] = {
            'pls': pls_result,
            'cv': cv_result,
            }
        return  multi_algorithm_results