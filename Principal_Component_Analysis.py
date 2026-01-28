import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class principal_component_analysis:
    """導入外部畫圖函式"""
    def __init__(self,):
        pass
    def run_PCA_analyzer(self, spectra_data: np.ndarray ,Training_has_ta:bool):
        # 假设spectra_data为预处理后的光谱数据集，维度为样本数 x 波长数
        # 数据标准化
        scaler = StandardScaler()
        spectra_data_std = scaler.fit_transform(spectra_data)

        # 执行PCA
        pca = PCA(n_components=0.95) # 选择累积贡献率为95%的主成分
        spectra_pca = pca.fit_transform(spectra_data_std)

        # # 绘制累积贡献率曲线
        # plt.plot(np.cumsum(pca.explained_variance_ratio_))
        # plt.xlabel('Number of Components')
        # plt.ylabel('Cumulative Explained Variance')
        # plt.show()

        # # 绘制前两个主成分的散点图
        # plt.scatter(spectra_pca[:, 0], spectra_pca[:, 1])
        # plt.xlabel('Principal Component 1')
        # plt.ylabel('Principal Component 2')
        # plt.show()
        # pca.components_: (n_components, n_features)
        loadings = pca.components_.T   # 變成 (37, 2)

        # 每個波長的整體貢獻度
        importance = np.sum(loadings**2, axis=1)
        # 排序
        idx = np.argsort(importance)[::-1]
        # 整體貢獻度為高前5
        top_wavelengths = idx[:5]
        # PC1 最像哪幾個波長？
        pc1_loading = np.abs(loadings[:, 0])
        idx_pc1 = np.argsort(pc1_loading)[::-1]
        # PC2 最像哪幾個波長？
        pc2_loading = np.abs(loadings[:, 1])
        idx_pc2 = np.argsort(pc2_loading)[::-1]
        if not Training_has_ta:
            print(f"建議不要剔除通道{(top_wavelengths)+1}")           
            # print(f"建議剔除通道{np.sort(idx_pc1[-10::])+1}")
            print(f"建議1:剔除通道{(idx_pc1[-10::])+1}")                       
            # print(f"建議剔除通道{np.sort(idx_pc2[-10::])+1}")
            print(f"建議2:剔除通道{(idx_pc2[-10::])+1}")
        else:
            print(f"建議不要剔除通道{(top_wavelengths)+1}")
            print(f"建議1:剔除通道{(idx_pc1[-10::])+1}")
            print(f"建議2:剔除通道{(idx_pc2[-10::])+1}")
        return spectra_pca
    
    def vip(self,factor_result, factor_table, del_idx,Training_has_ta:bool):#VariableImportanceinProjection
        vip_scores_all= []
        important_idx_ori_all = []
        important_idx_all = []
        if not Training_has_ta:
            all_idx = np.arange(0, 36) # [0,1,2,...,35] 
        else:
            all_idx = np.arange(1, 37)       
        keep_idx=[]

        for idx in range(len(factor_table)):
            if len(del_idx) > 0:
                keep_idx.append(np.delete(all_idx, del_idx[idx]))
            else: 
                keep_idx.append(all_idx)
            model_info = factor_result[factor_table[idx]]
            pls_model  = model_info.get('model')
            
            T = pls_model[0][idx].x_scores_
            W = pls_model[0][idx].x_weights_
            Q = pls_model[0][idx].y_loadings_

            p, h = W.shape
            s = np.diag(T.T @ T @ Q.T @ Q).reshape(h, -1)
            total_s = np.sum(s)

            vip_scores = np.sqrt(p * (W**2 @ s) / total_s)
            important_idx = np.where(vip_scores > 1)[0]
            important_idx_ori_all.append(important_idx)
            vip_scores_all.append(vip_scores)
            important_idx_tmp = keep_idx[idx][important_idx]
            if not Training_has_ta:
                important_idx_all.append(important_idx_tmp+1)
            else:
                important_idx_all.append(important_idx_tmp)
    #===================plot observation=============
        n_comp =len(factor_table)
        cols = min(n_comp, 2)
        rows = (n_comp + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows),dpi = 80)
        # fig, ax = plt.subplots(figsize=(10, 5))
        if len(factor_table) == 1:
                axes = [axes]
        else:
            axes = np.array(axes).flatten()
        for idx in range(len(factor_table)):
            ax = axes[idx]
            ax.scatter((keep_idx[idx]+(1 if not Training_has_ta else 0)),vip_scores_all[idx],marker='x')
            ax.scatter(important_idx_all[idx],vip_scores_all[idx][important_idx_ori_all[idx]].ravel(),color='r',marker='x')
            for n, i in enumerate(important_idx_all[idx], start=0):
                ax.text(
                    i ,                     # x 座標
                    vip_scores_all[idx][important_idx_ori_all[idx][n]]+ 0.03,      # y 座標# 往上偏一點             
                    f'X{i}',                 # 你要顯示的文字
                    fontsize=9,
                    color='r',
                    ha='left',
                    va='bottom'
                )
            ax.plot([1, max(keep_idx[idx])],[1, 1],linestyle='--',color='k')
            # xlabel / ylabel
            ax.set_xlabel('Predictor Variables')
            ax.set_ylabel('VIP Scores')
            # axis tight
            plt.tight_layout()
        plt.show()



        return important_idx_all