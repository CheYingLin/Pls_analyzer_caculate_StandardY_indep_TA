以各自的y進行各自標準化後各自的PLS訓練
增加各成分畫出factor與explained_variance的關係圖可觀察componets曲線
再回測算法是裡用y標準化後的尺度進行回測進行截距計算，與原版有差。
更動crossvalidation的數據，以標準化的值進行R^2與explained_variance計算
FAE版->pls_analyzer.py
line103~108:standard Scaler有問題應該是不需要的
可產生檢量線有TA、無TA都可設定
增加回測分數比較RMSE,MAE,MAPE
增加PCA分析推薦頻譜spectrum選擇
增加VIP(varible importantance in projection)分析PLSR裡頻譜的權重比例(視覺化)


