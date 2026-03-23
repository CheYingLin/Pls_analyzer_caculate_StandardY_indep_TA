import numpy as np
import pandas as pd
import scipy.signal as signal
import os
import math
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import medfilt
from scipy.signal import find_peaks

matplotlib.rc('font', family='serif', serif=['ABC', 'MingLiU']) 
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def run_plot_MW_vs_concentration(prefix,comp_cols,timeRef,df):
    plots_per_page = 6
    rows, cols = 2, 3

    fig, ax = plt.subplots(figsize=(17.5, 9.5),dpi = 80)
    # fig, axs = plt.subplots(rows, cols, figsize=(17.5, 9.5),dpi = 80)
    y = np.array(timeRef[comp_cols])
    x = np.array(timeRef['Time'])
    #===調整取樣觀察點,觀察減少滴定點是否影響趨勢====
    linespace =np.linspace(0, len(x), 
                            num=25, 
                            endpoint=False, 
                            retstep=False, 
                            dtype=int)
    #=============================================
    ax.plot(x[linespace], y[linespace], '-o')
    ax.grid(True, alpha=0.8)
    # plt.show()
    figs = []  # 收集 figure
    for page in range(int(36/plots_per_page)):
        fig, axs = plt.subplots(rows, cols, figsize=(17.5, 9.5),dpi = 80)
        axs = axs.flatten()

        for i in range(plots_per_page):
            idx = page * plots_per_page + i
            if idx >= 36:
                break

            col = f"{prefix}{idx+1}"
            ax = axs[i]
            y = df[f"{idx+1}-MW_NON"]
            x = df['Time']
            ax.plot(x, y)
            ax.grid(True, alpha=0.8)
            ax.set_title(col)
            ax.legend(fontsize=10)
    plt.show()
    
def run_plot_MW_zoom_out(prefix,comp_cols,timeRef,df):
    plots_per_page = 4
    rows, cols = 2, 2   
    ch_num = 24  #=============只觀察36

    fig, ax = plt.subplots(figsize=(17.5, 9.5),dpi = 80)
    y = np.array(timeRef[comp_cols])
    x = np.array(timeRef['Time'])
    ax.plot(x, y, '-o', markersize=4, color='red', alpha=0.7)
    ax.grid(True, alpha=0.8)
    
    # plt.show()

    figs = []  # 收集 figure
    for page in range(int(ch_num/plots_per_page)):
        fig, axs = plt.subplots(rows, cols, figsize=(17.5, 9.5),dpi = 80)
        axs = axs.flatten()

        for i in range(plots_per_page):
            idx = page * plots_per_page + i
            if idx >= ch_num:
                break

            col = f"{prefix}{idx+1}"
            ax = axs[i]
            x = timeRef['Time']
            # y = signal.medfilt(df, kernel_size=(3, 1))#使用中值率波
            # y_med= df[:,idx]
            y = df[:,idx]
            # y_med = signal.medfilt(y, kernel_size=3)#使用中值率波
            # ax.scatter(x, y)
            ax.plot(x, y ,'-o', markersize=3, color='blue', alpha=0.7)
            ax.grid(True, alpha=0.8)
            ax.set_title(col)
            ax.set_xlabel('concentation', fontsize=12)
            ax.set_ylabel('Intensity', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()    
    plt.show()

def run_plot_MW_zoom_out_process(prefix,comp_cols,timeRef,df):
    plots_per_page = 4
    rows, cols = 2, 2   
    ch_num = 24  #=============只觀察36

    fig, ax = plt.subplots(figsize=(17.5, 9.5),dpi = 80)
    y = np.array(timeRef[comp_cols])
    # #===========濾波滴定點保留波峰波谷值
    y_smooth = medfilt(y, kernel_size=3)
    peaks, _ = find_peaks(y_smooth, distance=10, prominence=0.02)
    valleys, _ = find_peaks(-y_smooth, distance=10, prominence=0.02)

    idx = np.sort(np.concatenate([peaks, valleys]))
    y_extreme = y[idx]
    
    x = np.array(timeRef['Time'])
    x_extreme = x[idx]
    ax.plot(x_extreme , y_extreme, '-o', markersize=4, color='red', alpha=0.7)
    ax.grid(True, alpha=0.8)
    
    # plt.show()

    figs = []  # 收集 figure
    for page in range(int(ch_num/plots_per_page)):
        fig, axs = plt.subplots(rows, cols, figsize=(17.5, 9.5),dpi = 80)
        axs = axs.flatten()

        for i in range(plots_per_page):
            idx = page * plots_per_page + i
            if idx >= ch_num:
                break

            col = f"{prefix}{idx+1}"
            ax = axs[i]
            x = timeRef['Time']
            # y = signal.medfilt(df, kernel_size=(3, 1))#使用中值率波
            # y_med= df[:,idx]
            y = df[:,idx]
            # y_med = signal.medfilt(y, kernel_size=3)#使用中值率波
            # ax.scatter(x, y)
            ax.plot(x, y ,'-o', markersize=3, color='blue', alpha=0.7)
            ax.grid(True, alpha=0.8)
            ax.set_title(col)
            ax.set_xlabel('concentation', fontsize=12)
            ax.set_ylabel('Intensity', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()    
    plt.show()   

def run_plot_MWLine_VS_MWwindow_scatter_new( prefix,timedata, comp_cols,training_has_ta,df, sel_df): 
    plots_per_page = 4
    rows, cols = 2, 2
    tot_P = int(36/plots_per_page)
    figs = []  # 收集 figure

    for page in range(tot_P):
        fig, axs = plt.subplots(rows, cols, figsize=(17.5, 9.5),dpi = 80)
        axs = axs.flatten()
        for i in range(plots_per_page):
            idx = page * plots_per_page + i
            if idx >= 36:
                break

            col = f"{prefix}{idx+1}"
            ax = axs[i]
            y = df[f"{idx +1}-MW_NON"]
            x = df['Time']

            if training_has_ta:
                yy = sel_df[:,idx+1 ]
            else:
                yy = sel_df[:,idx ]
            xx = timedata['Time']
            ax.plot(x, y,'-', markersize=3, label='MW',color='blue', marker='.', alpha=0.3)
            ax.scatter(xx, yy,label='avg.MW', color='red', marker='o',alpha=0.8)
            ax.grid(True, alpha=0.8)
            ax.set_title(col)
            ax.legend(fontsize=10)
            ax.set_xlabel('concentation', fontsize=12)
            ax.set_ylabel('Intensity', fontsize=12)
            ax.tick_params(axis='x', rotation=45)

        fig.tight_layout()
        figs.append(fig)

    # 🔥 一次顯示全部
    plt.show()
