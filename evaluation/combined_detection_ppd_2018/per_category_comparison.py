#!/usr/bin/env python3

# Modifications copyright (C) 2023 [ANONYMIZED]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

if __name__ == '__main__':

    tables = {
        "T-DBSCAN": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.53, 0.11, 0.16, 0.39, 0.06],
            ["Point Contextual", 0.55, 0.17, 0.22, 0.5, 0.1],
            ["Persistent Global", 1, 0.99, 0.99, 0.98, 1],
            ["Persistent Contextual", 1, 0.99, 0.99, 0.98, 1],
            ["Collective Global", 0.65, 0.45, 0.52, 0.92, 0.3],
            ["Collective Trend", 0.66, 0.48, 0.55, 0.94, 0.32],
            ["Intra-Subgroup", 1, 0.98, 0.98, 0.97, 1],
        ],
        "TranAD": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.75, 0.18, 0.23, 0.11, 0.51],
            ["Point Contextual", 0.60, 0.08, 0.10, 0.05, 0.23],
            ["Persistent Global", 0.99, 0.75, 0.76, 0.6, 1],
            ["Persistent Contextual", 0.99, 0.66, 0.69, 0.49, 1],
            ["Collective Global", 0.99, 0.64, 0.68, 0.47, 1],
            ["Collective Trend", 0.95, 0.65, 0.67, 0.51, 0.91],
            ["Intra-Subgroup", 0.62, 0.21, 0.19, 0.17, 0.26],
        ],
        "STRADA-TranAD": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.73, 0.2, 0.24, 0.13, 0.48],
            ["Point Contextual", 0.61, 0.10, 0.12, 0.07, 0.24],
            ["Persistent Global", 0.99, 0.79, 0.8, 0.66, 1],
            ["Persistent Contextual", 0.99, 0.71, 0.74, 0.55, 1],
            ["Collective Global", 0.99, 0.69, 0.72, 0.53, 1],
            ["Collective Trend", 0.95, 0.7, 0.71, 0.57, 0.91],
            ["Intra-Subgroup", 0.99, 0.68, 0.71, 0.51, 1],
        ],
        "Informer-MSE": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.80, 0.25, 0.3, 0.15, 0.62],
            ["Point Contextual", 0.65, 0.13, 0.15, 0.08, 0.31],
            ["Persistent Global", 0.99, 0.79, 0.8, 0.65, 1],
            ["Persistent Contextual", 0.99, 0.71, 0.73, 0.54, 1],
            ["Collective Global", 0.99, 0.69, 0.72, 0.52, 1],
            ["Collective Trend", 0.99, 0.74, 0.76, 0.58, 1],
            ["Intra-Subgroup", 0.67, 0.31, 0.3, 0.27, 0.37],
        ],
        "STRADA-MSE": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.81, 0.26, 0.32, 0.16, 0.64],
            ["Point Contextual", 0.65, 0.14, 0.16, 0.09, 0.32],
            ["Persistent Global", 0.99, 0.8, 0.81, 0.66, 1],
            ["Persistent Contextual", 0.99, 0.72, 0.74, 0.56, 1],
            ["Collective Global", 0.99, 0.7, 0.73, 0.54, 1],
            ["Collective Trend", 0.99, 0.75, 0.76, 0.6, 1],
            ["Intra-Subgroup", 0.99, 0.68, 0.71, 0.52, 1],
        ],
        "Informer-SMSE": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.82, 0.27, 0.32, 0.17, 0.65],
            ["Point Contextual", 0.66, 0.14, 0.17, 0.09, 0.34],
            ["Persistent Global", 0.99, 0.8, 0.81, 0.66, 1],
            ["Persistent Contextual", 0.99, 0.72, 0.74, 0.56, 1],
            ["Collective Global", 0.99, 0.7, 0.73, 0.54, 1],
            ["Collective Trend", 0.99, 0.75, 0.77, 0.6, 1],
            ["Intra-Subgroup", 0.69, 0.34, 0.33, 0.3, 0.4],
        ],
        "STRADA-SMSE": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.73, 0.2, 0.24, 0.13, 0.48],
            ["Point Contextual", 0.61, 0.1, 0.12, 0.07, 0.24],
            ["Persistent Global", 0.99, 0.79, 0.8, 0.66, 1],
            ["Persistent Contextual", 0.99, 0.71, 0.74, 0.55, 1],
            ["Collective Global", 0.99, 0.69, 0.72, 0.53, 1],
            ["Collective Trend", 0.95, 0.7, 0.71, 0.57, 0.91],
            ["Intra-Subgroup", 0.99, 0.68, 0.71, 0.51, 1],
        ],
    }

    t_dbscan_np = np.array(tables['T-DBSCAN'])

    t_dbscan_df = pd.DataFrame(t_dbscan_np[1:, 1:].astype(np.float64),
                                index=t_dbscan_np[1:, 0],
                                columns=t_dbscan_np[0, 1:])
    
    tranad_np = np.array(tables['TranAD'])

    tranad_df = pd.DataFrame(tranad_np[1:, 1:].astype(np.float64),
                                index=tranad_np[1:, 0],
                                columns=tranad_np[0, 1:])
    
    strada_tranad_np = np.array(tables['STRADA-TranAD'])

    strada_tranad_df = pd.DataFrame(strada_tranad_np[1:, 1:].astype(np.float64),
                                        index=strada_tranad_np[1:, 0],
                                        columns=strada_tranad_np[0, 1:])
    
    informer_mse_np = np.array(tables['Informer-MSE'])

    informer_mse_df = pd.DataFrame(informer_mse_np[1:, 1:].astype(np.float64),
                                    index=informer_mse_np[1:, 0],
                                    columns=informer_mse_np[0, 1:])
    
    strada_mse_np = np.array(tables['STRADA-MSE'])

    strada_mse_df = pd.DataFrame(strada_mse_np[1:, 1:].astype(np.float64),
                                    index=strada_mse_np[1:, 0],
                                    columns=strada_mse_np[0, 1:])
    
    
    informer_smse_np = np.array(tables['Informer-SMSE'])

    informer_smse_df = pd.DataFrame(informer_smse_np[1:, 1:].astype(np.float64),
                                        index=informer_smse_np[1:, 0],
                                        columns=informer_smse_np[0, 1:])
    
    strada_smse_np = np.array(tables['STRADA-SMSE'])

    strada_smse_df = pd.DataFrame(strada_smse_np[1:, 1:].astype(np.float64),
                                    index=strada_smse_np[1:, 0],
                                    columns=strada_smse_np[0, 1:])
    
    results_combined = pd.concat({'T-DBSCAN': t_dbscan_df,
                                    'TranAD': tranad_df,
                                    'STRADA-TranAD': strada_tranad_df,
                                    'Informer-MSE': informer_mse_df,
                                    'STRADA-MSE': strada_mse_df,
                                    'Informer-SMSE': informer_smse_df,
                                    'STRADA-SMSE': strada_smse_df},
                                    names=['Model', 'Anomaly Type'])
    
    SMALL_SIZE = 13
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 13

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    print('Results per category')

    print(results_combined.groupby(level='Anomaly Type').mean())
    print(results_combined.groupby(level='Anomaly Type').std())

    print('Results per model')

    print(results_combined.groupby(level='Model').mean())
    print(results_combined.groupby(level='Model').std())

    results_per_category = results_combined.reset_index()
    results_per_category = pd.melt(results_per_category,
                                    id_vars=['Model', 'Anomaly Type'],
                                    value_vars=['MCC'],
                                    value_name='MCC')

    plt.rcParams['figure.constrained_layout.use'] = True

    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=300)

    sns.boxplot(data=results_per_category,
                            y='Anomaly Type',
                            x='MCC',
                            orient='h',
                            palette='deep')

#     q1 = results_combined.groupby('Anomaly Type').quantile(0.25)['MCC'].to_numpy()
#     q2 = results_combined.groupby('Anomaly Type').quantile(0.75)['MCC'].to_numpy()
#     outlier_top_lim = q2 + 1.5*(q2 - q1)
#     outlier_bottom_lim = q1 - 1.5*(q2 - q1)
# 
#     print(outlier_top_lim)
#     print(outlier_bottom_lim)
# 
#     for count, row in enumerate(results_combined.itertuples()):
# 
#         count = count % 7
# 
#         val = row.MCC
#         if val > outlier_top_lim[count] or val < outlier_bottom_lim[count]:
#             plt.text(count, val, row.Index, ha='left', va='center')

    ax = plt.gca()

    ax.set_xlim(0, 1)

    plt.setp(ax.get_yticklabels(),
                        rotation=35,
                        ha="right",
                        rotation_mode="anchor")

    plt.savefig('combined_detection_boxplot_by_category.png')

    results_per_model = results_combined.reset_index()
    results_per_model = pd.melt(results_per_model,
                                    id_vars=['Model', 'Anomaly Type'],
                                    value_vars=['MCC'],
                                    value_name='MCC')

    plt.rcParams['figure.constrained_layout.use'] = True

    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=300)

    sns.boxplot(data=results_per_model,
                            y='Model',
                            x='MCC',
                            orient='h',
                            palette='deep')

    ax = plt.gca()

    ax.set_xlim(0, 1)

    plt.setp(ax.get_yticklabels(),
                        rotation=35,
                        ha="right",
                        rotation_mode="anchor")

    plt.savefig('combined_detection_boxplot_by_model.png')



    
