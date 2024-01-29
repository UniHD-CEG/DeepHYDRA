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
import matplotlib.ticker as mtick
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
        "DeepHYDRA-TranAD": [
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
        "DeepHYDRA-MSE": [
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
        "DeepHYDRA-SMSE": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.73, 0.2, 0.24, 0.13, 0.48],
            ["Point Contextual", 0.61, 0.1, 0.12, 0.07, 0.24],
            ["Persistent Global", 0.99, 0.79, 0.8, 0.66, 1],
            ["Persistent Contextual", 0.99, 0.71, 0.74, 0.55, 1],
            ["Collective Global", 0.99, 0.69, 0.72, 0.53, 1],
            ["Collective Trend", 0.95, 0.7, 0.71, 0.57, 0.91],
            ["Intra-Subgroup", 0.99, 0.68, 0.71, 0.51, 1],
        ],
        "DAGMM": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.612, 0.244, 0.241, 0.262, 0.229],
            ["Point Contextual", 0.553, 0.124, 0.121, 0.143, 0.109],
            ["Persistent Global", 0.998, 0.952, 0.951, 0.908, 1.000],
            ["Persistent Contextual", 0.998, 0.927, 0.928, 0.864, 1.000],
            ["Collective Global", 0.832, 0.726, 0.724, 0.795, 0.668],
            ["Collective Trend", 0.901, 0.830, 0.826, 0.857, 0.805],
            ["Intra Rack", 0.554, 0.172, 0.197, 0.375, 0.112],
        ],
        "DeepHYDRA-DAGMM": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.626, 0.258, 0.254, 0.259, 0.257],
            ["Point Contextual", 0.582, 0.175, 0.171, 0.183, 0.168],
            ["Persistent Global", 0.998, 0.945, 0.945, 0.896, 1.000],
            ["Persistent Contextual", 0.998, 0.918, 0.919, 0.848, 1.000],
            ["Collective Global", 0.981, 0.893, 0.893, 0.831, 0.966],
            ["Collective Trend", 0.901, 0.822, 0.817, 0.839, 0.805],
            ["Intra Rack", 0.996, 0.902, 0.904, 0.824, 0.996],
        ],
        "OmniAnomaly": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.620, 0.068, 0.091, 0.039, 0.279],
            ["Point Contextual", 0.553, 0.035, 0.040, 0.020, 0.146],
            ["Persistent Global", 0.980, 0.647, 0.677, 0.478, 1.000],
            ["Persistent Contextual", 0.980, 0.542, 0.597, 0.372, 1.000],
            ["Collective Global", 0.980, 0.519, 0.580, 0.351, 1.000],
            ["Collective Trend", 0.980, 0.579, 0.626, 0.408, 1.000],
            ["Intra Rack", 0.678, 0.232, 0.233, 0.164, 0.395],
        ],
        "DeepHYDRA-OmniAnomaly": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.616, 0.067, 0.089, 0.038, 0.271],
            ["Point Contextual", 0.576, 0.046, 0.057, 0.026, 0.190],
            ["Persistent Global", 0.981, 0.650, 0.680, 0.481, 1.000],
            ["Persistent Contextual", 0.981, 0.545, 0.600, 0.375, 1.000],
            ["Collective Global", 0.981, 0.523, 0.583, 0.354, 1.000],
            ["Collective Trend", 0.981, 0.583, 0.629, 0.411, 1.000],
            ["Intra Rack", 0.979, 0.501, 0.566, 0.335, 0.996]
        ],
        "USAD": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.656, 0.220, 0.226, 0.167, 0.321],
            ["Point Contextual", 0.565, 0.100, 0.098, 0.078, 0.139],
            ["Persistent Global", 0.995, 0.888, 0.890, 0.799, 1.000],
            ["Persistent Contextual", 0.995, 0.837, 0.844, 0.719, 1.000],
            ["Collective Global", 0.995, 0.824, 0.833, 0.701, 1.000],
            ["Collective Trend", 0.995, 0.856, 0.861, 0.749, 1.000],
            ["Intra Rack", 0.553, 0.147, 0.140, 0.200, 0.116],
        ],
        "DeepHYDRA-USAD": [
            ["Anomaly Type", "AUC-ROC", "F1", "MCC", "Prec", "Recall"],
            ["Point Global", 0.670, 0.230, 0.239, 0.171, 0.350],
            ["Point Contextual", 0.590, 0.130, 0.130, 0.099, 0.190],
            ["Persistent Global", 0.995, 0.882, 0.884, 0.789, 1.000],
            ["Persistent Contextual", 0.995, 0.829, 0.837, 0.708, 1.000],
            ["Collective Global", 0.995, 0.816, 0.826, 0.689, 1.000],
            ["Collective Trend", 0.995, 0.849, 0.855, 0.738, 1.000],
            ["Intra Rack", 0.995, 0.803, 0.815, 0.671, 1.000],
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
    
    deep_hydra_tranad_np = np.array(tables['DeepHYDRA-TranAD'])

    deep_hydra_tranad_df = pd.DataFrame(deep_hydra_tranad_np[1:, 1:].astype(np.float64),
                                        index=deep_hydra_tranad_np[1:, 0],
                                        columns=deep_hydra_tranad_np[0, 1:])
    
    informer_mse_np = np.array(tables['Informer-MSE'])

    informer_mse_df = pd.DataFrame(informer_mse_np[1:, 1:].astype(np.float64),
                                    index=informer_mse_np[1:, 0],
                                    columns=informer_mse_np[0, 1:])
    
    deep_hydra_mse_np = np.array(tables['DeepHYDRA-MSE'])

    deep_hydra_mse_df = pd.DataFrame(deep_hydra_mse_np[1:, 1:].astype(np.float64),
                                    index=deep_hydra_mse_np[1:, 0],
                                    columns=deep_hydra_mse_np[0, 1:])
    
    informer_smse_np = np.array(tables['Informer-SMSE'])

    informer_smse_df = pd.DataFrame(informer_smse_np[1:, 1:].astype(np.float64),
                                        index=informer_smse_np[1:, 0],
                                        columns=informer_smse_np[0, 1:])
    
    deep_hydra_smse_np = np.array(tables['DeepHYDRA-SMSE'])

    deep_hydra_smse_df = pd.DataFrame(deep_hydra_smse_np[1:, 1:].astype(np.float64),
                                    index=deep_hydra_smse_np[1:, 0],
                                    columns=deep_hydra_smse_np[0, 1:])
    
    omni_anomaly_np = np.array(tables['OmniAnomaly'])

    omni_anomaly_df = pd.DataFrame(omni_anomaly_np[1:, 1:].astype(np.float64),
                                                    index=omni_anomaly_np[1:, 0],
                                                    columns=omni_anomaly_np[0, 1:])
    
    deep_hydra_omni_anomaly_np = np.array(tables['DeepHYDRA-OmniAnomaly'])

    deep_hydra_omni_anomaly_df = pd.DataFrame(deep_hydra_omni_anomaly_np[1:, 1:].astype(np.float64),
                                                            index=deep_hydra_omni_anomaly_np[1:, 0],
                                                            columns=deep_hydra_omni_anomaly_np[0, 1:])
    
    dagmm_np = np.array(tables['DAGMM'])

    dagmm_df = pd.DataFrame(dagmm_np[1:, 1:].astype(np.float64),
                                index=dagmm_np[1:, 0],
                                columns=dagmm_np[0, 1:])
    
    deep_hydra_dagmm_np = np.array(tables['DeepHYDRA-DAGMM'])

    deep_hydra_dagmm_df = pd.DataFrame(deep_hydra_dagmm_np[1:, 1:].astype(np.float64),
                                                    index=deep_hydra_dagmm_np[1:, 0],
                                                    columns=deep_hydra_dagmm_np[0, 1:])
    
    usad_np = np.array(tables['USAD'])

    usad_df = pd.DataFrame(usad_np[1:, 1:].astype(np.float64),
                                index=usad_np[1:, 0],
                                columns=usad_np[0, 1:])
    
    deep_hydra_usad_np = np.array(tables['DeepHYDRA-USAD'])

    deep_hydra_usad_df = pd.DataFrame(deep_hydra_tranad_np[1:, 1:].astype(np.float64),
                                        index=deep_hydra_usad_np[1:, 0],
                                        columns=deep_hydra_usad_np[0, 1:])
    
    results_combined = pd.concat({'T-DBSCAN': t_dbscan_df,
                                    'TranAD': tranad_df,
                                    'DeepHYDRA-TranAD': deep_hydra_tranad_df,
                                    'Informer-MSE': informer_mse_df,
                                    'DeepHYDRA-MSE': deep_hydra_mse_df,
                                    'Informer-SMSE': informer_smse_df,
                                    'DeepHYDRA-SMSE': deep_hydra_smse_df,
                                    'OmniAnomaly': omni_anomaly_df,
                                    'DeepHYDRA-OmniAnomaly': deep_hydra_omni_anomaly_df,
                                    'DAGMM': dagmm_df,
                                    'DeepHYDRA-DAGMM': deep_hydra_dagmm_df,
                                    'USAD': usad_df,
                                    'DeepHYDRA-USAD': deep_hydra_usad_df},
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

    fig, ax = plt.subplots(figsize=(6.5, 3), dpi=300)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

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

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

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



    
