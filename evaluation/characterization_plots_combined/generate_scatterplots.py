import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.markers import MarkerStyle
import seaborn as sns

# plt.rcParams['figure.constrained_layout.use'] = True


def legend_without_duplicate_labels(ax, loc):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc=loc)


if __name__ == '__main__':

    parameters_hlt = pd.read_csv('data/parameters_hlt_dcm_2018.csv', index_col=0)
    activations_hlt = pd.read_csv('data/activations_hlt_dcm_2018.csv', index_col=0)
    flops_hlt = pd.read_csv('data/flops_hlt_dcm_2018.csv', index_col=0)
    auc_roc_hlt = pd.read_csv('data/auc_roc_hlt_dcm_2018.csv', index_col=0)
    mcc_hlt = pd.read_csv('data/mcc_hlt_dcm_2018.csv', index_col=0)

    unreduced_data_regex = r'\(Un|STRADA|T-DBSCAN'

    sizes_hlt = parameters_hlt['Parameters'].add(activations_hlt['Activations'], fill_value=0)

    sizes_hlt.rename('Size', inplace=True)

    flops_deep_hydra_reduction = flops_hlt.loc['Reduction'] + flops_hlt.loc['T-DBSCAN']

    flops_hlt.loc['Informer-MSE (Unreduced)'] = flops_hlt.loc['Informer-MSE (Reduced)'] + flops_hlt.loc['Reduction']
    flops_hlt.loc['Informer-SMSE (Unreduced)'] = flops_hlt.loc['Informer-SMSE (Reduced)'] + flops_hlt.loc['Reduction']
    flops_hlt.loc['TranAD (Unreduced)'] = flops_hlt.loc['TranAD (Reduced)'] + flops_hlt.loc['Reduction']
    flops_hlt.loc['DAGMM (Unreduced)'] = flops_hlt.loc['DAGMM (Reduced)'] + flops_hlt.loc['Reduction']
    flops_hlt.loc['USAD (Unreduced)'] = flops_hlt.loc['USAD (Reduced)'] + flops_hlt.loc['Reduction']
    flops_hlt.loc['OmniAnomaly (Unreduced)'] = flops_hlt.loc['OmniAnomaly (Reduced)'] + flops_hlt.loc['Reduction']

    flops_hlt.loc['STRADA-MSE'] = flops_hlt.loc['Informer-MSE (Reduced)'] + flops_deep_hydra_reduction
    flops_hlt.loc['STRADA-SMSE'] = flops_hlt.loc['Informer-SMSE (Reduced)'] + flops_deep_hydra_reduction
    flops_hlt.loc['STRADA-TranAD'] = flops_hlt.loc['TranAD (Reduced)'] + flops_deep_hydra_reduction
    flops_hlt.loc['T-DBSCAN/DAGMM'] = flops_hlt.loc['DAGMM (Reduced)'] + flops_deep_hydra_reduction
    flops_hlt.loc['T-DBSCAN/USAD'] = flops_hlt.loc['USAD (Reduced)'] + flops_deep_hydra_reduction
    flops_hlt.loc['T-DBSCAN/OmniAnomaly'] = flops_hlt.loc['OmniAnomaly (Reduced)'] + flops_deep_hydra_reduction

    sizes_reduced_hlt = sizes_hlt.loc[~(sizes_hlt.index.str.contains(unreduced_data_regex))]
    flops_reduced_hlt = flops_hlt.loc[~(flops_hlt.index.str.contains(unreduced_data_regex))]
    auc_roc_reduced_hlt = auc_roc_hlt.loc[~(auc_roc_hlt.index.str.contains(unreduced_data_regex))]
    mcc_reduced_hlt = mcc_hlt.loc[~(mcc_hlt.index.str.contains(unreduced_data_regex))]

    sizes_unreduced_hlt = sizes_hlt.loc[(sizes_hlt.index.str.contains(unreduced_data_regex))]
    flops_unreduced_hlt = flops_hlt.loc[(flops_hlt.index.str.contains(unreduced_data_regex))]
    auc_roc_unreduced_hlt = auc_roc_hlt.loc[(auc_roc_hlt.index.str.contains(unreduced_data_regex))]
    mcc_unreduced_hlt = mcc_hlt.loc[(mcc_hlt.index.str.contains(unreduced_data_regex))]

    parameters_smd = pd.read_csv('data/parameters_smd.csv', index_col=0)
    activations_smd = pd.read_csv('data/activations_smd.csv', index_col=0)
    flops_smd = pd.read_csv('data/flops_smd.csv', index_col=0)
    auc_roc_smd = pd.read_csv('data/auc_roc_smd.csv', index_col=0)
    mcc_smd = pd.read_csv('data/mcc_smd.csv', index_col=0)

    # These colors are specifically chosen to improve
    # accessibility for readers with colorblindness

    colors = {  '1L-Method 3 (Reduced)': '#D81B60',
                '1L-Method 3 (Unreduced)': '#D81B60',
                '1L-Method 4 (Reduced)': '#1E88E5',
                '1L-Method 4 (Unreduced)': '#1E88E5',
                'MERLIN-S (Reduced)': '#FFC107',
                'MERLIN-S (Unreduced)': '#FFC107',
                'MERLIN-P (Reduced)': '#663399',
                'MERLIN-P (Unreduced)': '#663399',
                'T-DBSCAN': '#000000',
                'Informer-MSE (Reduced)': '#1CB2C5',
                'Informer-MSE (Unreduced)': '#1CB2C5',
                'STRADA-MSE': '#1CB2C5',
                'Informer-SMSE (Reduced)': '#6F8098',
                'Informer-SMSE (Unreduced)': '#6F8098',
                'STRADA-SMSE': '#6F8098',
                'TranAD (Reduced)': '#D4FC14',
                'TranAD (Unreduced)': '#D4FC14',
                'STRADA-TranAD': '#D4FC14',
                'DAGMM (Reduced)': '#004D40',
                'DAGMM (Unreduced)': '#004D40',
                'T-DBSCAN/DAGMM': '#004D40',
                'USAD (Reduced)': '#C43F42',
                'USAD (Unreduced)': '#C43F42',
                'T-DBSCAN/USAD': '#C43F42',
                'OmniAnomaly (Reduced)': '#1164B3',
                'OmniAnomaly (Unreduced)': '#1164B3',
                'T-DBSCAN/OmniAnomaly': '#1164B3',}
    
    markers = {  '1L-Method 3 (Reduced)': 'o',
                '1L-Method 3 (Unreduced)': 'o',
                '1L-Method 4 (Reduced)': 'o',
                '1L-Method 4 (Unreduced)': 'o',
                'MERLIN-S (Reduced)': 'o',
                'MERLIN-S (Unreduced)': 'o',
                'MERLIN-P (Reduced)': 'o',
                'MERLIN-P (Unreduced)': 'o',
                'T-DBSCAN': 'o',
                'Informer-MSE (Reduced)': '>',
                'Informer-MSE (Unreduced)': '>',
                'STRADA-MSE': 'D',
                'Informer-SMSE (Reduced)': '>',
                'Informer-SMSE (Unreduced)': '>',
                'STRADA-SMSE': 'D',
                'TranAD (Reduced)': '>',
                'TranAD (Unreduced)': '>',
                'STRADA-TranAD': 'D',
                'DAGMM (Reduced)': '>',
                'DAGMM (Unreduced)': '>',
                'T-DBSCAN/DAGMM': 'D',
                'USAD (Reduced)': '>',
                'USAD (Unreduced)': '>',
                'T-DBSCAN/USAD': 'D',
                'OmniAnomaly (Reduced)': '>',
                'OmniAnomaly (Unreduced)': '>',
                'T-DBSCAN/OmniAnomaly': 'D',}

    data_reduced_hlt = pd.concat((sizes_reduced_hlt,
                                flops_reduced_hlt,
                                auc_roc_reduced_hlt,
                                mcc_reduced_hlt), axis=1)
    
    data_reduced_hlt.drop('Reduction', inplace=True)
    data_reduced_hlt['Color'] = [colors[name] for name in data_reduced_hlt.index]
    data_reduced_hlt['Marker'] = [markers[name] for name in data_reduced_hlt.index]

    data_reduced_hlt.index = [name.split(' (')[0] for name in data_reduced_hlt.index]
    
    data_unreduced_hlt = pd.concat((sizes_unreduced_hlt,
                                    flops_unreduced_hlt,
                                    auc_roc_unreduced_hlt,
                                    mcc_unreduced_hlt), axis=1)

    data_unreduced_hlt['Color'] = [colors[name] for name in data_unreduced_hlt.index]
    data_unreduced_hlt['Marker'] = [markers[name] for name in data_unreduced_hlt.index]

    data_unreduced_hlt.index = [name.split(' (')[0] for name in data_unreduced_hlt.index]

    sizes_smd = parameters_smd['Parameters'].add(activations_smd['Activations'], fill_value=0)

    sizes_smd.rename('Size', inplace=True)

    data_smd = pd.concat((sizes_smd,
                            flops_smd,
                            auc_roc_smd,
                            mcc_smd), axis=1)

    print(data_reduced_hlt)
    print(data_unreduced_hlt)
    print(data_smd)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=300)

    axes = (('FLOPs', 'Reduced', axes[0][0]),
            ('Size', 'Reduced', axes[0][1]),
            ('FLOPs', 'Unreduced', axes[1][0]),
            ('Size', 'Unreduced', axes[1][1]),)

    for count, (metric_type, dataset_type, ax) in enumerate(axes):

        data = data_reduced_hlt if dataset_type == 'Reduced'\
                                        else data_unreduced_hlt


        if dataset_type == 'Unreduced':
            ax.set_ylim(70, 100)
        else:
            ax.set_ylim(75, 100)
            # if metric_type == 'FLOPs':
            #     ax.set_xlim(1e-3, 5e3)
            # else:
            #     ax.set_xlim(1e3, 5e9)

        ax.set_xscale('log')
        ax.set_ylabel('MCC [%]')

        ax.grid(zorder=0)

        if dataset_type == 'Reduced':
            dataset_type = 'Pre-Reduced'

        # ax.set_title(f'{metric_type} and AUC-ROC {dataset_type} Test Set')
        ax.set_title(f'{metric_type} and MCC {dataset_type} Test Set')

        if metric_type == 'FLOPs':
            ax.set_xlabel('MFLOPs')
        else:
            ax.set_xlabel('Parameters + Activations')

        if (count % 2) != 0:
            ax.set_ylabel(None) 

        for element in data.itertuples():

            metric = element.Size if metric_type == 'Size' else element.FLOPs/1e6

            if metric_type == 'FLOPs':
                if element.Index == 'MERLIN-S':
                    fill_style = 'left'
                elif element.Index == 'MERLIN-P':
                    fill_style = 'right'
                else:
                    fill_style = 'full'

            scatter = ax.scatter(metric,
                                    element.MCC,
                                    55,
                                    element.Color,
                                    MarkerStyle(element.Marker,
                                                    fillstyle=fill_style),
                                    edgecolors='k',
                                    zorder=3)
            
            ax.minorticks_off()


    marker_types = {'Non-ML-Based': 'o',
                        'ML-Based': '>',
                        'DeepHYDRA': 'D'}
    
    marker_legends = [mlines.Line2D([0], [0], color='white',\
                                                    marker=v,\
                                                    linestyle='none',\
                                                    markersize=7,\
                                                    markeredgecolor='k',\
                                                    label=k) for k, v in marker_types.items()]
    
    color_types = {  '1L-Method 3': '#D81B60',
                        '1L-Method 4': '#1E88E5',
                        'MERLIN-S': '#FFC107',
                        'MERLIN-P': '#663399',
                        'T-DBSCAN': '#000000',
                        'Informer-MSE': '#1CB2C5',
                        'Informer-SMSE': '#6F8098',
                        'TranAD': '#D4FC14',
                        'DAGMM': '#004D40',
                        'USAD': '#C43F42',
                        'OmniAnomaly': '#1164B3',}
    
    color_legends = [mlines.Line2D([0], [0], color=v,\
                                                marker='s',\
                                                linestyle='none',\
                                                markersize=15,\
                                                markeredgecolor='k',\
                                                label=k) for k, v in color_types.items()]


    # legend_1 = plt.legend(color_legends, loc=1)
    fig.legend(handles=marker_legends, title='Categories', loc='lower left', bbox_to_anchor=(0.1, 0.025), ncol=1)
    fig.legend(handles=color_legends, title='Methods', loc='lower right', bbox_to_anchor=(0.9, -0.0085), ncol=3)
    # fig.add_artist(legend_1)

    plt.tight_layout(rect=[0, 0.163, 1, 1])

    # plt.savefig(f'plots/scatterplots_size_and_flops_over_auc_roc.png')
    plt.savefig(f'plots/scatterplots_size_and_flops_over_mcc.png')