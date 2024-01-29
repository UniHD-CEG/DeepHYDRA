
import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['figure.constrained_layout.use'] = True

data_dir = 'data/smd/by_operator/'

activation_display_lower_threshold = 2*8


def group_small_activation_counts(results: dict):

    activations_other = 0
    keys_to_remove = []

    for key, val in results.items():
        if val < activation_display_lower_threshold:
            activations_other += val
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del(results[key])
    
    results['other'] = activations_other

    return results


def legend_without_duplicate_labels(ax, loc):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc=loc)


if __name__ == '__main__':


    colors = {  'einsum': '#D81B60',
                'linear': '#1E88E5',
                'matmul': '#FFC107',
                'bmm': '#004D40',
                'conv': '#C43F42',
                'add': '#6F8098',
                'mean': '#D4FC14',
                'cumsum': '#1CB2C5',
                'sum': '#18F964',
                'layer_norm': '#1164B3',
                'batch_norm': '#F0E442',
                'gru': '#111111',
                'other': '#000000'}

    results_all = {}
    
    for file in os.listdir(data_dir):
        with open(os.path.join(data_dir, file), 'r') as input_file:
            results = json.load(input_file)

        model_params = file.split('.')[0].split('_')

        model_name = model_params[0]

        if model_name == 'tranad':
            model_name = 'TranAD'
        elif model_name == 'omnianomaly':
            model_name = 'OmniAnomaly'
        elif model_name in ['usad', 'mscred', 'dagmm']:
            model_name = model_name.upper()
        else:
            model_name = model_name.capitalize()

        if len(model_params) > 2:
            model_name = f'{model_name}-{model_params[2].upper()}'

        results_all[model_name] = group_small_activation_counts(results)

    results_all_pd = pd.DataFrame(index=results_all.keys(),
                                    columns=['Activations'])

    for model_name, results in results_all.items():

        results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

        activations_summed = int(np.sum(list(results.values())))

        results_all_pd.loc[model_name, 'Activations'] = activations_summed

    # results_all_pd.to_csv(
    #     '../characterization_plots_combined/data/activations_hlt_dcm_2018.csv')
    
    results_all_pd.to_csv(
        '../characterization_plots_combined/data/activations_smd.csv')

    # fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # ax.set_xscale('log')
    # ax.set_xlim(1, 1e12)

    # ax.set_title('Activations per Sample by Operator for HLT_DCM_2018 Dataset')
    # ax.set_xlabel('Activations')
    # ax.set_ylabel('Model')

    # for model_name, results in results_all.items():

    #     results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

    #     left = 0

    #     for operator, activations in results.items():
    #         ax.barh(model_name,
    #                         activations,
    #                         color=colors[operator],
    #                         height=0.8,
    #                         left=left,
    #                         label=operator)

    #         left += activations

    # legend_without_duplicate_labels(ax, 'upper right')

    # plt.savefig(f'plots/activation_comparison_by_operator.png')