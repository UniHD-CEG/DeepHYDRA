
import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['figure.constrained_layout.use'] = True

data_dir = 'data/smd/by_operator/'

flop_display_lower_threshold_percent = 0.001


def group_small_flop_counts(results: dict,
                                flops_sum):

    flops_other = 0
    keys_to_remove = []

    for key, val in results.items():
        if val < flop_display_lower_threshold_percent*flops_sum:
            flops_other += val
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del(results[key])
    
    results['other'] = flops_other

    return results


def legend_without_duplicate_labels(ax, loc):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc=loc)


def get_label_and_color(op_name: str):

    colors = ['#D81B60',
                '#1E88E5',
                '#FFC107',
                '#004D40',
                '#C43F42',
                '#6F8098',
                '#18F964',
                '#555555',
                '#000000']

    if op_name == 'einsum':
        return 'einsum', colors[0]
    elif op_name == 'linear':
        return 'linear', colors[1]
    elif op_name in ['bmm', 'matmul']:
        return 'bmm/matmul', colors[2]
    elif op_name == 'conv':
        return 'conv', colors[3]
    elif 'norm' in op_name:
        return 'batchnorm/layernorm', colors[4]
    elif op_name in ['add', 'sub']:
        return 'add/sub', colors[5]
    elif op_name in ['mul', 'div']:
        return 'mul/div', colors[6]
    elif op_name == 'gru':
        return 'gru', colors[7]
    elif op_name in ['sum', 'cumsum', 'mean']:
        return '(cum)sum/mean', colors[8]
    else:
        raise NotImplementedError('Label/color for operation '
                                    f'{op_name} is not implemented')


if __name__ == '__main__':

    results_all = {}
    
    for file in os.listdir(data_dir):

        print(file)

        with open(os.path.join(data_dir, file), 'r') as input_file:
            results = json.load(input_file)

        model_params = file.split('.')[0].split('_')

        model_name = model_params[0]

        if model_name == 'tranad':
            model_name = 'TranAD'
        elif model_name == 'omnianomaly':
            model_name = 'OmniAnomaly'
        elif model_name in ['usad', 'mscred', 'dagmm', 'merlin']:
            model_name = model_name.upper()
        else:
            model_name = model_name.capitalize()
        
        # dataset_name =\
        #     f'{model_params[1].upper()}_'\
        #     f'{model_params[2].upper()}_'\
        #     f'{model_params[3]}'

        if len(model_params) > 2:
            model_name = f'{model_name}-{model_params[2].upper()}'

        results_all[model_name] = results

    results_all_pd = pd.DataFrame(index=results_all.keys(),
                                    columns=['FLOPs'])

    for model_name, results in results_all.items():

        results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

        flops_summed = int(np.sum(list(results.values())))

        results_all_pd.loc[model_name, 'FLOPs'] = flops_summed

    # print(results_all_pd)

    # results_all_pd.to_csv(
    #     '../characterization_plots_combined/data/flops_hlt_dcm.csv')

    results_all_pd.to_csv(
        '../characterization_plots_combined/data/flops_smd.csv')
    
    # fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # ax.set_xscale('log')
    # ax.set_xlim(1, 5e16)

    # ax.set_title('FLOPs per Sample by Operator on HLT Dataset')
    # ax.set_xlabel('FLOPs')
    # ax.set_ylabel('Model')

    # for model_name, results in results_all.items():

    #     if model_name != 'MERLIN':

    #         results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

    #         left = 0

    #         for operator, flops in results.items():

    #             label, color = get_label_and_color(operator)

    #             ax.barh(model_name,
    #                             flops,
    #                             color=color,
    #                             height=0.8,
    #                             left=left,
    #                             label=label)

    #             left += flops

    #     else:
    #         results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

    #         left = 0

    #         for operator, flops in results.items():

    #             ax.barh(model_name,
    #                             flops,
    #                             color='#000000',
    #                             fill=False,
    #                             hatch='/',
    #                             height=0.8,
    #                             left=left,
    #                             label='ESTIMATED')

    #             left += flops

    # legend_without_duplicate_labels(ax, 'upper right')

    # plt.savefig(f'plots/computational_intensity_comparison_by_operator.png')