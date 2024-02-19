
import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['figure.constrained_layout.use'] = True

data_dir = 'data/by_module/'


class TreeNode:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.children = {}


    def insert(self, path, value):
        node = self
        for part in path:
            if part not in node.children:
                node.children[part] = TreeNode(part)
            node = node.children[part]
        node.value = value


    def get_level(self, level):
        if level == 0:
            return {self.key: self.value}
        else:
            values = {}
            for child in self.children.values():
                values.update(child.get_level(level - 1))
            return values


    def __repr__(self, level=0):
        ret = ' ' * level + repr(self.value) + '\n'
        for child in self.children.values():
            ret += child.__repr__(level+1)
        return ret


def build_tree(data):
    root = TreeNode('root')
    for key, value in data.items():
        path = key.split('.') if key else []
        root.insert(path, value)
    return root


def legend_without_duplicate_labels(ax, loc):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc=loc)


def get_label_and_color(module_name: str):

    colors = ['#D81B60',
                '#1E88E5',
                '#FFC107',
                '#004D40',
                '#C43F42',
                '#6F8098',
                '#D4FC14',
                '#1CB2C5',
                '#18F964',
                '#1164B3',
                '#F0E442',
                '#000000']

    if module_name == 'lstm':
        return 'lstm', colors[4]
    elif 'enc' in module_name:
        return 'encoder', colors[1]
    elif 'dec' in module_name:
        return 'decoder', colors[2]
    elif module_name in ['fcn', 'projection']:
        return 'projection', colors[3]
    elif module_name == 'estimate':
        return 'estimate', colors[5]
    else:
        raise NotImplementedError('Label/color for module '
                                    f'{module_name} is not implemented')


if __name__ == '__main__':

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
        elif model_name in ['usad', 'mscred', 'dagmm', 'merlin']:
            model_name = model_name.upper()
        else:
            model_name = model_name.capitalize()

        dataset_name =\
            f'{model_params[1].upper()}_'\
            f'{model_params[2].upper()}_'\
            f'{model_params[3]}'

        if len(model_params) > 4:
            model_name = f'{model_name}-{model_params[4].upper()}'

        results = build_tree(results)

        results_level_1 = results.get_level(1)

        activations_all = results.get_level(0)['root']

        activations_summed = 0

        for activations in results_level_1.values():
            activations_summed += activations

        print(f'{model_name}: Deviation from reported sum: '\
                f'{100*(activations_all - activations_summed)/activations_all:.3f} %')

        results_all[model_name] = results_level_1

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.set_xscale('log')
    ax.set_xlim(1, 1e12)

    ax.set_title('Activations per Sample by Layer for HLT Dataset')
    ax.set_xlabel('Activations')
    ax.set_ylabel('Model')

    for model_name, results in results_all.items():

        if model_name != 'MERLIN':

            results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

            left = 0

            for operator, activations in results.items():

                label, color = get_label_and_color(operator)

                ax.barh(model_name,
                                activations,
                                color=color,
                                height=0.8,
                                left=left,
                                label=label)

                left += activations

        else:
            results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

            left = 0

            for operator, activations in results.items():

                ax.barh(model_name,
                                activations,
                                color='#000000',
                                fill=False,
                                hatch='/',
                                height=0.8,
                                left=left,
                                label='ESTIMATED')

                left += activations

    legend_without_duplicate_labels(ax, 'upper right')

    plt.savefig(f'plots/activation_comparison_by_module.png')