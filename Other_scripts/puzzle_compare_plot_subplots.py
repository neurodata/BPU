import pandas as pd
import os
import yaml

from DPU_lib.evaluate_on_puzzles.plotting import plot_bar
import numpy as np
import matplotlib.pyplot as plt

_OUTPUT_TYPES = ['','_residual','_cumulative']
_MODEL_TYPES = ['DPU_Base_Unlearnable','DPU_CNN_Unlearnable','DPU_CNN_Unlearnable_RELU']

NUM_RECORDS = 2560000
n_TIMESTEPS = [2, 5, 10, 15]

RESULT_PATH = '../results-chess'
FIG_SAVEPATH = os.path.join(RESULT_PATH, 'figures')
os.makedirs(FIG_SAVEPATH, exist_ok=True)

with open("../config/droso_config_KC.yaml", "r") as f:
    base_config = yaml.safe_load(f)
experiments = base_config.pop('experiments')

for compare_type in _OUTPUT_TYPES + _MODEL_TYPES:
    try:
        if compare_type == '':
            fig_save_name = 'output_default'
            exp_list = _MODEL_TYPES
        elif compare_type in _OUTPUT_TYPES:
            fig_save_name = 'output'+ compare_type
            exp_list = [m + compare_type for m in _MODEL_TYPES]
        else:
            fig_save_name = compare_type
            exp_list = [compare_type + o for o in _OUTPUT_TYPES]

        fig, axs = plt.subplots(3,1,figsize=(15,15))
        axs = axs.ravel()
        plot_idx = 0

        for exp_name in exp_list: # loop through 3 subplots
            if 'filter_num' in experiments[exp_name].keys():
                exp_name = exp_name + f"_{experiments[exp_name]['filter_num']}filters"
            # fig_save_name = f"{exp_name}_{NUM_RECORDS}"
            percentage_list = []
            acc_list = []
            for timesteps in n_TIMESTEPS:
                folder_name = f"{exp_name}_{NUM_RECORDS}_trial1_{timesteps}Timesteps-signed"
                result_df = pd.read_pickle(os.path.join(RESULT_PATH,folder_name, 'puzzle_result.pkl'))
                x_labels, percentages, overall_acc = plot_bar(result_df, choice = 'elo',no_plot = True)
                percentage_list.append(percentages)
                acc_list.append(overall_acc)
            num_groups = len(percentage_list[0]) # elo score groups
            num_bars = len(percentage_list)  # num comparisons
            x = np.arange(num_groups)

            bar_width = 0.2
            offsets = np.linspace(-bar_width * (num_bars - 1) / 2, bar_width * (num_bars - 1) / 2, num_bars)

            cmap = plt.get_cmap('tab10')
            colors = [cmap(i) for i in range(num_bars)]

            ax = axs[plot_idx]

            for i in range(num_bars):
                ax.bar(x + offsets[i], percentage_list[i], width=bar_width, color=colors[i], label=f'{n_TIMESTEPS[i]}Timesteps - Overall Acc: {acc_list[i]:.4f}')

            # Label the axes and add a title
            ax.set_xlabel('Puzzle Rating (Elo) - Count')
            ax.set_ylabel('Accuracy (%)')
            ax.set_ylim([0.0, 1.0])
            ax.set_yticks(np.arange(0.0, 1.05, 0.05))
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plot_title = f"Percentage of Correct Results per Rating Interval\n {exp_name}"
            ax.set_title(plot_title)

            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)

            ax.legend()
            plot_idx += 1
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_SAVEPATH,fig_save_name + ".png"))
        # plt.show()
        plt.close()
    except:
        continue
