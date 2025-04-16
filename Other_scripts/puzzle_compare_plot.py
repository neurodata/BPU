import pandas as pd
import os
import yaml
from DPU_lib.evaluate_on_puzzles.plotting import plot_bar
import numpy as np
import matplotlib.pyplot as plt

result_path = '../results-chess-learn-KC'
fig_save_path = os.path.join(result_path, 'figures')
os.makedirs(fig_save_path, exist_ok=True)
with open("../config/droso_config_KC.yaml", "r") as f:
    base_config = yaml.safe_load(f)
num_records = base_config.pop("train_num_sample")
experiments = base_config.pop('experiments')
timesteps_list = [2, 5, 10]
learnable_list = ['All','betweenKC','withinKC']

for exp_name in experiments.keys():
    if 'filter_num' in experiments[exp_name].keys():
        exp_name = exp_name + f"_{experiments[exp_name]['filter_num']}filters"
    for leanrnable in learnable_list:

        fig_save_name = f"{exp_name}_{leanrnable}_{num_records}"
        percentage_list = []
        acc_list = []
        for timesteps in timesteps_list:
            folder_name = f"{exp_name}_{leanrnable}_{num_records}_trial1_{timesteps}Timesteps-signed"
            result_df = pd.read_pickle(os.path.join(result_path,folder_name, 'puzzle_result.pkl'))
            x_labels, percentages, overall_acc = plot_bar(result_df, choice = 'elo',no_plot = True)
            percentage_list.append(percentages)
            acc_list.append(overall_acc)
        # x_labels = ['Overall'] + x_labels
        num_groups = len(percentage_list[0]) # elo score groups
        num_bars = len(percentage_list)  # num comparisons
        x = np.arange(num_groups)

        bar_width = 0.2
        offsets = np.linspace(-bar_width * (num_bars - 1) / 2, bar_width * (num_bars - 1) / 2, num_bars)

        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(num_bars)]

        fig, ax = plt.subplots(figsize=(15,6))

        for i in range(num_bars):
            ax.bar(x + offsets[i], percentage_list[i], width=bar_width, color=colors[i], label=f'{timesteps_list[i]}Timesteps - Overall Acc: {acc_list[i]:.4f}')

        # Label the axes and add a title
        ax.set_xlabel('Puzzle Rating (Elo) - Count')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim([0.0, 1.0])
        ax.set_yticks(np.arange(0.0, 1.05, 0.05))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plot_title = f"Percentage of Correct Results per Rating Interval\n {fig_save_name}"
        ax.set_title(plot_title)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_path,fig_save_name + ".png"))
        # plt.show()
        plt.close()
