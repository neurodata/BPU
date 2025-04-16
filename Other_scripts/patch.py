import pandas as pd
import os
import torch
from DPU_lib.evaluate_on_puzzles.evaluate_DPU_on_puzzles import eval_DPU_on_puzzles
from DPU_lib.evaluate_on_puzzles.plotting import plot_puzzle_results

folder_path = '../results-chess-learn-KC'

folder_names = [
    name for name in os.listdir(folder_path)
    if os.path.isdir(os.path.join(folder_path, name))
]

for out_path in folder_names:
    if out_path == 'figures':
        continue
    path_name = os.path.join(folder_path, out_path)


    # model_path = os.path.join(path_name, 'model.pth')
    # checkpoint = torch.load(model_path, weights_only=False)
    # model = checkpoint["model"]
    # nonzero_params = sum(torch.count_nonzero(p).item() for p in model.parameters())
    # print(out_path, nonzero_params)

    print(out_path)
    eval_DPU_on_puzzles(path_name)

    # result_df = pd.read_pickle(os.path.join(path_name,'puzzle_result.pkl'))
    # plot_puzzle_results(result_df, path_name, f"puzzle_result")