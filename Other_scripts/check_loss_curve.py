import pandas as pd
import os
import pickle

from matplotlib import pyplot as plt

folder_path = '../results-chess'

folder_names = [
    name for name in os.listdir(folder_path)
    if os.path.isdir(os.path.join(folder_path, name)) and name != 'figures'
]

for out_path in folder_names:
    path_name = os.path.join(folder_path, out_path)

    with open(os.path.join(path_name, 'record.pkl'), "rb") as f:
        results = pickle.load(f)
    train_loss = results['epoch_train_loss']
    test_loss = results['epoch_test_loss']
    plt.plot(range(1,11),train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.ylim([0,1])
    plt.title(out_path)
    plt.legend()
    plt.show()
    plt.close()
    a = 1