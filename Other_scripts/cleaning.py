import os

root_dir = '../results-chess'  # change this to your folder path

for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith('.png'):
            full_path = os.path.join(dirpath, file)
            os.remove(full_path)