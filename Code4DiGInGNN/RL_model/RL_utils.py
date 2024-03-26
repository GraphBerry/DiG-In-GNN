import os
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    """
    Plot the learning curve
    """
    plt.figure()
    plt.plot(episodes, records, linestyle="-", color="r")
    plt.title(title)
    plt.xlabel("episode")
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig(figure_file)


def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + " is already exist!")
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + " create successfully!")


def scale_action(action, high, low):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias

    return action_
