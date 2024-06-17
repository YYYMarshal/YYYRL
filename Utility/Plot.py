import numpy as np
import matplotlib.pyplot as plt


def moving_average(y_list: [], window_size: int):
    cumulative_sum = np.cumsum(np.insert(y_list, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(y_list[:window_size - 1])[::2] / r
    end = (np.cumsum(y_list[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def plot(y_list: [], algorithm: str, env_name: str,
         is_plot_average=False,
         xlabel="Episode", ylabel="Episode Reward"):
    title = f"{algorithm} on {env_name}"
    x_list = list(range(len(y_list)))
    plt.plot(x_list, y_list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

    if is_plot_average is False:
        return

    mv_return = moving_average(y_list, 9)
    plt.plot(x_list, mv_return)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_multiple(lists: [], algorithm: str, env_name: str,
                  xlabel="Steps", ylabel="Reward"):
    title = f"{algorithm} on {env_name}"
    for y_list in lists:
        x_list = list(range(len(y_list)))
        plt.plot(x_list, y_list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
