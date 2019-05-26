"""Matplotlib.pyplot util functions."""
import logging

import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)


def save_bar_chart(data,
                   output_path,
                   y_label,
                   x_tick_labels,
                   title,
                   bar_width=.75,
                   size_inches=(10, 7)):
    """Save bar chart.

    Args:
        data (list(int or float)): Each entry in list refers to one bar. Should have same length as x_tick_labels.
        output_path (str or pathlib.Path): Path to save bar chart to.
        y_label (str): Label for y axis.
        x_tick_labels (list(str)): Name for each bar. This number will be displayed right below the bar.
        title (str): Title of bar
        bar_width (float): Bar width.
        size_inches (tuple(int or float)): Size of plot in inches.

    """
    LOGGER.debug("Plot file ...")
    ind = np.arange(len(x_tick_labels))

    fig, ax = plt.subplots()
    fig.set_size_inches(*size_inches, forward=True)

    rects = ax.bar(ind, data, bar_width, color='b')

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(x_tick_labels)

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
                '%d' % int(height),
                ha='center', va='bottom')
    LOGGER.debug("Save file ...")
    plt.savefig(output_path)


def save_scatter_plot_with_classes(output_path, types, data, class_color_dict):
    """Saves scatter to disk.
    
    Each point needs to have class (which is held in types). Plot each point of a class with the same color.

    Args:
        output_path (str or pathlib.Path): Path to save figure to
        types (list(str)): Each entry holds a class. Index must match data rows. 
        data (numpy.ndarray(numpy.float)):  Data of size [n_samples, 2]
        class_color_dict (dict): Each key refers to class while the entries refer to the color. 


    """
    for class_types, color in class_color_dict.items():
        subset = data[types == class_types, :]
        plt.scatter(subset[:, 0], subset[:, 1], c=color, label=class_types, s=10)
        plt.legend()
    plt.savefig(output_path)

