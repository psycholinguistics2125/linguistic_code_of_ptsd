""" 
Author: Quillivic Robin 
created date: 2022, 7 september
description :  useful function load config file or data
"""

import pandas as pd
import yaml
import os


def load_config(file_path: str = "config.yaml") -> dict:
    """
    load config file into dict python object
    """
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Fail to load config file because of {e}")
        config = {}
    return config


def flatten(list_of_lists: list) -> list:
    """
    Flatten a list of lists to a combined list

    Args:
        list_of_lists (list): _description_

    Returns:
        list: _description_
    """

    return [item for sublist in list_of_lists for item in sublist]


import matplotlib.patheffects as path_effects


def add_median_labels(ax, precision=".1f"):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4 : len(lines) : lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(
            x,
            y,
            f"{value:{precision}}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
        )
        # create median-colored border around white text for contrast
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground=median.get_color()),
                path_effects.Normal(),
            ]
        )
