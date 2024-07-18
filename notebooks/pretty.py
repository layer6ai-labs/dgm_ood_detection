"""
This file contains all the functions that visualize plots in a color-themed and pretty way!
"""

from typing import Any
import numpy as np
import typing as th
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
from math import log10
from enum import Enum, auto
import os
import functools
import inspect
from roc_analysis import get_convex_hull, get_auc, get_roc_graph

import matplotlib.lines as mlines

#################
# color palette #
#################

class ColorTheme(Enum):
    OOD = "#A02C30" 
    OOD_SECONDARY = "#DF8A8A" 
    IN_DISTR = "#1F78B4" 
    IN_DISTR_SECONDARY = "#A6CBE3" 
    GENERATED = "#dfc27d" 
    DENSITY = "#dfc27d"   
    DARK_DENSITY = '#AC8200'
    PIRATE_BLACK = '#363838'
    OOD_to_IN_DISTR_0 = "#A02C30"
    OOD_to_IN_DISTR_1 = "#863B4A"
    OOD_to_IN_DISTR_2 = "#6C4A65"
    OOD_to_IN_DISTR_3 = "#535A7F"
    OOD_to_IN_DISTR_4 = "#39699A"
    OOD_to_IN_DISTR_5 = "#1F78B4"

FONT_FAMILY = 'serif'

hashlines = ['////', '\\\\\\\\', '|||', '---', '+', 'x', 'o', '0', '.', '*']
line_styles = ['-', '--', '-.', ':']

######################################################
# Generic decorators for the visualization functions #
######################################################

def show_plot(
    func
):
    """
    When this decorator is set, the function calls plt.show() after it returns
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        plt.show()
        # terminal decorator
    
    return wrapper

def savable(func):
    """
    Takes in a function that presumably plots a figure and adds a parameter
    file_name to it so that it can save the output of the plot in a file.
    """
    # Obtain the signature of the function
    sig = inspect.signature(func)
    # Add the 'file_name' parameter to the signature
    new_params = list(sig.parameters.values()) + [inspect.Parameter('file_name', inspect.Parameter.KEYWORD_ONLY, default=None)]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        
        file_name = kwargs.pop('file_name', None)
        
        # Execute the function and get the plot
        result = func(*args, **kwargs)

        
        plt.tight_layout()
        # save it
        if file_name:
            plt.savefig(os.path.join('figures', f"{file_name}.png"), bbox_inches='tight')

        return result

    # Update the wrapper's signature
    sig2 = inspect.signature(wrapper)
    new_sig = sig2.replace(parameters=new_params)
    wrapper.__signature__ = new_sig
    return wrapper

class StyleDecorator:
    def __init__(
        self,
        font_scale,
        style,
        line_style: th.Optional[str] = None
    ):
        self.font_scale = font_scale
        self.style = style
        self.line_style = line_style
        
    
    def __call__(self, func) -> Any:
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sns.set(font_scale=self.font_scale, style=self.style)
            if self.line_style:
                plt.rcParams['grid.linestyle'] = self.line_style
                
            return func(*args, **kwargs)
        return wrapper


def plot_kde_raw(
    x_values: th.List[np.array], 
    labels: th.List[str],
    colors: th.List[str],
    x_label: th.Optional[str]=None,
    y_label: th.Optional[str]= None,
    scale: int = 0,
    figsize: tuple = (10, 6),
    fontsize: th.Optional[int] = None,
    xlim: th.Optional[tuple] = None,
    skip_xticks: th.Optional[int] = None,
    tick_fontsize: th.Optional[int] = None,
    no_legend: bool = False,
    legend_fontsize: th.Optional[int] = None,
    legend_loc: th.Optional[str] = None,
    
    clear_ylabel: bool = False,
    clear_yticks: bool = False,
    show_scale: bool = True,
):
    """
    Plots KDE for given data.

    Parameters:
    - x_values: List of numpy arrays
    - labels: List of labels corresponding to x_values
    - colors: List of colors corresponding to x_values
    - scale: Integer value to scale the ylabel
    
    Returns:
    the ax for the possible decorators
    """
    
    # Ensure input lists are of the same length
    assert len(x_values) == len(labels) == len(colors), "Input lists must have the same length"
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=figsize)

    idx = 0
    for x, label, color in zip(x_values, labels, colors):
        density = sns.kdeplot(x, bw_adjust=0.5, color=color).get_lines()[-1].get_data()
        ax.fill_between(density[0], 0, density[1] , color=color, label=label, alpha=0.5, hatch=hashlines[idx % len(hashlines)])
        idx += 1
    
    y_label = y_label or 'Density'
    # Adjust y-axis label based on the scale
    if scale != 0:
        ax.yaxis.set_major_formatter(lambda x, _: f'{x * 10 ** (scale):.1f}')
        if show_scale:
            y_label += f' $\\times 10^{{{scale}}}$'
        ax.set_ylabel(f'{y_label}', fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    else:
        ax.set_ylabel(f'{y_label}', fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    
       
        
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    if xlim is not None:
        ax.set_xlim(xlim)
    
    if tick_fontsize is not None:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    if skip_xticks:
        x_ticks = ax.get_xticks()
        if skip_xticks < 0:
            ax.set_xticks([])
        else:
            ax.set_xticks(x_ticks[::skip_xticks])  # Use every second tick
    
    if clear_yticks:
        ax.tick_params(axis='y', colors='white')
    if clear_ylabel:
        ax.yaxis.label.set_color('white')
        
    
    ax.legend(loc=legend_loc, prop={'family': FONT_FAMILY, 'size': legend_fontsize})
    if no_legend:
        ax.legend_.remove()
    
    return ax


@savable
@StyleDecorator(font_scale=1.5, style='ticks')
@functools.wraps(plot_kde_raw)
def plot_kde(*args, **kwargs):
    return plot_kde_raw(*args, **kwargs)
    
@savable
@StyleDecorator(font_scale=1.5, style='whitegrid', line_style='--')
@functools.wraps(plot_kde_raw)
def plot_kde_dotted(*args, **kwargs):
    return plot_kde_raw(*args, **kwargs)

@savable
@StyleDecorator(font_scale=2, style='whitegrid', line_style='--')
def plot_trends(
    t_values: np.array,
    mean_values: th.List[np.array],
    labels: th.List[str],
    colors: th.List,
    std_values: th.Optional[th.List[np.array]] = None,
    y_label: th.Optional[str] = None,
    figsize: th.Optional[tuple] = (10, 6),
    with_std: bool = False,
    vertical_lines: th.Optional[th.List[float]] = None,
    vertical_line_thickness: th.Optional[float] = None,
    horizontal_lines: th.Optional[th.List[float]] = None,
    horizontal_lines_thickness: th.Optional[float] = None,
    horizontal_lines_color: th.Optional[th.List] = None,
    smoothing_window: int = 1,
    fontsize: th.Optional[int] = None,
    tick_fontsize: th.Optional[int] = None,
    custom_xticks: th.Optional[int] = None,
    
    no_legend: bool = False,
    legend_fontsize: th.Optional[int] = None,
    legend_loc: th.Optional[str] = None,
):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    idx = 0
    
    smoothing_window_means = []
    smoothing_window_stds = []
    
    if std_values is None:
        std_values = [None for _ in mean_values]
        
    for means, stds, color, lbl in zip(mean_values, std_values, colors, labels):
        smoothing_window_means.append(means)
        if stds is not None:
            smoothing_window_stds.append(stds)
        if len(smoothing_window_stds) > smoothing_window:
            smoothing_window_means.pop(0)
            if stds is not None:
                smoothing_window_stds.pop(0)
        
        smooth_mean = sum(smoothing_window_means) / len(smoothing_window_means)
        if stds is not None:
            smooth_std = sum(smoothing_window_stds) / len(smoothing_window_stds)
        
        # Create a lineplot using Seaborn
        sns.lineplot(x=t_values, y=smooth_mean, color=color, ax=ax, label=f'Avg. {lbl}' if stds is not None else lbl)

        # Use fill_between to add the transparent area representing std
        if stds is not None:
            if with_std:
                ax.fill_between(
                    t_values, smooth_mean - smooth_std, smooth_mean + smooth_std, 
                    alpha=0.3, label=f'std {lbl}', color=color, hatch=hashlines[idx % len(hashlines)])
            else:
                ax.fill_between(
                    t_values, smooth_mean - smooth_std, smooth_mean + smooth_std, 
                    alpha=0.3, color=color, hatch=hashlines[idx % len(hashlines)])
        idx += 1
    
    if vertical_lines is not None:
        for vert in vertical_lines:
            # Add a vertical dotted line at x=0.5 with color green
            ax.axvline(x=vert, color=ColorTheme.DARK_DENSITY.value, linestyle='--', linewidth=vertical_line_thickness)
    
    if horizontal_lines is not None:
        for hor, col in zip(horizontal_lines, horizontal_lines_color):
            ax.axhline(y=hor, color=col, linestyle=':', linewidth = horizontal_lines_thickness)
            
    # Set labels, title, and legend
    if x_label:
        ax.set_xlabel(x_label, fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize, fontdict={'family': FONT_FAMILY})
        
    # Adjusting tick font size
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    if custom_xticks:
        ax.set_xticks(custom_xticks)  
    
    
    ax.legend(loc=legend_loc, prop={'family': FONT_FAMILY, 'size': legend_fontsize})
    if no_legend:
        ax.legend_.remove()
    return ax
    

@savable
@StyleDecorator(font_scale=2, style='whitegrid', line_style='--')
def plot_scatter(
    x_s: th.List[np.array],
    y_s: th.List[np.array],
    labels: th.List[str],
    colors: th.List,
    x_label: th.Optional[str] = None,
    y_scale: th.Optional[int] = None,
    y_label: th.Optional[str] = None,
    figsize: th.Optional[tuple] = (10, 6),
    fontsize: th.Optional[int] = None,

    no_legend: bool = False,
    legend_fontsize: th.Optional[int] = None,
    legend_loc: th.Optional[str] = None,
    
    dotsize: th.Optional[int] = None,
    
    tick_fontsize: th.Optional[int] = None,
):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    for x, y, color, lbl in zip(x_s, y_s, colors, labels):
        # Create a lineplot using Seaborn
        if y_scale:
            sns.scatterplot(x=x, y=y * (10 ** y_scale), color=color, ax=ax, label=lbl, alpha=0.5)
        else:
            sns.scatterplot(x=x, y=y, color=color, ax=ax, label=lbl, alpha=0.5)
            
    # Set labels, title, and legend
    if x_label:
        ax.set_xlabel(x_label, fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    if y_label:
        if y_scale:
            if y_label[0] == '$':
                y_label = f'${y_label[1:-1]} \\times 10^{{{y_scale}}}$'
            else:
                y_label = f'${y_label} \\times 10^{{{y_scale}}}$'
                
        ax.set_ylabel(y_label, fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    
    
    ax.legend(loc=legend_loc, prop={'family': FONT_FAMILY, 'size': legend_fontsize})
    if no_legend:
        ax.legend_.remove()
    
    if dotsize is not None:
        legend = ax.legend_
        for handle in legend.legendHandles:
            handle.set_sizes([dotsize])
    
    
    if tick_fontsize is not None:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
    return ax


@savable
@StyleDecorator(font_scale=2, style='whitegrid', line_style='--')
def plot_rog(
    x_graphs: th.List[np.array],
    y_graphs: th.List[np.array],
    labels: th.List[str],
    has_graph: th.List[bool],
    colors: th.List[str],
    figsize: th.Optional[tuple] = (10, 6),
    fontsize: th.Optional[int] = None,
    alpha_scatter: float = 0.02,
    alpha_surface: float = 0.01,
    no_legend: bool = False,
    legend_fontsize: th.Optional[int] = None,
    legend_loc: th.Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=figsize)
    
    idx = 0
    for x_graph, y_graph, has_graph, lbl, color in zip(x_graphs, y_graphs,  has_graph, labels, colors):
        if isinstance(color, tuple):
            color1, color2 = color
        else:
            color1 = color
            color2 = color
        x_curve, y_curve = get_convex_hull(x_graph, y_graph)
        print(f"AUC of {lbl} = {get_auc(x_curve, y_curve)}")
        
        sns.lineplot(x=x_curve, y=y_curve, color=color1, label=f"ROC {lbl}")
        ax.fill_between(x_curve, y_curve, np.zeros_like(x_curve), color=color2, alpha=alpha_surface, hatch=hashlines[idx % len(hashlines)])
        # ax.fill_between([0], [0], [0], label=f"ROC {lbl}", color=color2, hatch=hashlines[idx % len(hashlines)])
        
        # Create a lineplot using Seaborn
        if has_graph:
            sns.scatterplot(x=x_graph, y=y_graph, color=color1, ax=ax, alpha=alpha_scatter * y_graph)
            sns.scatterplot(x=[0], y=[0], color=color1, label=f"ROG {lbl}", ax=ax)
        idx += 1
        
    # Set labels, title, and legend
    ax.set_xlabel('FP Rate', fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    ax.set_ylabel('TP Rate', fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    
    ax.legend(loc=legend_loc, prop={'family': FONT_FAMILY, 'size': legend_fontsize})
    if no_legend:
        ax.legend_.remove()
    return ax

@savable
@StyleDecorator(font_scale=2, style='whitegrid', line_style='--')
def plot_linechart(
    x_values: th.Union[th.List, np.array],
    y_values: th.List[th.Union[np.array, th.List]],
    labels: th.List[str],
    colors: th.List[str],
    markers: th.List[str],
    trend_markers: th.List[str],
    figsize: th.Optional[tuple] = (10, 6),
    fontsize: th.Optional[int] = None,
    thickness_coeff: float = 1.0,
    relative_thickness: float = 2.5,
    alpha: float = 0.2,
    
    x_label: th.Optional[str] = None,
    y_label: th.Optional[str] = None,
    
    custom_xticks: th.Optional[th.List] = None,
    tick_fontsize: th.Optional[int] = None,
    
    no_legend: bool = False,
    legend_fontsize: th.Optional[int] = None,
    legend_loc: th.Optional[str] = None,
    legend_marker_size: th.Optional[int] = None,
):
    fig, ax = plt.subplots(figsize=figsize)
    
    idx = 0
    legend_lines = []
    for y, lbl, color, m, trend_marker in zip(y_values, labels, colors, markers, trend_markers):
        # Plot the trend line
        sns.lineplot(x=x_values, y=y, linewidth=thickness_coeff * relative_thickness * 2.5, alpha=alpha, color=color, linestyle=trend_marker, ax=ax)
        legend_lines.append(mlines.Line2D([], [], color=color, marker=m, linestyle=trend_marker, linewidth=relative_thickness * 0.3* legend_marker_size, markersize=legend_marker_size, markerfacecolor=color, label=lbl))


    for y, lbl, color, m, trend_marker in zip(y_values, labels, colors, markers, trend_markers):
       
        # Highlight the actual data points
        sns.scatterplot(x=x_values, y=y, color=color, s=thickness_coeff * 100, label=lbl, marker=m, ax=ax)

        
    # Set labels, title, and legend
    if x_label:
        ax.set_xlabel(x_label, fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize, fontdict={'family': FONT_FAMILY})
        
    # Adjusting tick font size
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    if custom_xticks:
        ax.set_xticks(custom_xticks)  
    
    ax.legend(loc=legend_loc, prop={'family': FONT_FAMILY, 'size': legend_fontsize}, handles=legend_lines)
    if no_legend:
        ax.legend_.remove()
    return ax
    
    
# def plot_histogram(
#     x_values: np.array, 
#     labels: th.List[str],
#     colors: th.List[str],
#     x_label: th.Optional[str]=None,
#     scale: int = 0,
#     figsize: tuple = (10, 6),
#     bins: int = 50,
#     binwidth: th.Optional[float] = None,
#     xlim: th.Optional[tuple] = None,
# ):
#     """
#     Plots overlapping histograms.
    
#     Parameters:
#         x_values (list of numpy arrays): Data values for histograms.
#         labels (list of str): Labels for each histogram.
#         colors (list of str): Colors for each histogram.
#     """
    
#     # Ensure input lists are of the same length
#     assert len(x_values) == len(labels) == len(colors), "Input lists must have the same length"
    
#     # Create figure and axis objects
#     fig, ax = plt.subplots(figsize=figsize)

#     idx = 0
#     for x, label, color in zip(x_values, labels, colors):
#         if binwidth is not None:
#             bin_args = {'binwidth': binwidth}
#         else:
#             bin_args = {'bins': bins}
#         # Plotting the histogram (density ensures it's normalized)
#         sns.histplot(x, kde=True, **bin_args,
#                      ax=ax, 
#                      color=color, label=label, element="step",
#                      stat="density", common_norm=False, 
#                     #  kde_kws=,
#                      hatch=hashlines[idx%len(hashlines)])
        
#         # sns.histplot(x, bins='auto', color=color, kde=False, label=label, stat='density', alpha=0.5, linewidth=0)
        
#         # Plotting the KDE on top
#         # sns.kdeplot(x, color=color, linestyle=line_styles[idx % len(line_styles)])
#         idx += 1
        
#     # Adjust y-axis label based on the scale
#     if scale != 0:
#         ax.yaxis.set_major_formatter(lambda x, _: f'{x * 10 ** (scale):.1f}')
#         ax.set_ylabel(f'Density $\\times 10^{{-{scale}}}$')
#     else:
#         ax.set_ylabel('Density')
    
#     if legend_loc:
#         # Add legend to the left of the plot
#         ax.legend(loc=legend_loc)
    
#     if x_label is not None:
#         ax.set_xlabel(x_label)
#     if xlim is not None:
#         ax.set_xlim(xlim)
        
#     plt.tight_layout()
    
#     plt.show()