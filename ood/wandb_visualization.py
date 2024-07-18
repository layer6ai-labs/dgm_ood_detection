"""
This code is intended for visualization purposes on weights and biases.
The tables and figures on weights and biases are then used to produce the
actual evaluation metrics in the paper.
"""
import numpy as np
import typing as th
import wandb
import decimal

# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 20

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    global ctx
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def visualize_scatterplots(
    scores: np.ndarray,
    column_names: th.Optional[th.List[str]] = None,
    title: str = 'Score-Scatterplot',
):
    """
    A set of scores with their corresponding column names are given.
    This would be visualized in a scatterplot in W&B. If we have 
    'k' columns, then we would have 'k choose 2' scatterplots. With each of the
    scatterplots being visualized in W&B showing the representation of each row 
    corresponding to those two metrics.
    
    Args:
        scores (np.ndarray): The scores that we want to visualize.
        column_names (th.Optional[th.List[str]], optional): The name of the columns. Defaults to score-i
        title (str, optional): The title of the scatterplots, defaults to 'Score-Scatterplot'.
    """
    data = []
    column_names = column_names or [f"score-{i}" for i in range(scores.shape[1])]
    for i in range(scores.shape[0]):
        row = []
        for j in range(scores.shape[1]):
            row.append(scores[i, j])
        data.append(row)
    
    table = wandb.Table(data=data, columns = column_names)
    
    for i in range(len(column_names)):
        for j in range(i+1,len(column_names)):
            wandb.log(
                {
                    f"scatter/{title}-{column_names[i]}-vs-{column_names[j]}": wandb.plot.scatter(table, column_names[i], column_names[j], title=f"{column_names[i]} and {column_names[j]}")
                }
            )

def visualize_trends(
    scores: np.ndarray,
    t_values: np.ndarray,
    title: str = 'scores',
    x_label: str = 't-values',
    y_label: str = 'scores',
    with_std: bool = False,
):
    """
    This function visualizes the trends in the scores. Scores is interpreted as an
    aggregation of scores.shape[0] trends. Each row represents a trend and the t_values
    is a 1D array of size scores.shape[1] representing the t-values of each trend.
    
    The visualization is done in W&B, and exports from them are useful.
    
    When with_std is set to False, the average of the trends are plotted.
    and when with_std is set to True the standard deviation of them is also taken into consideration.
    
    
    Args:
        scores (np.ndarray): An ndarray where each row represents the scores in a trend and each column is a specific t-value.
        t_values (np.ndarray): A monotonically increasing array of t-values in a trend.
        with_std (bool, optional): If set to true, then the std of the trends are also visualized.
        title (str, optional): The title of the plots. Defaults to 'scores'.
        x_label (str, optional): The x-axis name of the trend. Defaults to 't-values'.
        y_label (str, optional): The y-axis name of the trend. Defaults to 'scores'.
    """
    
    def _convert_float(t):
        try:
            a, b = float_to_str(t).split('.')
            a = int(a)
        except Exception as e:
            b = 0
            a = t
        return "{0:4d}".format(a) + f"_{b}"
    
    everything_columns = [f"t={_convert_float(t)}" for t in t_values]
    everything = wandb.Table(
        columns = everything_columns,
        data = [[val for val in row] for row in scores],
    )
    wandb.log({
        'all_trends': everything,
    })
    
    mean_scores = []
    mean_minus_std = []
    mean_plus_std = []
    
    for _, i in zip(t_values, range(scores.shape[1])):
        scores_ = scores[:, i]
        avg_scores = np.nanmean(scores_)
        std_scores = np.nanstd(scores_)
        
        mean_scores.append(avg_scores)
        mean_minus_std.append(avg_scores - std_scores)
        mean_plus_std.append(avg_scores + std_scores)
        
    if with_std:
        ys = [mean_scores, mean_minus_std, mean_plus_std]
        keys = [f"{y_label}-mean", f"{y_label}-std", f"{y_label}+std"]
        
        wandb.log({
            f"trend/{title}": wandb.plot.line_series(
                xs = t_values,
                ys = ys,
                keys = keys,
                title = title,
                xname = x_label,
            )
        })
    else:   
        # Instead of multi-line plots, just plot a single line with the average score values
        table = wandb.Table(data = [[x, y] for x, y in zip(t_values, mean_scores)], columns = [x_label, y_label])
        wandb.log({
            f"trend/{title}": wandb.plot.line(
                table,
                x_label,
                y_label,
                title=title,
            )
        })


def visualize_histogram(
    scores: np.ndarray,
    plot_using_lines: bool = False,
    bincount: int = 10,
    reject_outliers: th.Optional[float] = None,
    x_label: str = 'scores',
    y_label: str = 'density',
    title: str = 'Histogram of the scores',
):
    """
    The generic histogram in weights and biases does not give
    us the capability to manually adjust the binwidth and limits
    our representation capabilities.
    
    Therefore, we have this function that plots a lineplot with our
    own plotting scheme that resembles the histogram.
    Also, while calling this function without the plot_using_lines,
    it will provide you the normal W&B histogram.
    
    Args:
        scores: The scores that we want to visualize. 
        plot_using_lines: If set to True then we use our own binning
            and come up with an approxiate density and plot that density using
            a line.
        bincount: The number of bins used in the histogram.
        reject_outliers: The percentage of outliers that we want to reject from the
                    beginning and end. If set to None, no outlier rejection happens.
        x_label: The label of the x-axis.
        y_label: The label of the y-axis.
        title: The title of the histogram.
    Returns: None
        It visualizes the scores in W&B.
    """
    
    table1 = wandb.Table(columns=[x_label], data=[[x] for x in scores])
    wandb.log({f"scores_histogram/{x_label}": wandb.plot.histogram(table1, x_label, title=f"histogram of {x_label}")})
    
    if plot_using_lines:
        # Now plot a line wandb plot 
        # sort all_scores 
        all_scores = np.sort(scores)
        
        # reject the first and last quantiles of the scores for rejecting outliers
        if reject_outliers is not None:
            L = int(reject_outliers * len(all_scores))
            R = int((1 - reject_outliers) * len(all_scores))
            all_scores = all_scores[L: R]
        
        # create a density histogram out of all_scores
        # and store it as a line plot in (x_axis, density)
        hist, bin_edges = np.histogram(all_scores, bins=bincount, density=True)
        density = hist / np.sum(hist)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # get the average distance between two consecutive centers
        avg_dist = np.mean(np.diff(centers))
        # add two points to the left and right of the histogram
        # to make sure that the plot is not cut off
        centers = np.concatenate([[centers[0] - avg_dist], centers, [centers[-1] + avg_dist]])
        density = np.concatenate([[0], density, [0]])
        
        data = [[x, y] for x, y in zip(centers, density)]
        table = wandb.Table(data=data, columns = [x_label, y_label])
        dict_to_log = {f"histogram/{title}/{x_label}": wandb.plot.line(table, x_label, y_label, title=title)}
        
        wandb.log(dict_to_log)

