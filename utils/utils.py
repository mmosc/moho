import os
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt


def select_pareto_models(metrics_df_minmax):
    """
    Given a dataframe that stores the values of the evaluation metrics of the models, each model
    being represented by a set of different hyperparameter configurations, it selects those that
    are non-dominated for that model class.
    :param metrics_df_minmax: pd.Dataframe the dataframe with the minmax rescaled metrics
    :return pareto_metrics_minmax: the dataframe restricted to the pareto optimal configurations
    """
    pareto_dataframes = []
    for model_class in metrics_df_minmax.model_class.unique():
        model_class_df = metrics_df_minmax[metrics_df_minmax.model_class == model_class]
        model_class_df['is_pareto_val'] = is_pareto_efficient_simple(model_class_df[
                                                                         ['ndcg_val', 'coverage_val', 'novelty_val',
                                                                          'kl_item_val', 'kl_user_val']].to_numpy())
        model_class_df = model_class_df[model_class_df.is_pareto_val]
        pareto_dataframes += [model_class_df]
        # groupby(['model_class']).agg(is_pareto)
    pareto_metrics_minmax = pd.concat(pareto_dataframes)
    pareto_metrics_minmax = pareto_metrics_minmax.drop(columns=['is_pareto_val'])
    return pareto_metrics_minmax


def is_pareto_efficient_simple(costs):
    """
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)  # Keep any point with a higher cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def subsubplot_scatter_lines(
        datasets_df,
        model_class=['lightgcn', 'bpr', 'vae', 'iknn'],
        marker_list=['x', '+', '$\circ$', '^'],

        ndcg_only=True,
):
    """
    Given a dataframe storing the metrics for each model class and for each dataset
    Do a scatterplot of pair-wise metrics.
    and computes the slope of the line of best fit, as well as the pearson's correlation.
    If ndcg_only is true, only ndcg is used as
    x-axis
    :param dataset_df: dataset containing the metrics for the models and on different datasets
    :param model_class: list of names used for denoting each model class
    :param marker_list: markers to be used in the scatter plot

    :return fig, axs: the matplotlib figure and axes
    :return slopes: the slopes of the lines of best fit
    :return correlations: pairwise pearson's correlations
    """
    metrics = ['ndcg_test', 'coverage_test',
               'novelty_test', 'kl_item_test', 'kl_user_test']

    metric_pairs = []
    if ndcg_only:
        i = 0
        for j in range(1, len(metrics)):
            metric_pairs += [(metrics[i], metrics[j])]
    else:
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                metric_pairs += [(metrics[i], metrics[j])]

    num_pairs = len(metric_pairs)
    num_datasets = len(datasets_df)
    width = num_pairs * 9
    fig, axs = plt.subplots(num_pairs, num_datasets, figsize=(24, width), dpi=80)

    slopes = []
    correlations = []
    for row, metric_pair in enumerate(metric_pairs):
        x_name, y_name = metric_pair
        row_slopes = []
        row_correlations = []
        for column, metrics_df in enumerate(datasets_df):
            plot_list = []
            labels = []
            for model, marker in zip(model_class, marker_list):
                data = metrics_df[metrics_df.model_class == model].copy()
                sns.regplot(x=x_name, y=y_name, data=data, fit_reg=True, label=model, marker=marker,
                            ax=axs[row, column])
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=data[x_name], y=data[y_name])
                row_slopes += [slope]
                row_correlations += [r_value]
                labels += [model, f'{intercept:.2f} {slope:+.2f}*x', f'{std_err:.2f}']
                axs[row, column].legend(loc='upper center', labels=labels, ncol=4)
        slopes += [row_slopes]
        correlations += [row_correlations]
        # axs[row, column].set_ylim([-0.2, 1.2])
    return fig, axs, slopes, correlations


def subplot_scatter_lines(
        datasets_df,
        x_name='ndcg_test',
        y_name='novelty_test',
        model_class=['lightgcn', 'bpr', 'vae', 'iknn'],
        marker_list=['x', '+', '$\circ$', '^'],
):
    """
    Given a dataframe storing the metrics for each model class and for each dataset
    Do a scatterplot of two metrics. computes the slope of the line of best fit, as well as the pearson's correlation
    :param dataset_df: dataset containing the metrics for the models and on different datasets
    :param x_name: metric on the x-axis
    :param y_name: metric on the y axis
    :param model_class: list of names used for denoting each model class
    :param marker_list: markers to be used in the scatter plot

    :return fig, axs: the matplotlib figure and axes
    """
    num_datasets = len(datasets_df)
    fig, axs = plt.subplots(1, num_datasets, figsize=(24, 9), dpi=80)

    for i, metrics_df in enumerate(datasets_df):
        labels = []
        for model, marker in zip(model_class, marker_list):
            data = metrics_df[metrics_df.model_class == model].copy()
            sns.regplot(x=x_name, y=y_name, data=data, fit_reg=True, label=model, marker=marker, ax=axs[i])
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=data[x_name], y=data[y_name])
            labels += [model, f'{intercept:.2f} {slope:+.2f}*x', f'{std_err:.2f}']
            axs[i].legend(loc='upper center', labels=labels, ncol=4)

        axs[i].set_ylim([-0.2, 1.2])
    return fig, axs


def plot_scatter_lines(
        metrics_df,
        x_name='ndcg_test',
        y_name='novelty_test',
        model_class=['lightgcn', 'bpr', 'vae', 'iknn'],
        marker_list=['x', '+', '$\circ$', '^'],
):
    """
    Given a dataframe storing the metrics for each model class and for each dataset
    Do a scatterplot of two metrics. computes the slope of the line of best fit, as well as the pearson's correlation
    :param dataset_df: dataset containing the metrics for the models and on different datasets
    :param x_name: metric on the x-axis
    :param y_name: metric on the y axis
    :param model_class: list of names used for denoting each model class
    :param marker_list: markers to be used in the scatter plot

    :return plot_list: list of plots
    """
    plot_list = []
    labels = []
    plt.figure(figsize=(8, 9), dpi=80)

    for model, marker in zip(model_class, marker_list):
        data = metrics_df[metrics_df.model_class == model].copy()
        p = sns.regplot(x=x_name, y=y_name, data=data, fit_reg=True, label=model, marker=marker)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=data[x_name], y=data[y_name])
        labels += [model, f'{intercept:.2f} {slope:+.2f}*x', f'{std_err:.2f}']
        plot_list += [p]

    plt.legend(loc='upper center', labels=labels, ncol=4)

    return tuple(plot_list)


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def create_folder_if_not_exist(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"Directory {path} created!")


def assign_model(model_path):
    """
    """
    if 'MultiVAE' in model_path:
        model = 'multivae'
    elif 'BPR' in model_path:
        model = 'bpr'
    else:
        model = 'unknown_model'
    return model


def haolun_normalization(
        metrics_df,
        metrics_names,
        metrics_to_invert_names,
):
    """
    Perform metrics normalization as in
    https://arxiv.org/abs/2105.02951
    """

    metrics_df_ = metrics_df.copy()
    metrics_df_[metrics_to_invert_names] = (1 / metrics_df_[metrics_to_invert_names])
    metrics_df_[metrics_names] = metrics_df_[metrics_names] / metrics_df_[metrics_names].max()

    return metrics_df_


def minmax_(
        metrics_df_,
        columns_names,
        columns_to_invert_names,
):
    """
    MinMax for one (model, dataset) combination
    """

    metrics_df_[columns_names] = (metrics_df_[columns_names] - metrics_df_[columns_names].min()) / (
            metrics_df_[columns_names].max() - metrics_df_[columns_names].min())
    metrics_df_[columns_to_invert_names] = 1. - metrics_df_[columns_to_invert_names]

    return metrics_df_


def minmax_column(
        metric_series,
        columns_to_invert_names,
):
    """
    Minmax of a pd.series
    """

    metric_series = (metric_series - metric_series.min()) / (metric_series.max() - metric_series.min())
    if metric_series.name in columns_to_invert_names:
        metric_series = 1 - metric_series

    return metric_series


def minmax(
        metrics_df,
        columns_names,
        columns_to_invert_names,
        group_by_model=False,
):
    """
    Minmax for one dataset, across all models if group_by_model=False. If true, for each model separately
    """

    metrics_df_ = metrics_df.copy()
    if group_by_model:
        df_list = []
        for model_class in metrics_df_.model_class.unique():
            df_list += [minmax_(metrics_df_[metrics_df_.model_class == model_class], columns_names=columns_names,
                                columns_to_invert_names=columns_to_invert_names)]

        return pd.concat(df_list)

    else:
        return minmax_(metrics_df_, columns_names=columns_names, columns_to_invert_names=columns_to_invert_names)


def plot_correlation(
        metrics_df,
        mask=None,
        path=None,

):
    """
    Plot the heatmap of the correlation among metrics
    """
    correlation_df = metrics_df.corr()
    plt.figure(figsize=(16, 6))

    if mask:
        mask = np.triu(np.ones_like(correlation_df.corr(), dtype=np.bool_))

    heatmap = sns.heatmap(correlation_df,
                          mask=mask,
                          annot=True,
                          cmap=sns.cubehelix_palette(start=-.2, rot=-.3, dark=0, light=0.95, reverse=True, as_cmap=True)
                          )
    heatmap.set_title('Correlations among metrics', fontdict={'fontsize': 16}, pad=12);
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_metric_dists(
        full_df,
        DATASET='ml-100k',
        MODEL='BPR',
        METRIC="NDCG@10",
):
    """
    Plot of the metrics as a function of the sampling threshold
    """
    one_model_mult_thresh = full_df[full_df.model == MODEL]
    one_model_mult_thresh = one_model_mult_thresh[one_model_mult_thresh.dataset == DATASET]
    dist = sns.displot(one_model_mult_thresh,
                       x=METRIC,
                       hue="harm",
                       kind="kde",
                       bw_adjust=1.,
                       palette={
                           0.00: 'red',
                           0.01: 'orange',
                           0.05: 'green',
                           0.10: 'blue'}
                       )

    dist.fig.suptitle(f'{METRIC}, {DATASET}, {MODEL}')
    dist.savefig(f'{METRIC}_{DATASET}_{MODEL}.png')


def map_series_to_int(
        series: pd.Series,
        path_for_dict=None,
) -> pd.Series:
    import pickle
    int_to_ml_dict = {id_: ml for id_, ml in enumerate(series.unique())}
    ml_to_int_dict = {ml: id_ for id_, ml in int_to_ml_dict.items()}
    if path_for_dict:
        with open(path_for_dict, 'wb') as f:
            pickle.dump(ml_to_int_dict, f)
    return series.replace(ml_to_int_dict)
