import h5py
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

def dataread(PATH):
    file_name_features_train = PATH + r"\features\features_train.mat"
    file_name_features_test = PATH + r"\features\features_test.mat"
    file_name_labels_train = PATH + r"\labels\labels_train.mat"
    file_name_labels_test = PATH + r"\labels\labels_test.mat"

    features_train = h5py.File(file_name_features_train, 'r')
    x_train = np.transpose(features_train['x_train'][:])

    features_test = h5py.File(file_name_features_test, 'r')
    x_test = np.transpose(features_test['x_test'][:])

    labels_train = h5py.File(file_name_labels_train, 'r')
    y_train = np.transpose(labels_train['y_train'][:])

    labels_test = h5py.File(file_name_labels_test, 'r')
    y_test = np.transpose(labels_test['y_test'][:])

    return x_train, x_test, y_train, y_test

def dataread_inter(PATH):
    file_name_features_train = PATH + r"\X.mat"
    file_name_labels_train = PATH + r"\y.mat"

    features = h5py.File(file_name_features_train, 'r')
    x = np.transpose(features['X'][:])

    labels = h5py.File(file_name_labels_train, 'r')
    y = np.transpose(labels['y'][:])

    return x, y


def evaluation_metrics_old(y_pred, y_test):
    err = y_pred - y_test
    err_abs = abs(y_pred - y_test)

    err_mean = np.mean(err)
    err_std = np.std(err)

    err_mean_abs = np.mean(err_abs)
    err_std_abs = np.std(err_abs)

    return err_mean, err_std, err_mean_abs

def evaluation_metrics(y_pred, y_test):
    err = y_pred - y_test
    ME = np.mean(err)
    MAE = mean_absolute_error(y_pred, y_test)
    RMSE = np.sqrt(mean_squared_error(y_pred, y_test))
    R = r2_score(y_test, y_pred)

    return ME, RMSE, MAE, R



def bland_altman_plot(m1, m2,X_axis,
                      sd_limit=1.96,
                      ax=None,
                      scatter_kwds=None,
                      mean_line_kwds=None,
                      limit_lines_kwds=None,
                      ):
    """
    Bland-Altman Plot.
    A Bland-Altman plot is a graphical method to analyze the differences
    between two methods of measurement. The mean of the measures is plotted
    against their difference.
    Parameters
    ----------
    m1, m2: pandas Series or array-like
    sd_limit : float, default 1.96
        The limit of agreements expressed in terms of the standard deviation of
        the differences. If `md` is the mean of the differences, and `sd` is
        the standard deviation of those differences, then the limits of
        agreement that will be plotted will be
                       md - sd_limit * sd, md + sd_limit * sd
        The default of 1.96 will produce 95% confidence intervals for the means
        of the differences.
        If sd_limit = 0, no limits will be plotted, and the ylimit of the plot
        defaults to 3 standard deviatons on either side of the mean.
    ax: matplotlib.axis, optional
        matplotlib axis object to plot on.
    scatter_kwargs: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.scatter plotting method
    mean_line_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
    limit_lines_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
   Returns
    -------
    ax: matplotlib Axis object
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))

    # means = np.mean([m1, m2], axis=0)
    means = m2
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    if ax is None:
        ax = plt.gca()

    scatter_kwds = scatter_kwds or {}
    if 's' not in scatter_kwds:
        scatter_kwds['s'] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'black'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 3
    for kwds in [limit_lines_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'red'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 3

    if 'linestyle' not in mean_line_kwds:
        kwds['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kwds:
        kwds['linestyle'] = ':'

    ax.scatter(means, diffs, **scatter_kwds)
    ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

    # Annotate mean line with mean difference.
    ax.annotate('Mean:\n{}'.format(np.round(mean_diff, 2)),
                xy=(0.99, 0.5),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=20,
                xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kwds)
        ax.annotate('-SD{}: {}'.format(sd_limit, np.round(lower, 2)),
                    xy=(0.99, 0.07),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=20,
                    xycoords='axes fraction',
                    color='red')
        ax.annotate('+SD{}: {}'.format(sd_limit, np.round(upper, 2)),
                    xy=(0.99, 0.86),
                    horizontalalignment='right',
                    fontsize=20,
                    xycoords='axes fraction',
                    color='red')

    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

    ax.set_ylabel('Error (mmHg)', fontsize=20)
    ax.set_xlabel(X_axis, fontsize=20)
    ax.tick_params(labelsize=20)
    plt.tight_layout()
    return ax

def BHS_compute(predict, label):
    err = abs(predict - label)
    n = len(predict)
    CP_5 = len(predict[err <= 5]) / n
    CP_10 = len(predict[err <= 10]) / n
    CP_15 = len(predict[err <= 15]) / n
    return CP_5, CP_10, CP_15