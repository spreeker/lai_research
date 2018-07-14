""""
Plot of the all data (cru and lai and predicted lai).
of one location

Uses the predictive function below and calculates rmse.
"""
# import prepare_data
import os
import logging
from datetime import datetime

from load_datasets import load_data
# from load_datasets import calculate_moving_mean

from matplotlib import pyplot
import numpy as np
from functions_pred_lai import prediction_options
from settings import conf


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def _add_mesurement_plots(ds_to_plot, predictors, x, ax2, ax3, ax4, ax5):
    """Add subplots of measurements.

    x timestamps
    ax  graphs
    """

    handles = []

    # ax2.set_ylabel('C')
    if 'tmp' in predictors:
        y_tmp_values = ds_to_plot['tmp']
        tmp, = ax2.plot(x, y_tmp_values, color='r', label='Temperature')
        handles.append(tmp)

    if 'tmp_gdd' in predictors:
        y_tmp_values = ds_to_plot['tmp_gdd']
        tmp2, = ax2.plot(
            x, y_tmp_values, color='orange', label='Temperature > 5')
        handles.append(tmp2)

    if 'jolly_tmp' in predictors:
        y_tmp_values = ds_to_plot['jolly_tmp']
        tmp2, = ax2.plot(
            x, y_tmp_values, color='orange', label='Temperature (jolly)')
        handles.append(tmp2)

    # ax3.set_ylabel('mm')
    if 'pre' in predictors:
        y_pre_values = ds_to_plot['pre']
        pre, = ax3.plot(x, y_pre_values, color='b', label='Precipitation')
        handles.append(pre)

    if 'pre_one' in predictors:
        pre_one = ds_to_plot['pre_one']
        pre2, = ax3.plot(
            x, pre_one, color='orange', label='Precipitation memory')
        handles.append(pre2)

    if 'pre_two' in predictors:
        pre_two = ds_to_plot['pre_two']
        pre3, = ax3.plot(
            x, pre_two, color='red', label='Precipitation memory 2')
        handles.append(pre3)

    # ax4.set_ylabel('hPa')
    if 'vap' in predictors:
        y_vap_values = ds_to_plot['vap']
        vap, = ax4.plot(x, y_vap_values, color='y', label='Vapour pressure')
        handles.append(vap)

    # ax4.set_ylabel('hPa')
    if 'jolly_vap' in predictors:
        y_vap_values = ds_to_plot['jolly_vap']
        vap, = ax4.plot(
            x, y_vap_values, color='y', label='Vapour pressure (jolly)')
        handles.append(vap)

    # vap2, = ax4.plot(x[8:], y_vap_avg_values[8:], color='orange', label='T2')
    # ax5.set_ylabel('mm')
    if 'pet' in predictors:
        y_pet_values = ds_to_plot['pet']
        pet, = ax5.plot(x, y_pet_values, label='Potential Evapotranspiration')
        handles.append(pet)

    return handles


def plot(timestamps, mlai, plai, datasets,
         predictors=None, p_label=None, text='',
         valid=None, aic=None):

    assert len(timestamps) == len(mlai)
    assert len(mlai) == len(plai)

    TENYEAR = 120

    ds_to_plot = {}

    if not predictors:
        return ValueError()
        # predictors = ['lai', 'tmp', 'pre', 'vap', 'pet']

    time_x = timestamps[:120]

    y_lai_values = mlai

    for i, ds_var in enumerate(predictors):
        ds_to_plot[ds_var] = datasets[i][:TENYEAR]

    y_pred_lai = plai

    # Three subplots sharing both x/y axes
    f, (ax1, ax2, ax3, ax4, ax5) =  \
        pyplot.subplots(5, sharex=True, sharey=False)
    pyplot.title(f"{p_label}", y=5.08)

    x = time_x
    log.info(x.size)
    log.info(y_lai_values.size)
    lai, = ax1.plot(x, y_lai_values, label='LAI')
    pred, = ax1.plot(x, y_pred_lai, color='g', label='Predicted LAI')

    handles = _add_mesurement_plots(
        ds_to_plot, predictors, x, ax2, ax3, ax4, ax5)
    handles.append(lai)
    handles.append(pred)

    # pet2, = ax5.plot(x[8:], y_pet_avg_values[8:], color='orange', label='T2')

    pyplot.legend(
        handles=handles, bbox_to_anchor=(1.05, 1.05), prop={'size': 14})

    # units
    pyplot.xlabel('Time (Months)')

    for txt in pyplot.gca().xaxis.get_majorticklabels():
        txt.set_rotation(90)

    pyplot.tight_layout(h_pad=1.0, pad=2.6, w_pad=1.5)

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0)
    pyplot.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    rmse = calc_rmse(plai[:120], mlai[:120])

    pyplot.figtext(
        0.73, 0.01, f'RMSE {rmse:.4f}', fontsize=10,
        horizontalalignment='center',
        verticalalignment='center', bbox=dict(facecolor='white', alpha=1),
    )

    if aic:
        pyplot.figtext(
            0.93, 0.01, f'AIC {aic:.4f}', fontsize=10,
            horizontalalignment='center',
            verticalalignment='center', bbox=dict(facecolor='white', alpha=1),
        )

    if text:
        pyplot.figtext(0.3, .02, text, fontsize=10, ha='center')
        # pyplot.suptitle(text)

    fig1 = pyplot.gcf()
    # manager = pyplot.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    pyplot.show()
    _save_plot(fig1, f'{p_label}-graphs')
    plot_correlation(y_lai_values, y_pred_lai, title=p_label)


def _save_plot(fig, subject):
    d = datetime.now()
    date = f'{d.year}-{d.month}-{d.day}'
    tile = conf['groupname']
    imgtarget = os.path.join(
        'imgs', 'anna', f'{tile}{date}-{subject}.png')
    fig.savefig(imgtarget)


def plot_correlation(xarr, yarr, title='LAI - Predicted LAI'):
    correlation = np.corrcoef(xarr, yarr)[1, 0]
    group = conf['groupname']
    pyplot.title(f"Correlation plot {group} {title}")
    pyplot.scatter(xarr, yarr)

    pyplot.figtext(
        0.83, 0.84, f'correlation coefficient {correlation:.3f}', fontsize=10,
        horizontalalignment='center',
        verticalalignment='center', bbox=dict(facecolor='white', alpha=1),
    )

    fig1 = pyplot.gcf()
    # manager = pyplot.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    pyplot.show()
    _save_plot(fig1, f'correlation {title}')


def make_prediction(datasets):
    label = conf['prediction_option']
    prediction_function = prediction_options[label]
    pred_lai = prediction_function(datasets)
    datasets[f'pred_{label}'] = pred_lai


def calc_rmse(predictions, targets):

    if type(predictions) == list:
        predictions = np.array(predictions)
        targets = np.array(targets)

    differences = predictions - targets           # the DIFFERENCES.
    differences_squared = differences ** 2        # the SQUARES of ^
    mean_of_differences_squared = differences_squared.mean()  # the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)           # ROOT of ^
    return rmse_val


def main():
    timestamps, datasets = load_data(conf['groupname'])
    # calculate_moving_mean()
    make_prediction(datasets)
    plot(timestamps, datasets)


if __name__ == '__main__':
    main()
