import pathlib
import numpy as np
import sklearn.metrics as skl_m
from bogc import snapshots, plotting
from bogc.data_loading import DataConfig, DataLoader
from bogc.training import ResultT, batch_training_and_evaluation


def run(data_config: DataConfig,
        plot_results: bool = True, plot_color: int = 11) -> dict[tuple[int, ...], ResultT]:

    print('Started leave-one-out cross-validation')

    # prepare output folder
    snapshot_name = snapshots.get_snapshot_name(data_config)
    snapshot_hash = snapshots.get_snapshot_hash(snapshot_name)
    output_path = f'cross_validation/{snapshot_hash}'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # load previously trained models
    prev_models = snapshots.load_prev_models(
        data_path=output_path, snapshot_name=snapshot_name, train_iter=data_config.train_iter,
        model_type=data_config.model_type, kernel_type=data_config.kernel_type)

    # load the data
    data_loader = DataLoader(data_config)

    # run the training and save the models
    batch_results, prev_models = batch_training_and_evaluation(
        x=data_loader.x, y=data_loader.y, x_scaler=data_loader.x_scaler,
        train_iter=data_config.train_iter, test_size=1, prev_models=prev_models,
        model_type=data_config.model_type, kernel_type=data_config.kernel_type,
        show_progress=True)
    snapshots.save_prev_models(output_path, snapshot_name, prev_models)

    if plot_results:

        y_true = [data_loader.y_scaler.transform(result[0]) for result in batch_results.values()]
        y_pred = [data_loader.y_scaler.transform(result[1]) for result in batch_results.values()]

        fig, axs = plotting.plt.subplots(1, len(data_config.target_cols), figsize=(7 / 4 * len(data_config.target_cols), 4), squeeze=False)
        axs = axs.flatten()

        rmse = skl_m.root_mean_squared_error(np.concat(y_true), np.concat(y_pred), multioutput='raw_values')

        for i, ax in enumerate(axs):
            ax: plotting.plt.Axes = ax
            for y_true_i, y_pred_i in zip(y_true, y_pred):
                ax.errorbar(y_true_i[:, i].mean(), y_pred_i[:, i].mean(),
                            xerr=y_true_i[:, i].std(), yerr=y_pred_i[:, i].std(),
                            color=plotting.pretty_colors[plot_color], marker='.')
            ax.text(0.0, 1.0, f'RMSE={rmse[i]:.3f}', va='top')
            plotting.true_pred_ax(ax, plotting.pretty_target_label(data_config.target_cols[i], data_config.sample_grid))

        fig.tight_layout()
        plot_path = f'{output_path}/loocv.png'
        fig.savefig(plot_path, bbox_inches='tight')
        print(f'Saved figure to {plot_path}')

    return batch_results
