import pathlib
import torch
import numpy as np
import sklearn.metrics as skl_m
import sklearn.model_selection as skl_ms
from bogc import snapshots, plotting
from bogc.data_loading import DataConfig, DataLoader
from bogc.model import initialize_model
from bogc.training import ResultT, training, evaluation


def run(data_config: DataConfig,
        test_size: float = 0.1,
        plot_results: bool = True, plot_color: int = 11) -> tuple[ResultT, ResultT]:

    print('Started training-test split')

    # prepare output folder
    snapshot_name = snapshots.get_snapshot_name(data_config)
    snapshot_hash = snapshots.get_snapshot_hash(snapshot_name)
    output_path = f'training_test_split'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # load the data
    data_loader = DataLoader(data_config)

    # scale and concatenate the data
    x_cat = torch.from_numpy(data_loader.x_scaler.transform(torch.cat(data_loader.x))).float()
    y_cat = torch.cat(data_loader.y).float()

    # split the data
    x_train, x_test, y_train, y_test = skl_ms.train_test_split(
        x_cat, y_cat,
        test_size=test_size, shuffle=True)

    # initialize the model
    model = initialize_model(x_train, y_train, data_config.model_type, data_config.kernel_type)

    # run the training cycle
    training(model, data_config.train_iter, data_config.model_type, show_progress=True)

    # evaluate the model
    train_results = evaluation(model, x_train, y_train)
    test_results = evaluation(model, x_test, y_test)

    # store the model
    snapshots.save_model(output_path, snapshot_name, model)
    print(f'Finished training and stored the model in {output_path}/{snapshot_hash}.pickle')

    if plot_results:

        y_train_true = data_loader.y_scaler.transform(train_results[0])
        y_train_pred = data_loader.y_scaler.transform(train_results[1])
        y_test_true = data_loader.y_scaler.transform(test_results[0])
        y_test_pred = data_loader.y_scaler.transform(test_results[1])

        for y_true, y_pred, scope in zip((y_train_true, y_test_true), (y_train_pred, y_test_pred), ('training', 'test')):

            fig, axs = plotting.plt.subplots(1, len(data_config.target_cols), figsize=(7 / 4 * len(data_config.target_cols), 4), squeeze=False)
            axs = axs.flatten()

            rmse = skl_m.root_mean_squared_error(np.vstack(y_true), np.vstack(y_pred), multioutput='raw_values')

            for i, ax in enumerate(axs):
                ax: plotting.plt.Axes = ax
                ax.scatter(y_true[:, i], y_pred[:, i], color=plotting.pretty_colors[plot_color], marker='.')
                ax.text(0.0, 1.0, f'RMSE={rmse[i]:.3f}', va='top')
                plotting.true_pred_ax(ax, plotting.pretty_target_label(data_config.target_cols[i], data_config.sample_grid))

            fig.tight_layout()
            plot_path = f'{output_path}/{snapshot_hash}_{scope}.png'
            fig.savefig(plot_path, bbox_inches='tight')
            print(f'Saved figure to {plot_path}')

    return train_results, test_results
