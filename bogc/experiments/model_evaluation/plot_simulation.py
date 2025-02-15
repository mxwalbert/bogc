import json
import numpy as np
from bogc import snapshots
from bogc import plotting
from bogc.data_loading import DataConfig


def get_simulation_data(simulation_results: dict, case: str, column: str, target: int) -> list[np.ndarray]:
    return [np.array(data)[:, :, target].mean(1) for data in simulation_results[case][column]]


def plot_simulation_curve(ax: plotting.plt.Axes, mean_data: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                          parameter_bounds: tuple[float, float], fill_between: bool = True,
                          color: str = None, linestyle: str = None, label: str = None) -> None:
    x_data = np.linspace(*parameter_bounds, len(mean_data))
    ax.plot(x_data, mean_data, color=color, linestyle=linestyle, label=label)
    if fill_between:
        ax.fill_between(x_data, lower, upper, alpha=0.2, facecolor=color)


def run(data_config: DataConfig):

    # load the simulation results corresponding to the data configuration
    input_path = 'evaluate_model'
    snapshot_name = snapshots.get_snapshot_name(data_config)
    snapshot_hash = snapshots.get_snapshot_hash(snapshot_name)
    with open(f'{input_path}/{snapshot_hash}_simulation.json', 'r') as f:
        simulation_results = json.load(f)

    # get data config params
    single = data_config.data_points == 'single'
    parameter_space = data_config.parameter_space.copy()
    if single:
        del parameter_space['x cu']
        del parameter_space['y']
    dims = len(parameter_space)
    colors = plotting.cmap(np.linspace(0, 1, dims + 2))[1:-1]

    for target in data_config.target_cols:
        target_idx = data_config.target_cols.index(target)
        plot_rows = dims // 2 + dims % 2
        fig, axs = plotting.plt.subplots(plot_rows, 2, figsize=(7, 3 * plot_rows), squeeze=False)
        for case, color in zip(simulation_results, colors):
            for ax, column in zip(axs.flatten(), parameter_space):
                simulation_data = get_simulation_data(simulation_results, case, column, target_idx)
                plot_simulation_curve(ax, simulation_data[1], simulation_data[2], simulation_data[3],
                                      data_config.parameter_space[column],
                                      color=color, label=case)
                ax.set_xlim(parameter_space[column])
                ax.set_xlabel(column)
                ax.set_ylabel(target)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(simulation_results), bbox_to_anchor=(0.5, 0))
        fig.tight_layout(rect=(0., 0., 1., 1.))
        plot_path = f'{input_path}/{snapshot_hash}_[{target}]_simulation.png'
        fig.savefig(plot_path, bbox_inches='tight')
        print(f'Saved figure to {plot_path}')
