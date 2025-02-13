import pathlib
import torch
import statistics
from bogc import snapshots
from bogc.data_loading import DataConfig, DataLoader
from bogc.training import batch_training_and_evaluation
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties


def evaluate_model(params: dict[str, None | str | bool | float | int], data_config: DataConfig) -> dict:
    excluded_labels = [label for label in data_config.labels if params.get(label)]
    data_loader = DataLoader(data_config, excluded_labels)
    ranges = {}
    y_mean = torch.stack([y_i.mean(0) if len(y_i) > 1 else y_i[0] for y_i in data_loader.y])
    y_std = torch.stack([y_i.std(0) if len(y_i) > 1 else y_i[0] * 0. for y_i in data_loader.y])
    for i, target in enumerate(data_config.target_cols):
        ranges[f'{target} range'.replace(' ', '_')] = (float(y_mean.max(0).values[i] - y_mean.min(0).values[i]), float(y_std.mean(0)[i]))

    batch_results, _ = batch_training_and_evaluation(
        x=data_loader.x, y=data_loader.y, x_scaler=data_loader.x_scaler,
        train_iter=params.get('train_iter'), test_size=1, prev_models=None,
        model_type=data_config.model_type, kernel_type=data_config.kernel_type,
        show_progress=False
    )

    errors = [statistics.mean(br[3]) for br in batch_results.values()]

    results = {
        'max_error': (max(errors), 0.0),
        'mean_error': (statistics.mean(errors), 0.0),
        'num_selected': (len(errors), 0.0)
    }
    results.update(ranges)

    return results


def run(data_config: DataConfig, min_num_selected: int, trials: int, target_ranges: list[float] = None) -> None:
    snapshot_name = snapshots.get_snapshot_name(data_config)
    snapshot_hash = snapshots.get_snapshot_hash(snapshot_name)
    target_ranges = [] if target_ranges is None else target_ranges
    folder_path = 'model_optimization'
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
    experiment_name = f'{snapshot_hash}_mns{min_num_selected}'
    file_path = f'{folder_path}/{experiment_name}.json'
    try:
        ax_client = AxClient.load_from_json_file(file_path)
    except FileNotFoundError:
        ax_client = AxClient()

        selection_parameters = [
            {
                'name': label,
                'type': 'choice',
                'values': [True, False],
                'value_type': 'bool',
                'sort_values': False,
                'is_ordered': True
            }
            for label in data_config.labels]

        range_constraints = [f'{t.replace(' ', '_')}_range >= {r}'
                             for t, r in zip(data_config.target_cols, target_ranges) if r is not None]

        ax_client.create_experiment(
            name=experiment_name,
            parameters=selection_parameters + [
                {
                    'name': 'train_iter',
                    'type': 'range',
                    'bounds': [50, 1000],
                    'value_type': 'int'
                }
            ],
            objectives={
                'mean_error': ObjectiveProperties(minimize=True)
            },
            outcome_constraints=[
                f'num_selected >= {min_num_selected}'
            ] + range_constraints,
            tracking_metric_names=[
                'max_error'
            ] + [f'{target} range'.replace(' ', '_') for target in data_config.target_cols]
        )

    for i in range(trials):
        params, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate_model(params, data_config))
        ax_client.save_to_json_file(file_path)
