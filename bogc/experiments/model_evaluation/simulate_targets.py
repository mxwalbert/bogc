import pathlib
import json
import pandas as pd
import torch
import numpy as np
import sklearn.preprocessing as skl_pre
from bogc import snapshots
from bogc.data_loading import DataConfig, DataLoader
from bogc.model import ModelT


def scale_parameters(parameters: dict, x_scaler: skl_pre.MinMaxScaler, single: bool) -> dict:
    if single:
        scaled = x_scaler.transform([list(parameters.values()) + [4.5, 4.5]])[0][:-2]
    else:
        scaled = x_scaler.transform([list(parameters.values())])[0]
    return {k: v for k, v in zip(parameters, scaled)}


def simulate_targets(model: ModelT, column: str, scaled_parameters: dict, single: bool,
                     npoints: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # get into evaluation mode
    model.eval()
    model.likelihood.eval()

    params = []
    y_means = []
    y_lower = []
    y_upper = []

    # iterate over whole parameter range for selected target
    for xi in np.linspace(-1., 1., npoints):

        # generate input data
        x = torch.Tensor([[xi if k == column else v for k, v in scaled_parameters.items()]])
        if single:
            x = torch.cat((
                x.repeat(64, 1),
                torch.cartesian_prod(torch.linspace(-1., 1., 8), torch.linspace(-1., 1., 8))
            ), dim=1)
        params.append(x)

        # make predictions
        pred = model.likelihood(model(x))
        y_means.append(pred.mean.detach())
        lower, upper = pred.confidence_region()
        y_lower.append(lower.detach())
        y_upper.append(upper.detach())

    return torch.stack(params), torch.stack(y_means), torch.stack(y_lower), torch.stack(y_upper)


def run(data_config: DataConfig):

    # prepare output folder
    output_path = 'evaluate_model'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # restore the trained model
    input_path = 'final_models'
    snapshot_name = snapshots.get_snapshot_name(data_config)
    snapshot_hash = snapshots.get_snapshot_hash(snapshot_name)
    state = snapshots.load_model(input_path, snapshot_name)
    model = snapshots.restore_model(state, data_config.model_type, data_config.kernel_type)

    # get data config params
    data_loader = DataLoader(data_config)
    single = data_config.data_points == 'single'

    # get the parameters of the selected optima
    target_optima = pd.read_csv(f'{output_path}/{snapshot_hash}_unique.csv', index_col=0).drop(columns=['counts', 'value'])
    with open(f'{output_path}/{snapshot_hash}_selected.json', 'r') as f:
        selected_optima = json.load(f)
    parameter_cases = {
        so: scale_parameters(target_optima.loc[so.split(' & ')].mean().to_dict(), data_loader.x_scaler, single)
        for so in selected_optima
    }

    # simulate selected parameter cases
    print(f'Starting simulation of {selected_optima}')
    simulation_results = {}
    for case, parameters in parameter_cases.items():
        print(f'Simulating {case}')
        simulation_results[case] = {
            column: [t.tolist() for t in simulate_targets(model, column, parameters, single)]
            for column in target_optima.columns
        }

    print(f'Finished all target simulations')

    with open(f'{output_path}/{snapshot_hash}_simulation.json', 'w') as f:
        json.dump(simulation_results, f)
