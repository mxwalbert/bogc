import pathlib
import glob
import json
import re
import numpy as np
import pandas as pd
from bogc import snapshots
from bogc.data_loading import DataConfig, DataLoader


def find_unique_params(all_best: list[tuple], round_digit: int) -> tuple[list, list, list]:
    unique_params = []
    counts = []
    optima = []
    p, o, _ = zip(*[t for t in all_best])
    p_arr = np.array(p)
    o_arr = np.array(o)
    o_rounded = o_arr.round(round_digit)
    for u in np.unique(o_rounded):
        idxs = o_rounded == u
        unique_params.append(p_arr[idxs].mean(0))
        counts.append(idxs.sum())
        optima.append(o_arr[idxs].mean(0))
    return unique_params, counts, optima


def run(data_config: DataConfig) -> None:

    # prepare output folder
    output_path = 'evaluate_model'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # restore the name of the model
    snapshot_name = snapshots.get_snapshot_name(data_config)
    snapshot_hash = snapshots.get_snapshot_hash(snapshot_name)

    # get data config params
    parameter_space = data_config.parameter_space.copy()
    if data_config.data_points == 'single':
        del parameter_space['x cu']
        del parameter_space['y']
    data_loader = DataLoader(data_config)

    target_optima = {}

    for optimization in glob.glob(f'{output_path}/{snapshot_hash}_*.json'):
        t, e = re.search(r'.+\[(.+)]_(.+)\.json', optimization).groups()
        with open(optimization, 'r') as f:
            all_best = json.load(f)
        unique_params, counts, values = find_unique_params(all_best, 1)
        if data_config.data_points == 'single':
            unique_params = [np.hstack([p, np.array([0., 0.])]) for p in unique_params]
        unique_params = [data_loader.x_scaler.inverse_transform(p.reshape(1, -1))[0].tolist() for p in unique_params]
        for i in range(len(unique_params)):
            name = f'{t} {e} {i}'
            target_optima[name] = {
                k: v for k, v in zip(list(parameter_space), unique_params[i])
            }
            target_optima[name]['counts'] = counts[i]
            target_optima[name]['value'] = values[i]

    target_optima = pd.DataFrame(target_optima).T

    target_optima.to_csv(f'{output_path}/{snapshot_hash}_unique.csv')

    print('Found the following unique optima in the optimization results:')
    print(target_optima)
    print('')

    selected_optima = []

    print('Type the name of the optimum to include and press Enter.')
    print('Combine multiple optima by typing " & " between their names (including spaces).')
    print('Leave empty to finish.')
    while True:
        input_str = input('Include optimum: ')
        if input_str == '':
            break
        if not all([name in target_optima.index for name in input_str.split(' & ')]):
            print('Specified optimum name(s) not found.')
        else:
            selected_optima.append(input_str)

    print('')
    print(f'Saved the following optima: {selected_optima}')

    with open(f'{output_path}/{snapshot_hash}_selected.json', 'w') as f:
        json.dump(selected_optima, f)
