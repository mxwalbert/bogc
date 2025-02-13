import os
import json
import pickle
import hashlib
import torch
from bogc.data_loading import DataConfig
from bogc.model import ModelT, initialize_model


def get_snapshot_name(data_config: DataConfig) -> str:
    return (f'batch{data_config.bo_batch}--{data_config.sample_grid}--{data_config.data_points}'
            f'--{data_config.kernel_type}--{list(data_config.parameter_space.keys())}--{data_config.target_cols}'
            f'--{[s.replace('BOGC010', '') for s in data_config.labels]}'
            f'{"--approx" if data_config.model_type == 'approx' else ""}')

def get_snapshot_hash(snapshot_name: str) -> str:
    return hashlib.md5(snapshot_name.encode()).hexdigest()

def read_snapshots(data_path: str) -> dict[str, str]:
    try:
        with open(f'{data_path}/snapshots.json', 'r') as jf:
            snapshots = json.load(jf)
    except FileNotFoundError:
        snapshots = {}
        with open(f'{data_path}/snapshots.json', 'w') as jf:
            json.dump(snapshots, jf)
    return snapshots


def save_model(data_path: str, snapshot_name: str, model: ModelT) -> None:
    state = {
        'train_iter': model.steps,
        'state_dict': model.state_dict(),
        'train_inputs': model.train_inputs,
        'train_targets': model.train_targets
    }
    snapshots = read_snapshots(data_path)
    snapshots[snapshot_name] = get_snapshot_hash(snapshot_name)
    with open(f'{data_path}/snapshots.json', 'w') as jf:
        json.dump(snapshots, jf)
    pickle_name = f'{data_path}/{snapshots[snapshot_name]}.pickle'
    with open(pickle_name, 'wb') as pf:
        pickle.dump(state, pf)


def load_model(data_path: str, snapshot_name: str) -> dict:
    snapshots = read_snapshots(data_path)
    if snapshot_name not in snapshots:
        raise FileNotFoundError('Cannot find snapshot with provided name!')
    pickle_name = f'{data_path}/{snapshots[snapshot_name]}.pickle'
    with open(pickle_name, 'rb') as pf:
        state = pickle.load(pf)
    return state


def restore_model(state: dict,
                  model_type: str = 'exact', kernel_type: str = 'nugget') -> ModelT:
    model = initialize_model(torch.cat(state['train_inputs']).float(), state['train_targets'], model_type, kernel_type)
    model.load_state_dict(state['state_dict'])
    model.steps = state['train_iter']
    return model


def save_prev_models(data_path: str, snapshot_name: str, prev_models: dict[tuple[int, ...], ModelT]) -> None:
    for index_tuple, model in prev_models.items():
        save_model(data_path, f'{snapshot_name}--{index_tuple[0]}--{model.steps}', model)


def load_prev_models(data_path: str, snapshot_name: str, train_iter: int,
                     model_type: str = 'exact', kernel_type: str = 'nugget') -> dict[tuple[int, ...], ModelT]:
    prev_models = {}
    # find snapshot with most training iterations that are less than provided train_iter
    snapshots = read_snapshots(data_path)
    train_iters = set()
    for snapshot in snapshots:
        if snapshot_name in snapshot:
            existing_train_iter = int(snapshot.split('--')[-1])
            if existing_train_iter <= train_iter:
                train_iters.add(existing_train_iter)
    index = 0
    while len(train_iters) > 0:
        try:
            state = load_model(data_path, f'{snapshot_name}--{index}--{max(train_iters)}')
            prev_models[(index,)] = restore_model(state, model_type, kernel_type)
            index += 1
        except FileNotFoundError:
            break
    return prev_models


def delete_model(data_path: str, snapshot_name: str) -> None:
    snapshots = read_snapshots(data_path)
    os.remove(f'{data_path}/{snapshots[snapshot_name]}.pickle')
    del snapshots[snapshot_name]
    with open(f'{data_path}/snapshots.json', 'w') as jf:
        json.dump(snapshots, jf)


def delete_prev_models(data_path: str, snapshot_name: str) -> None:
    snapshots = read_snapshots(data_path)
    for snapshot in snapshots:
        if snapshot_name in snapshot:
            delete_model(data_path, snapshot)
