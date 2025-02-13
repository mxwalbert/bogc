import torch
import itertools
import numpy as np
import pandas as pd
import sklearn.utils as skl_u
import sklearn.preprocessing as skl_pre


class DataConfig:
    def __init__(self, data_points: str, sample_grid: str, target_cols: list[str],
                 batch_labels: list[list[str]], bo_batch: int, excluded_labels: list[str] = None,
                 train_iter: int = 200, model_type: str = 'exact', kernel_type: str = 'nugget'):
        self.filename = f'data/training_data_{data_points}_{sample_grid}.csv'
        self.data_points = data_points
        self.sample_grid = sample_grid
        self.target_cols = target_cols
        self.num_tasks = len(target_cols)

        self.batch_labels = batch_labels
        self.bo_batch = bo_batch
        self.excluded_labels = excluded_labels
        self.labels = self.select_labels(excluded_labels)

        self.train_iter = train_iter
        self.kernel_type = kernel_type
        self.model_type = model_type

        self.parameter_space = {
            'cycles': (30, 150),
            'height': (-750, 9250),
            'speed': (20000, 40000),
            'flow rate': (0.5, 1.5),
            'temperature': (220, 340),
            'concentration': (12.5, 50)
        }
        if data_points == 'columns':
            self.parameter_space['x cu'] = (1, 8)
        elif data_points == 'single':
            self.parameter_space['x cu'] = (1, 8)
            self.parameter_space['y'] = (1, 8)

        self.input_cols = list(self.parameter_space.keys())

    def select_labels(self, excluded_labels: list[str] = None) -> list[str]:
        excluded_labels = [] if excluded_labels is None else excluded_labels
        selected_labels = []
        for labels in self.batch_labels[:self.bo_batch + 1]:
            selected_labels.extend(labels)
        return [label for label in selected_labels if label not in excluded_labels]


class DataLoader:
    def __init__(self, data_config: DataConfig, excluded_labels: list[str] = None):
        self.data_config = data_config
        excluded_labels = data_config.excluded_labels if excluded_labels is None else excluded_labels
        self.selected_labels = data_config.select_labels(excluded_labels)

        df = pd.read_csv(self.data_config.filename, index_col=0)

        df['log jsc'] = np.log(df['jsc'].abs())
        self.labels, groups = zip(*[
            (label, group) for label, group in df.groupby('name') if label in self.selected_labels
        ])

        self.x = [torch.from_numpy(group[self.data_config.input_cols].to_numpy()) for group in groups]
        self.y = [torch.from_numpy(group[self.data_config.target_cols].to_numpy()) for group in groups]

        bounds = torch.Tensor(list(data_config.parameter_space.values())).T
        self.x_scaler = skl_pre.StandardScaler().fit(bounds)
        self.y_scaler = skl_pre.MinMaxScaler().fit(torch.cat(self.y))


class TrainingBatchGenerator:
    """Takes data in form of two ``torch.Tensor`` objects and constructs a generator for all possible batches of
    training and test data which can be composed, given the specified `test_size`."""

    def __init__(self, x: list[torch.Tensor], y: list[torch.Tensor], test_size: int = 1):
        self.index = 0
        self.x = x
        self.y = y
        self.index_combinations = list(itertools.combinations(range(len(x)), test_size))

    def __len__(self) -> int:
        return len(self.index_combinations)

    def __iter__(self) -> 'TrainingBatchGenerator':
        return self

    def __next__(self) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            test_indices = self.index_combinations[self.index]
            self.index += 1
        except IndexError:
            self.index = 0
            raise StopIteration

        train_indices = [i for i in range(len(self.x)) if i not in test_indices]

        x_train = torch.cat([self.x[idx] for idx in train_indices])
        y_train = torch.cat([self.y[idx] for idx in train_indices])
        x_train, y_train = skl_u.shuffle(x_train, y_train)

        x_test = torch.cat([self.x[idx] for idx in test_indices]) if test_indices else torch.Tensor([])
        y_test = torch.cat([self.y[idx] for idx in test_indices]) if test_indices else torch.Tensor([])
        x_test, y_test = skl_u.shuffle(x_test, y_test)

        return test_indices, x_train.float(), y_train.float(), x_test.float(), y_test.float()
