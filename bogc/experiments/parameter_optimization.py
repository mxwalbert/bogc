import pathlib
import datetime
import nubo
import gpytorch
import torch
import pandas as pd
from typing import Callable, Optional
from bogc import snapshots
from bogc.data_loading import DataConfig, DataLoader


class MultitaskMCUpperConfidenceBound(nubo.acquisition.MCUpperConfidenceBound):

    def __init__(self,
                 fom: Callable,
                 gp: gpytorch.models.GP,
                 beta: Optional[float] = 4.0,
                 x_pending: Optional[torch.Tensor] = None,
                 samples: Optional[int] = 512,
                 fix_base_samples: Optional[bool] = False) -> None:

        super().__init__(gp, beta, x_pending, samples, fix_base_samples)
        self.fom = fom

    def eval(self, x: torch.Tensor) -> torch.Tensor:

        # reshape tensor to (batch_size x dims)
        x = torch.reshape(x, (-1, self.dims)).float()

        # add pending points
        if isinstance(self.x_pending, torch.Tensor):
            x = torch.cat([x, self.x_pending], dim=0)

        # set Gaussian Process to eval mode
        self.gp.eval()

        # get predictive distribution
        pred = self.gp(x)
        mean = pred.mean
        covariance = pred.lazy_covariance_matrix

        # get samples from Multivariate Normal
        mvn = gpytorch.distributions.MultitaskMultivariateNormal(mean, covariance)
        if self.base_samples is None and self.fix_base_samples == True:
            self.base_samples = mvn.get_base_samples(torch.Size([self.samples]))
        samples = mvn.rsample(torch.Size([self.samples]), base_samples=None).double()

        # calculate figure of merit for prediction mean and distribution samples
        fom_mean = self.fom(mean)
        fom_samples = self.fom(samples)

        # compute Upper Confidence Bound
        ucb = fom_mean + self.beta_coeff * torch.abs(fom_samples - fom_mean)
        ucb = ucb.max(dim=1).values
        ucb = ucb.mean(dim=0, keepdim=True)  # average samples

        return -ucb


def fom(x: torch.Tensor) -> torch.Tensor:
    return x[..., 0] + x[..., 0] ** 2 / x[..., 1]


def run(data_config: DataConfig, batch_size: int = 4) -> pd.DataFrame:

    if data_config.model_type != 'exact':
        SystemExit('Parameter optimization only works for exact model')

    print('Started parameter optimization')

    # prepare output folder
    output_path = 'parameter_optimization'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # restore the trained model
    input_path = 'final_models'
    snapshot_name = snapshots.get_snapshot_name(data_config)
    state = snapshots.load_model(input_path, snapshot_name)
    model = snapshots.restore_model(state, data_config.model_type, data_config.kernel_type)

    # load the data
    data_loader = DataLoader(data_config)

    # specify acquisition function
    acq = MultitaskMCUpperConfidenceBound(fom=fom, gp=model, beta=1.96 ** 2, samples=512)

    # optimise acquisition function
    bounds = torch.from_numpy(data_loader.x_scaler.transform(torch.Tensor(list(data_config.parameter_space.values())).T))
    x_new, ucb = nubo.optimisation.multi_sequential(func=acq, method="Adam", batch_size=batch_size, bounds=bounds,
                                                    lr=0.1, steps=200, num_starts=5)

    # rescale new parameters
    x_new_rescaled = data_loader.x_scaler.inverse_transform(x_new)

    # generate DataFrame
    new_df = pd.DataFrame(x_new_rescaled, columns=list(data_config.parameter_space))
    new_df['ucb'] = ucb

    # generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # save the data
    snapshot_hash = snapshots.get_snapshot_hash(snapshot_name)
    csv_path = f'{output_path}/{timestamp}_{snapshot_hash}.csv'
    new_df.to_csv(csv_path)

    print(f'Finished parameter optimization and stored results in {csv_path}')

    return new_df
