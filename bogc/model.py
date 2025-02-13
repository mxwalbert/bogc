import torch
import gpytorch
import math
from typing import Optional, Type


class NuggetMaternKernel(gpytorch.kernels.Kernel):
    is_stationary = True
    has_lengthscale = True

    def __init__(self, nu: Optional[float] = 2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(NuggetMaternKernel, self).__init__(**kwargs)
        self.nu = nu
        self.register_parameter(
            name='raw_nugget', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_constraint("raw_nugget", gpytorch.constraints.Interval(0, 1e-4))

    @property
    def nugget(self):
        return self.raw_nugget_constraint.transform(self.raw_nugget)

    @nugget.setter
    def nugget(self, value):
        self._set_nugget(value)

    def _set_nugget(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.nugget)
        self.initialize(raw_nugget=self.raw_nugget_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        mean = x1.mean(dim=-2, keepdim=True)

        x1_ = (x1 - mean).div(self.lengthscale)
        x2_ = (x2 - mean).div(self.lengthscale)
        distance = self.covar_dist(x1_, x2_, diag=diag, **params)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        constant_component = 0
        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)

        if len(exp_component.shape) == 2:
            noise_cov = torch.eye(*exp_component.shape) * self.nugget.reshape([self.nugget.shape[0], -1])
        elif len(exp_component.shape) == 3:
            noise_cov = torch.eye(*exp_component.shape[1:]) * self.nugget
        else:
            raise RuntimeError('This kernel does not support higher dimensions than 3 for the covariance.')

        return constant_component * exp_component + noise_cov


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood, kernel: Type[gpytorch.kernels.Kernel] = gpytorch.kernels.RBFKernel):
        super(MultitaskGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=likelihood.num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(kernel(ard_num_dims=x_train.shape[-1])),
            num_tasks=likelihood.num_tasks, rank=1
        )
        self.steps = 0
        self.last_lr = None

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def add_step(self):
        self.steps += 1


class ApproximateMultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, x_train, y_train, likelihood, kernel: Type[gpytorch.kernels.Kernel] = gpytorch.kernels.RBFKernel):
        inducing_points = torch.rand(likelihood.num_tasks, 64, x_train.shape[1])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([likelihood.num_tasks])
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=likelihood.num_tasks,
        )
        super().__init__(variational_strategy)
        self.train_inputs = [x_train]
        self.train_targets = y_train
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([likelihood.num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel(ard_num_dims=x_train.shape[-1], batch_shape=torch.Size([likelihood.num_tasks])),
            batch_shape=torch.Size([likelihood.num_tasks])
        )
        self.steps = 0
        self.last_lr = None

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def add_step(self):
        self.steps += 1


ModelT = MultitaskGPModel | ApproximateMultitaskGPModel


def initialize_model(
        x_train: torch.Tensor, y_train: torch.Tensor,
        model_type: str, kernel_type: str) -> ModelT:

    if kernel_type == 'rbf':
        kernel_class = gpytorch.kernels.RBFKernel
    elif kernel_type == 'matern':
        kernel_class = gpytorch.kernels.MaternKernel
    elif kernel_type == 'nugget':
        kernel_class = NuggetMaternKernel
    else:
        raise RuntimeError

    if model_type == 'exact':
        model_class = MultitaskGPModel
    elif model_type == 'approx':
        model_class = ApproximateMultitaskGPModel
    else:
        raise ValueError(f'"{model_type}" is not a valid value for train_kind.')

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y_train.shape[1])
    return model_class(x_train, y_train, likelihood, kernel_class)
