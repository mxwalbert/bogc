import tqdm
import torch
import gpytorch
import sklearn.preprocessing as skl_pre
from bogc.data_loading import TrainingBatchGenerator
from bogc.model import ModelT, initialize_model


ResultT = tuple[torch.Tensor, torch.Tensor, float, list[float]]


def training(model: ModelT, train_iter: int, model_type: str = 'exact', show_progress: bool = True):
    # Get data
    x_train = model.train_inputs[0]
    y_train = model.train_targets

    # Get into training mode
    model.train()
    model.likelihood.train()

    # Setup optimizer and loss
    lr = 0.2 if model.last_lr is None else model.last_lr
    if model_type == 'exact':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    elif model_type == 'approx':
        optimizer = torch.optim.Adam([{'params': model.parameters()},], lr=lr)
        mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=y_train.size(0))
    else:
        raise ValueError(f'"{model_type}" is not a valid value for `kind`.')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=50, min_lr=1e-4)

    iterator = range(train_iter - model.steps)
    if show_progress:
        iterator = tqdm.tqdm(iterator, desc="Iteration", leave=False)

    for _ in iterator:
        # train
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        model.add_step()

        # schedule
        scheduler.step(loss)
        model.last_lr = scheduler.get_last_lr()[0]

        # update progressbar
        if show_progress:
            iterator.set_postfix({'loss': loss.item()})

    # print result
    if show_progress:
        final_loss = -mll(model(x_train), y_train)
        print(f'Loss after total {model.steps} iterations: {final_loss.item()}')


def evaluation(model: ModelT,
               x_test: torch.Tensor, y_test: torch.Tensor) -> ResultT:
    # Get into evaluation (predictive posterior) mode
    model.eval()
    model.likelihood.eval()

    # Make predictions
    y_pred_dist = model.likelihood(model(x_test))

    # quantify predictions
    nlpd = gpytorch.metrics.negative_log_predictive_density(y_pred_dist, y_test)
    mse = gpytorch.metrics.mean_squared_error(y_pred_dist, y_test)

    y_true = y_test.detach()
    y_pred = y_pred_dist.mean.detach()

    return y_true, y_pred, float(nlpd), mse.tolist()


def batch_training_and_evaluation(
        x: list[torch.Tensor], y: list[torch.Tensor], x_scaler: skl_pre.MinMaxScaler,
        train_iter: int = 200, test_size: int = 1, prev_models: dict[tuple[int, ...], ModelT] = None,
        model_type: str = 'exact', kernel_type: str = 'nugget',
        show_progress: bool = True) -> tuple[dict[tuple[int, ...], ResultT], dict[tuple[int, ...], ModelT]]:

    batches = TrainingBatchGenerator(x, y, test_size)

    batch_results = {}
    models = {} if prev_models is None else prev_models

    for test_indices, x_train, y_train, x_test, y_test in batches:

        # scale the data
        x_train = torch.from_numpy(x_scaler.transform(x_train)).float()
        x_test = torch.from_numpy(x_scaler.transform(x_test)).float()

        # initialize the model
        if test_indices not in models:
            model = initialize_model(x_train, y_train, model_type, kernel_type)
            models[test_indices] = model
        else:
            model = models[test_indices]

        # perform the training cycle
        training(model, train_iter=train_iter, model_type=model_type, show_progress=show_progress)

        # evaluate the model
        batch_results[test_indices] = evaluation(model, x_test, y_test)

        # store model for further training
        models[test_indices] = model

    return batch_results, models
