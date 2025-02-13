import time
import nubo
import tqdm
import pathlib
import torch
import json
import itertools
from unittest.mock import Mock
from bogc import snapshots
from bogc.model import ModelT
from bogc.data_loading import DataConfig


def predict_target(model: ModelT, target: int, x: torch.Tensor,) -> torch.Tensor:
    model.eval()
    model.likelihood.eval()
    prediction = model.likelihood(model(x.float()))
    return prediction.mean[:, target]


def calc_loss(model: ModelT, target: int, extremum: str, parameters: torch.Tensor,
              bounds: torch.Tensor, single: bool) -> tuple[torch.Tensor, torch.Tensor]:
    factor = -1 if extremum == 'max' else 1
    high_diff = torch.clamp(parameters - bounds.T.max(1).values, min=0)
    low_diff = torch.clamp(bounds.T.min(1).values - parameters, min=0)
    x = torch.reshape(parameters - high_diff + low_diff, (1, -1))
    if single:
        x = torch.cat((
            x.repeat(64, 1),
            torch.cartesian_prod(torch.linspace(-1., 1., 8), torch.linspace(-1., 1., 8))
        ), dim=1)
    prediction = predict_target(model, target, x).mean()
    return factor * (prediction + factor * (prediction.abs() * high_diff).sum() + factor * (prediction.abs() * low_diff).sum()), prediction


def optimize(model: ModelT, target: int, extremum: str, dims: int, single: bool,
             num_per_dim: int = 5, min_steps: int = 100, max_steps: int = 1000, tolerance: float = 1e-4,
             show_progress: bool = True) -> list[tuple]:

    bounds = torch.tensor([[-1.0, 1.0] for _ in range(dims)]).T
    parameters_pool = nubo.utils.gen_inputs(num_points=dims*num_per_dim, num_dims=dims, bounds=bounds)

    all_best = []

    # Perform multiple starts
    for start in range(len(parameters_pool)):
        # Initial parameters for each start
        parameters = parameters_pool[start].detach()
        parameters.requires_grad = True

        # Set up the Adam optimizer
        optimizer = torch.optim.Adam([parameters], lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, min_lr=1e-4)

        # Best solution for the current start
        best = [None], None, float('inf')

        # Store the previous loss for convergence check
        prev_loss = best[2]

        # Initialize tqdm progress bar
        if show_progress:
            progress_bar = tqdm.tqdm(total=max_steps, leave=True)
            progress_print = print
        else:
            progress_bar = Mock()
            progress_print = Mock().print

        # Perform optimization for each start
        for step in range(max_steps):
            optimizer.zero_grad()  # Zero out the gradients
            loss, prediction = calc_loss(model, target, extremum, parameters, bounds, single)  # Calculate loss
            loss.backward()  # Compute gradients
            optimizer.step()  # Update the parameters using Adam
            scheduler.step(loss)  # Update the optimizer

            # Update best for this start if we find a better solution
            if loss < best[2]:
                x_new = parameters.tolist()
                y_new = prediction.item()
                best = x_new, y_new, loss.item()

                # Check for convergence (if the change in loss is smaller than tolerance)
                if step > min_steps and abs(loss.item() - prev_loss) < tolerance:
                    progress_print(f"Converged at step {step} for start {start}")
                    progress_bar.update(max_steps - step)
                    time.sleep(0.05)
                    break

            # Update previous loss for the next iteration
            prev_loss = loss.item()

            # Update progress bar
            progress_bar.update(1)

        progress_bar.close()

        progress_print(
            f"Best at start {start}: "
            f"Inputs: {[round(b, 4) for b in best[0]]}, "
            f"Output: {round(best[1], 4)}, "
            f"Loss: {round(best[2], 4)}"
        )

        # Update all solutions
        all_best.append(best)

    return all_best


def run(data_config: DataConfig) -> None:

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
    single = data_config.data_points == 'single'
    dims = len(data_config.parameter_space) - 2 if single else len(data_config.parameter_space)

    for target, extremum in itertools.product(data_config.target_cols, ['min', 'max']):

        optimization_path = f'{output_path}/{snapshot_hash}_[{target}]_{extremum}.json'

        print(f'Starting optimization to find {extremum} of {target}')
        print(f'****************************************************')

        all_best = optimize(model, data_config.target_cols.index(target), extremum, dims, single)

        # Find overall best solutions across all starts
        overall_best = 0
        for i in range(len(all_best)):
            if all_best[i][2] < all_best[overall_best][2]:
                overall_best = i

        # Final best solution after all starts
        print(f'-----------------------------------------------------')
        print(
            f"Best solution found after {len(all_best)} starts: "
            f"Inputs: {[round(b, 4) for b in all_best[overall_best][0]]}, "
            f"Output: {round(all_best[overall_best][1], 4)}, "
            f"Loss: {round(all_best[overall_best][2], 4)}"
        )

        # store optimization results
        with open(optimization_path, 'w') as f:
            json.dump(all_best, f)

        print(f'Saved results to {optimization_path}\n')