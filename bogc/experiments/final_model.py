import pathlib
import torch
from bogc import snapshots
from bogc.data_loading import DataConfig, DataLoader
from bogc.model import initialize_model
from bogc.training import training


def run(data_config: DataConfig) -> None:

    print('Started training of final model')

    # prepare output folder
    output_path = 'final_models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # load the data
    data_loader = DataLoader(data_config)

    # scale and concatenate the data
    x_train = torch.from_numpy(data_loader.x_scaler.transform(torch.cat(data_loader.x))).float()
    y_train = torch.cat(data_loader.y).float()

    # initialize the model
    model = initialize_model(x_train, y_train, data_config.model_type, data_config.kernel_type)

    # run the training cycle
    training(model, data_config.train_iter, data_config.model_type, show_progress=True)

    # store the trained model
    snapshot_name = snapshots.get_snapshot_name(data_config)
    snapshots.save_model(output_path, snapshot_name, model)
    snapshot_hash = snapshots.get_snapshot_hash(snapshot_name)

    print(f'Finished training and stored the model in {output_path}/{snapshot_hash}.pickle')
