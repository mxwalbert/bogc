# Bayesian optimization of spray parameters for the deposition of Ga2O3-Cu2O heterojunctions


## Introduction

This repository contains the Python source code used in our publication, put together into a single commandline 
program. It provides tools for conducting various experiments related to Gaussian Process training, Bayesian 
Optimization, and Model Evaluation.


## Getting Started

### Dependencies

This package is tested for Python 3.12 and requires the following dependencies:

- PyYAML~=6.0 
- torch~=2.6 
- numpy~=2.2 
- pandas~=2.2 
- scikit-learn~=1.6 
- gpytorch~=1.13 
- matplotlib~=3.10 
- tqdm~=4.67 
- ax-platform~=0.4 
- nubopy~=1.2

### Installation

You can install the package including its dependencies from the GitHub repository. Follow these steps:
1. **Clone** the repository to your local machine:
    ```console
    git clone https://github.com/mxwalbert/bogc.git
    ```
2. **Navigate** into the cloned directory:
    ```console
    cd bogc
    ```
3. **Install** the package and its dependencies:
    ```console
    pip install .
    ```

### Configuration

The experiments are configured via a YAML file (`config.yml`) which should be placed in the path where the commandline 
program is invoked. For example:

<a name="configyml"></a>
```yaml
train_iter: 200
data_points: 'grouped'
sample_grid: '6x6'
target_cols: ['voc', 'voc max']
batch_labels:
  - ['BOGC01000', 'BOGC01001', 'BOGC01002', 'BOGC01003', 'BOGC01004', 'BOGC01005', 
     'BOGC01006', 'BOGC01007', 'BOGC01008', 'BOGC01009', 'BOGC01010', 'BOGC01011']
  - ['BOGC01012', 'BOGC01013', 'BOGC01014', 'BOGC01015']
  - ['BOGC01016', 'BOGC01017', 'BOGC01018', 'BOGC01019']
  - ['BOGC01020', 'BOGC01021', 'BOGC01022', 'BOGC01023']
  - ['BOGC01024', 'BOGC01025', 'BOGC01026', 'BOGC01027']
bo_batch: 0
excluded_labels: []
model_type: 'exact'
kernel_type: 'nugget'
```

See the [Configuration Reference](#configref) for a complete description of the available options.

The training data is published as supplementary information of the related paper (see [Citation](#citation)). In order 
for the data loader to find the CSV files, copy them into a `./data/` folder next to the `config.yml`.

After configuration, your source folder should look like this:

```
src
 ├── data
 |    ├── training_data_grouped_6x6.csv
 |    ├── training_data_grouped_8x8.csv
 |    ├── training_data_single_6x6.csv
 |    └── training_data_single_8x8.csv
 └── config.yml
```

### Basic Usage

The installed package is executed through a command-line interface (CLI) with the following options:

```sh
python -m bogc [-h] [-cnf CONFIG_FILE] [-exp EXPERIMENT] [other arguments]
```

Use the `-h` flag to get a detailed description of the commandline arguments.


## Experiments

### Training of Final Model

To only train a [`GPyTorch`](https://github.com/cornellius-gp/gpytorch) Gaussian process with the specified options 
in the `config.yml`, simply run the package without any commandline arguments:

```sh
python -m bogc
```

This will store the state of the model as `<hash>.pickle` in the folder `./final_models/`. The `hash` is a hashed 
representation of the (very long) model name which is written in clear text to the `snapshots.json`.

### Random Training-Test Split

A random split of the loaded data into a training and a test set can be done with this command:

```sh
python -m bogc -exp tt --test_size 0.1
```

The `test_size` argument specifies the fractional size of the test set (10% in this case). As expected, the model will 
be trained on the training set and validated with the test set. The `<hash>.pickle` file inside the 
`./training_test_split/` folder contains the state of the model, and the results are plotted in `<hash>_training.png` 
and `<hash>_test.png`.

### Leave-one-out Cross-Validation

Instead of random splitting, the data of a single sample is used for validating the model. This can be a single data 
point for the `grouped` data or up to 64 data points for the `single` data on the `8x8` grid. The leave-one-out 
cross-validation is started with the `cv` option:

```sh
python -m bogc -exp cv
```

A new folder with the hashed model name will be created in `./cross_validation/` and contain the states of the multiple 
models (each using another sample for the test data) as well as the resulting plot in `loocv.png`.

### Parameter Optimization

> [!NOTE]  
> This experiment requires a final model. Run `python -m bogc` before continuing.

The Bayesian optimization of input parameters with the [`NUBO`](https://github.com/mikediessner/nubo) package is 
carried out like so:

```sh
python -m bogc -exp po --batch_size 4
```

The number of sequentially generated evaluation points can be set with the `batch_size` argument. After completion, 
the parameter batch will be written to `./parameter_optimization/<datetime>_<hash>.csv`, with the current date and time 
followed by the hashed model name.

> [!WARNING]  
> The parameter optimization only works with `model_type: 'exact'` (see [Configuration Reference](#configref)).

### Model Optimization

The Bayesian optimization of sample data selection with the [`Ax`](https://github.com/facebook/Ax) platform is based on 
minimizing the leave-one-out cross-validation error. It can be started with a command like this:

```sh
python -m bogc -exp mo --min_num_selected 6 --trials 100 --target_ranges 0.5 0.7
```

For parametrizing the experiment, three arguments can be used:

- `min_num_selected` specifies the minimum number of selected samples as a constraint of the model evaluation. In the 
shown example, models which use less than 6 samples for the training data are considered invalid.
- `trials` specifies the total number of tested parameter combinations, i.e., 100 in the shown example.
- `target_ranges` specifies the lower range limits for the ordered list of targets in the `target_cols` configuration 
option (see [Configuration Reference](#configref)). In the shown example, and using the `config.yml` from above, if the 
ranges of `voc` and `voc_max` must be more than 0.5 V and 0.7 V, respectively.

The state of the `AxClient` is stored in `./model_optimization/<hash>_mns<X>.json` where `X` is the specified minimum 
number of selected samples. This JSON file will be used to reload the optimization when the experiment is restarted. 
For evaluating the results, refer to the documentation of the [`Ax`](https://github.com/facebook/Ax) platform.


### Model Evaluation

> [!NOTE]  
> This experiment requires a final model. Run `python -m bogc` before continuing.

The evaluation of a final model is separated into four individual modules which must be run one after the other:

#### Optimize Targets

First, the target surfaces are probed for optimum values:

```sh
python -m bogc -exp ot
```

This will try to find the (local) minima and maxima of the targets in a couple of semi-random starts (number depends on 
the dimensions of the parameter space). For each target, the discovered extreme points from all starts will be stored 
in `./evaluate_model/` as `<hash>_[<target>]_min.json` and `<hash>_[<target>]_max.json`, respectively.

#### Select Optima

Then, the aforementioned results are grouped into unique optima:

```sh
python -m bogc -exp so
```

A table of grouped extreme points will be printed, showing the averaged parameter values, number of occurrence in the 
set of semi-random starts (`counts`), as well as the predicted target `value`. Subsequently, the program asks for 
optimum names to be entered which are then written to `./evaluate_model/<hash>_selected.json`.

#### Simulate Targets

Further, calculate the input parameter dependencies of the selected target optima:

```sh
python -m bogc -exp st
```

The resulting dictionary contains a variation of each input parameter over the whole parameter range (while keeping the 
other inputs at the value of the selected optimum) and the corresponding predicted target values. The dictionary is 
stored in `./evaluate_model/<hash>_simulation.json`.

#### Plot Simulation

Finally, the last command reads the simulation results and generates a plot of the dependencies:

```sh
python -m bogc -exp ps
```

The figures will be saved to `./evaluate_model/<hash>_[<target>]_simulation.png` for each `target`.


## <a name="configref"></a>Configuration Reference

|      Option       |      Type       | Description                                                                                                                                                                                                                                                                                                                                                                                                                   |
|:-----------------:|:---------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   `train_iter`    |       int       | Number of model training cycles.                                                                                                                                                                                                                                                                                                                                                                                              |
|   `data_points`   |       str       | Specifies which kind of data points should be used: <ul><li>`grouped` uses the average of the selected sample grid and provides statistical targets ( e.g., `voc max`, `voc std`, ...).</li><li>`single` is the position-augmented data without statistical targets but with additional features `x cu` and `y`.</li></ul>                                                                                                    |
|   `sample_grid`   |       str       | Can be one of `8x8` or `6x6` to select from which grid the data points are considered.                                                                                                                                                                                                                                                                                                                                        |
|   `target_cols`   |    list[str]    | A list of table columns from the data CSV which should be used for the training.                                                                                                                                                                                                                                                                                                                                              |
|  `batch_labels`   | list[list[str]] | Ordered list of sample label lists which correspond to the consecutive batches of measured samples.                                                                                                                                                                                                                                                                                                                           |
|    `bo_batch`     |       int       | Largest index of the `batch_labels` used for loading the data. E.g., `bo_batch: 1` loads the data of the first two lists of sample labels.                                                                                                                                                                                                                                                                                    |
| `excluded_labels` |    list[str]    | A list of sample labels to exclude from the data loading.                                                                                                                                                                                                                                                                                                                                                                     |
|   `model_type`    |       str       | Specifies if an `exact` or `approx` model should be used.                                                                                                                                                                                                                                                                                                                                                                     |
|   `kernel_type`   |       str       | Specifies which kernel to use: `rbf`, `matern`, or `nugget`.                                                                                                                                                                                                                                                                                                                                                                  |


## <a name="citation"></a>Citation

If you use something from this package in your research, please cite:

```
[Citation information will be added after the publication of the related paper.]
```


## License

The source code of this project is licensed under the [MIT license](LICENSE).


## Contact

This project is developed by AIT Austrian Institute of Technology and TU Wien.

For questions, feedback, or support, feel free to [open an issue](https://github.com/mxwalbert/bogc/issues) 
or reach out via email at [maximilian.wolf@ait.ac.at](mailto:maximilian.wolf@ait.ac.at?subject=BOGC).
