print('Loading dependencies...')


import argparse
import yaml
from bogc import experiments
from bogc.data_loading import DataConfig


def none_or_float(value: str) -> None | float:
    if value.lower() == "none":
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-cnf', '--config_file', type=str, default='config.yml',
                        help="(str) path of the configuration yaml file")
    parser.add_argument('-exp', '--experiment', choices=['fm', 'tt', 'cv', 'po', 'mo', 'ot', 'so', 'st', 'ps'], default='fm',
                        help="(str) select which experiment to run:\n"
                             "  [fm: Final Model]\n"
                             "   tt:  Training-Test Split\n"
                             "   cv:  Leave-one-out Cross-Validation\n"
                             "   po:  Parameter Optimization\n"
                             "   mo:  Model Optimization\n"
                             "   ot:  Model Evaluation: Optimize Targets\n"
                             "   so:  Model Evaluation: Select Optima\n"
                             "   st:  Model Evaluation: Simulate Targets\n"
                             "   ps:  Model Evaluation: Plot Simulation"
                        )
    parser.add_argument('--test_size', type=float, default=0.1,
                        help="(float) the fractional size of the test set for the training-test split [0.1]")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="(int) the number of sequential samples to draw for the parameter optimization [4]")
    parser.add_argument('--min_num_selected', type=int, default=0,
                        help="(int) minimum number of selected samples for the model optimization [0]")
    parser.add_argument('--trials', type=int, default=1,
                        help="(int) number of trials for the model optimization [1]")
    parser.add_argument('--target_ranges', type=none_or_float, nargs='+', default=None,
                        help="(float [float ...]) range constraints of the targets for the model optimization [None]")
    parser.add_argument('--plot_results', type=bool, default=True,
                        help="(bool) flag if results should be plotted where applicable [True]")
    parser.add_argument('--plot_color', choices=range(21), default=11,
                        help="(int) select which color of the viridis color map to use for plotting [11]")
    args = parser.parse_args()
    return args


def read_config(config_file: str) -> DataConfig:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return DataConfig(**config)


def main():
    args = parse_args()
    data_config = read_config(args.config_file)
    if args.experiment == 'fm':
        experiments.final_model.run(data_config)
    elif args.experiment == 'tt':
        experiments.training_test_split.run(data_config,
               test_size=args.test_size,
               plot_results=args.plot_results, plot_color=args.plot_color)
    elif args.experiment == 'cv':
        experiments.cross_validation.run(data_config,
               plot_results=args.plot_results, plot_color=args.plot_color)
    elif args.experiment == 'po':
        experiments.parameter_optimization.run(data_config,
               batch_size=args.batch_size)
    elif args.experiment == 'mo':
        experiments.model_optimization.run(data_config,
               min_num_selected=args.min_num_selected, trials=args.trials, target_ranges=args.target_ranges)
    elif args.experiment == 'ot':
        experiments.model_evaluation.optimize_targets.run(data_config)
    elif args.experiment == 'so':
        experiments.model_evaluation.select_optima.run(data_config)
    elif args.experiment == 'st':
        experiments.model_evaluation.simulate_targets.run(data_config)
    elif args.experiment == 'ps':
        experiments.model_evaluation.plot_simulation.run(data_config)


if __name__ == '__main__':
    main()
