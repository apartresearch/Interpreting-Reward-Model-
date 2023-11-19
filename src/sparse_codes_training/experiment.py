import argparse
from time import sleep

from sparse_codes_training.experiment_configs import (
    ExperimentConfig, grid_experiment_configs
)
from sparse_codes_training.experiment_runner import ExperimentRunner

parser = argparse.ArgumentParser(description="Choose which experiment config you want to run.")

parser.add_argument("--fast", action="store_true", help="Whether to run in fast mode or not.", required=False)
parser.add_argument("--l1_coef", default=None, type=float, help="The l1_coef you want to use.", required=False)
parser.add_argument("--base_model_name", default='pythia-70m', type=str, help="The model name you want to use.", required=False)
parser.add_argument("--reward_function", default='utility_reward', type=str, help="The reward function you want to leverage.", required=False)

def run_experiment(experiment_config: ExperimentConfig):
    '''
    Part 1 of IMDb experiment:
    1. Compute parameter divergence and sorts layers by parameter divergence between m_base and m_rlhf.
    2. Extract activations for the train split of IMDb prefixes.
    3. Train autoencoders on the extracted activations.
    4. Measure loss of the autoencoder on the IMDb test dataset.
    '''
    sleep(5)
    experiment_runner = ExperimentRunner(experiment_config=experiment_config)
    experiment_runner.run_experiment()


def parse_args():
    args = parser.parse_args()
    base_model_name = args.base_model_name
    reward_function = args.reward_function
    chosen_experiment_config = grid_experiment_configs[(base_model_name, reward_function)]

    # Override default experiment config with parsed command line args.
    parsed_hyperparams = {
        "fast": args.fast,
        "l1_coef": args.l1_coef
    }
    for key, value in parsed_hyperparams.items():
        if value is not None:
            chosen_experiment_config.hyperparameters[key] = value
    return chosen_experiment_config

chosen_experiment_config = parse_args()
print(f'Running experiment now for config {chosen_experiment_config}')
run_experiment(experiment_config=chosen_experiment_config)