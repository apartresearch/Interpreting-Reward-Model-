<p align="center">
    <br>
    <img src="./pics/apart_banner_reward.png" width="900"/>
    <br>
<p>

## Official Repository for Interpreting Learned Feedback Patterns in Large Language Models
#### By Luke Marks, Amir Abdullah, Clement Neo, Rauno Arike, David krueger, Philip Torr, Fazl Barez 
1. This repository provides scripts to train LLMs with RLHF.
2. The repository also supports training sparse autoencoders for feature extraction on the MLP layers of LLMs.
3. As well as classifying and probing those features.

## Installation
From source

```bash
git clone https://github.com/apartresearch/Interpreting-Learned-Feedback-Patterns.git
cd Interpreting-Learned-Feedback-Patterns
pip install .
```

## Repository structure.
This repo is structured so that RLHF models are trained under `src/rlhf_model_training`.
The autoencoder training code is under `src/sparse_codes_training`.

As such we divide this repository into two major components, `rlhf_model_training` and `sparse_codes_training`.

The structure looks like this:

```
requirements.txt
scripts/
    ppo_training/
        run_experiment.sh
    sparse_codes_training/
        experiment.sh
    setup_environment.sh

src/
    rlhf_model_training
        reward_class.py
        rlhf_model_pipeline.py
        rlhf_training_utils/
    sparse_codes_training
        metrics/
        models/
            sparse_autoencoder.py
	experiment_helpers/
            autoencoder_trainer_and_preparer.py
            experiment_runner.py
            layer_activations_handler.py
        experiment.py
        experiment_configs.py
    utils/
```

`experiment.py` is the main script entrypoint for autoencoder training where we parse command line arguments and select/launch autoencoder training. `experiment_runner.py` has most of the actual logic of the paper, where we extract divergent layers, initialize models and train autoencoders on activations.

The LayerActivationsHandler class carries out necessary primitives of extracting activations from a layer, and calculating divergences between the corresponding layers of two neural nets.

## Getting started.
1. Run `source scripts/setup_environment.sh` to set your python path. Run the script as `source scripts/setup_environment.sh -v` if you also want to create and activate the appropriate virtual environment with all dependencies.
2. The main script for training PPO models is under `scripts/ppo_training/run_experiment.sh`.
3. The script for training autoencoders is under `scripts/sparse_codes_training/experiment.sh`. Modify these two scripts as needed to launch new PPO model or autoencoder training runs. We use other `experiment_x` scripts in the same directory to explore other parameter choices.

##  Refrence
If you use this work, please cite:

```bibtex
@misc{marks2023interpreting,
      title={Interpreting Learned Feedback Patterns in Large Language Models}, 
      author={Luke Marks and Amir Abdullah and Clement Neo and Rauno Arike and Philip Torr and Fazl Barez},
      year={2023},
      eprint={2310.08164},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
