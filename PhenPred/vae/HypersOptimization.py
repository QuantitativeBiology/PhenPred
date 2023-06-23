# %load_ext autoreload
# %autoreload 2

import os
import sys

proj_dir = "/home/egoncalves/PhenPred"
if not os.path.exists(proj_dir):
    proj_dir = "/Users/emanuel/Projects/PhenPred"
sys.path.extend([proj_dir])

import json
import torch
import optuna
from datetime import datetime
from optuna.trial import TrialState
from PhenPred.vae import plot_folder
from PhenPred.vae.Hypers import Hypers
from PhenPred.vae.Train import CLinesTrain
from sklearn.model_selection import StratifiedShuffleSplit
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2


class OptunaOptimization:
    def __init__(self, data, hypers, n_splits=3, test_size=0.15, random_state=42):
        self.data = data
        self.hypers = hypers
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def __call__(self, trial):
        # Cross validation
        cv = StratifiedShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            random_state=self.random_state,
        ).split(self.data, self.data.samples_by_tissue("Haematopoietic and Lymphoid"))

        # Sample hyperparameters
        hypers = self.sample_params(trial)

        # Train
        loss_val = CLinesTrain(self.data, hypers).training(cv)

        return loss_val

    def sample_params(self, trial):
        hypers = self.hypers.copy()

        hypers["optimizer_type"] = trial.suggest_categorical("optimizer_type", ["adam"])
        hypers["activation_function"] = trial.suggest_categorical(
            "activation_function", ["relu", "tanh", "leaky_relu", "sigmoid"]
        )
        hypers["hidden_dims"] = trial.suggest_categorical(
            "hidden_dims", [[0.4], [0.7], [0.7, 0.4], [0.7, 0.4, 0.4]]
        )

        hypers["w_kl"] = trial.suggest_float("w_kl", 0.0, 0.1)
        hypers["w_decay"] = trial.suggest_float("w_decay", 1e-8, 1e-3, log=True)
        hypers["probability"] = trial.suggest_float("probability", 0.0, 0.5)
        hypers["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 1e-1, log=True
        )
        hypers["latent_dim"] = trial.suggest_float("latent_dim", 0.02, 0.3, log=True)
        hypers["scheduler_threshold"] = trial.suggest_float(
            "scheduler_threshold", 1e-6, 1e-3, log=True
        )
        hypers["scheduler_factor"] = trial.suggest_float("scheduler_factor", 0.3, 0.85)
        hypers["scheduler_min_lr"] = trial.suggest_float(
            "scheduler_min_lr", 1e-6, 1e-4, log=True
        )

        hypers["num_epochs"] = trial.suggest_int("num_epochs", 150, 500)
        hypers["batch_size"] = trial.suggest_int("batch_size", 32, 256)
        hypers["scheduler_patience"] = trial.suggest_int("scheduler_patience", 5, 15)

        hypers = Hypers.parse_torch_functions(hypers)

        return hypers


if __name__ == "__main__":
    # Class variables - Hyperparameters
    hyperparameters = Hypers.read_hyperparameters()

    # Load dataset
    clines_db = CLinesDatasetDepMap23Q2(
        label=hyperparameters["label"],
        datasets=hyperparameters["datasets"],
        feature_miss_rate_thres=hyperparameters["feature_miss_rate_thres"],
    )

    # Optuna optimization
    opt = optuna.create_study(
        direction="minimize",
        study_name="MOVE",
        load_if_exists=True,
        storage=f"sqlite:///{plot_folder}/files/optuna.db",
    )
    opt.optimize(
        OptunaOptimization(clines_db, hyperparameters),
        n_trials=5,
        show_progress_bar=True,
        n_jobs=-1,
    )

    # Print results
    trial = opt.best_trial
    pruned_trials = opt.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = opt.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(
        "Study statistics: "
        + f"\n\tNumber of finished trials = {len(opt.trials):,}"
        + f"\n\tNumber of pruned trials = {len(pruned_trials):,}"
        + f"\n\tNumber of complete trials = {len(complete_trials):,}"
        + f"\nBest trial: Value = {trial.value:.5f}"
        + f"\nParams:\n"
    )

    for k, v in trial.params.items():
        print(f"\t{k} = {v}")

    # Write the hyperparameters to json file
    hyperparameters_opt = Hypers.read_hyperparameters(parse_torch_functions=False)
    hyperparameters_opt.update(trial.params)

    json.dump(
        hyperparameters_opt,
        open(f"{plot_folder}/files/optuna_hyperparameters.json", "w"),
        indent=4,
        default=lambda o: "<not serializable>",
    )

    fig = optuna.visualization.plot_param_importances(opt)
    fig.write_image(f"{plot_folder}/files/optuna_best_param_plot.pdf")
