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
    def __init__(self, data, hypers, n_splits=1, test_size=0.20, random_state=42):
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
        gmvae_args_dict = (
            dict(
                k=100,
                init_temp=1.0,
                decay_temp=1.0,
                hard_gumbel=0,
                min_temp=0.5,
                decay_temp_rate=0.013862944,
            )
            if hyperparameters["model"] == "GMVAE"
            else None
        )

        loss_val = CLinesTrain(
            self.data,
            hypers,
            early_stop_patience=10,
            gmvae_args_dict=gmvae_args_dict,
        ).training(cv)

        return loss_val

    def sample_params(self, trial):
        hypers = self.hypers.copy()

        hypers["model"] = trial.suggest_categorical("model", ["MOVE", "GMVAE"])
        hypers["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        )
        hypers["batch_size"] = trial.suggest_int("batch_size", 16, 256)
        hypers["view_latent_dim"] = trial.suggest_float("view_latent_dim", 0.01, 0.1)
        hypers["latent_dim"] = trial.suggest_int("latent_dim", 10, 100)
        hypers["hidden_dims"] = trial.suggest_categorical(
            "hidden_dims", ["0.4", "0.5", "0.6", "0.7", "0.7,0.4"]
        )
        hypers["optimizer_type"] = trial.suggest_categorical(
            "optimizer_type", ["adam", "sgd"]
        )
        hypers["activation_function"] = trial.suggest_categorical(
            "activation_function", ["relu", "leaky_relu", "prelu"]
        )
        hypers["scheduler_factor"] = trial.suggest_float("scheduler_factor", 0.5, 0.85)
        hypers = Hypers.parse_torch_functions(hypers)

        return hypers


if __name__ == "__main__":
    # Class variables - Hyperparameters
    hyperparameters = Hypers.read_hyperparameters()
    hyperparameters["num_epochs"] = 150

    # Load dataset
    clines_db = CLinesDatasetDepMap23Q2(
        labels_names=hyperparameters["labels"],
        datasets=hyperparameters["datasets"],
        feature_miss_rate_thres=hyperparameters["feature_miss_rate_thres"],
        standardize=hyperparameters["standardize"],
        filter_features=hyperparameters["filter_features"],
        filtered_encoder_only=hyperparameters["filtered_encoder_only"],
    )

    # Optuna optimization
    opt = optuna.create_study(
        direction="minimize",
        study_name="MOVE",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        storage=f"sqlite:///{plot_folder}/files/optuna.db",
    )

    opt.optimize(
        OptunaOptimization(clines_db, hyperparameters),
        n_trials=1000,
        show_progress_bar=True,
        n_jobs=1,
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

    fig = optuna.visualization.plot_optimization_history(opt)
    fig.write_image(f"{plot_folder}/files/optuna_optimization_history.pdf")

    fig = optuna.visualization.plot_slice(opt)
    fig.write_image(f"{plot_folder}/files/optuna_slice_plot.pdf")

    fig = optuna.visualization.plot_edf(opt)
    fig.write_image(f"{plot_folder}/files/optuna_edf_plot.pdf")

    fig = optuna.visualization.plot_intermediate_values(opt)
    fig.write_image(f"{plot_folder}/files/optuna_intermediate_values_plot.pdf")

    fig = optuna.visualization.plot_parallel_coordinate(opt)
    fig.write_image(f"{plot_folder}/files/optuna_parallel_coordinate_plot.pdf")

    fig = optuna.visualization.plot_contour(opt, params=["learning_rate", "batch_size"])
    fig.write_image(f"{plot_folder}/files/optuna_contour_plot.pdf")
