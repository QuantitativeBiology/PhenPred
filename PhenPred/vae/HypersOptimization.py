# %load_ext autoreload
# %autoreload 2

import os
import sys

from sympy import hyper

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
        cv = StratifiedShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            random_state=self.random_state,
        ).split(self.data, self.data.samples_by_tissue("Haematopoietic and Lymphoid"))

        hypers = self.sample_params(trial)

        loss_val = CLinesTrain(self.data, hypers, early_stop_patience=10).training(cv)

        return loss_val

    def sample_params(self, trial):
        hypers = self.hypers.copy()

        # Optimizer
        hypers["batch_size"] = trial.suggest_int("batch_size", 16, 256)
        hypers["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        )
        hypers["view_dropout"] = trial.suggest_float("view_dropout", 0.0, 0.8)
        hypers["optimizer_type"] = trial.suggest_categorical(
            "optimizer_type",
            ["adam", "adamw", "adagrad"],
        )

        # Layers
        hypers["probability"] = trial.suggest_float("probability", 0.0, 0.6)

        # Dimensions
        hypers["view_latent_dim"] = trial.suggest_float("view_latent_dim", 0.01, 0.15)
        hypers["latent_dim"] = trial.suggest_int("latent_dim", 10, 256)
        hypers["hidden_dims"] = trial.suggest_categorical(
            "hidden_dims", ["0.3", "0.4", "0.5", "0.6", "0.7", "0.7,0.4", "0.6, 0.3"]
        )

        # Scheduler
        hypers["scheduler_factor"] = trial.suggest_float("scheduler_factor", 0.5, 0.85)

        # Loss terms weights
        hypers["w_contrastive"] = trial.suggest_float(
            "w_contrastive", 1e-6, 1.0, log=True
        )

        # Contrastive loss margins
        hypers["contrastive_pos_margin"] = trial.suggest_float(
            "contrastive_pos_margin", 0.65, 0.95
        )

        hypers["contrastive_neg_margin"] = trial.suggest_float(
            "contrastive_neg_margin", 0.05, 0.35
        )

        # GMVAE
        if hypers["model"] == "GMVAE":
            hypers["gmvae_k"] = trial.suggest_int("gmvae_k", 1, 200)

            hypers["w_gauss"] = trial.suggest_float("w_gauss", 1e-6, 1.0, log=True)
            hypers["w_cat"] = trial.suggest_float("w_cat", 1e-6, 1.0, log=True)

            hypers["gmvae_views_logits"] = trial.suggest_int(
                "gmvae_views_logits", 1, 1024
            )
            hypers["gmvae_hidden_size"] = trial.suggest_int(
                "gmvae_hidden_size", 1, 1024
            )
            hypers["gmvae_decay_temp"] = trial.suggest_categorical(
                "gmvae_decay_temp", [True, False]
            )
            hypers["gmvae_init_temp"] = trial.suggest_float(
                "gmvae_init_temp",
                0.5,
                1.0,
            )
            hypers["gmvae_min_temp"] = trial.suggest_float(
                "gmvae_min_temp",
                0.1,
                0.5,
            )
            hypers["gmvae_hard_gumbel"] = trial.suggest_float(
                "gmvae_hard_gumbel", 0.0, 1.0
            )
            hypers["gmvae_decay_temp_rate"] = trial.suggest_float(
                "gmvae_decay_temp_rate", 0.0001, 0.2
            )

        hypers = Hypers.parse_torch_functions(hypers)

        return hypers


if __name__ == "__main__":
    # Class variables - Hyperparameters
    hyperparameters = Hypers.read_hyperparameters()
    hyperparameters["num_epochs"] = 100

    # Load dataset
    clines_db = CLinesDatasetDepMap23Q2(
        datasets=hyperparameters["datasets"],
        labels_names=hyperparameters["labels"],
        standardize=hyperparameters["standardize"],
        filter_features=hyperparameters["filter_features"],
        filtered_encoder_only=hyperparameters["filtered_encoder_only"],
        feature_miss_rate_thres=hyperparameters["feature_miss_rate_thres"],
    )

    # Optuna optimization
    study_name = "MOVE"

    opt = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        load_if_exists=True,
        storage=f"sqlite:///{plot_folder}/files/optuna_{study_name}.db",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=15),
    )

    opt.optimize(
        OptunaOptimization(clines_db, hyperparameters),
        n_trials=1000,
        show_progress_bar=True,
        n_jobs=1,
    )

    # Filter outlier trials
    value_thres = 10
    filtered_trials = [t for t in opt.trials if t.value and t.value < value_thres]
    filtered_opt = optuna.create_study(direction="minimize", study_name=study_name)
    filtered_opt.add_trials(filtered_trials)
    print(
        f"Filtering {len(opt.trials) - len(filtered_trials)} trials with value > {value_thres}"
    )

    # Print results
    trial = filtered_opt.best_trial
    pruned_trials = filtered_opt.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = filtered_opt.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE]
    )

    print(
        "Study statistics: "
        + f"\n\tNumber of finished trials = {len(filtered_opt.trials):,}"
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

    # Save best hyperparameters
    json.dump(
        hyperparameters_opt,
        open(
            f"{plot_folder}/files/optuna_{filtered_opt.study_name}_hyperparameters.json",
            "w",
        ),
        indent=4,
        default=lambda o: "<not serializable>",
    )

    # Plot results
    fig = optuna.visualization.plot_param_importances(filtered_opt)
    fig.write_image(
        f"{plot_folder}/files/optuna_{filtered_opt.study_name}_best_param_plot.pdf"
    )

    fig = optuna.visualization.plot_optimization_history(filtered_opt)
    fig.write_image(
        f"{plot_folder}/files/optuna_{filtered_opt.study_name}_optimization_history.pdf"
    )

    fig = optuna.visualization.plot_slice(filtered_opt)
    fig.write_image(
        f"{plot_folder}/files/optuna_{filtered_opt.study_name}_slice_plot.pdf"
    )

    fig = optuna.visualization.plot_edf(filtered_opt)
    fig.write_image(
        f"{plot_folder}/files/optuna_{filtered_opt.study_name}_edf_plot.pdf"
    )

    fig = optuna.visualization.plot_parallel_coordinate(filtered_opt)
    fig.write_image(
        f"{plot_folder}/files/optuna_{filtered_opt.study_name}_parallel_coordinate_plot.pdf"
    )

    fig = optuna.visualization.plot_contour(
        filtered_opt, params=["latent_dim", "batch_size"]
    )
    fig.write_image(
        f"{plot_folder}/files/optuna_{filtered_opt.study_name}_contour_plot.pdf"
    )
