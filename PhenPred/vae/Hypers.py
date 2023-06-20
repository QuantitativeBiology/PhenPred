import json
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.Losses import CLinesLosses


class Hypers:
    @classmethod
    def read_hyperparameters(cls, hypers_json=None):
        if hypers_json is None:
            hypers_json = f"{plot_folder}/files/hyperparameters.json"

        with open(hypers_json, "r") as f:
            hypers = json.load(f)

        hypers["activation_function"] = CLinesLosses.activation_function(
            hypers["activation_function"]
        )

        hypers["reconstruction_loss"] = CLinesLosses.reconstruction_loss_method(
            hypers["reconstruction_loss"]
        )

        hypers["datasets"] = {
            k: f"{data_folder}/{v}" for k, v in hypers["datasets"].items()
        }

        return hypers
