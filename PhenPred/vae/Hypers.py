import json
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.Losses import CLinesLosses


class Hypers:
    @classmethod
    def read_hyperparameters(cls, hypers_json=None):
        if hypers_json is None:
            hypers_json = f"{plot_folder}/files/hyperparameters.json"

        # Read json file and convert to dict
        with open(hypers_json, "r") as f:
            hypers = json.load(f)

        # Parse activation function
        hypers["activation_function"] = CLinesLosses.activation_function(
            hypers["activation_function_name"]
        )

        # Parse datasets
        hypers["datasets"] = {
            k: f"{data_folder}/{v}" for k, v in hypers["datasets"].items()
        }

        return hypers
