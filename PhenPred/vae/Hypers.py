import json
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.Losses import CLinesLosses


class Hypers:
    @classmethod
    def read_json(cls, json_file):
        with open(json_file, "r") as f:
            hypers = json.load(f)
        return hypers

    @classmethod
    def read_hyperparameters(cls, hypers_json=None, parse_torch_functions=True):
        if hypers_json is None:
            hypers_json = f"{plot_folder}/files/hyperparameters.json"

        hypers = cls.read_json(hypers_json)

        if "model" not in hypers:
            hypers["model"] = "MOVE"

        if "standardize" not in hypers:
            hypers["standardize"] = False

        hypers["datasets"] = {
            k: f"{data_folder}/{v}" for k, v in hypers["datasets"].items()
        }

        if parse_torch_functions:
            hypers = cls.parse_torch_functions(hypers)

        return hypers

    @classmethod
    def parse_torch_functions(cls, hypers):
        hypers["activation_function"] = CLinesLosses.activation_function(
            hypers["activation_function"]
        )

        hypers["reconstruction_loss"] = CLinesLosses.reconstruction_loss_method(
            hypers["reconstruction_loss"]
        )

        if type(hypers["hidden_dims"]) == str:
            hypers["hidden_dims"] = [float(l) for l in hypers["hidden_dims"].split(",")]

        return hypers
