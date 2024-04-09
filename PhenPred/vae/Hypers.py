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
    def read_hyperparameters(
        cls, hypers_json=None, parse_torch_functions=True, timestamp=None
    ):
        if timestamp is not None:
            hypers_json = f"{plot_folder}/files/{timestamp}_hyperparameters.json"
        elif hypers_json is None:
            hypers_json = f"{plot_folder}/files/hyperparameters.json"

        hypers = cls.read_json(hypers_json)

        if timestamp is not None:
            hypers["load_run"] = timestamp

        if "model" not in hypers:
            hypers["model"] = "MOSA"

        if "standardize" not in hypers:
            hypers["standardize"] = False

        if "w_rec" not in hypers:
            hypers["w_rec"] = 1

        if "w_gauss" not in hypers:
            hypers["w_gauss"] = 0.01

        if "w_cat" not in hypers:
            hypers["w_cat"] = 0.01

        if hypers["view_loss_weights"] is None:
            hypers["view_loss_weights"] = [1.0] * len(hypers["views"])

        if hypers["use_conditionals"] is None:
            hypers["use_conditionals"] = True

        if timestamp is None:  # full path is already stored in previous json config
            hypers["datasets"] = {
                k: f"{data_folder}/{v}" for k, v in hypers["datasets"].items()
            }
        print(f"# ---- Hyperparameters")
        print(json.dumps(hypers, indent=4, sort_keys=True))

        if parse_torch_functions:
            hypers = cls.parse_torch_functions(hypers)

        return hypers

    @classmethod
    def parse_torch_functions(cls, hypers):
        if type(hypers["activation_function"]) == str:
            hypers["activation_function"] = CLinesLosses.activation_function(
                hypers["activation_function"]
            )

        if type(hypers["reconstruction_loss"]) == str:
            hypers["reconstruction_loss"] = CLinesLosses.reconstruction_loss_method(
                hypers["reconstruction_loss"]
            )

        if type(hypers["hidden_dims"]) == str:
            hypers["hidden_dims"] = [float(l) for l in hypers["hidden_dims"].split(",")]

        return hypers
