import yaml
import joblib
import torch


class CustomException(Exception):
    def __init__(self, message: str):
        super(CustomException, self).__init__()
        self.message = message


def dump(value: str, filename: str):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)

    else:
        raise CustomException("Cannot be dump into pickle file".capitalize())


def load(filename: str):
    if filename is not None:
        joblib.load(filename=filename)

    else:
        raise CustomException("Cannot be load the pickle file".capitalize())


def device_init(self, device: str = "mps"):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    else:
        return torch.device("cpu")


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)
