import torch
from torch.utils.data import Dataset
from torchvision import datasets
import json

JSON_CONFIG_PATH = "dataset.json"


def open_config():
    with open(JSON_CONFIG_PATH) as config_file:
        return json.load(config_file)



config = open_config()

def get_labels(config):
    words = []

    for sample in config:
        if not sample["word"] in words:
            words.append(sample["word"])

    return words

    

class SignsDataset(Dataset):
    def __init__(self):
        self.config = config
        self.labels = get_labels(config)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.config[index]
        label = self.labels.index(sample["word"])

        return torch.FloatTensor(sample["frames_compiled_points"]), label
