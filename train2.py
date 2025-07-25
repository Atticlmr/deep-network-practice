import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
import os

def train_CAV(args_cli, model, configure,log_dir):
    """
    params::
    model: networl model
    configure: config json for training

    optimizer: Adam

    """
    device = args_cli.device
    writer = SummaryWriter(log_dir=log_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=configure['lr'])
    criterion = MSELoss()
    
    print("training finished!")
    return