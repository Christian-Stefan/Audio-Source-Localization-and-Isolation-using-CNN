import ctypes as ct
import random as rdm
import torch.nn as nn
import torch
import pandas as pd
import matplotlib.pyplot as PLT
from torchinfo import summary
from utils.assignment1_data.mnist_dataloader import create_dataloaders
import argparse
from torch.optim import (
    SGD,RAdam
)
from torch.nn import (
    BCELoss, MSELoss
)
from torchvision.transforms import v2
import os