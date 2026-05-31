import ctypes as ct
import random as rdm
import torch.nn as nn
import torch
import pandas as pd
import matplotlib.pyplot as PLT
from torchinfo import summary
import argparse
from torch.optim import (
    SGD,RAdam, Adam
)
from torch.nn import (
    BCELoss, MSELoss
)
from torchvision.transforms import v2
import os
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchcodec.decoders import AudioDecoder
from torchscan import summary
import matplotlib.pyplot as plt
import matplotlib.image as mpimg