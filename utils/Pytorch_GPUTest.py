# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import torch
from torch import cuda

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

use_cuda = cuda.is_available()
print(use_cuda)

