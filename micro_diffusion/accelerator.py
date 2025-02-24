import torch
from accelerate import Accelerator

#accelerate launch --multi_gpu --num_processes 1 accelerator.py


acc = Accelerator()
device = acc.device
device_idx = int(acc.process_index)

print (f"device:{device}\ndevice_idx:{device_idx}")
device= f"{device}:{device_idx}"
tensor = torch.randn(1).to(device)
print (f"tensor shipped")
print (tensor.device)