import torch
import os
import re
import pprint

# read file
file_dir = './ckpts/adni_struct_count'
file_name = 'checkpoint_best.pt'
data = os.path.join(file_dir, file_name)

# load checkpoint
checkpoint = torch.load(data)

print("file name is: ", data)

pprint.pprint(checkpoint['extra_state'])

print("valid loss is: ", float(checkpoint['extra_state']['val_loss'])) #return validation loss
# print("valid loss is: ", checkpoint['extra_state']['metrics']['valid'][0][-1])#['val'])) #return train loss


# #['args', 'cfg', 'model', 'criterion', 'optimizer_history', 'task_state', 'extra_state', 'last_optimizer_state']
# print("train loss is: ", checkpoint['extra_state']['metrics']['train'][0][-1]) #['val'])) #return train loss

print("train loss is: ", float(checkpoint['extra_state']['metrics']['train'][0][-1]['val'])) #return train loss

# it will return score (validation loss) of best model, and you should run test dataset with that model