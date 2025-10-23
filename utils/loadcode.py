import torch

checkpoint = torch.load()

# model.load_state_dict(checkpoint['model_state_dict'])

loaded_codebook = checkpoint['codebook']
import pdb;pdb.set_trace()
print(loaded_codebook.shape) 