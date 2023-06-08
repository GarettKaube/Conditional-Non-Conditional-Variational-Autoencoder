import torch
import numpy as np

def one_hot(x, max_x, device):
    return torch.eye(max_x + 1, device=device)[x]



def denormalize(x):
    """Denomalize a normalized image back to uint8.
    Args:
        x: torch.Tensor, in [0, 1].
    Return:
        x_denormalized: denormalized image as numpy.uint8, in [0, 255].
    """
    
    minimum = torch.min(x)
    maximum = torch.max(x)
    
    x = x*((255)/(maximum - minimum)) + 0
    x = x.detach().cpu().numpy()
    x = np.transpose(x, (0, 2, 3, 1)) 
    return x.astype(np.uint8)