import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt

def get_save_path(conditional):
    if conditional:
        name = "CVAE"
    else:
        name = "VAE"
    
    if not os.path.exists(os.path.join(os.path.curdir, "visualize", name)):
        os.makedirs(os.path.join(os.path.curdir, "visualize", name))
    save_path = os.path.join(os.path.curdir, "visualize", name)
    return save_path


def plot_test(epoch, l2_losses, bce_losses, ssim_losses, kld_losses, total_losses, total_losses_train, conditional):
    
    save_path = get_save_path(conditional)
    

    plt.plot(l2_losses, label="L2 Reconstruction")
    plt.plot(bce_losses, label="BCE")
    plt.plot(ssim_losses, label="SSIM")
    plt.plot(kld_losses, label="KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim([1, epoch])
    plt.legend()
    plt.savefig(os.path.join(os.path.join(save_path, "losses.png")), dpi=300)
    plt.clf()
    plt.close('all')

    plt.plot(total_losses, label="Total Loss Test")
    plt.plot(total_losses_train, label="Total Loss Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim([1, epoch])
    plt.legend()
    plt.savefig(os.path.join(os.path.join(save_path, "total_loss.png")), dpi=300)
    plt.clf()
    plt.close('all')


def plot_image(epoch, xg, x, y, classes, conditonal):
    save_path = get_save_path(conditonal)

    plt.figure(figsize=(10, 5))
    for p in range(10):
        plt.subplot(4, 5, p+1)
        
        plt.imshow(xg[p])
        plt.subplot(4, 5, p + 1 + 10)
        plt.imshow(x[p])
        plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                    backgroundcolor='white', fontsize=8)
        plt.axis('off')
        
    plt.savefig(os.path.join(os.path.join(save_path, "Epoch{:d}.png".format(epoch))), dpi=300)
    plt.clf()
    plt.close('all')
    print("Figure saved at epoch {}.".format(epoch))