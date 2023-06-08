import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import argparse
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
from utils import denormalize
from plotting import get_save_path
from model import VAE, CVAE
from train import Trainer
from loss import Loss

parser = argparse.ArgumentParser(description='Conditional/Non conditional auto encoder', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--conditional", help="Conditional or non conditiona", default=False)


def main():
    torch.manual_seed(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on: {}".format(device))

    args = parser.parse_args()
    conditional = bool(args.conditional)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    batch_size = 256
    workers = 2
    latent_dim = 300
    lr = 0.0005
    num_epochs = 60
    validate_every = 1
    print_every = 100


    if conditional:
        name = "cvae"
    else:
        name = "vae"


    # Data loaders
    tfms = transforms.Compose([
        transforms.ToTensor(),
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=tfms)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True,
        transform=tfms,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=workers)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=workers)

    subset = torch.utils.data.Subset(
        test_dataset, 
        [0, 380, 500, 728, 1000, 2300, 3400, 4300, 4800, 5000])

    loader = torch.utils.data.DataLoader(
        subset, 
        batch_size=10)
    

    if conditional:
        # contidtional variational autoencoder 
        model = CVAE(latent_dim=latent_dim)
    else:
        # variational autoencoder
        model = VAE(latent_dim=latent_dim)

    model.to(device)

    loss = Loss()

    # KL Annealing
    kl_annealing = np.linspace(0,1, num_epochs+1) 


    trainer = Trainer(model,
                      lr,
                      loss,
                      num_epochs,
                      train_dataset,
                      train_loader,
                      test_loader,
                      loader,
                      validate_every,
                      device,
                      conditional,
                      classes)
    
    trainer.train(print_every, kl_annealing)
    
    if conditional:
        model = CVAE(latent_dim=latent_dim)
    else:
        model = VAE(latent_dim=latent_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    ckpt = torch.load(name+'.pt')
    model.load_state_dict(ckpt)

    # Generate 20 random images
    x_reconstucted, y = model.generate(20)

    x_reconstucted = denormalize(x_reconstucted)
    if y is not None:
        y = y.cpu().numpy()

    plt.figure(figsize=(10, 5))
    for p in range(20):
        plt.subplot(4, 5, p+1)
        if y is not None:
            plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                    backgroundcolor='white', fontsize=8)
        plt.imshow(x_reconstucted[p])
        plt.axis('off')

    save_path = get_save_path("")
    plt.savefig(os.path.join(os.path.join(save_path, "random.png")), dpi=300)
    plt.clf()
    plt.close('all')


if __name__ == "__main__":
     main()
