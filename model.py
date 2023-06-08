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
from utils import one_hot



class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        in_channels: int = 3,
        conditional = False
        ):
        super(Encoder, self).__init__()
        self.conditional = conditional
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(self.in_channels, 64, 4, stride = 2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(64, 128, 4, stride = 2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(128, 256, 4, stride = 2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=256)

        if conditional:
            self.fc1 = nn.Linear(4*5*256, self.latent_dim)
            self.fc2 = nn.Linear(4*5*256, self.latent_dim)
        else:
            self.fc1 = nn.Linear(4*4*256, self.latent_dim)
            self.fc2 = nn.Linear(4*4*256, self.latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        if self.conditional:
            x = x.view(-1, 256*4*5)
        else:
            x = x.view(-1, 256*4*4)
        
        mean = self.fc1(x)
        logvar = self.fc2(x)
        
        return mean, logvar
    

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        conditional = False
        ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        self.fc1 = nn.Linear(self.latent_dim, 448*2*2)
        self.bn1 = nn.BatchNorm1d(num_features=448*2*2)
        self.tconv1 = nn.ConvTranspose2d(448, 256, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.tconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.tconv4 = nn.ConvTranspose2d(64, self.out_channels, 4, stride=2, padding=1)
        
    def forward(self, z):
        
        z = F.relu(self.bn1(self.fc1(z)))
        z = z.view(-1, 448, 2, 2)
        z = F.relu(self.bn2(self.tconv1(z)))
        z = F.relu(self.bn3(self.tconv2(z)))
        z = F.relu(self.tconv3(z))
        z = torch.sigmoid(self.tconv4(z))
        
        return z
    


class VAE(nn.Module):
    def __init__(
        self, 
        latent_dim: int = 128,
        device=None):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.device = device

        self.encode = Encoder(conditional = False, latent_dim=latent_dim)
        self.decode = Decoder(conditional = False, latent_dim=latent_dim)

    def reparameterize(self, mu, log_var):
        """Reparameterization Tricks to sample latent vector z
        from distribution w/ mean and variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * log_var + mu
        return z

    def forward(self, x, y):

        """Forward for CVAE.
        Returns:
            x_reconstucted: reconstructed image from decoder.
            mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
            z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
        """
        mu , log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstucted = self.decode(z)
        return x_reconstucted, mu, log_var, z

    def generate(
        self,
        n_samples: int,
        ):


        """Randomly sample from the latent space and return
        the reconstructed samples.
        Returns:
            x_reconstucted: reconstructed image
            None: a placeholder simply.
        """
        

        x = torch.randn(n_samples, self.latent_dim, device=self.device)
        x_reconstucted = self.decode(x)
        return x_reconstucted, None



class CVAE(nn.Module):
    def __init__(self, latent_dim: int = 128, num_classes: int = 10, img_size: int = 32, device=None):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = device

        self.encode = Encoder(conditional = True, latent_dim=latent_dim+num_classes, in_channels=3)
        self.decode = Decoder(conditional = True, latent_dim=latent_dim+num_classes)



    def reparameterize(self, mu, log_var):
        """Reparameterization Tricks to sample latent vector z
        from distribution w/ mean and variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * log_var + mu
        return z

    def forward(self, x, y):

        """Forward for CVAE.
        Returns:
            x_reconstucted: reconstructed image from decoder.
            mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
            z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
        """

        y_oh = one_hot(y, 9, self.device)
        y_ = y_oh.view(x.shape[0], 1, 1, 10)*torch.ones((x.shape[0], x.shape[1], x.shape[2], 10)).to(self.device)

        x = torch.cat((x, y_), dim=3).to(self.device)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu[:,:self.latent_dim], log_var[:,:self.latent_dim])
        
        z  = torch.cat((z, y_oh), dim=1).to(self.device)
        
        x_reconstucted = self.decode(z)
        
        
        return x_reconstucted , mu, log_var, z


    def generate(self,n_samples: int,y: torch.Tensor = None):

        """Randomly sample from the latent space and return
        the reconstructed samples.
        NOTE: Randomly generate some classes here, if not y is provided.
        Returns:
            x_reconstucted : reconstructed image
            y: classes for xg. 
        """

        

        x = torch.randn((n_samples, self.latent_dim),device=self.device).to(self.device)
        y = torch.randint(low = 0, high = self.num_classes-1, size = (n_samples,),device=self.device).to(self.device)
        y_oh = one_hot(y, 9, self.device)
        
        z = torch.cat([x,y_oh], dim=1)
        
        x_reconstucted  = self.decode(z)

        
        return x_reconstucted , y
