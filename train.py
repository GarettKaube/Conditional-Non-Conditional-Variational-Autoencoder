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
from utils import one_hot, denormalize
from plotting import plot_test, plot_image


class Trainer:
    def __init__(self, 
                 model, 
                 lr, 
                 loss, 
                 num_epoch, 
                 train_data,
                 train_loader, 
                 test_loader, 
                 image_loader, 
                 validate_every, 
                 device, 
                 conditional,
                 classes) -> None:
        
        self.model = model

        self.conditional = conditional

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.schedule = optim.lr_scheduler.MultiStepLR(self.optimizer, [40, 50], gamma=0.1, verbose=False)
        self.loss = loss
        self.best_total_loss = float("inf")
        self.num_epochs = num_epoch
        self.validate_every = validate_every

        self.train_data = train_data
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.image_loader = image_loader

        self.device = device

        self.classes = classes

        self.l2_losses = []
        self.bce_losses = []
        self.ssim_losses = []
        self.kld_losses = []
        self.total_losses = []
        self.total_losses_train = []

    def train_step(self, x, y):
        """One train step for VAE/CVAE.
        x, y: one batch (images, labels) from Cifar10 train set.
        Returns:
            loss: total loss per batch.
            l2_loss: MSE loss for reconstruction.
            bce_loss: binary cross-entropy loss for reconstruction.
            ssim_loss: ssim loss for reconstruction.
            kldiv_loss: kl divergence loss.
        """

        self.optimizer.zero_grad()
        
        output, mu, log_var, z = self.model(x, y)

        loss, l2_loss, bce_loss, ssim_loss, kldiv_loss = self.loss.compute_loss(x, output, mu, log_var)
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss, l2_loss / x.shape[0], bce_loss  / x.shape[0], ssim_loss / x.shape[0], kldiv_loss / x.shape[0]
    
    def train(self, print_every, kl_annealing):
        for epoch in range(1, self.num_epochs + 1):
            self.total_loss_train = 0.0
            for i, (x, y) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                
                y = one_hot(y, 9, self.device)
                y = y.view(x.shape[0], 1, 1, 10)*torch.ones((x.shape[0], x.shape[1], x.shape[2], 10)).to('cuda')
                

                # Train step
                self.model.train()
                loss, recon_loss, bce_loss, ssim_loss, kldiv_loss = self.train_step(x, y)
                self.total_loss_train += loss * x.shape[0]
                
                
                if i % print_every == 0:
                    print("Epoch {}, Iter {}: Total Loss: {:.6f} MSE: {:.6f}, SSIM: {:.6f}, BCE: {:.6f}, KLDiv: {:.6f}".format(epoch, i, loss, recon_loss, ssim_loss, bce_loss, kldiv_loss))
                
            self.total_losses_train.append((self.total_loss_train / len(self.train_data)).detach().cpu().numpy())

            if epoch % self.validate_every == 0:
                self.model.eval()

                self.l2_losses.append(recon_loss.detach().cpu().numpy())
                self.bce_losses.append(bce_loss.detach().cpu().numpy())
                self.ssim_losses.append(ssim_loss.detach().cpu().numpy())
                self.kld_losses.append(kldiv_loss.detach().cpu().numpy())

                self.test(epoch, self.test_loader)
                
            self.loss.KLDiv_criterion.lambd = kl_annealing[epoch]

            print("Lambda:", self.loss.KLDiv_criterion.lambd)
            
            self.schedule.step()
            self.generate_image(epoch)


    def test(self, epoch, test_loader):
        ckpt_path = 'CVAE.pt' if self.conditional else 'VAE.pt'  # path for saving models
        # Loop through test set
        with torch.no_grad():
            for x, y in test_loader:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                
                y = one_hot(y, 9, self.device)

                y = y.view(x.shape[0], 1, 1, 10)*torch.ones((x.shape[0], x.shape[1], x.shape[2], 10)).to(self.device)    
                xg, mu, log_var, _ = self.model(x, y)

                loss, mse, bce_loss, ssim_loss, kldiv_loss = self.loss.compute_loss(x, xg, mu, log_var)
                avg_total_recon_loss_test = (mse+ bce_loss + ssim_loss).detach().cpu().numpy()
                total_loss_ = avg_total_recon_loss_test
                self.total_losses.append(total_loss_)

            if epoch > 1:
                plot_test(epoch, self.l2_losses, self.bce_losses, self.ssim_losses, self.kld_losses, self.total_losses, self.total_losses_train, self.conditional)

            if avg_total_recon_loss_test < self.best_total_loss:
                torch.save(self.model.state_dict(), ckpt_path)
                self.best_total_loss = avg_total_recon_loss_test
                print("Best model saved w/ Total Reconstruction Loss of {:.6f}.".format(self.best_total_loss))


    def generate_image(self, epoch):
        self.model.eval()
        with torch.no_grad():
            x, y = next(iter(self.image_loader))
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            y = one_hot(y, 9, self.device)
            y = y.view(x.shape[0], 1, 1, 10)*torch.ones((x.shape[0], x.shape[1], x.shape[2], 10)).to(self.device) 

            x_reconstucted , _, _, _ = self.model(x, y)
           
            # Visualize
            x_reconstucted  = denormalize(x_reconstucted )
            x = denormalize(x)

            y = torch.max(torch.max(torch.argmax(y, dim=3),dim = 2)[0], dim=1)[0]
            
            y = y.cpu().numpy()
            plot_image(epoch, x_reconstucted , x, y, self.classes, self.conditional)
            
    