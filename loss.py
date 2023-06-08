
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

class KLDivLoss(nn.Module):
    def __init__(
        self,
        lambd: float = 1.0,
        ):
        super(KLDivLoss, self).__init__()
        self.lambd = lambd

    def forward(
        self, 
        mu, 
        log_var,
        ):
        loss = 0.5 * torch.sum(-log_var - 1 + mu ** 2 + log_var.exp(), dim=1)
        self.lambd = max(0.001, self.lambd)
        return self.lambd * torch.mean(loss)


class Loss:
    def __init__(self) -> None:
        self.KLDiv_criterion = KLDivLoss()
        self.ssim = SSIM().to('cuda')
        self.mse_loss = nn.MSELoss()
        self.bce = nn.BCELoss(reduce='sum')

    def compute_loss(self, x, output, mu, log_var):

        l2_loss = self.mse_loss(x, output)
        bce_loss = torch.mean(F.binary_cross_entropy(output, x, reduction='none'))
        ssim_loss = (1 - self.ssim(x, output))
        kldiv_loss = self.KLDiv_criterion(mu, log_var) 
        loss =  (l2_loss + bce_loss + ssim_loss + kldiv_loss) / len(x)
        
        return loss, l2_loss, bce_loss, ssim_loss, kldiv_loss


