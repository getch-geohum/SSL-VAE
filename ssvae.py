import numpy as np
import random
import torch
from torch import nn
from torchvision.models.resnet import resnet18, resnet34, resnet50 # just resnet 50 is used as encoder backbone
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class SSVAE(nn.Module):
    
    def __init__(self, img_size, nb_channels, latent_img_size, z_dim, beta=0.1):
        '''
        '''
        super(SSVAE, self).__init__()

        self.img_size = img_size
        self.nb_channels = nb_channels
        self.latent_img_size = latent_img_size
        self.z_dim = z_dim
        self.beta = beta

        self.nb_conv = int(np.log2(img_size // latent_img_size))
        self.max_depth_conv = 2 ** (4 + self.nb_conv)
        print(f'Maximum depth conv: {self.max_depth_conv}')
        
        self.resnet = resnet34(pretrained=False) # resnet18(pretrained=False)
        self.resnet_entry = nn.Sequential(
            nn.Conv2d(self.nb_channels, 64, kernel_size=7,
                stride=2, padding=3, bias=False),
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.resnet_layer_list = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4 
        ]
        self.encoder_layers = [self.resnet_entry] 
        for i in range(1, self.nb_conv): 
            try:
                self.encoder_layers.append(self.resnet_layer_list[i - 1])
            except IndexError: 
                depth_in = 2 ** (4 + i)
                depth_out = 2 ** (4 + i + 1)
                self.encoder_layers.append(nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, 4, 2, 1),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU()
                    ))
        self.conv_encoder = nn.Sequential(
            *self.encoder_layers,
        )
        self.final_encoder = nn.Sequential(
            nn.Conv2d(128, self.z_dim * 2, kernel_size=1,
            stride=1, padding=0)
        ) # self.max_depth_conv  repace 2048   128

        self.initial_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, self.max_depth_conv,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.max_depth_conv),
            nn.ReLU()
        )
            
        nb_conv_dec = self.nb_conv

        self.decoder_layers = []
        for i in reversed(range(nb_conv_dec)):
            depth_in = 2 ** (4 + i + 1)
            depth_out = 2 ** (4 + i)
            if i == 0:
                depth_out = self.nb_channels
                self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
                ))
            else:
                self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU()
                ))
        self.conv_decoder = nn.Sequential(
            *self.decoder_layers
        )

    def encoder(self, x):
        x = self.conv_encoder(x)
        x = self.final_encoder(x)
        return x[:, :self.z_dim], x[:, self.z_dim:]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(torch.mul(logvar, 0.5))
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu
    
    def decoder(self, z):
        z = self.initial_decoder(z)
        x = self.conv_decoder(z)
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        self.mu = mu
        self.logvar = logvar
        return self.decoder(z), (mu, logvar)

    def xent_continuous_ber(self, recon_x, x, gamma=1):
        ''' p(x_i|z_i) a continuous bernoulli '''
        eps = 1e-6
        def log_norm_const(x):
            # numerically stable computation
            x = torch.clamp(x, eps, 1 - eps)
            x = torch.where((x < 0.49) | (x > 0.51), x, 0.49 *
                    torch.ones_like(x))
            return torch.log((2 * self.tarctanh(1 - 2 * x)) /
                            (1 - 2 * x) + eps)
        
        recon_x = gamma * recon_x # like if we multiplied the lambda ?
        return torch.sum(
                    (x * torch.log(recon_x + eps) +
                    (1 - x) * torch.log(1 - recon_x + eps) +
                    log_norm_const(recon_x)),
                    dim=(1)   # it use to be (1)
                )

    def mean_from_lambda(self, l):
        ''' because the mean of a continuous bernoulli is not its lambda '''
        l = torch.clamp(l, 10e-6, 1 - 10e-6)
        l = torch.where((l < 0.49) | (l > 0.51), l, 0.49 *
            torch.ones_like(l))
        return l / (2 * l - 1) + 1 / (2 * self.tarctanh(1 - 2 * l))

    def kld_m(self, beta=1): # kld loss with mask
        # NOTE -kld actually
        mu_ = self.mu.pow(2) * beta
        logvar_ = self.logvar * beta
        return 0.5 * torch.sum(
                (1 + logvar_ - mu_ - logvar_.exp()),
            dim=(1)#, 2, 3)
        )

    def kld(self): # kld loss without mask
        return 0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp(),dim=(1))


    def tarctanh(self, x):
        return 0.5 * torch.log((1+x)/(1-x))

    def compute_loss(self, x, xm, Mm, Mn, recon_x):
        '''x: original input
            xm: modified version of input x
            recon_x: recontructed image
            Mm: mask signifying modified regions
            Mn: Mask signifying normal areas
        '''
        # A beta coefficient that will be pixel wise and that will be bigger
        # for pixels of the mask which are very different between the original
        # and modified version of x
        gamma = Mn #self.beta + Mm * torch.abs(xm - x)
        beta = Mn # Mm * torch.abs(xm - x)
        beta_inv = Mm
        for i in range(self.nb_conv):
            beta = nn.functional.max_pool2d(beta, 2)
            beta_inv = nn.functional.max_pool2d(beta_inv, 2)
        beta = torch.mean(beta, axis=1)[:, None]
        beta_inv = torch.mean(beta_inv, axis=1)[:, None]
        #print(f'beta shape: {beta.shape}')
        #plt.imshow(beta[0, 0].cpu().numpy())
        #plt.savefig("mask.png")
        #fs
        
        
        #rec_term = self.xent_continuous_ber(recon_x, x, gamma=gamma)
        #rec_term = torch.mean(rec_term) # mean over the batch
        #kld = torch.mean(self.kld_m(beta=beta)) # mean over the batch
        #kld_inv = torch.mean(self.kld_m(beta=beta_inv))

        base = 256*256
        lambda_ = 0.9
        w_n = torch.sum(Mn[:,0,:,:],dim=(1,2))/base # [batch_n,] weight to balance contribution from unmodified region
        w_m = torch.sum(Mm[:,0,:,:],dim=(1,2))/base # [batch_n,] weight to balance contribution from modified region
        rec_normal = torch.mean(torch.mean(self.xent_continuous_ber(recon_x, x, gamma=Mn),dim=(1,2))*w_n)
        rec_modified = torch.mean(torch.mean(self.xent_continuous_ber(recon_x, x, gamma=Mm),dim=(1,2))*w_m)
        rec_term = lambda_*rec_normal-(1-lambda_)*rec_modified  # just to follow Boers work
        kld = torch.mean(self.kld())
        
        L = (rec_term + 0.0001 * kld)

        loss = L

        loss_dict = {
            'loss': loss,
            'rec_term': rec_term,
            'beta*kld': kld
        }
        #print(loss_dict)

        return loss, loss_dict
    
    def new_rec_loss(self, x, xm, Mm, Mn, recon_x, lamda=0.1):# trested but not used 
        '''x: original input
        xm: modified version of input x
        recon_x: recontructed image
        Mm: mask signifying modified regions
        Mn: Mask signifying normal areas
        '''
        rec_dif = recon_x-x   # Xrec-X
        mod_dif = xm-x        # Xm-X
        mod_dif_abs = torch.abs(mod_dif)   # |Xm-X|
        
        
        top = Mn*rec_dif
        buttom_ = Mm*rec_dif
        buttom = mod_dif_abs*rec_dif
        a = (lamda/torch.norm(Mn,1))*torch.norm(top,2)
        b = ((1-lamda)/torch.norm(buttom_,1))*torch.norm(Mm*buttom,2)
        loss = a-b

        return loss


    def step(self, inputs): # inputs contain, modified image, normal image and mask
        X, Xm, M, Mn = inputs
        rec, _ = self.forward(Xm)
        loss, loss_dict = self.compute_loss(x=X, xm=Xm, Mm=M, Mn=Mn, recon_x=rec)

        rec = self.mean_from_lambda(rec)

        return loss, rec, loss_dict
