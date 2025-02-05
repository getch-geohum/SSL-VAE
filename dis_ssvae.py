import numpy as np
import random
import torch
from torch import nn
from torchvision.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
)  # just resnet 50 is used as encoder backbone
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from mlp import MLP

class Projector(nn.Module):
    def __init__(self, c=256):
        self.c = c
        super(Projector, self).__init__()

    def forward(self, x):
        xx = torch.zeros(x.shape[0],self.c,self.c).to(x.device)
        for i in range(x.shape[0]):
            xx[i,:,:] = x[i]
        return xx


class DIS_SSVAE(nn.Module):
    def __init__(
        self,
        img_size,
        nb_channels,
        latent_img_size,
        z_dim,
        nb_dataset,
        beta=0.1,
        lr_scheduler=None,
        z_dim_constrained=2
    ):
        """ """
        super(DIS_SSVAE, self).__init__()

        #self.fill = Projector()
        self.img_size = img_size
        self.nb_channels = nb_channels
        self.latent_img_size = latent_img_size
        self.z_dim = z_dim
        self.beta = beta

        self.nb_conv = int(np.log2(img_size // latent_img_size))
        self.max_depth_conv = 2 ** (4 + self.nb_conv)
        print(f"Maximum depth conv: {self.max_depth_conv}")

        self.resnet = resnet34(pretrained=False)
        self.resnet_entry = nn.Sequential(
            nn.Conv2d(
                self.nb_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            ),
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
        )
        self.resnet_layer_list = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
        ]
        self.encoder_layers = [self.resnet_entry]
        for i in range(1, self.nb_conv):
            try:
                self.encoder_layers.append(self.resnet_layer_list[i - 1])
            except IndexError:
                depth_in = 2 ** (4 + i)
                depth_out = 2 ** (4 + i + 1)
                self.encoder_layers.append(
                    nn.Sequential(
                        nn.Conv2d(depth_in, depth_out, 4, 2, 1),
                        nn.BatchNorm2d(depth_out),
                        nn.ReLU(),
                    )
                )
        self.conv_encoder = nn.Sequential(
            *self.encoder_layers,
        )
        self.final_encoder = nn.Sequential(
            #    nn.Linear(self.max_depth_conv * self.latent_img_size ** 2,
            #        self.z_dim * 2)
            nn.Conv2d(
                self.max_depth_conv, self.z_dim * 2, kernel_size=1, stride=1, padding=0
            )
        )

        self.initial_decoder = nn.Sequential(
            # nn.Linear(self.z_dim, self.max_depth_conv * self.latent_img_size **
            #    2),
            # nn.BatchNorm1d(self.max_depth_conv * self.latent_img_size ** 2),
            nn.ConvTranspose2d(
                self.z_dim, self.max_depth_conv, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(self.max_depth_conv),
            nn.ReLU(),
        )

        nb_conv_dec = self.nb_conv

        self.decoder_layers = []
        for i in reversed(range(nb_conv_dec)):
            depth_in = 2 ** (4 + i + 1)
            depth_out = 2 ** (4 + i)
            if i == 0:
                depth_out = self.nb_channels
                self.decoder_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
                    )
                )
            else:
                self.decoder_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
                        nn.BatchNorm2d(depth_out),
                        nn.ReLU(),
                    )
                )
        self.conv_decoder = nn.Sequential(*self.decoder_layers)

        self.nb_dataset = nb_dataset
        self.z_dim_constrained = z_dim_constrained

        self.dis_mlp = MLP(
            self.z_dim_constrained * self.latent_img_size**2, [128],1)  # self.nb_dataset  or batch_size?
        # self.dis_cnn = nn.Sequential(
        #    nn.Linear(self.z_dim_constrained, self.z_dim_constrained *
        #        self.latent_img_size ** 2),
        #    #nn.BatchNorm1d(self.z_dim_constrained * self.latent_img_size ** 2),
        #    #nn.ReLU(),
        #    nn.Unflatten(1, (self.z_dim_constrained, self.latent_img_size,
        #        latent_img_size)),
        #    nn.ConvTranspose2d(self.z_dim_constrained, self.z_dim_constrained, 4, 2, 1),
        #    nn.BatchNorm2d(self.z_dim_constrained),
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(self.z_dim_constrained, self.z_dim_constrained, 4, 2, 1),
        #    nn.BatchNorm2d(self.z_dim_constrained),
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(self.z_dim_constrained, self.z_dim_constrained, 4, 2, 1),
        #    nn.BatchNorm2d(self.z_dim_constrained),
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(self.z_dim_constrained, 4, 4, 2, 1),
        #    #nn.BatchNorm2d(self.z_dim_constrained),
        #    #nn.ReLU(),
        #    #nn.ConvTranspose2d(self.z_dim_constrained, 1, 4, 2, 1),
        #    #nn.BatchNorm2d(self.z_dim_constrained),
        #    #nn.ReLU(),
        #    )

        # self.max_pooling_2d = torch.nn.MaxPool2d(4, 2, 1)

    def encoder(self, x):
        x = self.conv_encoder(x)
        # x = x.view(x.shape[0], -1) # NOTE that this is needed if Linear latent space
        x = self.final_encoder(x)
        #print(f'Encider shape-->: {x.shape}')
        return x[:, : self.z_dim], x[:, self.z_dim :]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(torch.mul(logvar, 0.5))
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decoder(self, z):
        print('latent space shape-->:', z.shape)
        z = self.initial_decoder(z)
        # z = z.view(z.shape[0], self.max_depth_conv, self.latent_img_size,
        #        self.latent_img_size) # NOTE that this is needed if Linear
        #    # latent space
        x = self.conv_decoder(z)
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        self.mu = mu
        self.logvar = logvar
        return self.decoder(z), (self.mu, self.logvar)

    def xent_continuous_ber(self, recon_x, x, gamma=1):
        """p(x_i|z_i) a continuous bernoulli"""
        eps = 1e-6

        def log_norm_const(x):
            # numerically stable computation
            x = torch.clamp(x, eps, 1 - eps)
            x = torch.where((x < 0.49) | (x > 0.51), x, 0.49 * torch.ones_like(x))
            return torch.log((2 * self.tarctanh(1 - 2 * x)) / (1 - 2 * x) + eps)
        x = gamma * x 
        recon_x = gamma * recon_x  # like if we multiplied the lambda ?
        return torch.mean(
            (
                x * torch.log(recon_x + eps)
                + (1 - x) * torch.log(1 - recon_x + eps)
                + log_norm_const(recon_x)
            ),
            dim=(1),  # it use to be (1)
        )
        #print(f'xent shape: {val.shape}')
    
    def xent_continuous_ber_vae(self, recon_x, x, pixelwise=False):
        ''' p(x_i|z_i) a continuous bernoulli '''
        eps = 1e-6
        def log_norm_const(x):
            # numerically stable computation
            x = torch.clamp(x, eps, 1 - eps)
            x = torch.where((x < 0.49) | (x > 0.51), x, 0.49 *
                    torch.ones_like(x))
            return torch.log((2 * self.tarctanh(1 - 2 * x)) /
                            (1 - 2 * x) + eps)
        if pixelwise:
            return torch.mean((x * torch.log(recon_x + eps) +
                            (1 - x) * torch.log(1 - recon_x + eps) +
                            log_norm_const(recon_x)), dim=(1,2,3)) # mean is aded and readjusted
        else:
            return torch.sum(x * torch.log(recon_x + eps) +
                            (1 - x) * torch.log(1 - recon_x + eps) +
                            log_norm_const(recon_x), dim=(1, 2, 3))
    
    
    def mean_from_lambda(self, l):
        """because the mean of a continuous bernoulli is not its lambda"""
        l = torch.clamp(l, 10e-6, 1 - 10e-6)
        l = torch.where((l < 0.49) | (l > 0.51), l, 0.49 * torch.ones_like(l))
        return l / (2 * l - 1) + 1 / (2 * self.tarctanh(1 - 2 * l))

    def kld(self, dataset_lbl, test=True):  # kld(self, dataset_lbl) # kld loss without mask
        # start_dim = (dataset_lbl * self.z_dim_constrained).long()
        # end_dim = ((dataset_lbl + 1) * self.z_dim_constrained).long()
        # seems hard to index with batched slice
        if test:
            return 0.5 * torch.mean(
                    -1
                    - self.logvar
                    + self.mu.pow(2)
                    + self.logvar.exp(),
                    dim=(1),
                )
        else:
            b, c, h, w = self.logvar.shape
            #print(f'b={b}, c={c}, h={h},w={w},shp={self.logvar.shape},zdim_cons={self.z_dim_constrained}')
            #print(f'logvar: {self.logvar.shape}, mu: {self.mu.shape}')
            logvar = self.logvar.reshape((b, self.nb_dataset, self.z_dim_constrained, h, w))
            b, c, h, w = self.mu.shape
            mu = self.mu.reshape((b, self.nb_dataset, self.z_dim_constrained, h, w))
            #print(f'logvar: {logvar.shape}, mu: {mu.shape}')

            mu = torch.mean(mu,dim=1)
            logvar = torch.mean(logvar, dim=1)


            #print(f'logvar: {logvar.shape}, mu: {mu.shape}')
            return 0.5 * torch.mean(
                -1
                - logvar
                + mu.pow(2)
                + logvar.exp(),
                dim=(1),
                )

        #return 0.5 * torch.mean(
        #    -1
        #    - logvar[:, dataset_lbl]
        #    + mu[:, dataset_lbl].pow(2)
        #    + logvar[:, dataset_lbl].exp(),
        #    dim=(1),
        #)

    def tarctanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def compute_loss(self, x, xm, Mm, Mn, P, recon_x, dataset_lbl):
        """x: original input
        xm: modified version of input x
        recon_x: recontructed image
        Mm: mask signifying modified regions
        Mn: Mask signifying normal areas
        """

        ### VAE ELBO
        # A beta coefficient that will be pixel wise and that will be bigger
        # for pixels of the mask which are very different between the original
        # and modified version of x
        #gamma = Mn  # self.beta + Mm * torch.abs(xm - x)
        #beta = Mn  # Mm * torch.abs(xm - x)
        #beta_inv = Mm
        #for i in range(self.nb_conv):
        #    beta = nn.functional.max_pool2d(beta, 2)
        #    beta_inv = nn.functional.max_pool2d(beta_inv, 2)
        #beta = torch.mean(beta, axis=1)[:, None]
        #beta_inv = torch.mean(beta_inv, axis=1)[:, None]

        base = 256 * 256
        lambda_ = 1
        #w_n = (
        #    torch.sum(Mn[:, 0, :, :], dim=(1, 2)) / base
        # )  # [batch_n,] weight to balance contribution from unmodified region
        #w_m = (
        #    torch.sum(Mm[:, 0, :, :], dim=(1, 2)) / base
        # )  # [batch_n,] weight to balance contribution from modified region

        #rec_normal = torch.mean(
        #    torch.mean(self.xent_continuous_ber(recon_x, x, gamma=Mn), dim=(1, 2)) * w_n
        # )
        #rec_modified = torch.mean(
        #    torch.mean(self.xent_continuous_ber(recon_x, x, gamma=Mm), dim=(1, 2)) * w_m
        # )
        #rec_term = (
        #    lambda_ * rec_normal - (1 - lambda_) * rec_modified
        # )  # just to follow Boers work

        #rec_term = torch.mean(
        #    torch.mean(self.xent_continuous_ber(recon_x, x), dim=(1, 2))
        #    )

        # based on the derivation from the overlef document

        #rec = self.xent_continuous_ber(recon_x,xm)
        ##print('rec shape after: ', rec.shape)
        #rec_term = torch.mean(torch.mean(Mn[:,0,:,:]*(P[:,0,:,:]+rec),dim=(1,2)) + (1-lambda_)*torch.mean(Mm[:,0,:,:]*(P[:,0,:,:]+rec),dim=(1,2)))
        
        #print(f'shape of normal and modified images: {Mm.shape} --> {Mn.shape}')

        #base = 256 * 256

        base = x.shape[0]
        lambda_ = 1 #      # start with lambda = 1, maybe modify it later
        w_n = (
            torch.sum(Mn[:, 0, :, :], dim=(1,2)) / base # use to be dim=(1,2)
         )  # [batch_n,]
        w_m = (
            torch.sum(Mm[:, 0, :, :], dim=(1,2)) / base # use to be dim=(1,2)
         )  # [batch_n,]

        rec_normal = torch.mean(self.xent_continuous_ber(recon_x, xm, gamma=Mn) + torch.log(w_n[:,None,None])) # , dim=(1, 2))
        rec_modified = torch.mean(self.xent_continuous_ber(recon_x, xm, gamma=Mm) + torch.log(1 - torch.clip(w_n[:,None,None],0.001,0.999))) #, dim=(1, 2))
        rec_term = (
                lambda_*rec_normal + (2 - lambda_) * rec_modified
        )
        # rec_term = rec_normal + rec_modified

        #print(f'normal loss: {rec_normal.item()}, modif loss: {rec_modified.item()}') # maxw: {w_n.max()},min:{w_n.min()}')
    

        kld = torch.mean(self.kld(dataset_lbl, test=False))

        # Can we imagine different beta for the constrained and unconstrained
        # dim ?
        beta = 0.0001
        L = rec_term - beta * kld

        ### DISENTANGLEMENT MODULE
        # NOTE y a til une maj des poids de l'encodeur ici ?
        dl = torch.zeros_like(kld)


        #print('-->Label size<-- ', dataset_lbl.shape)

        #dis_loss = nn.MSELoss(reduction="mean")#nn.CrossEntropyLoss(reduction="mean")
        #dl = dis_loss(
        #        self.dis_mlp(
        #            torch.reshape(
        #                self.mu[:, : self.z_dim_constrained, :, :],
        #                (self.mu.shape[0], -1))), dataset_lbl.float().reshape(-1,1))
        #self.dis_cnn(
        #        self.mu[:, :self.z_dim_constrained]
        #        ),
        #    self.max_pooling_2d(Mm[:, :1] * x)
        #)

        loss = L - dl # 1 * dl

        loss_dict = {
            "loss": loss,
            "rec_term": rec_term,
            "kld": kld,
            "beta*kld": beta * kld,
            "dis_loss": dl,
        }

        return loss, loss_dict

    def step(self, inputs):  # inputs contain, modified image, normal image and mask
        X, Xm, M, Mn, proba, dataset_lbl = inputs
        rec, _ = self.forward(Xm)
        #print('x shape: ', X.shape)
        #print('xm shape: ', Xm.shape)
        #print('M shape: ', M.shape)
        #print('P shape: ', proba.shape)
        #print('rec shape: ', rec.shape)


        loss, loss_dict = self.compute_loss(X, Xm, M, Mn, proba, rec, dataset_lbl)

        rec = self.mean_from_lambda(rec)

        return loss, rec, loss_dict
