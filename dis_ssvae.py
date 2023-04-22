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


class DIS_SSVAE(nn.Module):
    def __init__(
        self, img_size, nb_channels, latent_img_size, z_dim, nb_dataset, beta=0.1
    ):
        """ """
        super(DIS_SSVAE, self).__init__()

        self.img_size = img_size
        self.nb_channels = nb_channels
        self.latent_img_size = latent_img_size
        self.z_dim = z_dim
        self.beta = beta

        self.nb_conv = int(np.log2(img_size // latent_img_size))
        self.max_depth_conv = 2 ** (4 + self.nb_conv)
        print(f"Maximum depth conv: {self.max_depth_conv}")

        self.resnet = resnet18(pretrained=False)
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
        self.z_dim_constrained = 2

        # self.dis_mlp = MLP(
        #    self.z_dim_constrained * self.latent_img_size**2, [128], self.nb_dataset
        # )
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
        return x[:, : self.z_dim], x[:, self.z_dim :]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(torch.mul(logvar, 0.5))
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decoder(self, z):
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

        recon_x = gamma * recon_x  # like if we multiplied the lambda ?
        return torch.mean(
            (
                x * torch.log(recon_x + eps)
                + (1 - x) * torch.log(1 - recon_x + eps)
                + log_norm_const(recon_x)
            ),
            dim=(1),  # it use to be (1)
        )

    def mean_from_lambda(self, l):
        """because the mean of a continuous bernoulli is not its lambda"""
        l = torch.clamp(l, 10e-6, 1 - 10e-6)
        l = torch.where((l < 0.49) | (l > 0.51), l, 0.49 * torch.ones_like(l))
        return l / (2 * l - 1) + 1 / (2 * self.tarctanh(1 - 2 * l))

    def kld(self, dataset_lbl):  # kld loss without mask
        # start_dim = (dataset_lbl * self.z_dim_constrained).long()
        # end_dim = ((dataset_lbl + 1) * self.z_dim_constrained).long()
        # seems hard to index with batched slice
        b, c, h, w = self.logvar.shape
        logvar = self.logvar.reshape((b, self.nb_dataset, self.z_dim_constrained, h, w))
        b, c, h, w = self.mu.shape
        mu = self.mu.reshape((b, self.nb_dataset, self.z_dim_constrained, h, w))
        return 0.5 * torch.mean(
            -1
            - logvar[:, dataset_lbl]
            + mu[:, dataset_lbl].pow(2)
            + logvar[:, dataset_lbl].exp(),
            dim=(1),
        )

    def tarctanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def compute_loss(self, x, xm, Mm, Mn, recon_x, dataset_lbl):
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
        # gamma = Mn  # self.beta + Mm * torch.abs(xm - x)
        # beta = Mn  # Mm * torch.abs(xm - x)
        # beta_inv = Mm
        # for i in range(self.nb_conv):
        #    beta = nn.functional.max_pool2d(beta, 2)
        #    beta_inv = nn.functional.max_pool2d(beta_inv, 2)
        # beta = torch.mean(beta, axis=1)[:, None]
        # beta_inv = torch.mean(beta_inv, axis=1)[:, None]

        # base = 256 * 256
        # lambda_ = 0.9
        # w_n = (
        #    torch.sum(Mn[:, 0, :, :], dim=(1, 2)) / base
        # )  # [batch_n,] weight to balance contribution from unmodified region
        # w_m = (
        #    torch.sum(Mm[:, 0, :, :], dim=(1, 2)) / base
        # )  # [batch_n,] weight to balance contribution from modified region

        # rec_normal = torch.mean(
        #    torch.mean(self.xent_continuous_ber(recon_x, x, gamma=Mn), dim=(1, 2)) * w_n
        # )
        # rec_modified = torch.mean(
        #    torch.mean(self.xent_continuous_ber(recon_x, x, gamma=Mm), dim=(1, 2)) * w_m
        # )
        # rec_term = (
        #    lambda_ * rec_normal + (1 - lambda_) * rec_modified
        # )  # just to follow Boers work

        rec_term = torch.mean(
            torch.mean(self.xent_continuous_ber(recon_x, x), dim=(1, 2))
        )
        kld = torch.mean(self.kld(dataset_lbl))

        # Can we imagine different beta for the constrained and unconstrained
        # dim ?
        beta = 0.0001
        L = rec_term - beta * kld

        ### DISENTANGLEMENT MODULE
        # NOTE y a til une maj des poids de l'encodeur ici ?
        dl = torch.zeros_like(kld)
        # dis_loss = nn.MSELoss(reduction="mean")#nn.CrossEntropyLoss(reduction="mean")
        # dl = dis_loss(
        #    #self.dis_mlp(
        #    #    torch.reshape(
        #    #        self.mu[:, : self.z_dim_constrained, :, :],
        #    #        (self.mu.shape[0], -1),
        #    #    )
        #    #),
        #    #dataset_lbl,
        #    self.dis_cnn(
        #        self.mu[:, :self.z_dim_constrained]
        #        ),
        #    self.max_pooling_2d(Mm[:, :1] * x)
        # )

        loss = L - 1 * dl

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

        loss, loss_dict = self.compute_loss(X, Xm, M, Mn, rec, dataset_lbl)

        rec = self.mean_from_lambda(rec)

        return loss, rec, loss_dict
