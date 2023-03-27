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
import copy

class SS_CVAE(nn.Module):
    
    def __init__(self, img_size, nb_channels, latent_img_size, z_dim, mask_nb_channel=4, with_condition=True):
        '''
        '''
        super(SS_CVAE, self).__init__()

        self.img_size = img_size
        self.mask_nb_channel = mask_nb_channel  # by default this would be the same as the number of channels of input image
        self.nb_channels = nb_channels          # because the mask(y) will be be ingested to network
        self.latent_img_size = latent_img_size
        self.z_dim = z_dim

        self.nb_conv = int(np.log2(img_size // latent_img_size))
        self.max_depth_conv = 2 ** (4 + self.nb_conv)
        print(f'Maximum depth conv: {self.max_depth_conv}')
        
        if with_condition:
            self.exit_nb_channel = self.nb_channels - 1 # self.mask_nb_channel   # changed part
        else:
            self.exit_nb_channel = self.nb_channels
        self.code_nb_channel = self.z_dim + self.mask_nb_channel # number of channels when we concatenate q(z|x,y) with y

        self.resnet = resnet34(weights=None) # resnet18(pretrained=False)
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
        ) # self.max_depth_conv  repace 2048 128 256

        self.initial_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, self.max_depth_conv,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.max_depth_conv),
            nn.ReLU()
        ) # self.z_dim is changed by self.code_nb_channel
            
        nb_conv_dec = self.nb_conv

        self.decoder_layers = []
        for i in reversed(range(nb_conv_dec)):
            depth_in = 2 ** (4 + i + 1)
            depth_out = 2 ** (4 + i)
            if i == 0:
                depth_out = self.exit_nb_channel  # self.entry_nb_channel
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
        # x = torch.cat((x,y), dim=1)
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
    
    def decoder(self, z): # add how to input y to the decoder
        # y_copy = copy.deepcopy(y)  # to inegst y to gether with z
        # for i in range(self.nb_conv):
        #     y_copy = nn.functional.max_pool2d(y_copy, 2)   # by assuming maxpooling is deterministic function
        # z = torch.cat((z,y_copy), dim=1)
        z = self.initial_decoder(z)
        x = self.conv_decoder(z)
        x = nn.Sigmoid()(x)  # this is p(x|y, zl) 
        return x

    def forward(self, x): # add how to include y to the saystem
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)  # this is q(z|x,y)
        self.mu = mu
        self.logvar = logvar
        return self.decoder(z), (mu, logvar)  # q(x|z,q(z|x,y))

    def xent_continuous_ber(self, recon_x, x):   # remove gamma may be
        ''' p(x_i|z_i) a continuous bernoulli '''
        eps = 1e-6
        def log_norm_const(x):
            # numerically stable computation
            x = torch.clamp(x, eps, 1 - eps)
            x = torch.where((x < 0.49) | (x > 0.51), x, 0.49 *
                    torch.ones_like(x))
            return torch.log((2 * self.tarctanh(1 - 2 * x)) /
                            (1 - 2 * x) + eps)
        
        return x * torch.log(recon_x + eps) + (1 - x) * torch.log(1 - recon_x + eps) + log_norm_const(recon_x)

    def mean_from_lambda(self, l):
        ''' because the mean of a continuous bernoulli is not its lambda '''
        l = torch.clamp(l, 10e-6, 1 - 10e-6)
        l = torch.where((l < 0.49) | (l > 0.51), l, 0.49 *
            torch.ones_like(l))
        return l / (2 * l - 1) + 1 / (2 * self.tarctanh(1 - 2 * l))


    def kld(self): # kld loss without mask
        return 0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp(),dim=(1,2,3))


    def tarctanh(self, x):
        return 0.5 * torch.log((1+x)/(1-x))

    def compute_loss(self,X, Mm, Mn, prob, recon_x, beta):
        '''x: modified input image, in Bauers paper both modified and normal image for loss computation |Rec_x-X|
            recon_x: recontructed image
            Mm: mask signifying modified regions
            Mn: Mask signifying normal areas
            prob: probability of a pixel being 1 or 0, which is randomly generated values within [0,1]
            as all pixels have equali probability of being 1 or 0
        '''

        #base = 256 * 256
        #w_n = torch.sum(Mn[:, 0, :, :], dim=(1,2)) / base  # [batch_n,] weight
        #w_m = torch.sum(Mm[:, 0, :, :], dim=(1,2)) / base

        P  = torch.where(Mm==0, torch.log((1-prob)), torch.log(prob))
        rec = self.xent_continuous_ber(recon_x, X)

        rec_raw = torch.sum(Mn*(P+rec),dim=(1,2,3)) + torch.sum(Mm*(P+rec),dim=(1,2,3))  # the norm is computed on channel dim
        #rec_raw = torch.mean(torch.sum(Mn*(rec+P), dim=(1)), dim=(1,2))*w_n + torch.mean(torch.sum(Mm * (rec+P), dim=(1)),dim=(1,2))*w_m  # ton channel dim

        #rec_term = torch.mean(rec_raw)
        #kld = torch.mean(self.kld())

        #L = rec_term + beta*kld # this should be discussed

        # L = torch.mean(rec_term + beta*self.kld())

        kld = self.kld()

        L = torch.mean(rec_raw + beta*kld)


        loss = L


        loss_dict = {
            'loss': loss,
            'rec_term': torch.mean(rec_raw),
            'kld': torch.mean(kld),
            'beta*kld':torch.mean(beta*kld)
        } # the key is left not to modif entire workflow

        return loss, loss_dict
    

    def step(self, inputs, beta): # inputs contain, modified image, normal image and mask
        X, Xm, Mm, Mn, prob = inputs

        #print(f'shape of X: {X.shape}')
        #print(f'shape of Xm: {Xm.shape}')
        #print(f'shape of Mm: {Mm.shape}')
        #print(f'shape of Mn: {Mn.shape}')
        #print(f'shape of prob: {prob.shape}')

        rec, _ = self.forward(Xm)
        #print(f'shape of rec: {rec.shape}')

        loss, loss_dict = self.compute_loss(X=Xm[:,:self.exit_nb_channel,:,:], Mm=Mm[:,:self.exit_nb_channel,:,:], Mn=Mn[:,:self.exit_nb_channel,:,:], prob=prob[:,:self.exit_nb_channel,:,:], recon_x=rec, beta=beta)  # 

        rec = self.mean_from_lambda(rec)

        return loss, rec, loss_dict

