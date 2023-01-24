import numpy as np
import random
import torch
from torch import nn
from torchvision.models.resnet import resnet18, resnet50 # just resnet 50 is used as encoder backbone
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SSAE(nn.Module):
    
    def __init__(self, img_size, nb_channels, latent_img_size, z_dim, lamda=0.5):
        '''
        '''
        super(SSAE, self).__init__()

        self.img_size = img_size
        self.nb_channels = nb_channels
        self.latent_img_size = latent_img_size
        self.z_dim = z_dim
        self.lamda = lamda

        self.nb_conv = int(np.log2(img_size // latent_img_size))
        self.max_depth_conv = 2 ** (4 + self.nb_conv)
        print(f'Maximum depth conv: {self.max_depth_conv}')
        
        self.resnet = resnet50(pretrained=False) # resnet18(pretrained=False)
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
            nn.Conv2d(2048, self.z_dim, kernel_size=1,
            stride=1, padding=0)
        ) # self.max_depth_conv  repace 2048

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
        return x 
    
    def decoder(self, z):
        z = self.initial_decoder(z)
        x = self.conv_decoder(z)
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return  out # self.decoder(z), (mu, logvar)
    
    def compute_loss(self, x, xm, Mm, Mn, recon_x):
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

        a = (self.lamda/torch.norm(Mn,1))*torch.norm(top,2)
        b = ((1-self.lamda)/torch.norm(buttom_,1))*torch.norm(Mm*buttom,2) 
        loss = a-b

        print(f'{a.item()}--> {b.item()}--> {loss.item()}')
        
        loss_dict = {'normal':a.mean(), 'modified':b.mean(), 'total':loss.mean()}

        return loss.mean(), loss_dict
    
    def step(self, inputs): # inputs contain, modified image, normal image and mask
        X, Xm, M, Mn = inputs
        rec = self.forward(Xm)
        loss, loss_dict = self.compute_loss(x=X, xm=Xm, Mm=M, Mn=Mn, recon_x=rec)
        return loss, rec, loss_dict
