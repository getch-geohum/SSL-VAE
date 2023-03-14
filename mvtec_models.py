from torch import nn
import torch

class DiskDown(nn.Module):
    def __init__(self, inc, outc, k = 3):
        super(DiskDown, self).__init__()
        self.inc = inc
        self.outc = outc
        self.k = k
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=k, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=k, stride=1, padding=1, dilation=1)
        self.batch_norm = nn.BatchNorm2d(num_features=outc)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0, dilation=1)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x
    
class SDC(nn.Module):
    def __init__(self, inch=128, outc=64, c=5):
        super(SDC, self).__init__()
        self.inch = inch
        self.outc = outc
        self.conv1 = nn.Conv2d(in_channels=self.inch, out_channels=self.outc, kernel_size=5, stride=1, dilation=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=self.inch, out_channels=self.outc, kernel_size=5, stride=1, dilation=2, padding=4)
        self.conv4 =  nn.Conv2d(in_channels=self.inch, out_channels=self.outc, kernel_size=5, stride=1, dilation=4, padding=8)
        self.conv8 =  nn.Conv2d(in_channels=self.inch, out_channels=self.outc, kernel_size=5, stride=1, dilation=8, padding=16)
        self.conv16 =  nn.Conv2d(in_channels=self.inch, out_channels=self.outc, kernel_size=5, stride=1, dilation=16, padding=32)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x4 = self.conv4(x)
        x8 = self.conv8(x)
        x16 = self.conv16(x)
        return torch.cat([x1, x2, x4, x8, x16], dim=1)

class DiskUP(nn.Module):
    def __init__(self, inc, outc, final_outc):
        super(DiskUP, self).__init__()
        self.inc = inc
        self.outc = outc
        self.final_outc = final_outc
        self.transConv = nn.ConvTranspose2d(in_channels=self.inc, out_channels=self.outc, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(in_channels=self.outc, out_channels=self.outc, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.outc, out_channels=self.final_outc, kernel_size=3, stride=1, dilation=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=self.outc)
        self.batch_norm2 = nn.BatchNorm2d(num_features=self.final_outc)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        x = self.transConv(x)
        #print(x.shape)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x) 
        return x
    

class Encoder(nn.Module):
    def __init__(self, z_dim=32):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.disk1 =  DiskDown(inc=3, outc=32, k = 3)
        self.disk2 = DiskDown(inc=32, outc=64, k = 3)
        self.disk3 = DiskDown(inc=64, outc=128, k = 3)
        
        self.sdc1 = SDC(inch=128, outc=64, c=5)
        self.sdc2 = SDC(inch=320, outc=64, c=5)
        self.sdc3 = SDC(inch=320, outc=64, c=5)
        self.sdc4 = SDC(inch=320, outc=64, c=5)
        self.z_conv = nn.Conv2d(in_channels=320, out_channels=self.z_dim, kernel_size=3, stride=1, padding=1)
            
    def forward(self, x):
        x = self.disk1(x)
        x = self.disk2(x)
        x = self.disk3(x)
        
        x = self.sdc1(x) 
        x = self.sdc2(x)
        x = self.sdc3(x) 
        x = self.sdc4(x)
        
        x = self.z_conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, z_dim=32):
        super(Decoder, self).__init__()
        self.updisk1 = DiskUP(inc=z_dim, outc=256, final_outc=128)
        self.updisk2 = DiskUP(inc=128, outc=128, final_outc=64)
        self.updisk3 = DiskUP(inc=64, outc=64, final_outc=32)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.updisk1(x)
        x = self.updisk2(x)
        x = self.updisk3(x) 
        x = self.final_conv(x)
        return x


######################## Self-supervised AE for Mvtec dataset ###################
class SS_AEmvtec(nn.Module):
    def __init__(self, zdim=256, lamda=0.9):
        super(SS_AEmvtec, self).__init__()
        self.encoder = Encoder(z_dim=zdim)
        self.decoder = Decoder(z_dim=zdim)
        self.lamda = lamda
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def compute_loss(self, x, xm, Mm, Mn, recon_x, beta=None):
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
        b = ((1-self.lamda)/torch.norm(buttom_,1))*torch.norm(buttom,2)  # Mm*buttom = buttom
        loss = a-b

        print(f'{a.item()}--> {b.item()}--> {loss.item()}')

        loss_dict = {'normal':a.mean(), 'modified':b.mean(), 'total':loss.mean()}

        return loss.mean(), loss_dict
    
    def step(self, inputs, beta=None):
        X, Xm, M, Mn = inputs
        rec = self.forward(Xm)
        loss, loss_dict = self.compute_loss(x=X, xm=Xm, Mm=M, Mn=Mn, recon_x=rec)
        return loss, rec, loss_dict



##############Self-supervised conditional VAE for MvTech dataset #####################################
class SS_CVAEmvtec(nn.Module):
    def __init__(self, zdim=256):
        super(SS_CVAEmvtec, self).__init__()
        self.encoder = Encoder(z_dim=zdim*2)
        self.decoder = Decoder(z_dim=zdim)
        #self.training = training

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(torch.mul(logvar, 0.5))
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu
        
    def forward(self, x):
        outs = self.encoder(x)
        mu, logvar = outs[:,:self.zdim,:], outs[:,self.zdim:,:]
        self.mu = mu
        self.logvar = logvar
        z = self.reparameterize(mu, logvar)  
        return self.decoder(z), (mu, logvar)
    
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

    def compute_loss(self,X, Mm, Mn, prob, recon_x, beta=None):
        '''x: modified input image, in Bauers paper both modified and normal image for loss computation |Rec_x-X|
            recon_x: recontructed image
            Mm: mask signifying modified regions
            Mn: Mask signifying normal areas
            prob: probability of a pixel being 1 or 0, which is randomly generated values within [0,1]
            as all pixels have equali probability of being 1 or 0
        '''
        
        P  = torch.where(Mm==0, torch.log((1-prob)), torch.log(prob))
        rec = self.xent_continuous_ber(recon_x, X)

        rec_raw = torch.sum(Mn*(P+rec),dim=(1,2,3)) + torch.sum(Mm*(P+rec),dim=(1,2,3))  # the norm is computed on channel dim

        rec_term = torch.mean(rec_raw)
        kld = torch.mean(self.kld())

        L = sum([rec_term, beta*kld])

       
        loss = L
        
        loss_dict = {
            'loss': loss,
            'rec_term': rec_term,
            'beta*kld': kld
        } 

        return loss, loss_dict
    

    def step(self, inputs, beta=None): # inputs contain, modified image, normal image and mask
        X, Xm, Mm, Mn, prob = inputs
        rec, _ = self.forward(Xm)
        loss, loss_dict = self.compute_loss(X=Xm, Mm=Mm, Mn=Mn, prob=prob, recon_x=rec, beta=beta)
        rec = self.mean_from_lambda(rec)
        return loss, rec, loss_dict
