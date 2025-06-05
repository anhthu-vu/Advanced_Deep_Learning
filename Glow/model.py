from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
from tqdm import tqdm
import time
import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ActNorm2d(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.nchannels = nchannels
        self.bias = nn.Parameter(torch.zeros((1, self.nchannels, 1, 1)))
        self.scale = nn.Parameter(torch.ones((1, self.nchannels, 1, 1)))
        self.initialization = False

    def encoder(self, x):
        batch_size, _, height, width = x.shape
        
        if not self.initialization:
            with torch.no_grad():
                mean = torch.mean(x.transpose(0, 1).reshape(self.nchannels, -1), dim=-1)
                std = torch.std(x.transpose(0, 1).reshape(self.nchannels, -1), dim=-1)

            self.bias.data.copy_(mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
            self.scale.data.copy_((std+1e-8).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
            self.initialization = True
            
        z = (x - self.bias)/self.scale
        log_Jacob = -height*width*torch.sum(torch.log(torch.abs(self.scale)))*torch.ones(batch_size, device=x.device)
        
        return [z], log_Jacob, None

    def decoder(self, z):
        x = z*self.scale + self.bias
        
        return x


class Conv2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        
        # Decompose Q in P (L + Id) (S + U)
        P, L, U = torch.lu_unpack(*Q.lu())
        
        # Not optimizized
        self.P = nn.Parameter(P, requires_grad=False)

        # Lower triangular
        self.L = nn.Parameter(L)

        # Diagonal
        self.S = nn.Parameter(U.diag()) 

        self.U = nn.Parameter(torch.triu(U, diagonal=1)) 

    def _assemble_W(self):
        """Computes W from P, L, S and U"""

        # Exclude the diagonal
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=self.L.device))

        # Exclude the diagonal
        U = torch.triu(self.U, diagonal=1)

        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def encoder(self, x):
        batch_size, _, height, width = x.shape
        w = self._assemble_W()
        y = torch.matmul(w.unsqueeze(0), x.reshape(batch_size, self.dim, -1)).reshape(batch_size, self.dim, height, width)
        log_Jacob = height*width*torch.sum(torch.log(torch.abs(self.S)))*torch.ones(batch_size, device=x.device)

        return [y], log_Jacob, None

    def decoder(self, y):
        batch_size, _, height, width = y.shape
        w = self._assemble_W()
        w_inverse = torch.inverse(w)
        x = torch.matmul(w_inverse.unsqueeze(0), y.reshape(batch_size, self.dim, -1)).reshape(batch_size, self.dim, height, width)
        
        return x


class ZeroConv2d(nn.Module): # See 3.3. Zero initialization in the original paper
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        
    def forward(self, x):
        return self.conv(x)
        

class NN2d(nn.Module): # See the first part of 5. Quantitative experiments in the original paper
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1, padding=0),
            nn.ReLU(),
            ZeroConv2d(512, in_channels*2),
        )

    def forward(self, x):
        return self.net(x)

    
class AffineCoupling2d(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.nchannels = nchannels
        self.scale_shift = NN2d(self.nchannels//2)

    def encoder(self, x):
        batch_size = x.shape[0]

        x_a, x_b = x.chunk(2, 1)
        scale_shift = self.scale_shift(x_b)
        scale, shift = scale_shift.chunk(2, 1)
        # For numerical stability, 
        # see https://github.com/openai/glow/blob/master/model.py#L376 (original implementation)
        scale = nn.functional.sigmoid(scale + 2.) 
        x_a = x_a * scale + shift
        log_Jacob = torch.sum(torch.log(torch.abs(scale)).reshape(batch_size, -1), dim=-1)
        
        return [torch.cat([x_a, x_b], dim=1)], log_Jacob, None
        
    def decoder(self, z):
        z_a, z_b = z.chunk(2, 1)
        scale_shift = self.scale_shift(z_b)
        scale, shift = scale_shift.chunk(2, 1)
        scale = nn.functional.sigmoid(scale + 2.)
        z_a = (z_a - shift)/(scale+1e-8)
        
        return torch.cat([z_a, z_b], dim=1)


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def encoder(self, x):
        batch_size, nchannels, height, width = x.shape
        x = x.reshape(batch_size, 
                      nchannels, 
                      height//2, 2,
                      width//2, 2
                     )
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(batch_size, nchannels*4,  height//2, width//2)
        
        return [x], None, None

    def decoder(self, z):
        batch_size, nchannels, height, width = z.shape
        z = z.reshape(batch_size, nchannels//4, 2, 2, height, width)
        z = z.permute(0, 1, 4, 2, 5, 3)
        z = z.reshape(batch_size, nchannels//4, height*2, width*2)

        return z


class Split(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.nchannels = nchannels 
        self.prior = ZeroConv2d(self.nchannels//2, self.nchannels) # Learn the prior distribution

    def encoder(self, x):
        z, x_new = x.chunk(2, 1)
        prior = self.prior(x_new)
        
        return [z, x_new], None, prior

    def decoder(self, z):
        prior = self.prior(z)
        log_std, mean = prior.chunk(2, 1)
        z_new = torch.randn(mean.shape, device=DEVICE) * torch.exp(log_std) + mean

        return torch.cat([z_new, z], dim=1)

    def reconstruct(self, z_latent, z):
        """
        Reconstruct image from the outputs of the encoder:
            - z_latent: the ouput that is not passed to the next layer 
            - z: the ouput that is passed to the next layer
        """
        return torch.cat([z_latent, z], dim=1)


class GlowModel(nn.Module):
    def __init__(self, *flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        
        # Learn the prior distribution
        nchannels = self.flows[-1].nchannels
        self.top_prior = ZeroConv2d(nchannels, nchannels*2)
        
    def encoder(self, x):
        m = x.shape[0]
        logdet = torch.zeros(m, device=x.device)
        prior = torch.zeros(m, device=x.device)
        zs = [] # contains the latent outputs of Split layers (those that are not passed to the next layers) and the whole model
        for i, module in enumerate(self.flows):
            gx, _logdet, _prior = module.encoder(x)

            if _logdet is not None:
                logdet += _logdet
                
            if _prior is not None:
                zs.append(gx[0])
                log_std, mean = _prior.chunk(2, 1)
                prior += (-log_std - 0.5*torch.log(2*torch.tensor(torch.pi)) - 0.5*(gx[0]-mean)**2/torch.exp(2*log_std)).reshape(m, -1).sum(dim=-1)

            x = gx[-1]
            if i == len(self.flows) - 1:
                zs.append(gx[-1])
            
        zl = zs[-1]
        top_prior = self.top_prior(torch.zeros(zl.shape, device=DEVICE))
        log_std, mean = top_prior.chunk(2, 1)
        prior += (-log_std - 0.5*torch.log(2*torch.tensor(torch.pi)) - 0.5*(zl-mean)**2/torch.exp(2*log_std)).reshape(zl.shape[0], -1).sum(dim=-1)
            
        return zs, logdet, prior

    def decoder(self, y):
        for module in reversed(self.flows):
            y = module.decoder(y)
        return y

    def reconstruct(self, zs):
        """
        Reconstruct image
        Args:
            - zs: latent outputs of the encoder 
        """
        i = len(zs) -1
        y = zs[i]
        for module in reversed(self.flows):
            if module.__class__.__name__ == 'Split':
                i = i-1
                y = module.reconstruct(zs[i], y)
            else: 
                y = module.decoder(y)
        return y
            

def preprocess(batch_img, num_bits=8): # Dequantization
    batch_img = torch.clamp(batch_img, 0, 1) * 255
    
    if num_bits < 8:
        batch_img = torch.floor(batch_img / 2 ** (8 - num_bits))
    
    num_bins = 2 ** num_bits
    batch_img = batch_img / num_bins - 0.5
    noise = torch.rand_like(batch_img, device=DEVICE) * (1. / num_bins)
    batch_img += noise

    return batch_img


def postprocess(batch_img, num_bits=8):
    num_bins = 2 ** num_bits
    
    batch_img = torch.floor((batch_img + 0.5) * num_bins)
    batch_img *= 256. / num_bins
    
    batch_img = torch.clamp(batch_img, 0, 255)
    
    return batch_img


class State:
    def __init__(self, path: Path, model, optim):
        self.path = path
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0

    @staticmethod
    def load(path: Path):
        if path.is_file():
            with path.open("rb") as fp:
                state = torch.load(fp, map_location=DEVICE)
                logging.info("Starting back from epoch %d", state.epoch)
                return state
        return State(path, None, None)

    def save(self):
        savepath_tmp = self.path.parent / ("%s.tmp" % self.path.name)
        with savepath_tmp.open("wb") as fp:
            torch.save(self, fp)
        os.rename(savepath_tmp, self.path)


if __name__ == "__main__":
    BATCH_SIZE = 256
    NB_EPOCHS = 100
    LEARNING_RATE = 1e-3
    num_warmup_epochs = 10
    num_bits = 5

    train_dataset = torchvision.datasets.CIFAR10(
        root='./cifar_data/',  
        transform=transforms.ToTensor(),
        train=True,
        download=True,
    )
    
    data_loader = DataLoader(dataset=train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)
    
    K = 16
    L = 3
    n_channels = 3
    img_size = 32
    n_pixel = img_size * img_size * n_channels
    n_bins = 2**num_bits
    layers = []
    for i in range (L):
            layers.append(Squeeze())
            n_channels = n_channels*4
            img_size = img_size//2
            for _ in range(K):
                layers.append(ActNorm2d(n_channels))
                layers.append(Conv2d(n_channels))
                layers.append(AffineCoupling2d(n_channels))
            if i < L-1: 
                layers.append(Split(n_channels))
                n_channels = n_channels//2

    savepath = Path('Glow2d.pth')
    state = State.load(savepath)

    if state.model is None:
        state.model = GlowModel(*layers).to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=LEARNING_RATE)

    model = state.model
    optim = state.optim
    writer = SummaryWriter('runs/section5_test' + time.asctime())
    steps_per_epoch = len(data_loader)
    img_init = torch.randn((8, n_channels, img_size, img_size), device=DEVICE)
    
    for i in range(state.epoch, NB_EPOCHS):
        total_loss = 0.
        for x, _ in tqdm(data_loader, desc=f'Epoch {state.epoch+1}'):
            # Linear warmup of the learning rate to LEARNING_RATE
            lr =  LEARNING_RATE * torch.minimum(torch.tensor(1.), torch.tensor(state.iteration / (num_warmup_epochs * steps_per_epoch)))
            for param_group in optim.param_groups:
                param_group['lr'] = lr
                
            optim.zero_grad()
            x = x.to(DEVICE)
            x = preprocess(x, num_bits=num_bits)

            # If this is the first iteration, we do not update the model's parameters because we initialize the 
            # parameters of the ActNorm layers using the first batch of data
            if state.iteration == 0: 
                with torch.no_grad():
                    _, logdet, logprob = model.encoder(x)
                    state.iteration += 1
                    continue
            else:
                _, logdet, logprob = model.encoder(x)

            loss = (torch.log(torch.tensor(n_bins)) * n_pixel - logprob - logdet)/(torch.log(torch.tensor(2.)) * n_pixel)
            loss = loss.mean()
            total_loss += loss.item()
            loss.backward()
            optim.step()
            
            writer.add_scalar('Loss', loss.item(), state.iteration)
            state.iteration += 1

        # Generate images
        prior = model.top_prior(torch.zeros((n_channels, img_size, img_size), device=DEVICE)).squeeze(0)
        log_std, mean = prior.chunk(2, 0)
        img = img_init * torch.exp(log_std.unsqueeze(0)) + mean.unsqueeze(0)
        img = model.decoder(img) 
        img = postprocess(img, num_bits=num_bits)
        img_grid = vutils.make_grid(img.detach().cpu(), normalize=True, scale_each=True)
        writer.add_image('Generative_images', img_grid, state.epoch)
        
        state.epoch += 1
        state.save() 
        print (f'Loss: {total_loss/len(data_loader)}') 

    writer.close()
                