# TODO
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import torchvision.utils as vutils
import logging
logging.basicConfig(level=logging.INFO) 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
In this practical session, I implement a variational autoencoder model under the following assumptions:
    - The prior latent distribution is N(0, I).
    - The distributions of the encoder and decoder are Gaussians with diagonal covariances. 
    
The model is then trained on the MNIST dataset.
"""

class Encoder(nn.Module):
    def __init__(self, dim_in, dim_latent):
        super(Encoder, self).__init__()
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.backbone = nn.Sequential(nn.Linear(self.dim_in, 500),
                                      nn.ReLU(),
                                     )
        self.last_mean_layer = nn.Linear(500, self.dim_latent)
        self.last_logvar_layer = nn.Linear(500, self.dim_latent)
        
    def forward(self, x):
        backbone_output = self.backbone(x)
        mean = self.last_mean_layer(backbone_output)
        log_var = self.last_logvar_layer(backbone_output)

        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, dim_out, dim_latent):
        super(Decoder, self).__init__()
        self.dim_out = dim_out
        self.dim_latent = dim_latent
        self.backbone = nn.Sequential(nn.Linear(self.dim_latent, 500),
                                      nn.ReLU(),
                                     )
        self.last_mean_layers = nn.Sequential(nn.Linear(500, self.dim_out),
                                              nn.Sigmoid(),
                                             )
        self.last_logvar_layers = nn.Sequential(nn.Linear(500, self.dim_out),
                                                nn.Sigmoid(),
                                               )
        
    def forward(self, z):
        backbone_output = self.backbone(z)
        mean = self.last_mean_layers(backbone_output)
        log_var = self.last_logvar_layers(backbone_output)

        return mean, log_var
        
class VAE(nn.Module):
    def __init__(self, dim_in, dim_latent):
        super(VAE, self).__init__()
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.encoder = Encoder(self.dim_in, self.dim_latent)
        self.decoder = Decoder(self.dim_in, self.dim_latent)
        
    def sampling(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        normal = Normal(mean, std) # dim_mean = dim_std = batch x dim_latent/dim_in 
        dist = Independent(normal, 1)

        return dist

BATCH_SIZE = 100
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: torch.flatten(img)),
])
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


class State:
    """
    This class serves for checkpointing purpose.
    """
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


def train_loop(data, state):
    total_loss = 0.
    
    model = state.model
    optimizer = state.optim
    
    for img, _ in tqdm(data, desc=f'Epoch {state.epoch+1}'):
        optimizer.zero_grad()

        img = img.to(DEVICE)
        mean_latent, logvar_latent = model.encoder(img) 
        dist_latent = model.sampling(mean_latent, logvar_latent)
        z = dist_latent.rsample()
        mean_out, logvar_out = model.decoder(z)
        dist_out = model.sampling(mean_out, logvar_out)
        
        kl = 0.5*(-logvar_latent + torch.exp(logvar_latent) + mean_latent**2)
        kl = kl.sum(dim=-1)
        reconstruction_loss = dist_out.log_prob(img)
        loss = (kl - reconstruction_loss).mean()
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        state.iteration +=1

    return total_loss/len(data)


def test_loop(data, state):
    total_loss = 0.
    
    model = state.model
    
    for img, _ in data:
        with torch.no_grad():
            img = img.to(DEVICE)
            mean_latent, logvar_latent = model.encoder(img) 
            dist_latent = model.sampling(mean_latent, logvar_latent)
            z = dist_latent.rsample()
            mean_out, logvar_out = model.decoder(z)
            dist_out = model.sampling(mean_out, logvar_out)
            
            kl = 0.5*(-logvar_latent + torch.exp(logvar_latent) + mean_latent**2)
            kl = kl.sum(dim=-1)
            reconstruction_loss = dist_out.log_prob(img)
            loss = (kl - reconstruction_loss).mean()
    
            total_loss += loss.item()

    return total_loss/len(data)


def add_logging(state, dataloader, writer, prefix=''):
    """
    This function visualizes the original and reconstructed images.
    """
    
    img, label = next(iter(dataloader))
    img = img.to(DEVICE)
    model = state.model
    
    with torch.no_grad():
        mean_latent, logvar_latent = model.encoder(img) 
        dist_latent = model.sampling(mean_latent, logvar_latent)
        z = dist_latent.rsample()
        mean_out, logvar_out = model.decoder(z[:10])

        img_grid_orig = vutils.make_grid(img[:10].reshape(10, 1, 28, 28).cpu(), nrow=10, normalize=True, scale_each=True)
        img_grid_reconstruct = vutils.make_grid(mean_out.reshape(10, 1, 28, 28).cpu(), nrow=10, normalize=True, scale_each=True)

        writer.add_image(f'Original Images/{prefix}', img_grid_orig, state.epoch)
        writer.add_image(f'Reconstructed Images/{prefix}', img_grid_reconstruct, state.epoch)
        writer.add_embedding(mean_latent, label.tolist(), img.reshape(-1, 1, 28, 28), state.epoch, f'Latent_space/{prefix}')
    

def run(train_loader, test_loader, model, nb_epochs, lr):
    writer = SummaryWriter('runs/vae-' + time.asctime())

    savepath = Path('vae.pth')
    state = State.load(savepath)
    if state.model is None:
        state.model =  model.to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=lr)

    for epoch in range(state.epoch, nb_epochs):
        loss_train = train_loop(train_loader, state)
        loss_test = test_loop(test_loader, state)
        
        writer.add_scalar('Loss_train', loss_train, state.epoch)
        writer.add_scalar('Loss_test', loss_test, state.epoch)
        
        print (f'Train loss: {loss_train} \t Test loss: {loss_test}')
        
        state.epoch = epoch + 1
        state.save()

        if state.epoch % 5 == 0:
            add_logging(state, train_loader, writer, prefix='Train')
            add_logging(state, test_loader, writer, prefix='Test')


NB_EPOCHS = 40
LEARNING_RATE = 1e-3
model = VAE(784, 2)
run(train_loader, test_loader, model, NB_EPOCHS, LEARNING_RATE) 
# END TODO