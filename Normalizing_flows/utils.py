from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import seaborn as sns

sns.set_theme()

from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.utils.tensorboard import SummaryWriter
from sklearn import datasets
import time
import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)


def toDataFrame(t: torch.Tensor, origin: str):
    t = t.cpu().detach().numpy()
    df = pd.DataFrame(data=t, columns=(f"x{ix}" for ix in range(t.shape[1])))
    df['ix'] = df.index * 1.
    df["origin"] = origin
    return df



def scatterplots(samples: List[Tuple[str, torch.Tensor]], col_wrap=4):
    """Draw the 

    Args:
        samples (List[Tuple[str, torch.Tensor]]): The list of samples with their names
        col_wrap (int, optional): Number of columns in the graph. Defaults to 4.

    Raises:
        NotImplementedError: If the dimension of the data is not supported
    """
    # Convert data into pandas dataframes
    _, dim = samples[0][1].shape
    samples = [toDataFrame(sample, name) for name, sample in samples]
    data = pd.concat(samples, ignore_index=True)

    g = sns.FacetGrid(data, height=2, col_wrap=col_wrap, col="origin", sharex=False, sharey=False)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    if dim == 1:
        g.map(sns.kdeplot, "distribution")
        plt.show()
    elif dim == 2:
        g.map(sns.scatterplot, "x0", "x1", alpha=0.6)
        plt.show()
    else:
        raise NotImplementedError()


def iter_data(dataset: Dataset, bs):
    """Infinite iterator on dataset"""
    while True:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        yield from iter(loader)


class MLP(nn.Module):
    """Simple 4 layer MLP"""
    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )
    def forward(self, x):
        return self.net(x)


# --- Modules de base
"""
The following code implements a naive normalizing flow model with only one layer, where the decoder is an affine transformation defined as x = f(z) = z*e^s + t. The model is trained on a dataset containing samples drawn from the distribution N(-1., 1.5). The prior distribution is chosen to be N(0., 1.)
"""
class FlowModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(1))
        self.t = nn.Parameter(torch.zeros(1))

    def decoder(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns [f(x)] and log |det J_f(x)|"""
        # TODO
        batch_size = z.shape[0]
        x = z*torch.exp(self.s) + self.t
        log_Jacob = self.s*torch.ones(batch_size, device=z.device)

        return [x], log_Jacob
        # END TODO

    def encoder(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns [f^{-1}(x)] and log |det J_f^{-1}(x)|"""
        # TODO
        batch_size = x.shape[0]
        z = (x-self.t)*torch.exp(-self.s)
        log_Jacob = -self.s*torch.ones(batch_size, device=x.device)

        return [z], log_Jacob
        # END TODO

    def check(self, x: torch.Tensor):
        with torch.no_grad():
            (y, ), logdetj_1 = self.encoder(x)
            (hat_x, ), logdetj = self.decoder(y)

            # Check inverse
            delta = (x - hat_x).abs().mean()
            assert  delta < 1e-6, f"f(f^{-1}(x)) not equal to x (mean abs. difference = {delta})"

            # Check logdetj
            delta_logdetj = (logdetj_1 + logdetj).abs().mean()
            assert  delta_logdetj < 1e-6, f"log | J | not equal to -log |J^-1| (mean abs. difference = {delta_logdetj})"


class FlowSequential(FlowModule):
    """A container for a succession of flow modules"""
    def __init__(self, *flows: FlowModule):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def apply(self, modules_iter, caller, x):
        m, _ = x.shape
        logdet = torch.zeros(m, device=x.device)
        zs = [x]
        for module in modules_iter:
            gx, _logdet = caller(module, x)
            zs.extend(gx)
            logdet += _logdet

            x = gx[-1]
        return zs, logdet            

    def modulenames(self, decoder=False):
        return [f"L{ix} {module.__class__.__name__}" for ix, module in enumerate(reversed(self.flows) if decoder else self.flows)]

    def encoder(self, x):
        """Returns the sequence (z_K, ..., z_0) and the log det"""
        zs, logdet = self.apply(self.flows, (lambda m, x: m.encoder(x)), x)
        return zs, logdet

    def decoder(self, y):
        """Returns the sequence (z_0, ..., z_K) and the log det"""
        zs, logdet = self.apply(reversed(self.flows), (lambda m, y: m.decoder(y)), y)
        return zs, logdet


class FlowModel(FlowSequential):
    """Flow model = prior + flow modules"""
    def __init__(self, prior: torch.distributions.Distribution, *flows: FlowModule):
        super().__init__(*flows)
        self.prior = prior

    def encoder(self, x):
        # Computes [z_K, ..., z_0] and the sum of log det | f |
        zs, logdet = super().encoder(x)

        # Just computes the prior of $z_0$
        logprob = self.prior.log_prob(zs[-1])

        return logprob, zs, logdet

    def plot(self, data: torch.Tensor, n: int):
        """Plot samples together with ground truth (data)"""
        with torch.no_grad():
            d = data[list(np.random.choice(range(len(data)), n)), :]
            z0 = self.prior.sample((n, ))
            zs, _ = self.decoder(z0)

            data = [("data", d), ("dist", z0)] + list(zip(self.modulenames(decoder=True), zs[1:]))    
            scatterplots(data, col_wrap=4)

# TODO 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class normal_dataset(Dataset):
    def __init__(self, data):
        super(normal_dataset, self).__init__()
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, ix):
        return self.data[ix]

if __name__ == "__main__":
    mu = -1.
    sigma = 1.5
    true_dist = Normal(mu, sigma)
    data = true_dist.sample(torch.Size([500, 1]))
    dataset = normal_dataset(data)
    latent_dist = Normal(0., 1.)
    NB_EPOCHS = 600
    BATCH_SIZE = 60
    model = FlowModel(latent_dist, FlowModule()).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    writer = SummaryWriter('runs/section3' + time.asctime())
    for i in range(NB_EPOCHS):
        total_loss = 0.
        for x in data_loader:
            optim.zero_grad()
            x = x.to(DEVICE)
            logprob, _, logdet = model.encoder(x)
            loss = -logprob.mean() - logdet.mean()
            total_loss += loss.item()
            loss.backward()
            optim.step()
        writer.add_scalar('Loss', total_loss/len(data_loader), i)
        if i%50 == 49:  
            print (f'Loss: {total_loss/len(data_loader)}') 

    writer.close()

    print (f'Estimation of mu: {model.flows[0].t}')
    print (f'Estimation of sigma: {torch.exp(model.flows[0].s)}')


"""
The following code implements the Glow model and trains it on Scikit-learn's moon dataset (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html). The prior distribution is chosen to be a multivariate Gaussian N(0, I_2), where I_2 is the 2x2 identity matrix.
"""
class ActNorm(FlowModule):
    def __init__(self, nb_features):
        super().__init__()
        self.nb_features = nb_features
        self.bias = nn.Parameter(torch.zeros(self.nb_features))
        self.log_scale = nn.Parameter(torch.zeros(self.nb_features))
        self.initialization = False

    def encoder(self, x):
        if not self.initialization:
            with torch.no_grad():
                mean = torch.mean(x, dim=0)
                std = torch.std(x, dim=0)

            self.bias.data.copy_(mean)
            self.log_scale.data.copy_(torch.log(std))
            self.initialization = True
            
        batch_size = x.shape[0]
        z = (x - self.bias)*torch.exp(-self.log_scale) 
        log_Jacob = -torch.sum(self.log_scale)*torch.ones(batch_size, device=x.device)
        
        return [z], log_Jacob
        
    def decoder(self, z):
        batch_size = z.shape[0]
        x = z*torch.exp(self.log_scale) + self.bias
        log_Jacob = torch.sum(self.log_scale)*torch.ones(batch_size, device=z.device)

        return [x], log_Jacob


class AffineCoupling(FlowModule):
    def __init__(self, nb_features, change_first):
        super().__init__()
        self.nb_features = nb_features
        self.change_first = change_first
        self.scale = MLP(self.nb_features//2, self.nb_features//2, self.nb_features*2)
        self.shift = MLP(self.nb_features//2, self.nb_features//2, self.nb_features*2)

    def encoder(self, x):
        batch_size = x.shape[0]
        
        if self.change_first:
            x_change = x[:, :self.nb_features//2]
            x_unchange = x[:, self.nb_features//2:]
        else:
            x_change = x[:, self.nb_features//2:]
            x_unchange = x[:, :self.nb_features//2]

        shift = self.shift(x_unchange)
        scale = self.scale(x_unchange)
        x_change = x_change * torch.exp(scale) + shift
        log_Jacob = torch.sum(scale, dim=-1)*torch.ones(batch_size, device=x.device)
        
        if self.change_first:
            return [torch.cat([x_change, x_unchange], dim=-1)], log_Jacob
        else:
            return [torch.cat([x_unchange, x_change], dim=-1)], log_Jacob
        
    def decoder(self, z):
        batch_size = z.shape[0]

        if self.change_first:
            z_change = z[:, :self.nb_features//2]
            z_unchange = z[:, self.nb_features//2:]
        else:
            z_change = z[:, self.nb_features//2:]
            z_unchange = z[:, :self.nb_features//2]

        shift = self.shift(z_unchange)
        scale = self.scale(z_unchange)
        z_change = (z_change - shift) * torch.exp(-scale)
        log_Jacob = -torch.sum(scale, dim=-1)*torch.ones(batch_size, device=z.device)

        if self.change_first:
            return [torch.cat([z_change, z_unchange], dim=-1)], log_Jacob
        else:
            return [torch.cat([z_unchange, z_change], dim=-1)], log_Jacob
# END TODO

        
class Invertible1x1Conv(FlowModule):
    """ 
    As introduced in Glow paper.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))

        # Decompose Q in P (L + Id) (S + U)
        
        # https://pytorch.org/docs/stable/generated/torch.lu_unpack.html
        P, L, U = torch.lu_unpack(*Q.lu())
        
        # Not optimizated
        self.P = nn.Parameter(P, requires_grad=False)

        # Lower triangular
        self.L = nn.Parameter(L)

        # Diagonal
        self.S = nn.Parameter(U.diag()) # dim_S = torch.Size([dim])

        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # U_prime

    def _assemble_W(self):
        """Computes W from P, L, S and U"""

        # https://pytorch.org/docs/stable/generated/torch.tril.html
        # Excludes the diagonal
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=self.L.device))

        # https://pytorch.org/docs/stable/generated/torch.triu.html
        # Excludes the diagonal
        U = torch.triu(self.U, diagonal=1)

        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def decoder(self, y) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns [f^-1(x)] and log |det J_f^-1(x)|"""
        # TODO
        batch_size = y.shape[0]
        w = self._assemble_W()
        w_inverse = torch.inverse(w)
        x = torch.matmul(y, w_inverse)
        log_Jacob = -torch.sum(torch.log(torch.abs(self.S)))*torch.ones(batch_size, device=x.device)
        
        return [x], log_Jacob
        # END TODO
        
    def encoder(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns [f(x)] and log |det J_f(x)|"""
        # TODO
        batch_size = x.shape[0]
        w = self._assemble_W()
        y = torch.matmul(x, w)
        log_Jacob = torch.sum(torch.log(torch.abs(self.S)))*torch.ones(batch_size, device=x.device)

        return [y], log_Jacob
        # END TODO


# TODO
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

prior = Independent(Normal(torch.tensor([0., 0.], device=DEVICE), torch.tensor([1., 1.], device=DEVICE)), 1)
n_samples = 2000
data, _ = datasets.make_moons(n_samples=n_samples, shuffle=True, noise=0.05, random_state=0)
data = torch.tensor(data).float()


if __name__ == "__main__":
    BATCH_SIZE = 64
    NB_EPOCHS = 5000
    LEARNING_RATE = 1e-4
    dataset = normal_dataset(data)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    layers = (ActNorm(2), Invertible1x1Conv(2), AffineCoupling(2, True), 
              ActNorm(2), Invertible1x1Conv(2), AffineCoupling(2, False),
              ActNorm(2), Invertible1x1Conv(2), AffineCoupling(2, True),
              ActNorm(2), Invertible1x1Conv(2), AffineCoupling(2, False),
              ActNorm(2), Invertible1x1Conv(2), AffineCoupling(2, True),
              ActNorm(2), Invertible1x1Conv(2), AffineCoupling(2, False),
              ActNorm(2), Invertible1x1Conv(2), AffineCoupling(2, True),
              ActNorm(2), Invertible1x1Conv(2), AffineCoupling(2, False),
             )

    savepath = Path('GlowModel.pth')
    state = State.load(savepath)

    if state.model is None:
        state.model = FlowModel(prior, *layers).to(DEVICE)
        state.optim = torch.optim.Adam(state.model.parameters(), lr=LEARNING_RATE)

    model = state.model
    optim = state.optim
    writer = SummaryWriter('runs/section4-' + time.asctime())
    for i in range(state.epoch, NB_EPOCHS):
        total_loss = 0.
        for x in data_loader:
            optim.zero_grad()
            x = x.to(DEVICE)
            logprob, _, logdet = model.encoder(x)
            loss = -logprob.mean() - logdet.mean()
            total_loss += loss.item()
            loss.backward()
            optim.step()
            state.iteration += 1
        writer.add_scalar('Loss', total_loss/len(data_loader), i)
        state.epoch += 1
        state.save()
        if i%50 == 49:  
            print (f'Loss: {total_loss/len(data_loader)}') 

    writer.close()
# END TODO