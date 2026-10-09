"""Independent MMW-style Euler adaptation, not original-paper replication.

Two sigmoid hidden layers; consumption share and exp(h) heads. Full joint
histogram replaces the author's finite-agent vector. Model B shocks are drawn
twice independently; aggregate shocks are common within each simulated economy.
Complementarity follows the paper's 1-h convention, not the opposite-sign
R_mu in the public notebook. No existing frozen module is modified.
"""
import torch
from torch import nn
from jedc_distribution_v1 import IntervalEconomy, rebin, midpoint_grid
import numpy as np


def official_mu(path, edges):
    with np.load(path) as d:
        mass = d['mu'].astype(float)
        if not np.allclose(d['grid'], midpoint_grid(np.arange(1001)*.1)):
            raise ValueError('official initial grid differs')
    mass *= (np.array([.1, .9])/mass.sum(1))[:, None]
    return rebin(mass, np.arange(1001)*.1, edges)


def complementarity(savings, consumption, h):
    # Eq. (44): 1-c/w = s/w, where w=s+c. At s=0, h<=1.
    # s/c has the same zero set but changes the training loss and gradients.
    a, b = savings/(savings+consumption), 1-h
    # vector_norm has a well-defined zero subgradient in PyTorch.
    return torch.linalg.vector_norm(torch.stack((a, b), -1), dim=-1)-a-b


class MMWNetwork(nn.Module):
    def __init__(self, grid, asset_max, hidden, share_bias):
        super().__init__()
        self.asset_max = float(asset_max)
        self.register_buffer('grid', grid.clone())
        self.register_buffer('share_bias', grid.new_tensor(share_bias))
        self.first = nn.Linear(2*len(grid)+5, hidden)
        self.second = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, 2)
        for layer in (self.first, self.second, self.output):
            nn.init.trunc_normal_(layer.weight, std=.01, a=-.02, b=.02)
            nn.init.zeros_(layer.bias)

    def forward(self, mu, z, cash, employment):
        K = (mu*self.grid).sum((1, 2))/self.asset_max
        macro = torch.cat((2*mu.flatten(1)-1,
                           torch.nn.functional.one_hot(z.long(), 2).to(mu.dtype),
                           K[:, None]), 1)
        # Model B cash-on-hand takes the place of the author's w.
        micro = torch.stack((2*cash/self.asset_max-1,
                             2*employment.to(mu.dtype)-1), -1)
        split = macro.shape[1]
        value = (torch.nn.functional.linear(macro, self.first.weight[:, :split], self.first.bias)
                 [:, None, None, :] +
                 torch.nn.functional.linear(micro, self.first.weight[:, split:]))
        value = torch.sigmoid(self.second(torch.sigmoid(value)))
        value = self.output(value)
        return torch.sigmoid(value[..., 0]+self.share_bias), torch.exp(value[..., 1])


class MMWEconomy(IntervalEconomy):
    def decisions(self, network, mu, z, assets, employment):
        cash = self.cash(mu, z, assets, employment)
        share, h = network(mu, z, cash, employment)
        # Same 500 upper cap and consumption floor as the existing implementation.
        upper = torch.minimum((cash-1e-4).clamp_min(0), cash.new_full(cash.shape, self.asset_max))
        savings = torch.minimum((cash*(1-share)).clamp_min(0), upper)
        return savings, cash-savings, h

    def policy(self, network, mu, z, assets, employment):
        s, c, _ = self.decisions(network, mu, z, assets, employment)
        return s, c

    def advance(self, network, mu, z, generator):
        """One common aggregate realization per batch economy; no BPTT."""
        with torch.no_grad():
            a, e = self.edge_queries(len(mu))
            s, _ = self.policy(network, mu, z, a, e)
            next_z = (torch.rand(len(mu), device=mu.device, generator=generator) >=
                      self.aggregate_probability(z, 0)).long()
            alternatives = torch.stack([self.push_distribution(mu, z, s, nz) for nz in (0, 1)], 1)
            next_mu = alternatives[torch.arange(len(mu), device=mu.device), next_z]
            # Repeated float32 transport can accumulate roundoff over the
            # 200,000 online advances in the frozen 100k-update configuration.
            # Correct only the scalar mass drift of each simulated economy;
            # reject a material one-step discrepancy instead of hiding it.
            total = next_mu.sum((1, 2))
            if not bool(torch.isfinite(total).all()) or bool((total <= 0).any()):
                raise FloatingPointError('nonfinite or nonpositive online transport mass')
            correction = float((total-1).abs().max())
            if correction > 1e-5:
                raise FloatingPointError(f'online one-step mass error {correction:.9g} exceeds 1e-5')
            self.last_online_mass_correction = correction
            next_mu = next_mu / total[:, None, None]
        return next_mu.detach(), next_z

    def objective(self, network, mu, z, generator, estimator='double_draw', mass_weight=.5,
                  high_multiplier=4., high_cutoff=20.):
        a, e = self.queries(len(mu))
        saving, c, h = self.decisions(network, mu, z, a, e)
        ea, ee = self.edge_queries(len(mu))
        se, _ = self.policy(network, mu, z, ea, ee)
        next_mus = [self.push_distribution(mu, z, se, nz) for nz in (0, 1)]
        if estimator == 'double_draw':
            draws = []
            for _ in range(2):
                # Same aggregate draw for all household queries in an economy.
                nz = (torch.rand(len(mu), device=mu.device, generator=generator) >=
                      self.aggregate_probability(z, 0)).long()
                nm = torch.stack(next_mus, 1)[torch.arange(len(mu), device=mu.device), nz]
                rows = z[:, None, None]*2+e.long()
                p0 = self.transition[rows, (nz*2)[:, None, None]]
                p1 = self.transition[rows, (nz*2+1)[:, None, None]]
                ne = (torch.rand(saving.shape, device=mu.device, generator=generator) >= p0/(p0+p1)).long()
                _, cp = self.policy(network, nm, nz, saving, ne)
                _, _, r, _ = self.prices(nm, nz)
                draws.append(self.beta*(1+r[:, None, None])*c/cp-h)
            euler = draws[0]*draws[1]  # unbiased conditional squared residual; may be negative
        elif estimator == 'exact':
            expected = torch.zeros_like(c)
            for nz, nm in enumerate(next_mus):
                zz = torch.full_like(z, nz)
                _, _, r, _ = self.prices(nm, zz)
                for ne in (0, 1):
                    _, cp = self.policy(network, nm, zz, saving, torch.full_like(e, ne))
                    p = self.transition[z[:, None, None]*2+e.long(), nz*2+ne]
                    expected = expected+p*(1+r[:, None, None])*c/cp
            euler = (self.beta*expected-h).square()
        else:
            raise ValueError('unknown estimator')
        fb = complementarity(saving, c, h)
        weight = mass_weight*mu+(1-mass_weight)/(2*len(self.grid))
        multiplier = 1+(high_multiplier-1)*(self.grid>=high_cutoff).to(mu.dtype)
        weight = weight*multiplier
        weight = weight/weight.sum((1, 2), keepdim=True)
        fb_loss = (weight*fb.square()).sum((1, 2)).mean()
        euler_loss = (weight*euler).sum((1, 2)).mean()
        loss = fb_loss+euler_loss
        return loss, dict(loss=loss, fb=fb_loss, euler_estimate=euler_loss,
                          K=(mu*self.grid).sum((1, 2)).mean(),
                          max_mass_error=torch.stack([(nm.sum((1, 2))-1).abs().max() for nm in next_mus]).max(),
                          min_consumption=c.min(), max_h=h.max())
