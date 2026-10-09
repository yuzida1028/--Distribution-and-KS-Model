"""Atom + uniform positive intervals; shared, versioned Model B transport.

The push uses an affine saving function on each source interval. It reprojects
positive output onto uniform destination intervals. This preserves mass and the
exact zero atom, but midpoint capital has a recorded discretization error.
No positive saving is deposited at the zero atom.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ks_model import KSEconomy

VERSION = 'atom_uniform_intervals_v1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def make_edges(fine=False, upper=500.):
    step = .05 if fine else .1
    tail = .5 if fine else 1.
    return np.r_[np.arange(round(100 / step) + 1) * step,
                 np.arange(100 + tail, upper + tail / 2, tail)]

def midpoint_grid(edges):
    return np.r_[0., (edges[:-1] + edges[1:]) / 2]

def rebin(mu, source_edges, target_edges):
    """Exact interval overlap conversion, no conversion of positives to atom."""
    out = np.empty((2, len(target_edges)))
    out[:, 0] = mu[:, 0]
    for e in range(2):
        cdf = np.r_[0., np.cumsum(mu[e, 1:])]
        out[e, 1:] = np.diff(np.interp(target_edges, source_edges, cdf))
    if target_edges[-1] < source_edges[-1] and np.sum(mu[:, 1:][:, source_edges[1:] > target_edges[-1]]) > 1e-12:
        raise ValueError('conversion would truncate positive mass')
    return out

def push_assets_numpy(mass, saving_edges, edges):
    """Push one conditional employment row without employment mixing."""
    if not np.isfinite(saving_edges).all() or saving_edges.min() < 0 or saving_edges.max() > edges[-1] + 1e-10:
        raise ValueError('saving outside asset domain; no silent clipping')
    out = np.zeros_like(mass)
    atom_choice = saving_edges[0]
    if atom_choice == 0: out[0] += mass[0]
    else: out[min(np.searchsorted(edges, atom_choice, side='right'), len(edges)-1)] += mass[0]
    a = np.minimum(saving_edges[:-1], saving_edges[1:])
    b = np.maximum(saving_edges[:-1], saving_edges[1:])
    flat = b - a <= 1e-12
    zero = flat & (b == 0)
    out[0] += mass[1:][zero].sum()
    positive_flat = flat & ~zero
    np.add.at(out, np.minimum(np.searchsorted(edges, a[positive_flat], side='right'), len(edges)-1), mass[1:][positive_flat])
    active = np.flatnonzero(~flat & (mass[1:] > 0))
    if len(active):
        lo = np.clip(np.searchsorted(edges, a[active], side='right')-1, 0, len(edges)-2)
        hi = np.clip(np.searchsorted(edges, b[active], side='left')-1, 0, len(edges)-2)
        count = hi-lo+1
        source = np.repeat(active, count)
        start = np.repeat(np.cumsum(count)-count, count)
        dest = np.repeat(lo, count) + np.arange(len(source))-start
        overlap = np.maximum(0., np.minimum(b[source], edges[dest+1])-np.maximum(a[source], edges[dest]))
        np.add.at(out, dest+1, mass[1:][source]*overlap/(b[source]-a[source]))
    return out

def transition_numpy(mu, z, next_z, saving_edges, edges, transition):
    moved = np.stack([push_assets_numpy(mu[e], saving_edges[e], edges) for e in range(2)])
    block = transition[2*z:2*z+2, 2*next_z:2*next_z+2]
    p = block.sum(1)
    if not np.allclose(p, p[0], atol=1e-7): raise ValueError('aggregate probabilities differ')
    return (block / p[:, None]).T @ moved

def push_assets_torch(mass, saving_edges, edges):
    """Sparse overlaps; differentiable on each fixed interval-intersection set."""
    if not bool(torch.isfinite(saving_edges).all()) or bool((saving_edges < 0).any()) or bool((saving_edges > edges[-1]+1e-6).any()):
        raise ValueError('saving outside interval domain')
    out = torch.zeros_like(mass)
    atom_dest = torch.searchsorted(edges, saving_edges[:1].detach(), right=True).clamp_max(len(edges)-1)
    atom_dest = torch.where(saving_edges[:1].detach() == 0, torch.zeros_like(atom_dest), atom_dest)
    out = out.scatter_add(0, atom_dest, mass[:1])
    a = torch.minimum(saving_edges[:-1], saving_edges[1:])
    b = torch.maximum(saving_edges[:-1], saving_edges[1:])
    flat = (b-a).detach() <= 1e-12
    flat_ids = torch.nonzero(flat).flatten()
    dest = torch.searchsorted(edges, a[flat_ids].detach(), right=True).clamp_max(len(edges)-1)
    dest = torch.where(b[flat_ids].detach() == 0, torch.zeros_like(dest), dest)
    out = out.scatter_add(0, dest, mass[1:][flat_ids])
    # Keep zero-mass intervals in the graph: their transport can acquire mass
    # during the continuation distribution, and derivatives of mass matter.
    ids = torch.nonzero(~flat).flatten()
    if ids.numel():
        lo = (torch.searchsorted(edges, a[ids].detach(), right=True)-1).clamp(0,len(edges)-2)
        hi = (torch.searchsorted(edges, b[ids].detach(), right=False)-1).clamp(0,len(edges)-2)
        count = hi-lo+1
        source = torch.repeat_interleave(ids, count)
        dest = torch.repeat_interleave(lo, count) + torch.arange(len(source),device=mass.device) - torch.repeat_interleave(torch.cumsum(count,0)-count,count)
        overlap = (torch.minimum(b[source], edges[dest+1])-torch.maximum(a[source],edges[dest])).clamp_min(0)
        out = out.scatter_add(0, dest+1, mass[1:][source]*overlap/(b[source]-a[source]))
    return out

class IntervalEconomy(KSEconomy):
    def __init__(self, calibration_path, edges, device, **kwargs):
        self.edges = edges.to(device)
        grid = torch.cat((edges.new_zeros(1), (edges[:-1]+edges[1:])/2))
        super().__init__(calibration_path, grid, device, **kwargs)
        self.asset_max = float(edges[-1])

    def edge_queries(self, batch):
        assets = self.edges[None,None,:].expand(batch,2,-1)
        employment = torch.arange(2,device=self.grid.device)[None,:,None].expand_as(assets)
        return assets, employment

    def push_distribution(self, mu, z, saving_edges, next_z):
        """Here saving_edges is queried at EDGES, never at midpoint grid."""
        moved = torch.stack([torch.stack([push_assets_torch(mu[i,e],saving_edges[i,e],self.edges) for e in range(2)]) for i in range(len(mu))])
        p = self.aggregate_probability(z, next_z)
        rows = []
        for ne in range(2):
            rows.append(sum(moved[:,e,:]*(self.transition[z.long()*2+e,next_z*2+ne]/p)[:,None] for e in range(2)))
        return torch.stack(rows,1)

    def euler_terms(self, network, mu, z, policy_fn=None):
        policy_fn = self.policy if policy_fn is None else policy_fn
        assets, employment = self.queries(len(mu))
        savings, consumption = policy_fn(network,mu,z,assets,employment)
        edge_assets, edge_employment = self.edge_queries(len(mu))
        saving_edges, _ = policy_fn(network,mu,z,edge_assets,edge_employment)
        expected = torch.zeros_like(consumption)
        next_masses = []
        for nz in range(2):
            nm = self.push_distribution(mu,z,saving_edges,nz)
            next_masses.append(nm)
            zz = torch.full_like(z,nz)
            _,_,r,_ = self.prices(nm,zz)
            for ne in range(2):
                _, c = policy_fn(network,nm,zz,savings,torch.full_like(employment,ne))
                p = torch.stack([self.transition[z.long()*2+e,nz*2+ne] for e in range(2)],1)
                expected = expected + p[:,:,None]*(1+r[:,None,None])/c
        return savings,consumption,1-self.beta*expected*consumption,next_masses

class YoungReference:
    """Verified column order and in-domain linear table interpolation."""
    def __init__(self, path):
        self.table = np.loadtxt(path).reshape(150,4,10)
        self.assets = self.table[:,0,0]
        self.capitals = self.table[0,:,1]
        if not np.all(np.diff(self.assets)>0) or not np.all(np.diff(self.capitals)>0): raise ValueError('table grids')
        if not np.allclose(self.table[:,:,0],self.assets[:,None]) or not np.allclose(self.table[:,:,1],self.capitals[None,:]): raise ValueError('table layout')
        self.cache = {}

    def choices(self, assets, K, z):
        if not self.capitals[0] <= K <= self.capitals[-1]: raise ValueError('K %.12g outside Young [%g,%g]' % (K,self.capitals[0],self.capitals[-1]))
        if assets.min()<self.assets[0] or assets.max()>self.assets[-1]: raise ValueError('assets outside Young table')
        key = (z, assets.shape, assets.tobytes())
        if key not in self.cache:
            columns = (9,7) if z==0 else (8,6)
            self.cache[key] = np.asarray([[np.interp(assets,self.assets,self.table[:,j,c]) for j in range(4)] for c in columns])
            if len(self.cache)>16: self.cache.pop(next(iter(self.cache)))
        j = np.searchsorted(self.capitals,K,side='right').clip(1,3)
        f = (K-self.capitals[j-1])/(self.capitals[j]-self.capitals[j-1])
        return self.cache[key][:,j-1]*(1-f)+self.cache[key][:,j]*f

    def adapter(self, economy):
        def policy(_network, mu, z, assets, employment):
            result = np.empty(assets.shape, dtype=np.float64)
            for i in range(len(mu)):
                K = float((mu[i]*economy.grid).sum())
                for row in range(assets.shape[1]):
                    a = assets[i,row].detach().cpu().numpy()
                    e = employment[i,row].detach().cpu().numpy().astype(int)
                    both = self.choices(a,K,int(z[i]))
                    result[i,row] = both[e,np.arange(len(a))]
            s = torch.as_tensor(result,dtype=mu.dtype,device=mu.device)
            return s,economy.cash(mu,z,assets,employment)-s
        return policy
